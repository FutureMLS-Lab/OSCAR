"""
MLA / NSA latent KV "fake-quant" INT2 pool.

Storage layout is identical to ``MLATokenToKVPool`` / ``NSATokenToKVPool``
(concatenated ``[c_kv | k_pe]`` in BF16/FP16 for the main ``kv_buffer``;
NSA additionally keeps its own ``index_k_with_scale_buffer``). On every
``set_mla_kv_buffer`` / ``set_kv_buffer`` we apply an optional rotation
to ``c_kv``, quantize it groupwise to INT2, immediately dequantize, then
inverse-rotate before storing. ``k_pe`` and NSA's index buffer are
unchanged.

This is a quality-only measurement: it shows the loss of running MLA
attention against a latent KV cache that has been forced through INT2,
without requiring a full int2-storage pool / FlashMLA dequant rewrite.
Memory footprint is unchanged from BF16.

Used for OSCAR INT2 latent KV experiments on GLM-5.1-FP8 / DeepseekV2
style MLA models (and the NSA-classified ``GlmMoeDsaForCausalLM``).

BF16 windows
------------
The same two BF16 windows ``UnifiedInt2HPKVPool`` gives a per-head model — a
sink of the first ``P`` positions and a ring of the last ``R`` positions of
every sequence — carved out of the shared latent, and on by default at the
project floor of 64/256 (``SGLANG_MIXED_KV_PREFIX_TOKENS`` /
``SGLANG_MIXED_KV_RECENT_TOKENS`` to raise them). No INT2 KV configuration
2-bits the attention sink or the newest tokens; this pool used to, which made
every NSA/MLA model the exception.

Because this pool fake-quantizes into a plain float cache, a "window" is not
a second arena: every token already owns a BF16 row, so a windowed token is
simply one that was never pushed through the INT2 round trip. A token
*leaving* the recent window is re-read from its own row and quantized in
place, which is what keeps the window a window instead of "the whole
generation stays BF16". See :meth:`_Int2HPMixin._init_latent_windows`.
"""

from __future__ import annotations

import atexit
import logging
import math
import os
from typing import Dict, List, Optional

import torch

from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool, NSATokenToKVPool

logger = logging.getLogger(__name__)

# ── Lloyd-Max INT2 constants (N(0,1) optimal, 4-level) ──────────────────────
# Decision boundaries between levels 0-1, 1-2, 2-3.
_LM_THRESHOLDS = (-0.9810652732849121, 0.0, 0.9810652732849121)
# Centroids of each level.
_LM_CENTROIDS = (-1.5095585584640503, -0.4527800381183624, 0.4527800381183624, 1.5095585584640503)
_LM_SPAN = _LM_CENTROIDS[3] - _LM_CENTROIDS[0]   # ≈ 3.019
_LM_RATIO = 1.16   # empirical scale to keep uniform dequant stable in-context

# ── BF16 window floor ───────────────────────────────────────────────────────
# Every INT2 KV configuration keeps the attention-sink tokens at the start of
# the sequence and the newest tokens in BF16; 2-bit'ing either degrades
# generation. 64/256 is the floor, not a tuning knob -- models that need more
# raise the recent window (Gemma-4 and Qwen3-8B use 512). These are the
# defaults for the latent path because the env vars' own defaults (32/128) are
# the per-head pool's older, lower pair.
DEFAULT_LATENT_SINK_TOKENS = 64
DEFAULT_LATENT_RECENT_TOKENS = 256


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_or_make_rotations(
    rotation_path: str,
    layer_num: int,
    start_layer: int,
    kv_lora_rank: int,
    device: str,
    dtype: torch.dtype,
    rotation_layer_ids: Optional[List[int]] = None,
) -> Optional[Dict[int, torch.Tensor]]:
    """Load per-layer rotations from a directory or generate Hadamard.

    ``rotation_path``:
      * ``""`` / missing → return ``None`` (no rotation).
      * ``"hadamard"`` → generate one random-sign 512×512 Hadamard-style
        orthogonal matrix shared across layers (no calibration data).
      * any existing directory → load ``layer_<i>.pt`` for ``i`` in the
        effective layer range; missing layers fall back to identity.

    ``rotation_layer_ids`` exists for the **hybrid** case. A mambaish model's
    full-attention layers are a sparse subset of the model's layers (e.g.
    3, 7, 11, ...), and the inner pool behind ``HybridLinearKVPool`` is indexed
    0..N-1 after the layer-id remap — so keying the file lookup off the local
    index would load ``layer_0.pt`` for what is really layer 3. Pass the global
    ids in full-attention order and local index ``j`` loads
    ``layer_<rotation_layer_ids[j]>.pt``. ``UnifiedInt2HPKVPool`` already does
    exactly this; this is the MLA-side equivalent.
    """
    if not rotation_path:
        return None

    d = kv_lora_rank

    if rotation_path == "hadamard":
        # Random-sign Hadamard (same matrix for all layers, reproducible).
        rng = torch.Generator()
        rng.manual_seed(0)
        signs = torch.where(
            torch.rand(d, generator=rng) > 0.5,
            torch.ones(d),
            -torch.ones(d),
        ).view(d, 1)
        # QR of a random Gaussian gives a Haar-uniform orthogonal matrix.
        base = torch.randn(d, d, generator=rng)
        R, _ = torch.linalg.qr(base)
        R = (R * signs).to(dtype=dtype, device=device).contiguous()
        return {i: R for i in range(start_layer, start_layer + layer_num)}

    if not os.path.isdir(rotation_path):
        logger.info("[MLAInt2] rotation_path=%s not found — no rotation", rotation_path)
        return None

    if rotation_layer_ids is not None and len(rotation_layer_ids) != layer_num:
        raise ValueError(
            f"rotation_layer_ids has {len(rotation_layer_ids)} entries but "
            f"layer_num={layer_num}"
        )

    rotations: Dict[int, torch.Tensor] = {}
    for j in range(layer_num):
        i = start_layer + j
        # File id: the global layer for a hybrid inner pool, else the pool's own
        # index. Getting this wrong loads a valid-but-wrong rotation, which does
        # not raise -- it just quantizes in the wrong frame.
        file_id = rotation_layer_ids[j] if rotation_layer_ids is not None else i
        p = os.path.join(rotation_path, f"layer_{file_id}.pt")
        if os.path.exists(p):
            rotations[i] = torch.load(p, map_location="cpu").to(dtype=dtype, device=device).contiguous()
        else:
            rotations[i] = torch.eye(d, dtype=dtype, device=device)
    # A rotation is only invertible by its transpose while it is orthogonal, and
    # 2-bit latents amplify any drift instead of absorbing it. Report the drift
    # of the first layer so a rotation that a dtype or a bad fit has quietly
    # de-orthogonalized is visible in the log rather than in the score.
    _r0 = rotations[start_layer].to(torch.float32)
    logger.info(
        "[MLAInt2] loaded %d rotations from %s in %s (layer %d |R R^T - I|max=%.2e)",
        len(rotations),
        rotation_path,
        dtype,
        start_layer,
        (_r0 @ _r0.T - torch.eye(d, device=_r0.device)).abs().max().item(),
    )
    return rotations


def _load_hp_subspaces(
    hp_subspace_path: str,
    layer_num: int,
    start_layer: int,
    device: str,
    dtype: torch.dtype,
) -> Optional[Dict[int, torch.Tensor]]:
    """Load per-layer HP-subspace matrices (``layer_<i>.pt``, each [k, kv_lora_rank]
    with orthonormal rows = top-k most sensitive latent directions).

    Returns ``None`` if no path given. Missing layers are skipped (those layers
    just get the plain rotation+INT2 path).
    """
    if not hp_subspace_path or not os.path.isdir(hp_subspace_path):
        return None
    subspaces: Dict[int, torch.Tensor] = {}
    for i in range(start_layer, start_layer + layer_num):
        p = os.path.join(hp_subspace_path, f"layer_{i}.pt")
        if os.path.exists(p):
            subspaces[i] = (
                torch.load(p, map_location="cpu").to(dtype=torch.float32, device=device).contiguous()
            )
    logger.info(
        "[MLAInt2] loaded HP subspaces for %d layers from %s (k=%s)",
        len(subspaces),
        hp_subspace_path,
        next(iter(subspaces.values())).shape[0] if subspaces else 0,
    )
    return subspaces or None


def _fake_quant_int2_groupwise(
    x: torch.Tensor,
    group_size: int,
    lloyd_max: bool = False,
) -> torch.Tensor:
    """INT2 groupwise quant-then-dequant on the last dim.

    Operates in groups of ``group_size`` elements along the last axis.
    When ``lloyd_max=True``, uses MSE-optimal Lloyd-Max buckets for N(0,1)
    instead of uniform min-max affine quantization.
    """
    orig_shape = x.shape
    x = x.reshape(-1, group_size).to(torch.float32)

    if lloyd_max:
        # Per-group mean/std normalization → bucketize with LM thresholds → dequant.
        mean = x.mean(dim=-1, keepdim=True)
        diff = x - mean
        std = (diff.pow(2).mean(dim=-1, keepdim=True) + 1e-8).sqrt()
        z = diff / std

        t0, t1, t2 = _LM_THRESHOLDS
        q = ((z >= t0).to(torch.uint8)
             + (z >= t1).to(torch.uint8)
             + (z >= t2).to(torch.uint8))  # 0..3

        # Dequant: LM centroids rescaled and unshifted back to original space.
        uniform_scale = (_LM_SPAN / 3.0) * _LM_RATIO * std
        uniform_zero = -_LM_CENTROIDS[0] / (_LM_SPAN / 3.0) - mean / uniform_scale
        x_deq = (q.to(torch.float32) - uniform_zero) * uniform_scale
    else:
        # Affine per-group min-max (original implementation).
        x_min = x.amin(dim=-1, keepdim=True)
        x_max = x.amax(dim=-1, keepdim=True)
        scale = torch.where(
            (x_max - x_min).abs() > 1e-8,
            (x_max - x_min) / 3.0,
            torch.ones_like(x_min),
        )
        q = (x - x_min) / scale
        q = q.round().clamp_(0.0, 3.0)
        x_deq = q * scale + x_min

    return x_deq.reshape(orig_shape)


def _real_kernel_enabled() -> bool:
    """Use the packed-INT2 Triton kernels instead of the torch fake-quant.

    The arithmetic is the same (verified to relL2 ~4e-06 on real GLM-5.2 c_kv
    through the rotate->quantize->unrotate path), so accuracy should be
    unchanged; this exercises the kernel in the real serving path. It does not
    yet save memory -- the codes are unpacked straight back into the BF16 pool,
    because the MLA attention path has no INT2 read support.
    """
    # ``from sglang.srt import environ as envs`` binds the *module*, but the
    # env descriptors live on the ``Envs`` class and the module defines no
    # ``__getattr__`` -- so the attribute lookup raised AttributeError, the
    # bare ``except`` swallowed it, and this returned False for every possible
    # value of the variable including "1". The kernel was unreachable, not
    # merely disabled, and every GLM-5.2 number in this branch came from
    # ``_fake_quant_int2_groupwise``. Two sites used this idiom against 128
    # files using the correct ``from sglang.srt.environ import envs``.
    #
    # The bare except is also gone: it existed to tolerate an import cycle,
    # but it is exactly what hid this for the life of the flag. If the import
    # ever does fail, failing loudly is the lesser harm.
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_OSCAR_MLA_KV_REAL_KERNEL.get())


def _quant_requested() -> bool:
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_OSCAR_MLA_KV_ROTATION_PATH.get())


# ── mixin ────────────────────────────────────────────────────────────────────

class _Int2HPMixin:
    """Helpers for MLA/NSA INT2 fake-quant pools.

    Mixed into the actual pool class which must provide ``kv_lora_rank``,
    ``qk_rope_head_dim``, ``dtype``, ``layer_num``, ``start_layer``,
    ``device``.
    """

    def _init_int2(
        self,
        rotation_path: str,
        group_size: int,
        dump_c_kv_dir: str,
        dump_max_tokens_per_layer: int,
        lloyd_max: bool,
        hp_subspace_path: str = "",
        rotation_layer_ids: Optional[List[int]] = None,
    ) -> None:
        if self.kv_lora_rank % group_size != 0:
            raise ValueError(
                f"group_size={group_size} must divide kv_lora_rank={self.kv_lora_rank}"
            )
        self._group_size = group_size
        self._lloyd_max = lloyd_max
        # Rotations and latents must live in a dtype that can actually hold them,
        # which is not always the pool's store dtype. sglang defaults a
        # DeepSeek-DSA model's KV cache to fp8_e4m3 on SM100+ (bfloat16 only on
        # Hopper and below), and a 512x512 orthogonal matrix has entries around
        # 1/sqrt(512) = 0.044 -- below fp8_e4m3's smallest normal (0.0156), so
        # about a quarter of them land in the subnormal range and lose most of
        # their mantissa. The matrix stops being orthogonal, the inverse
        # rotation stops inverting, and GLM-5.2-FP8 on B200 dropped to GPQA
        # 5.6% with 82% of generations emitting no answer at all. Keep the
        # store dtype whenever it is at least 16-bit float, so bf16/fp16 pools
        # (every result measured so far) are bit-for-bit unchanged.
        self._compute_dtype = (
            self.dtype
            if self.dtype
            in (torch.float64, torch.float32, torch.bfloat16, torch.float16)
            else torch.bfloat16
        )
        if self._compute_dtype != self.dtype:
            logger.warning(
                "[Int2HPKVPool] KV store dtype %s cannot hold rotations or "
                "latents; rotating in %s instead",
                self.dtype,
                self._compute_dtype,
            )
        self.rotations = _load_or_make_rotations(
            rotation_path,
            layer_num=self.layer_num,
            start_layer=self.start_layer,
            kv_lora_rank=self.kv_lora_rank,
            device=self.device,
            dtype=self._compute_dtype,
            rotation_layer_ids=rotation_layer_ids,
        )
        # Precompute float32 rotations to avoid per-step dtype casting overhead.
        self.rotations_f32: Optional[Dict[int, torch.Tensor]] = (
            {k: v.to(torch.float32) for k, v in self.rotations.items()}
            if self.rotations is not None else None
        )
        # OSCAR-for-latent high-precision subspace (top-k sensitive directions
        # kept in BF16; only the residual is rotated + INT2-quantized).
        self.hp_subspaces: Optional[Dict[int, torch.Tensor]] = _load_hp_subspaces(
            hp_subspace_path,
            layer_num=self.layer_num,
            start_layer=self.start_layer,
            device=self.device,
            dtype=self.dtype,
        )
        self._init_latent_windows()
        self.dump_c_kv_dir = dump_c_kv_dir
        self.dump_max_tokens_per_layer = dump_max_tokens_per_layer
        self._dump_counts: Dict[int, int] = {}
        self._dump_buffers: Dict[int, list] = {}
        if dump_c_kv_dir:
            os.makedirs(dump_c_kv_dir, exist_ok=True)
            logger.info(
                "[Int2HPKVPool] dumping up to %d c_kv tokens/layer to %s",
                dump_max_tokens_per_layer,
                dump_c_kv_dir,
            )
            atexit.register(self.flush_dumps)

    # ── BF16 sink + BF16 recent windows ─────────────────────────────────────

    def _init_latent_windows(self) -> None:
        """Set up the BF16 sink / BF16 recent windows over the shared latent.

        Same two knobs as ``UnifiedInt2HPKVPool``
        (``SGLANG_MIXED_KV_PREFIX_TOKENS`` = sink ``P``,
        ``SGLANG_MIXED_KV_RECENT_TOKENS`` = recent ``R``), same meaning:
        positions ``[0, P)`` and ``[L-R, L)`` of a length-``L`` sequence are
        served in BF16, ``[P, L-R)`` at 2 bits.

        This is the INT2 floor, not a tuning option: no INT2 KV config
        2-bits the attention sink or the newest tokens. Until now only the
        per-head pool read those two vars, so a ``GlmMoeDsaForCausalLM``
        routed to this pool quantized every latent token and was the one
        model in the project below the floor. Hence 64/256 by default here
        rather than opt-in.

        What differs from the unified pool is *where the BF16 lives*, and it
        matters for one specific hazard. The unified pool needs a second
        arena with its own slot-id namespace, because its quant tier really
        is 2 bits wide; a decode write there lands at
        ``hp_buffer[loc - hp_global_offset]`` unmasked (masking is unsafe
        under graph capture), so its page 0 has to be reserved as the sink
        for CUDA-graph padded replays. This pool fake-quantizes into an
        ordinary float cache: a windowed token keeps the very row it already
        owns, no offset is subtracted from any ``loc``, and the current-token
        write is still the stock parent setter. The only new write is the
        in-place demote, whose target comes from ``req_to_token`` and is
        masked *inside* the indexing (a device-side predicate, not a host
        boolean index) with masked-off entries pointed at KV slot 0 — the
        slot both allocators already hold back for "writing dummy outputs
        from padded tokens" and which ``req_to_token`` also yields for any
        position a request has not written yet.

        ``SGLANG_MIXED_KV_PREFIX_TOKENS=0 SGLANG_MIXED_KV_RECENT_TOKENS=0``
        restores the previous behaviour exactly -- every code path below is
        skipped and the write path is statement-for-statement what it was --
        which is what makes the no-window arm A/B-able, not a supported
        serving configuration.
        """
        from sglang.srt.environ import envs

        # On by default at the project-wide floor. Every INT2 KV config keeps
        # the attention sink and the newest tokens out of 2 bits; a model that
        # needs more raises the recent window (Gemma-4 and Qwen3-8B use 512),
        # nobody goes below 64/256. The two env vars' own defaults are the
        # per-head pool's older 32/128, so read them only when the operator
        # actually set them rather than inheriting a below-floor value.
        self.hp_prefix_tokens = (
            int(envs.SGLANG_MIXED_KV_PREFIX_TOKENS.get())
            if envs.SGLANG_MIXED_KV_PREFIX_TOKENS.is_set()
            else DEFAULT_LATENT_SINK_TOKENS
        )
        self.hp_recent_tokens = (
            int(envs.SGLANG_MIXED_KV_RECENT_TOKENS.get())
            if envs.SGLANG_MIXED_KV_RECENT_TOKENS.is_set()
            else DEFAULT_LATENT_RECENT_TOKENS
        )
        self._fb_window_meta: Optional[dict] = None
        self._window_fallback_logged = False
        self._latent_windows = (
            self.hp_prefix_tokens > 0 or self.hp_recent_tokens > 0
        ) and bool(self.rotations)

        # An fp8 NSA cache does not store the latent as floats at all -- the
        # row is ``[nope_fp8(512) | per-block scales(16) | rope bytes]``, so
        # neither "leave this row alone" nor "re-read and requantize this
        # row" means what it means below. Refuse rather than silently
        # corrupt; bf16 is the pinned recipe anyway (fp8 already destroys
        # rotation orthogonality).
        if self._latent_windows and (
            getattr(self, "nsa_kv_cache_store_fp8", False)
            or self.dtype
            not in (torch.float64, torch.float32, torch.bfloat16, torch.float16)
        ):
            logger.warning(
                "[Int2HPKVPool] BF16 latent windows need a float KV cache; "
                "store dtype=%s nsa_fp8=%s -> windows disabled",
                self.dtype,
                getattr(self, "nsa_kv_cache_store_fp8", False),
            )
            self._latent_windows = False

        if self._latent_windows:
            logger.info(
                "[Int2HPKVPool] BF16 latent windows: sink=%d recent=%d "
                "(positions [0,%d) and [L-%d,L) skip the INT2 round trip; "
                "a token leaving the recent window is quantized in place)",
                self.hp_prefix_tokens,
                self.hp_recent_tokens,
                self.hp_prefix_tokens,
                self.hp_recent_tokens,
            )
        elif self.rotations:
            logger.warning(
                "[Int2HPKVPool] BF16 latent windows OFF (sink=%d recent=%d): "
                "every latent token is INT2, including the attention sink and "
                "the newest tokens. Below the %d/%d floor every other INT2 KV "
                "config holds to -- only do this to A/B the windows",
                self.hp_prefix_tokens,
                self.hp_recent_tokens,
                DEFAULT_LATENT_SINK_TOKENS,
                DEFAULT_LATENT_RECENT_TOKENS,
            )

    def latent_windows_enabled(self) -> bool:
        """True when the BF16 sink / recent windows are active.

        Deliberately *not* named ``mixed_kv_enabled``: that name routes the
        scheduler into ``_alloc_for_decode_mixed``, which only the unified
        two-arena pool can serve, and makes the CUDA-graph runner aim padded
        ``out_cache_loc`` at ``hp_global_offset``. This pool wants neither.
        """
        return bool(getattr(self, "_latent_windows", False))

    def note_forward_batch(self, forward_batch) -> None:
        """Stash the per-forward metadata the windows need.

        Called once per forward (and once per CUDA-graph capture) from the
        two places a ``ForwardBatch`` is built. Nothing here is consumed
        outside the same forward's ``set_*_kv_buffer`` calls, and under
        CUDA-graph capture the tensors recorded are the graph's own static
        buffers, so the ops built on them replay correctly.
        """
        if not getattr(self, "_latent_windows", False):
            return
        try:
            is_decode = bool(forward_batch.forward_mode.is_decode_or_idle())
        except Exception:
            self._fb_window_meta = None
            return
        req_to_token_pool = getattr(forward_batch, "req_to_token_pool", None)
        self._fb_window_meta = {
            "is_decode": is_decode,
            "positions": forward_batch.positions,
            "seq_lens": forward_batch.seq_lens,
            "req_pool_indices": forward_batch.req_pool_indices,
            "req_to_token": (
                None if req_to_token_pool is None else req_to_token_pool.req_to_token
            ),
            "extend_seq_lens": getattr(forward_batch, "extend_seq_lens", None),
            "extend_seq_lens_cpu": getattr(forward_batch, "extend_seq_lens_cpu", None),
            "extend_prefix_lens_cpu": getattr(
                forward_batch, "extend_prefix_lens_cpu", None
            ),
        }

    def _window_fallback(self, why: str) -> None:
        """Log once and fall back to quantizing every token (old behaviour).

        Sticky for the rest of the forward: if the write path could not place
        the windows, the demote must not run either. Demoting on a forward
        whose rows were all quantized anyway is at best redundant, and on a
        multi-token-per-request decode (spec verify, where positions are not
        derivable from ``seq_lens``) it would advance the demote cursor by one
        while the sequence advanced by several, leaving a BF16 tail that grows
        without bound. Quantizing everything is the pre-window behaviour and
        is always safe.
        """
        if self._fb_window_meta is not None:
            self._fb_window_meta["fallback"] = True
        if not self._window_fallback_logged:
            self._window_fallback_logged = True
            logger.warning(
                "[Int2HPKVPool] BF16 latent windows inactive for this forward "
                "(%s); quantizing every latent token instead",
                why,
            )

    def _latent_window_keep(self, n_tokens: int):
        """Which of the ``n_tokens`` rows being written must stay BF16.

        Returns ``(keep_all, keep_mask)``. ``keep_all`` short-circuits the
        common decode case: the token just generated is by definition the
        newest, so with ``R >= 1`` a decode write is never quantized.
        ``keep_mask`` is a ``[n_tokens]`` bool tensor otherwise, or ``None``
        to mean "quantize everything" (the pre-window behaviour).
        """
        meta = self._fb_window_meta
        if meta is None:
            self._window_fallback("no ForwardBatch metadata")
            return False, None
        P, R = self.hp_prefix_tokens, self.hp_recent_tokens

        if meta["is_decode"]:
            seq_lens = meta["seq_lens"]
            if seq_lens is None or seq_lens.numel() != n_tokens:
                # Multi-token-per-request decode (spec verify): one position
                # per row is no longer derivable from seq_lens alone.
                self._window_fallback(
                    f"decode wrote {n_tokens} rows for {0 if seq_lens is None else seq_lens.numel()} requests"
                )
                return False, None
            if R >= 1:
                return True, None
            pos = seq_lens.to(torch.int64) - 1
            return False, pos < P

        positions = meta["positions"]
        extend_seq_lens = meta["extend_seq_lens"]
        if (
            positions is None
            or extend_seq_lens is None
            or positions.numel() != n_tokens
            or meta["seq_lens"] is None
        ):
            self._window_fallback("extend without per-token positions")
            return False, None
        # ``output_size`` keeps repeat_interleave off the host: without it the
        # output shape depends on the values of ``extend_seq_lens`` and torch
        # has to sync to learn it.
        tok_seq_len = torch.repeat_interleave(
            meta["seq_lens"].to(torch.int64),
            extend_seq_lens.to(torch.int64),
            output_size=n_tokens,
        )
        pos = positions.to(torch.int64)
        return False, (pos < P) | (pos >= tok_seq_len - R)

    def _windowed_fake_int2(self, layer_id: int, c_kv: torch.Tensor) -> torch.Tensor:
        """``_apply_fake_int2_c_kv`` on the rows outside the BF16 windows."""
        keep_all, keep = self._latent_window_keep(c_kv.shape[0])
        if keep_all:
            return c_kv
        c_kv_q = self._apply_fake_int2_c_kv(layer_id, c_kv)
        if keep is None:
            return c_kv_q
        return torch.where(
            keep.reshape(-1, *([1] * (c_kv.dim() - 1))), c_kv, c_kv_q.to(c_kv.dtype)
        )

    def _demote_slots(self, layer_id: int, slots: torch.Tensor, valid: torch.Tensor) -> None:
        """Push the latents already stored at ``slots`` through INT2 in place.

        ``slots`` must already be a legal index for every entry (masked-off
        entries point at the reserved padded slot 0); ``valid`` decides which
        ones actually change. Masked-off rows are written back byte-for-byte
        as read, so a padded graph replay whose stale request index resolves
        to slot 0 is a no-op instead of a second writer.
        """
        buf = self.get_key_buffer(layer_id)
        R = self.kv_lora_rank
        # Advanced indexing gathers into a fresh tensor, so ``rows`` is a copy
        # and the read-modify-write below cannot alias the pool.
        rows = buf[slots]                            # [n, 1, kv_cache_dim]
        c = rows[..., :R]
        c_q = self._apply_fake_int2_c_kv(layer_id, c.reshape(-1, R)).reshape(c.shape)
        rows[..., :R] = torch.where(
            valid.reshape(-1, *([1] * (c.dim() - 1))), c_q.to(rows.dtype), c
        )
        # k_pe rides along untouched: it is written back exactly as it was
        # read. Rewriting the whole row avoids a masked strided scatter into a
        # partial last dimension, which is what makes this one plain
        # ``index_put_`` and therefore safe to capture.
        buf[slots] = rows

    def _demote_aged_latents(self, layer_id: int) -> None:
        """Quantize every latent that has just fallen out of the recent window.

        After a forward that grew a request from ``L0`` to ``L`` tokens,
        positions ``[P, L-R)`` must be INT2. Positions written in this very
        forward were decided at write time; the ones that were resident and
        BF16 and have now aged out are ``[max(P, L0-R), min(L0, L-R))`` --
        exactly one position per request per decode step, and non-empty
        during extend only for a chunked prefill or a reused prefix.
        """
        R_win = self.hp_recent_tokens
        if R_win <= 0:
            return
        meta = self._fb_window_meta
        if meta is None or meta["req_to_token"] is None or meta.get("fallback"):
            return
        req_to_token = meta["req_to_token"]
        req_pool_indices = meta["req_pool_indices"]
        P = self.hp_prefix_tokens

        if meta["is_decode"]:
            seq_lens = meta["seq_lens"]
            if (
                seq_lens is None
                or req_pool_indices is None
                or seq_lens.numel() != req_pool_indices.numel()
                or seq_lens.numel() == 0
            ):
                return
            # L = seq_lens (decode seq_lens already counts the new token),
            # L0 = L-1, so the aged-out set is the single position L-1-R.
            demote_pos = seq_lens.to(torch.int64) - 1 - R_win
            valid = demote_pos >= P
            req_rows = req_pool_indices.to(torch.int64)
            slots = req_to_token[req_rows, demote_pos.clamp(min=0)].to(torch.int64)
            # Padded graph replays keep a stale req_pool_indices but get
            # seq_lens = the graph fill value, so they land here as invalid;
            # send them at the reserved slot 0 rather than a live row.
            slots = torch.where(valid, slots, torch.zeros_like(slots))
            self._demote_slots(layer_id, slots, valid)
            return

        # Extend. Only reachable when a request arrives with KV already
        # resident that has now aged out -- a chunked prefill's earlier chunk
        # or a reused prefix -- so it is usually empty. The range comes from
        # the CPU-side extend bookkeeping, which makes it a *constant* if it
        # were ever captured, so refuse under capture rather than bake in the
        # slots of whichever batch happened to be captured.
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            self._window_fallback("extend KV write inside a graph capture")
            return
        prefix_cpu = meta["extend_prefix_lens_cpu"]
        ext_cpu = meta["extend_seq_lens_cpu"]
        if not prefix_cpu or not ext_cpu or req_pool_indices is None:
            return
        # Identical for every layer of the forward, and the Python range walk
        # is O(recent_tokens * bs); build it once.
        slots = meta.get("extend_demote_slots", False)
        if slots is False:
            batch_rows: list = []
            cols: list = []
            for i, (l0, ext) in enumerate(zip(prefix_cpu, ext_cpu)):
                lo = max(P, int(l0) - R_win)
                hi = min(int(l0), int(l0) + int(ext) - R_win)
                batch_rows.extend([i] * max(hi - lo, 0))
                cols.extend(range(lo, hi))
            if cols:
                dev = req_to_token.device
                row_t = torch.tensor(batch_rows, dtype=torch.int64, device=dev)
                col_t = torch.tensor(cols, dtype=torch.int64, device=dev)
                slots = req_to_token[
                    req_pool_indices.to(torch.int64)[row_t], col_t
                ].to(torch.int64)
            else:
                slots = None
            meta["extend_demote_slots"] = slots
        if slots is None:
            return
        self._demote_slots(
            layer_id, slots, torch.ones_like(slots, dtype=torch.bool)
        )

    def _apply_fake_int2_c_kv(
        self,
        layer_id: int,
        c_kv: torch.Tensor,
    ) -> torch.Tensor:
        """Rotate → fake-quant INT2 → unrotate.

        When an HP subspace is configured for this layer, the projection of c_kv
        onto the top-k sensitive directions is kept in full precision and only
        the residual is quantized (OSCAR-for-latent mixed precision).
        """
        # Never round an intermediate of the rotate/unrotate round trip to the
        # pool's store dtype: on an fp8 KV cache that discards most of the
        # mantissa mid-round-trip, and the caller's own dtype is what the parent
        # setter expects anyway (the NSA/MLA write path takes the bf16 latent and
        # does its own cast). Identical to the old behaviour on a bf16 pool,
        # where the latent arrives in exactly the store dtype.
        out_dtype = c_kv.dtype
        Uk = self.hp_subspaces.get(layer_id) if self.hp_subspaces is not None else None
        c_hp = None
        if Uk is not None:
            x = c_kv.to(torch.float32)
            c_hp = (x @ Uk.T) @ Uk          # [N, R] projection onto sensitive subspace
            c_kv = x - c_hp                 # residual to be quantized

        Rf = self.rotations_f32.get(layer_id) if self.rotations_f32 else None
        if Rf is not None:
            c_kv = torch.matmul(c_kv.to(torch.float32), Rf)
        if _real_kernel_enabled() and c_kv.is_cuda and (
            c_kv.shape[-1] % self._group_size == 0
        ):
            from sglang.QuantKernel.mla_latent_int2 import (
                quantize_dequantize_reuse,
            )

            # One launch on reused scratch: 2.6x the torch fake-quant on the full
            # write path, bit-identical output. The result is a view into
            # module-level scratch that the next call overwrites, so it must be
            # copied out here. The MLA setter hands this straight to
            # parent_setter with no intervening op, and the `.to(self.dtype)`
            # below is a no-op whenever the kernel already returns the store
            # dtype — so without this clone all 78 layers alias one buffer and
            # every layer's c_kv is whatever the last layer wrote.
            _codes, _params, c_kv_q = quantize_dequantize_reuse(
                c_kv, self._group_size, self._lloyd_max
            )
            c_kv_q = c_kv_q.clone()
        else:
            c_kv_q = _fake_quant_int2_groupwise(c_kv, self._group_size, self._lloyd_max)
        c_kv_q = c_kv_q.to(torch.float32) if c_hp is not None else c_kv_q.to(out_dtype)
        if Rf is not None:
            c_kv_q = torch.matmul(c_kv_q.to(torch.float32), Rf.T)
        if c_hp is not None:
            c_kv_q = c_kv_q + c_hp          # add back the full-precision subspace
        return c_kv_q.to(out_dtype)

    def _maybe_dump_c_kv(self, layer_id: int, c_kv: torch.Tensor) -> None:
        if not self.dump_c_kv_dir:
            return
        count = self._dump_counts.get(layer_id, 0)
        if count >= self.dump_max_tokens_per_layer:
            return
        # Normalize to 2D [tokens, kv_lora_rank]: GLM-5.2 emits c_kv with an
        # extra leading dim in some calls (3D) vs 2D in others, which made the
        # per-layer buffer mix ranks → torch.cat(dim=0) "got 3 and 2".
        c_kv = c_kv.reshape(-1, c_kv.shape[-1])
        n = min(c_kv.shape[0], self.dump_max_tokens_per_layer - count)
        chunk = c_kv[:n].detach().to(torch.float32).cpu()
        buf = self._dump_buffers.setdefault(layer_id, [])
        buf.append(chunk)
        self._dump_counts[layer_id] = count + n
        if self._dump_counts[layer_id] >= self.dump_max_tokens_per_layer:
            path = os.path.join(self.dump_c_kv_dir, f"layer_{layer_id}.pt")
            torch.save(torch.cat(buf, dim=0), path)
            self._dump_buffers.pop(layer_id)
            logger.info(
                "[Int2HPKVPool] flushed layer %d c_kv dump (%d tokens) -> %s",
                layer_id,
                self._dump_counts[layer_id],
                path,
            )

    def flush_dumps(self) -> None:
        if not getattr(self, "dump_c_kv_dir", ""):
            return
        for layer_id, buf in list(self._dump_buffers.items()):
            path = os.path.join(self.dump_c_kv_dir, f"layer_{layer_id}.pt")
            torch.save(torch.cat(buf, dim=0), path)
            logger.info(
                "[Int2HPKVPool] flushed partial layer %d (%d tokens) -> %s",
                layer_id,
                self._dump_counts.get(layer_id, 0),
                path,
            )
            self._dump_buffers.pop(layer_id)

    def _log_write_path_once(self, setter_name: str, **dtypes) -> None:
        """Name the write path and its dtypes once per setter.

        Once per *setter*, not once per pool: a model can reach the latent
        through ``set_mla_kv_buffer`` on one forward mode and
        ``set_kv_buffer`` on the other (GLM-5.2 on the triton MLA backend
        logs the latter), and with a single flag the second one never
        appeared -- so the log could not tell you whether the path you
        reasoned about is the path that ran.

        Which of the two setters a model uses, and whether the tensors arriving
        there are the store dtype or the model's compute dtype, decides whether
        this pool quantizes anything meaningful. Both have already differed by
        platform (fp8 KV cache on SM100+ vs bf16 below), and both failures were
        silent -- they showed up as a score, not an exception.
        """
        seen = getattr(self, "_write_paths_logged", None)
        if seen is None:
            seen = self._write_paths_logged = set()
        if setter_name in seen:
            return
        seen.add(setter_name)
        logger.info(
            "[Int2HPKVPool] write path=%s windows=%s group=%s lloyd_max=%s "
            "store_dtype=%s pool_dtype=%s compute_dtype=%s nsa_fp8=%s %s",
            setter_name,
            (
                f"sink={self.hp_prefix_tokens}/recent={self.hp_recent_tokens}"
                if self._latent_windows
                else "off"
            ),
            # The codebook knobs decide which method the score belongs to, and
            # nothing else logs them: SGLANG_LLOYD_MAX flips the buckets between
            # uniform and MSE-optimal with no other observable trace, so an
            # archived run's server.log could not say which of the two produced
            # it. That is not hypothetical -- Lloyd-Max has been the difference
            # between a win and a regression on other families in this project
            # (it measurably hurt Gemma-4 and M3 long generations), so an arm
            # whose LM setting cannot be read back cannot be compared to one
            # whose can, and the pair has to be re-run to mean anything.
            self._group_size,
            self._lloyd_max,
            getattr(self, "store_dtype", None),
            self.dtype,
            self._compute_dtype,
            getattr(self, "nsa_kv_cache_store_fp8", None),
            " ".join(f"{k}={v}" for k, v in dtypes.items()),
        )

    def _int2_set_mla_kv_buffer(
        self,
        parent_setter,
        layer,
        loc,
        cache_k_nope,
        cache_k_rope,
    ):
        """Intercept MLA set_mla_kv_buffer: dump and/or fake-quant c_kv."""
        layer_id = layer.layer_id
        self._log_write_path_once(
            "set_mla_kv_buffer",
            k_nope=cache_k_nope.dtype,
            k_rope=cache_k_rope.dtype,
        )
        self._maybe_dump_c_kv(layer_id, cache_k_nope)
        if self._latent_windows:
            cache_k_nope = self._windowed_fake_int2(layer_id, cache_k_nope)
        elif self.rotations:  # only degrade quality when rotation is requested
            cache_k_nope = self._apply_fake_int2_c_kv(layer_id, cache_k_nope)
        parent_setter(layer, loc, cache_k_nope, cache_k_rope)
        if self._latent_windows:
            self._demote_aged_latents(layer_id)

    def _int2_set_kv_buffer(
        self,
        parent_setter,
        layer,
        loc,
        cache_k,
        cache_v,
    ):
        """Intercept NSA set_kv_buffer: dump and/or fake-quant the c_kv slice."""
        # NSA kv_buffer layout: [c_kv | k_pe] concatenated along last dim.
        c_kv_dim = self.kv_lora_rank
        c_kv = cache_k[..., :c_kv_dim]
        k_pe = cache_k[..., c_kv_dim:]
        layer_id = layer.layer_id
        self._log_write_path_once("set_kv_buffer", cache_k=cache_k.dtype)
        self._maybe_dump_c_kv(layer_id, c_kv.reshape(-1, c_kv_dim))
        if self._latent_windows:
            c_kv_q = self._windowed_fake_int2(
                layer_id, c_kv.reshape(-1, c_kv_dim)
            )
            cache_k = torch.cat([c_kv_q.reshape(c_kv.shape), k_pe], dim=-1)
        elif self.rotations:  # only degrade quality when rotation is requested
            c_kv_q = self._apply_fake_int2_c_kv(layer_id, c_kv.reshape(-1, c_kv_dim))
            cache_k = torch.cat([c_kv_q.reshape(c_kv.shape), k_pe], dim=-1)
        parent_setter(layer, loc, cache_k, cache_v)
        if self._latent_windows:
            self._demote_aged_latents(layer_id)


# ── pool classes ─────────────────────────────────────────────────────────────

class MLAInt2HPKVPool(_Int2HPMixin, MLATokenToKVPool):
    """MLA pool with INT2 fake-quant applied to c_kv (kv_lora_rank dim)."""

    def __init__(
        self,
        *args,
        rotation_path: str = "",
        group_size: int = 128,
        dump_c_kv_dir: str = "",
        dump_max_tokens_per_layer: int = 8192,
        lloyd_max: bool = False,
        hp_subspace_path: str = "",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._init_int2(rotation_path, group_size, dump_c_kv_dir,
                        dump_max_tokens_per_layer, lloyd_max, hp_subspace_path)

    def set_mla_kv_buffer(self, layer, loc, cache_k_nope, cache_k_rope):
        self._int2_set_mla_kv_buffer(
            super().set_mla_kv_buffer, layer, loc, cache_k_nope, cache_k_rope
        )

    def set_kv_buffer(self, layer, loc, cache_k, cache_v):
        self._int2_set_kv_buffer(
            super().set_kv_buffer, layer, loc, cache_k, cache_v
        )


class NSAInt2HPKVPool(_Int2HPMixin, NSATokenToKVPool):
    """NSA pool with INT2 fake-quant applied to c_kv. ``index_k_with_scale``
    buffer is unchanged — only the main MLA latent ``kv_buffer`` is
    fake-quantized.
    """

    def __init__(
        self,
        *args,
        rotation_path: str = "",
        group_size: int = 128,
        dump_c_kv_dir: str = "",
        dump_max_tokens_per_layer: int = 8192,
        lloyd_max: bool = False,
        hp_subspace_path: str = "",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._init_int2(rotation_path, group_size, dump_c_kv_dir,
                        dump_max_tokens_per_layer, lloyd_max, hp_subspace_path)

    def set_mla_kv_buffer(self, layer, loc, cache_k_nope, cache_k_rope):
        self._int2_set_mla_kv_buffer(
            super().set_mla_kv_buffer, layer, loc, cache_k_nope, cache_k_rope
        )

    def set_kv_buffer(self, layer, loc, cache_k, cache_v):
        self._int2_set_kv_buffer(
            super().set_kv_buffer, layer, loc, cache_k, cache_v
        )
