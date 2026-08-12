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
"""

from __future__ import annotations

import atexit
import logging
import math
import os
from typing import Dict, Optional

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


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_or_make_rotations(
    rotation_path: str,
    layer_num: int,
    start_layer: int,
    kv_lora_rank: int,
    device: str,
    dtype: torch.dtype,
) -> Optional[Dict[int, torch.Tensor]]:
    """Load per-layer rotations from a directory or generate Hadamard.

    ``rotation_path``:
      * ``""`` / missing → return ``None`` (no rotation).
      * ``"hadamard"`` → generate one random-sign 512×512 Hadamard-style
        orthogonal matrix shared across layers (no calibration data).
      * any existing directory → load ``layer_<i>.pt`` for ``i`` in the
        effective layer range; missing layers fall back to identity.
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

    rotations: Dict[int, torch.Tensor] = {}
    for i in range(start_layer, start_layer + layer_num):
        p = os.path.join(rotation_path, f"layer_{i}.pt")
        if os.path.exists(p):
            rotations[i] = torch.load(p, map_location="cpu").to(dtype=dtype, device=device).contiguous()
        else:
            rotations[i] = torch.eye(d, dtype=dtype, device=device)
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
    try:
        from sglang.srt import environ as envs
        return bool(envs.SGLANG_OSCAR_MLA_KV_REAL_KERNEL.get())
    except Exception:
        return False


def _quant_requested() -> bool:
    try:
        from sglang.srt import environ as envs
        return bool(envs.SGLANG_OSCAR_MLA_KV_ROTATION_PATH.get())
    except Exception:
        return False


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
    ) -> None:
        if self.kv_lora_rank % group_size != 0:
            raise ValueError(
                f"group_size={group_size} must divide kv_lora_rank={self.kv_lora_rank}"
            )
        self._group_size = group_size
        self._lloyd_max = lloyd_max
        self.rotations = _load_or_make_rotations(
            rotation_path,
            layer_num=self.layer_num,
            start_layer=self.start_layer,
            kv_lora_rank=self.kv_lora_rank,
            device=self.device,
            dtype=self.dtype,
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
            from sglang.QuantKernel.mla_latent_int2 import dequantize, quantize_pack

            codes, params = quantize_pack(c_kv, self._group_size, self._lloyd_max)
            c_kv_q = dequantize(
                codes, params, c_kv.shape, self._group_size,
                torch.float32, self._lloyd_max,
            )
        else:
            c_kv_q = _fake_quant_int2_groupwise(c_kv, self._group_size, self._lloyd_max)
        c_kv_q = c_kv_q.to(torch.float32) if c_hp is not None else c_kv_q.to(self.dtype)
        if Rf is not None:
            c_kv_q = torch.matmul(c_kv_q.to(torch.float32), Rf.T)
        if c_hp is not None:
            c_kv_q = c_kv_q + c_hp          # add back the full-precision subspace
        return c_kv_q.to(self.dtype)

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
        self._maybe_dump_c_kv(layer_id, cache_k_nope)
        if self.rotations:  # only degrade quality when rotation is requested
            cache_k_nope = self._apply_fake_int2_c_kv(layer_id, cache_k_nope)
        parent_setter(layer, loc, cache_k_nope, cache_k_rope)

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
        self._maybe_dump_c_kv(layer_id, c_kv.reshape(-1, c_kv_dim))
        if self.rotations:  # only degrade quality when rotation is requested
            c_kv_q = self._apply_fake_int2_c_kv(layer_id, c_kv.reshape(-1, c_kv_dim))
            cache_k = torch.cat([c_kv_q.reshape(c_kv.shape), k_pe], dim=-1)
        parent_setter(layer, loc, cache_k, cache_v)


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
