"""MLA / NSA latent KV pool with **real packed INT2 storage**.

``mla_int2_kv_pool.py`` is the fake-quant pool: it rounds ``c_kv`` through INT2
and writes the result back into an ordinary BF16 cache, which measures the
accuracy of 2-bit latents but saves nothing -- ``max_total_num_tokens`` is
identical to the BF16 arm's. This module is the storage version.

Layout, per token per layer
---------------------------
=====================  =========  ===============================================
buffer                 bytes      contents
=====================  =========  ===============================================
``c_codes``            R/4 = 128  packed 2-bit codes for the latent, 4 per byte
``c_params``           8*NG = 32  (scale, zero) fp32 per quantization group
``rope_buf``           2*P = 128  ``k_pe``, BF16, never quantized
=====================  =========  ===============================================

288 B against the BF16 pool's ``(512 + 64) * 2 = 1152``, a 4.0x reduction on the
latent itself. ``k_pe`` staying BF16 is load-bearing, not an oversight: it is
the positional half of the MLA key and 2-bit'ing it destroys the rope term.

Rotated storage and where the rotation went
-------------------------------------------
The codes are stored in the *rotated* frame, because that is the frame the
quantizer is calibrated for. The fake-quant pool could undo the rotation inline
(it dequantized on the write path anyway); a storage pool cannot, and applying
R^T to every row an attention kernel touches would cost more than the read.

So the rotation is moved to the two ends, where it is free:

    scores  q . c^T = q . (c' R^T)^T = (q R) . c'^T
    output  sum_j p_j c_j = (sum_j p_j c'_j) R^T

i.e. rotate the (already ``w_kc``-absorbed) query, and un-rotate the attention
output before ``w_vc``. Both are ``[tokens, heads, 512] @ [512, 512]`` -- a
rounding error next to the attention itself, and both fold into the projection
weights outright as a later optimization. :meth:`rotate_latent` /
:meth:`unrotate_output` are those two hooks.

BF16 windows without a second allocator
----------------------------------------
The sink/recent windows (64/512 measured optimal for GLM-5.2, and non-monotone
-- 1024 is *worse* than 256) cannot be "the row stays BF16" here, because rows
are 288 bytes. They live in a separate arena, but unlike ``UnifiedInt2HPKVPool``
this one needs no allocator, no free list and no scheduler support, because the
window is **positional** and therefore statically addressable:

    ring(req, pos) = 1 + req * (P + R) + (pos if pos < P else P + (pos-P) % R)

One row per request per window position, assigned by arithmetic. Nothing is
allocated at run time, so the whole thing is CUDA-graph safe.

Ageing out is free as a consequence. Position ``L-1`` maps to the same ring row
as ``L-1-R``, so writing the new token *is* the eviction of the one leaving the
window. An ``owner`` tag per ring row records which pool slot last wrote it; a
reader takes the BF16 row only when ``owner[ring[slot]] == slot``. That tag is
also what makes prefix-cache reuse safe: a cached slot whose ring row has since
been re-issued to another request fails the check and falls back to its packed
row, rather than reading another sequence's latent.

Known behavioural difference from the fake-quant pool, stated because it is a
difference and not a bug: on a **prefix-cache hit** the reused tokens were never
written through this forward, so their window rows are not in the ring and they
are served from the packed tier. The fake-quant pool keeps a reused prefix's
first ``P`` positions in BF16. Everything else -- every token a request writes
itself -- is identical.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch

from sglang.srt.mem_cache.memory_pool import (
    GPU_MEMORY_TYPE_KV_CACHE,
    MLATokenToKVPool,
    NSATokenToKVPool,
)
from sglang.srt.mem_cache.mla_int2_kv_pool import _Int2HPMixin

logger = logging.getLogger(__name__)


def _is_capturing() -> bool:
    """Whether sglang is currently capturing a CUDA graph.

    Deliberately not ``torch.cuda.is_current_stream_capturing()``: that is a
    CUDA call and dynamo cannot trace it, so on a compiled attention layer it
    is a graph break, not a branch.
    """
    from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

    return bool(get_is_capture_mode())


def _tracing() -> bool:
    """True while dynamo is tracing this call.

    The self-check syncs (it reads a max back to log it), and a sync is
    untraceable: inside a compiled attention layer it is a hard graph break, not
    a slow path. ``torch.compiler.is_compiling()`` is the one guard dynamo
    specializes instead of breaking on.
    """
    try:
        return bool(torch.compiler.is_compiling())
    except Exception:
        return False


def envs_selfcheck_budget() -> int:
    from sglang.srt.environ import envs

    return int(envs.SGLANG_OSCAR_MLA_PACKED_SELFCHECK_BUDGET.get())


def packed_latent_bytes_per_token(kv_lora_rank: int, qk_rope_head_dim: int,
                                  group_size: int, param_dtype_bytes: int = 4) -> int:
    """Bytes one token of latent occupies per layer under packed storage.

    Kept next to the buffers it describes so the pool-size arithmetic in
    ``pool_configurator`` and the allocation here cannot drift apart -- a
    ``cell_size`` that disagrees with what the pool actually allocates either
    wastes the difference or OOMs at the end of the pool.
    """
    n_groups = kv_lora_rank // group_size
    return (
        kv_lora_rank // 4                       # codes
        + n_groups * 2 * param_dtype_bytes      # (scale, zero) per group
        + qk_rope_head_dim * 2                  # k_pe, bf16
    )


class _PackedLatentMixin(_Int2HPMixin):
    """Packed-storage latent, mixed into the MLA and NSA pools.

    Reuses ``_Int2HPMixin`` for rotation loading, the window knobs and
    ``note_forward_batch``; replaces the write path and adds the read path.
    """

    # ── construction ────────────────────────────────────────────────────────

    def _init_packed(
        self,
        rotation_path: str,
        group_size: int,
        lloyd_max: bool,
        max_reqs: int,
        selfcheck: bool = False,
    ) -> None:
        # _init_int2 loads the rotations, resolves the compute dtype and reads
        # the window knobs. The dump/HP-subspace features are fake-quant-only.
        self._init_int2(
            rotation_path,
            group_size,
            dump_c_kv_dir="",
            dump_max_tokens_per_layer=0,
            lloyd_max=lloyd_max,
            hp_subspace_path="",
        )
        if not self.rotations:
            raise ValueError(
                "Packed MLA latent storage needs a rotation path: without one "
                "there is no calibrated frame to quantize in, and the pool "
                "would silently store 2-bit garbage."
            )
        if self.hp_subspaces:
            raise NotImplementedError(
                "SGLANG_OSCAR_MLA_KV_HP_SUBSPACE_PATH is a fake-quant-only "
                "feature; it keeps a dense BF16 projection per token, which "
                "packed storage has nowhere to put."
            )
        if self.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(
                f"Packed MLA latent storage needs a >=16-bit float KV dtype for "
                f"k_pe and the window arena; got {self.dtype}. Pin "
                f"--kv-cache-dtype bfloat16 (sglang defaults a DSA model to "
                f"fp8_e4m3 on SM100+, which also destroys the rotation)."
            )

        R = self.kv_lora_rank
        self._n_groups = R // group_size
        self._selfcheck = selfcheck
        n_rows = self.size + self.page_size

        P, W = self.hp_prefix_tokens, self.hp_recent_tokens
        self._win_p, self._win_r = P, W
        self._per_req_hp = P + W
        self._max_reqs = max_reqs
        # Ring row 0 is the dummy every masked-off write is aimed at, so the
        # rings start at 1 -- otherwise request 0's sink position 0 would share
        # a row with the padding sink and a padded graph replay could evict a
        # live window entry.
        n_hp = 1 + max_reqs * self._per_req_hp if self._latent_windows else 1

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.c_codes = [
                torch.zeros((n_rows, R // 4), dtype=torch.uint8, device=self.device)
                for _ in range(self.layer_num)
            ]
            self.c_params = [
                torch.zeros(
                    (n_rows, self._n_groups * 2),
                    dtype=torch.float32,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]
            self.rope_buf = [
                torch.zeros(
                    (n_rows, self.qk_rope_head_dim),
                    dtype=self.dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]
            self.hp_c = [
                torch.zeros((n_hp, R), dtype=self.dtype, device=self.device)
                for _ in range(self.layer_num)
            ]
            self.hp_row_of_slot = torch.full(
                (n_rows,), -1, dtype=torch.int32, device=self.device
            )
            self.hp_owner_of_row = torch.full(
                (n_hp,), -1, dtype=torch.int32, device=self.device
            )

        # Read scratch, grown on demand and reused across layers. Under graph
        # capture the address must be stable, which it is: capture happens after
        # the warmup forward has already grown it to the largest decode shape.
        self._read_scratch: Optional[torch.Tensor] = None
        self._deq_scratch: Optional[torch.Tensor] = None
        # The self-check verifies each write immediately, against a reference
        # computed from the values in hand. The obvious design -- shadow the
        # whole pool with the BF16 fake-quant result -- is not available here
        # and the reason is the point of the change: the packed pool is ~3x
        # more rows than the BF16 one, so its shadow is ~157 GB and OOMs during
        # construction. Verifying at the write covers every row that is ever
        # read, because a row's packed bytes are written exactly once.
        self._selfcheck = selfcheck
        self._selfcheck_budget = (
            envs_selfcheck_budget() if selfcheck else 0
        )
        self._selfcheck_worst = 0.0
        self._selfcheck_n = 0
        self._selfcheck_gate_logged = False
        # ``from sglang.srt.environ import envs`` -- importing the *module* as
        # ``envs`` binds a module whose attributes are not the descriptors, and
        # a bare except around it once turned every flag into False silently.
        from sglang.srt.environ import envs as _envs

        self._audit_budget = (
            _envs.SGLANG_OSCAR_MLA_PACKED_AUDIT_BUDGET.get()
            if _envs.SGLANG_OSCAR_MLA_PACKED_AUDIT.get()
            else 0
        )
        self._audit_stride = max(
            1, _envs.SGLANG_OSCAR_MLA_PACKED_AUDIT_STRIDE.get()
        )
        self._audit_every_n = 0
        self._audit_worst = 1.0

        bytes_tok = packed_latent_bytes_per_token(
            R, self.qk_rope_head_dim, group_size
        )
        logger.info(
            "[MLAPacked] packed latent storage: %d B/token/layer (BF16 would be "
            "%d B, %.2fx), rows=%d layers=%d, windows sink=%d/recent=%d over "
            "%d ring rows (%.2f GB), group=%d lloyd_max=%s selfcheck=%s",
            bytes_tok,
            (R + self.qk_rope_head_dim) * 2,
            (R + self.qk_rope_head_dim) * 2 / bytes_tok,
            n_rows,
            self.layer_num,
            P if self._latent_windows else 0,
            W if self._latent_windows else 0,
            n_hp,
            n_hp * R * 2 * self.layer_num / (1 << 30),
            group_size,
            lloyd_max,
            selfcheck,
        )
        # The parent constructor already logged an allocation line, but it ran
        # before any of these buffers existed and therefore reported 0.00 GB.
        # Restate it now that the numbers are real -- this line is the one a
        # reader will use to decide whether the storage change landed.
        self._finalize_allocation_log(self.size)

    # ── accounting ──────────────────────────────────────────────────────────

    def get_kv_size_bytes(self):
        # Called once from ``_finalize_allocation_log`` inside the *parent*
        # constructor, i.e. before ``_init_packed`` has allocated anything --
        # hence the getattr defaults rather than an attribute access that would
        # turn a log line into a startup crash.
        total = 0
        for name in ("c_codes", "c_params", "rope_buf", "hp_c",
                     "index_k_with_scale_buffer"):
            for t in getattr(self, name, None) or ():
                total += t.numel() * t.element_size()
        for name in ("hp_row_of_slot", "hp_owner_of_row"):
            t = getattr(self, name, None)
            if t is not None:
                total += t.numel() * t.element_size()
        return total

    def get_contiguous_buf_infos(self):
        raise NotImplementedError(
            "packed MLA latent has three buffers per layer; disaggregation "
            "would transfer only the codes and silently drop scales and k_pe"
        )

    # ── the buffers the stock MLA path expects, deliberately absent ─────────

    def latent_row_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    def get_key_buffer(self, layer_id: int):
        raise NotImplementedError(
            "packed MLA latent has no BF16 kv_buffer to hand out. Readers must "
            "go through materialize_rows() or the packed attention kernels; "
            "returning a stand-in here would let a consumer read uninitialised "
            "memory and score it."
        )

    def get_value_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id)

    # ── rotation hooks used by the model's MLA forward ──────────────────────

    def latent_rotation(self, layer_id: int) -> Optional[torch.Tensor]:
        if not self.rotations:
            return None
        return self.rotations.get(layer_id)

    def rotate_latent(self, layer_id: int, x: torch.Tensor) -> torch.Tensor:
        """``x @ R`` -- the query side, and the fresh keys handed to a kernel."""
        R = self.latent_rotation(layer_id)
        if R is None:
            return x
        shp = x.shape
        return torch.matmul(x.reshape(-1, shp[-1]), R.to(x.dtype)).view(shp)

    def unrotate_output(self, layer_id: int, x: torch.Tensor) -> torch.Tensor:
        """``x @ R^T`` -- the attention output, before ``w_vc``."""
        R = self.latent_rotation(layer_id)
        if R is None:
            return x
        shp = x.shape
        return torch.matmul(x.reshape(-1, shp[-1]), R.to(x.dtype).T).view(shp)

    # ── window addressing ───────────────────────────────────────────────────

    def _ring_rows(self, n_tokens: int):
        """``(ring_row, keep)`` for the ``n_tokens`` rows this forward writes.

        Both are ``[n_tokens]`` int64/bool device tensors, or ``(None, None)``
        when the windows cannot be placed for this forward (the pre-window
        behaviour: everything is 2-bit, which is always safe).
        """
        if not self._latent_windows:
            return None, None
        meta = self._fb_window_meta
        if meta is None or meta.get("fallback"):
            return None, None
        P, W = self._win_p, self._win_r
        req_pool_indices = meta["req_pool_indices"]
        if req_pool_indices is None:
            return None, None

        if meta["is_decode"]:
            seq_lens = meta["seq_lens"]
            if seq_lens is None or seq_lens.numel() != n_tokens:
                self._window_fallback(
                    f"decode wrote {n_tokens} rows for "
                    f"{0 if seq_lens is None else seq_lens.numel()} requests"
                )
                return None, None
            pos = seq_lens.to(torch.int64) - 1
            req = req_pool_indices.to(torch.int64)
            seq = seq_lens.to(torch.int64)
        else:
            positions = meta["positions"]
            extend_seq_lens = meta["extend_seq_lens"]
            if (
                positions is None
                or extend_seq_lens is None
                or positions.numel() != n_tokens
                or meta["seq_lens"] is None
            ):
                self._window_fallback("extend without per-token positions")
                return None, None
            ext = extend_seq_lens.to(torch.int64)
            pos = positions.to(torch.int64)
            seq = torch.repeat_interleave(
                meta["seq_lens"].to(torch.int64), ext, output_size=n_tokens
            )
            req = torch.repeat_interleave(
                req_pool_indices.to(torch.int64), ext, output_size=n_tokens
            )

        keep = (pos < P) | (pos >= seq - W)
        # Requests past the ring arena would alias each other's windows; drop
        # them to the packed tier rather than corrupt a neighbour. Sizing the
        # arena from req_to_token_pool means this is unreachable in practice.
        keep = keep & (req < self._max_reqs)
        # Slot 0 is the reserved dummy every padded row is aimed at (the paged
        # allocator hands out page 1 upward), so a row landing there is a padded
        # CUDA-graph replay. Its ``req_pool_indices`` entry is stale -- left over
        # from whichever batch was captured -- and letting it write a ring row
        # would knock a *live* request's sink out of the window on every replay.
        keep = keep & (self._write_loc > 0)
        in_sink = pos < P
        ring = 1 + req * self._per_req_hp + torch.where(
            in_sink, pos, P + (pos - P).clamp(min=0) % max(W, 1)
        )
        ring = torch.where(keep, ring, torch.zeros_like(ring))
        return ring, keep

    # ── write path ──────────────────────────────────────────────────────────

    def _packed_store(self, layer_id: int, loc: torch.Tensor,
                      c_kv: torch.Tensor, k_pe: torch.Tensor) -> None:
        """Rotate and pack ``c_kv`` into the pool at ``loc``.

        The rotation lives here, on the write, so that *every* write site is
        covered by construction -- the MHA prefill path reaches the pool through
        ``_set_mla_kv_buffer`` and the MLA path through the attention layer's
        ``save_kv_cache``, and a rotation applied at only one of them would give
        a cache half in each frame with nothing to catch it. The read side
        cancels it once, on the attention output.
        """
        from sglang.QuantKernel.mla_latent_int2 import scatter_pack_rows

        li = layer_id - self.start_layer
        R = self.kv_lora_rank
        c = self.rotate_latent(layer_id, c_kv.reshape(-1, R).to(torch.float32))
        pe = k_pe.reshape(-1, self.qk_rope_head_dim)
        loc64 = loc.reshape(-1).to(torch.int64)

        scatter_pack_rows(
            c, loc64.to(torch.int32), self.c_codes[li], self.c_params[li],
            self._group_size, self._lloyd_max,
        )
        self.rope_buf[li][loc64] = pe.to(self.dtype)

        self._write_loc = loc64
        ring, keep = self._ring_rows(c.shape[0])
        if ring is not None:
            self.hp_c[li][ring] = c.to(self.dtype)
            self.hp_owner_of_row[ring] = torch.where(
                keep, loc64.to(torch.int32), torch.full_like(loc64, -1, dtype=torch.int32)
            )
            self.hp_row_of_slot[loc64] = torch.where(
                keep, ring.to(torch.int32), torch.full_like(ring, -1, dtype=torch.int32)
            )
        else:
            self.hp_row_of_slot[loc64] = -1

        # The self-check reads a max back to the host, which is illegal inside a
        # graph capture ("operation not permitted when stream is capturing") and
        # would also bake a one-off comparison into the replayed graph. Skip it
        # there; the same rows are re-verified on the next eager forward.
        #
        # Gate on sglang's own capture flag, not on
        # ``torch.cuda.is_current_stream_capturing()``: the latter is a CUDA
        # call that dynamo cannot trace, so on a model whose attention layers go
        # through torch.compile (DeepSeek-V2-Lite does; GLM-5.2 is on the
        # piecewise-disabled list) it becomes a hard graph break rather than a
        # branch. sglang's flag is a plain Python global and traces fine.
        # The arena audit reads a count back to the host, so it carries the same
        # capture/trace restriction as the self-check. Unlike the self-check its
        # budget is spent on *late* decode steps, not early writes: the tag it
        # checks is only interesting after the ring has wrapped, which needs
        # thousands of steps. Layer 0 only, or it fires once per layer per step.
        if (
            self._audit_budget > 0
            and layer_id == self.start_layer
            and not _is_capturing()
            and not _tracing()
        ):
            self._audit_every_n += 1
            if self._audit_every_n % self._audit_stride == 0:
                self._audit_budget -= 1
                self.audit_window_arena(f"step~{self._audit_every_n}")

        if self._selfcheck_budget > 0 and not _is_capturing() and not _tracing():
            self._selfcheck_write(layer_id, loc64, c, keep)
        elif self._selfcheck and not self._selfcheck_gate_logged:
            # The self-check silently never fired on two runs, and "it ran and
            # matched perfectly" is indistinguishable from "it never ran" unless
            # the gate says which. Report the three guard values once.
            self._selfcheck_gate_logged = True
            logger.warning(
                "[MLAPacked] selfcheck requested but gated off: budget=%d "
                "capturing=%s tracing=%s",
                self._selfcheck_budget, _is_capturing(), _tracing(),
            )

    def set_kv_buffer(self, layer, loc, cache_k, cache_v):
        """NSA/triton write path: ``cache_k`` is ``[c_kv | k_pe]`` concatenated."""
        R = self.kv_lora_rank
        self._log_write_path_once("set_kv_buffer(packed)", cache_k=cache_k.dtype)
        self._packed_store(layer.layer_id, loc, cache_k[..., :R], cache_k[..., R:])

    def set_mla_kv_buffer(self, layer, loc, cache_k_nope, cache_k_rope):
        self._log_write_path_once(
            "set_mla_kv_buffer(packed)",
            k_nope=cache_k_nope.dtype,
            k_rope=cache_k_rope.dtype,
        )
        self._packed_store(layer.layer_id, loc, cache_k_nope, cache_k_rope)

    # ── read path ───────────────────────────────────────────────────────────

    def packed_read_operands(self, layer_id: int):
        """Everything a packed attention kernel needs for one layer."""
        li = layer_id - self.start_layer
        return (
            self.c_codes[li],
            self.c_params[li],
            self.rope_buf[li],
            self.hp_c[li] if self._latent_windows else None,
            self.hp_row_of_slot if self._latent_windows else None,
            self.hp_owner_of_row if self._latent_windows else None,
            self._group_size,
            self._lloyd_max,
        )

    def materialize_rows(self, layer_id: int, slots: torch.Tensor,
                         out: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Dequantize ``slots`` into a dense ``[n, 1, R+rope]`` BF16 block.

        This is the reference read: whatever a fused kernel does, it must agree
        with this. It is also the production path for extend-with-prefix, where
        the row set is small (the reused prefix) and a dense staging buffer is
        cheaper than a second specialised kernel.
        """
        from sglang.QuantKernel.mla_latent_int2 import (
            assemble_rows,
            gather_dequant_rows,
        )

        li = layer_id - self.start_layer
        R = self.kv_lora_rank
        D = self.latent_row_dim()
        slots = slots.reshape(-1)
        n = slots.numel()
        slots32 = slots.to(torch.int32)

        if self._deq_scratch is None or self._deq_scratch.shape[0] < n:
            self._deq_scratch = torch.empty(
                (max(n, 1024), R), dtype=self.dtype, device=self.device
            )
        c_rot = self._deq_scratch[:n]
        gather_dequant_rows(
            slots32, self.c_codes[li], self.c_params[li], c_rot,
            self._group_size, self._lloyd_max,
        )
        if out is None:
            if self._read_scratch is None or self._read_scratch.shape[0] < n:
                self._read_scratch = torch.empty(
                    (max(n, 1024), 1, D), dtype=self.dtype, device=self.device
                )
            out = self._read_scratch[:n]
        assemble_rows(
            c_rot,
            slots32,
            self.rope_buf[li],
            self.hp_c[li] if self._latent_windows else None,
            self.hp_row_of_slot if self._latent_windows else None,
            self.hp_owner_of_row if self._latent_windows else None,
            out.view(n, D),
        )
        return out.view(n, 1, D)

    def _selfcheck_write(self, layer_id: int, loc: torch.Tensor,
                         c_rot: torch.Tensor, keep) -> None:
        """Read back the rows just written and compare to the fake-quant value.

        This is the equivalence statement that makes a packed score comparable
        to the fake-quant 80.30: for the same input latent, the packed store
        plus the read path return what ``_fake_quant_int2_groupwise`` returns --
        except on window rows, which must come back untouched.

        It runs on a budget and then stops, so it covers the first few hundred
        writes (every layer, both forward modes) rather than the whole run. A
        divergence that only appears later would be missed; the claim is
        "equivalent over the covered writes", not "equivalent for all time".
        """
        from sglang.QuantKernel.mla_latent_int2 import quantize_dequantize_reuse

        self._selfcheck_budget -= 1
        self._selfcheck_n += 1
        _c, _p, deq = quantize_dequantize_reuse(
            c_rot, self._group_size, self._lloyd_max
        )
        want = deq.to(self.dtype)
        if keep is not None:
            want = torch.where(keep.unsqueeze(-1), c_rot.to(self.dtype), want)
        got = self.materialize_rows(layer_id, loc)[:, 0, : self.kv_lora_rank]
        diff = (got.float() - want.float()).abs()
        rel = float(
            diff.max() / want.float().abs().max().clamp(min=1e-6)
        )
        if (self._selfcheck_n == 1 or rel > self._selfcheck_worst
                or self._selfcheck_budget == 0):
            self._selfcheck_worst = max(self._selfcheck_worst, rel)
            logger.info(
                "[MLAPacked] selfcheck layer=%d rows=%d max_abs=%.3e rel=%.3e "
                "worst_rel=%.3e over %d writes (budget left %d)",
                layer_id, int(loc.numel()), float(diff.max()), rel,
                self._selfcheck_worst, self._selfcheck_n, self._selfcheck_budget,
            )

    # ── arena audit ─────────────────────────────────────────────────────────

    def audit_window_arena(self, tag: str = "") -> None:
        """Count the BF16 window rows a *reader* would actually accept.

        The write-side self-check and the teacher-forced NLL comparison both
        clear the packed path, and the NLL run puts it at 5% of the cost of
        quantizing at all -- on a corpus calibrated to reproduce the known
        +2.7% PPL. But teacher forcing is one prefill: the ring is written once
        and the ``owner`` tag, the wrap and the eviction are never exercised.
        A GPQA answer is 10,000+ tokens, so the ring wraps ~20 times and every
        decode step depends on that tag being right.

        This audits the invariant a reader relies on, using only pool state:
        for each live request, the positions that *should* be BF16 are
        ``[0, P)`` and ``[seq-W, seq)``, so the count of slots satisfying
        ``hp_row_of_slot[slot] >= 0 and hp_owner_of_row[row] == slot`` should be
        ``min(P, seq) + min(W, max(seq-P, 0))``. Anything materially below that
        means the window silently degraded to INT2 during decode, which is
        exactly the shape of the observed loss: long generations, lower
        accuracy-given-answered, never uniquely right.
        """
        if not self._latent_windows:
            return
        meta = self._fb_window_meta
        if meta is None or meta.get("fallback") or meta["req_to_token"] is None:
            return
        seq_lens = meta["seq_lens"]
        req_idx = meta["req_pool_indices"]
        if seq_lens is None or req_idx is None or seq_lens.numel() == 0:
            return

        P, W = self._win_p, self._win_r
        r2t = meta["req_to_token"]
        want_total = 0
        got_total = 0
        # One request is enough to detect a broken tag and keeps this cheap
        # enough to leave on; a per-request loop over a batch of 64 inside a
        # decode step is not.
        for i in range(min(1, int(req_idx.numel()))):
            seq = int(seq_lens[i].item())
            row = int(req_idx[i].item())
            if seq <= 0:
                continue
            pos = torch.cat([
                torch.arange(min(P, seq), device=r2t.device),
                torch.arange(max(seq - W, P), seq, device=r2t.device),
            ])
            if pos.numel() == 0:
                continue
            slots = r2t[row, pos].to(torch.int64)
            rows = self.hp_row_of_slot[slots].to(torch.int64)
            ok = rows >= 0
            owner = self.hp_owner_of_row[rows.clamp(min=0)]
            ok = ok & (owner == slots.to(owner.dtype))
            want_total += int(pos.numel())
            got_total += int(ok.sum().item())

        if want_total == 0:
            return
        frac = got_total / want_total
        self._audit_worst = min(getattr(self, "_audit_worst", 1.0), frac)
        logger.info(
            "[MLAPacked] arena audit%s seq_lens[0]=%d expected=%d accepted=%d "
            "(%.1f%%, worst %.1f%%)",
            f" {tag}" if tag else "", int(seq_lens[0].item()),
            want_total, got_total, 100.0 * frac, 100.0 * self._audit_worst,
        )


class MLAPackedInt2KVPool(_PackedLatentMixin, MLATokenToKVPool):
    """Plain-MLA (DeepSeek-V2 style) pool with packed INT2 latent storage."""

    def __init__(self, *args, rotation_path: str = "", group_size: int = 128,
                 lloyd_max: bool = False, max_reqs: int = 64,
                 selfcheck: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_packed(rotation_path, group_size, lloyd_max, max_reqs, selfcheck)

    def _create_buffers(self):
        # The BF16 latent cache this class replaces. Allocating it would defeat
        # the entire point, so stand in with an empty list: everything that
        # walked ``kv_buffer`` is overridden above and fails loudly otherwise.
        self.kv_buffer = []


class NSAPackedInt2KVPool(_PackedLatentMixin, NSATokenToKVPool):
    """NSA/DSA pool (GLM-5.2) with packed INT2 latent storage.

    The indexer's ``index_k_with_scale_buffer`` is untouched: it is a separate
    fp8 buffer, it is what selects tokens rather than what attention reads, and
    2-bit'ing the selector is a different experiment.
    """

    def __init__(self, *args, rotation_path: str = "", group_size: int = 128,
                 lloyd_max: bool = False, max_reqs: int = 64,
                 selfcheck: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_packed(rotation_path, group_size, lloyd_max, max_reqs, selfcheck)

    def _create_buffers(self):
        self.kv_buffer = []
