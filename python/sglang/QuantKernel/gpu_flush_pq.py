"""Decode-time HP -> quant demotion for pools whose K or V tier is
product-quantized. Reuses the INT2 flush plan and remap; only the row
encoding differs."""

from __future__ import annotations

import torch

from sglang.QuantKernel.gpu_flush_int2 import FlushPlan, _launch_flush_remap
from sglang.QuantKernel.oscar_pq_kv import pq_decode_at_locs, pq_encode
from sglang.QuantKernel.oscar_rotation_clip_int2_kv import (
    _launch_grouped_clip_int2,
    _launch_single_clip_int2,
)
from sglang.srt.mem_cache.kv_quant_kernels import _get_num_scale_groups


def _encode_pq_rows(rows, dst, codes, codes2, book_set, local: int, head_dim: int) -> None:
    pq_encode(rows, dst, codes, book_set.codebooks[local], book_set.norms[local])
    if book_set.residual:
        recon = pq_decode_at_locs(
            codes, dst, book_set.codebooks[local], head_dim=head_dim
        ).to(rows.dtype)
        pq_encode(
            (rows - recon).contiguous(),
            dst,
            codes2,
            book_set.stage2_codebooks[local],
            book_set.stage2_norms[local],
        )


def _encode_int2_rows(rows, dst, buf, sz_buf, clip_ratio: float, lloyd_max: bool) -> None:
    if _get_num_scale_groups(sz_buf) == 1:
        _launch_single_clip_int2(
            rows, dst, buf, sz_buf, clip_ratio, hp_global_offset=None, lloyd_max=lloyd_max
        )
    else:
        _launch_grouped_clip_int2(rows, dst, buf, sz_buf, clip_ratio, hp_global_offset=None)


def gpu_flush_pq_apply(
    plan: FlushPlan,
    *,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    kv_pool,
    apply_remap: bool = True,
) -> None:
    """Apply phase for a PQ-quantized pool: encode the planned HP rows into
    their quant slots, then point ``req_to_token`` at them.

    Every planned row is encoded, invalid ones from HP slot 0 into a
    destination slot nothing references; that keeps the step free of a
    host-side row mask, like the fused INT2 kernel.
    """
    src = plan.src_hp_slot.clamp(min=0)
    dst = plan.dst_quant_slots
    k_set = kv_pool.pq_k_set
    v_set = kv_pool.pq_v_set
    for local in range(kv_pool.layer_num):
        k_rows = kv_pool.hp_k_buffer[local][src]
        if k_set is not None:
            _encode_pq_rows(
                k_rows,
                dst,
                kv_pool.k_buffer[local],
                kv_pool.k_buffer2[local] if kv_pool.k_buffer2 is not None else None,
                k_set,
                local,
                kv_pool.head_dim,
            )
        else:
            _encode_int2_rows(
                k_rows,
                dst,
                kv_pool.k_buffer[local],
                kv_pool.k_scales_zeros[local],
                kv_pool._k_clip_ratio,
                kv_pool._lloyd_max,
            )
        v_rows = kv_pool.hp_v_buffer[local][src]
        if v_set is not None:
            _encode_pq_rows(
                v_rows, dst, kv_pool.v_buffer[local], None, v_set, local, kv_pool.v_head_dim
            )
        else:
            _encode_int2_rows(
                v_rows,
                dst,
                kv_pool.v_buffer[local],
                kv_pool.v_scales_zeros[local],
                kv_pool._v_clip_ratio,
                kv_pool._lloyd_max,
            )
    if apply_remap:
        _launch_flush_remap(plan, req_pool_indices, req_to_token, plan.bs, plan.flush_interval)
