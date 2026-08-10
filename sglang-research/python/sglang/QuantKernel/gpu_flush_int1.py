"""
GPU-only HP -> int1 flush for the unified mixed KV pool.

INT1 variant of :mod:`sglang.QuantKernel.gpu_flush_int2`. Same plan/apply
structure and same FlushPlan dataclass shape. The only differences are:

  * The quant cache holds ``head_dim // 8`` bytes per (token, head)
    instead of ``head_dim // 4``.
  * The quant kernel packs 8 quant slots per byte (bit ``i`` ↔ position
    ``i * BLOCK_OCTANT + bo`` of the row).
  * ``max_q == 1`` so ``scale = range`` (no divisor) and the quant slot
    holds {0, 1}.

Scale/zero layout (interleaved ``(scale, zero)`` per group) is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import triton
import triton.language as tl


@dataclass
class FlushPlanInt1:
    """Output of ``gpu_flush_int1_plan``; consumed by ``gpu_flush_int1_apply``.

    Same shape as the INT2 ``FlushPlan``; duplicated as its own dataclass so
    the int1 / int2 plan objects don't accidentally cross paths.
    """

    returned_slot_ids: torch.Tensor
    src_hp_slot: torch.Tensor
    flush_pos: torch.Tensor
    valid_mask: torch.Tensor
    dst_quant_slots: torch.Tensor
    bs: int
    flush_interval: int


# ---------------------------------------------------------------------------
# Plan kernel — identical to int2 (it does not touch the quant arena).
# ---------------------------------------------------------------------------


@triton.jit
def _flush_plan_kernel_int1(
    seq_lens_ptr,
    prefix_lens_ptr,
    req_pool_indices_ptr,
    dst_quant_slot_ptr,
    req_to_token_ptr,
    flush_mask_ptr,
    src_hp_slot_out_ptr,
    returned_slot_ids_ptr,
    flush_pos_out_ptr,
    valid_mask_out_ptr,
    max_ctx,
    rtt_stride_row,
    HP_PREFIX_TOKENS: tl.constexpr,
    HP_RECENT_TOKENS: tl.constexpr,
    HP_OFFSET: tl.constexpr,
    FLUSH_INTERVAL: tl.constexpr,
):
    i = tl.program_id(0)
    do_flush = tl.load(flush_mask_ptr + i).to(tl.int32)
    seq_len = tl.load(seq_lens_ptr + i).to(tl.int32)
    prefix_len = tl.load(prefix_lens_ptr + i).to(tl.int32)
    req_pool_idx = tl.load(req_pool_indices_ptr + i).to(tl.int64)

    for j in tl.static_range(FLUSH_INTERVAL):
        out_idx = i * FLUSH_INTERVAL + j
        dst_q = tl.load(dst_quant_slot_ptr + out_idx).to(tl.int64)

        valid = 0
        src_hp = tl.full((), -1, tl.int64)
        flush_pos = tl.full((), -1, tl.int32)

        if do_flush == 1 and HP_RECENT_TOKENS > 0:
            fp = seq_len - HP_RECENT_TOKENS - (FLUSH_INTERVAL - 1) + j
            if fp >= prefix_len and fp >= 0:
                loc = tl.load(
                    req_to_token_ptr + req_pool_idx * rtt_stride_row + fp.to(tl.int64)
                ).to(tl.int64)
                if loc >= HP_OFFSET:
                    src_hp = loc - HP_OFFSET
                    valid = 1
                    flush_pos = fp

        tl.store(valid_mask_out_ptr + out_idx, tl.full((), valid, tl.int8))
        tl.store(flush_pos_out_ptr + out_idx, flush_pos)

        if valid == 1:
            tl.store(returned_slot_ids_ptr + out_idx, src_hp + HP_OFFSET)
            tl.store(src_hp_slot_out_ptr + out_idx, src_hp)
        else:
            tl.store(returned_slot_ids_ptr + out_idx, dst_q)
            tl.store(src_hp_slot_out_ptr + out_idx, -1)


# ---------------------------------------------------------------------------
# Fused INT1 quant kernel
# ---------------------------------------------------------------------------


@triton.jit
def _fused_flush_quant_body_int1(
    hp_base,
    quant_base,
    sz_base,
    src_hp_slot,
    dst_quant_slot,
    active,
    head_idx,
    HP_STRIDE_LOC: tl.constexpr,
    HP_STRIDE_HEAD: tl.constexpr,
    HP_STRIDE_DIM: tl.constexpr,
    Q_STRIDE_LOC: tl.constexpr,
    Q_STRIDE_HEAD: tl.constexpr,
    Q_STRIDE_DIM: tl.constexpr,
    SZ_STRIDE_LOC: tl.constexpr,
    SZ_STRIDE_HEAD: tl.constexpr,
    SZ_STRIDE_DIM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_OCTANT: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_TOK: tl.constexpr,
    CLIP_INDEX: tl.constexpr,
    BSEARCH_ITERS: tl.constexpr,
    LLOYD_MAX: tl.constexpr,
):
    """Quantize ``BLOCK_TOK`` (src_hp_slot, head) HP rows into int1 at the
    matching ``dst_quant_slot``s. 8-way pack mirrors INT2's quartered split.

    When ``LLOYD_MAX`` is set the per-group binary quantizer matches the
    prefill grouped set_kv kernel: boundary at the group mean, centroids at
    ``mean ± std*sqrt(2/pi)`` (scale = ``2*sqrt(2/pi)*std``). Without it the
    legacy uniform binary (centroids at the clipped min/max) is used, which is
    catastrophic for 1-bit since both levels land at the distribution extremes.
    """
    full_offs = tl.arange(0, HEAD_DIM)
    base = (
        src_hp_slot[:, None] * HP_STRIDE_LOC
        + head_idx * HP_STRIDE_HEAD
        + full_offs[None, :] * HP_STRIDE_DIM
    )
    acc = tl.load(
        hp_base + base,
        mask=active[:, None],
        other=0.0,
    ).to(tl.float32)

    if CLIP_INDEX >= 0:
        abs_acc = tl.abs(acc)
        if BSEARCH_ITERS > 0:
            target_above = HEAD_DIM - CLIP_INDEX
            thr_lo = tl.zeros([BLOCK_TOK], dtype=tl.float32)
            thr_hi = tl.max(abs_acc, axis=1)
            for _ in tl.static_range(BSEARCH_ITERS):
                thr_mid = (thr_lo + thr_hi) * 0.5
                cnt_above = tl.sum((abs_acc > thr_mid[:, None]).to(tl.int32), axis=1)
                too_many = cnt_above > target_above
                thr_lo = tl.where(too_many, thr_mid, thr_lo)
                thr_hi = tl.where(too_many, thr_hi, thr_mid)
            thr = thr_hi
        else:
            sorted_acc = tl.sort(abs_acc)
            pick = (full_offs == CLIP_INDEX)[None, :]
            thr = tl.sum(tl.where(pick, sorted_acc, 0.0), axis=1)
        acc = tl.minimum(
            tl.maximum(acc, -thr[:, None]),
            thr[:, None],
        )

    grouped = tl.reshape(acc, (BLOCK_TOK, NUM_GROUPS, GROUP_SIZE))
    if LLOYD_MAX:
        LM_C_1BIT: tl.constexpr = 0.79788456  # sqrt(2/pi)
        group_mean = tl.sum(grouped, axis=2) / GROUP_SIZE
        group_diff = grouped - group_mean[:, :, None]
        group_var = tl.sum(group_diff * group_diff, axis=2) / GROUP_SIZE
        group_std = tl.sqrt(group_var + 1e-8)
        scale = tl.maximum(2.0 * LM_C_1BIT * group_std, 1e-8)
        zero = 0.5 - group_mean / scale
    else:
        val_min = tl.min(grouped, axis=2)
        val_max = tl.max(grouped, axis=2)
        scale = tl.maximum(val_max - val_min, 1e-8)  # max_q = 1
        zero = tl.math.div_rn(-val_min, scale)

    # Octant split via reshape + permute + split×3 on fp32 acc, broadcasting
    # per-group scale/zero through the same pipeline. Mirrors INT2's quartered
    # split (split×2); just one more level deep.
    acc_r = tl.reshape(acc, (BLOCK_TOK, 8, BLOCK_OCTANT))
    acc_p = tl.permute(acc_r, (0, 2, 1))
    acc_s = tl.reshape(acc_p, (BLOCK_TOK, BLOCK_OCTANT, 2, 2, 2))
    a_lo, a_hi = tl.split(acc_s)
    a_e0, a_e2 = tl.split(a_lo)
    a_e1, a_e3 = tl.split(a_hi)
    vals0, vals4 = tl.split(a_e0)
    vals2, vals6 = tl.split(a_e2)
    vals1, vals5 = tl.split(a_e1)
    vals3, vals7 = tl.split(a_e3)

    scale_3d = tl.broadcast_to(scale[:, :, None], (BLOCK_TOK, NUM_GROUPS, GROUP_SIZE))
    zero_3d = tl.broadcast_to(zero[:, :, None], (BLOCK_TOK, NUM_GROUPS, GROUP_SIZE))
    scale_flat = tl.reshape(scale_3d, (BLOCK_TOK, HEAD_DIM))
    zero_flat = tl.reshape(zero_3d, (BLOCK_TOK, HEAD_DIM))

    sr = tl.reshape(scale_flat, (BLOCK_TOK, 8, BLOCK_OCTANT))
    sp = tl.permute(sr, (0, 2, 1))
    ss = tl.reshape(sp, (BLOCK_TOK, BLOCK_OCTANT, 2, 2, 2))
    s_lo, s_hi = tl.split(ss)
    s_e0, s_e2 = tl.split(s_lo)
    s_e1, s_e3 = tl.split(s_hi)
    s0, s4 = tl.split(s_e0)
    s2, s6 = tl.split(s_e2)
    s1, s5 = tl.split(s_e1)
    s3, s7 = tl.split(s_e3)

    zr = tl.reshape(zero_flat, (BLOCK_TOK, 8, BLOCK_OCTANT))
    zp = tl.permute(zr, (0, 2, 1))
    zs = tl.reshape(zp, (BLOCK_TOK, BLOCK_OCTANT, 2, 2, 2))
    z_lo, z_hi = tl.split(zs)
    z_e0, z_e2 = tl.split(z_lo)
    z_e1, z_e3 = tl.split(z_hi)
    z0, z4 = tl.split(z_e0)
    z2, z6 = tl.split(z_e2)
    z1, z5 = tl.split(z_e1)
    z3, z7 = tl.split(z_e3)

    # Clamp the rounding result to {0, 1} before packing. For the uniform path
    # this is a no-op (qf in [0.5, 1.5] -> trunc gives {0, 1}); for the LM path
    # it is required, since the LM scale = 2*c*std makes qf = (val-mean)/scale +
    # 1.0 land in [-1, 2] for tail values, which would otherwise pack as 2 or
    # wrap to 255 and corrupt the byte. Clamped, the boundary sits exactly at
    # val == mean (q = 1 iff val >= mean), matching the prefill grouped kernel.
    q0 = tl.minimum(tl.maximum(tl.math.div_rn(vals0, s0) + z0 + 0.5, 0.0), 1.0).to(
        tl.uint8
    )
    q1 = tl.minimum(tl.maximum(tl.math.div_rn(vals1, s1) + z1 + 0.5, 0.0), 1.0).to(
        tl.uint8
    )
    q2 = tl.minimum(tl.maximum(tl.math.div_rn(vals2, s2) + z2 + 0.5, 0.0), 1.0).to(
        tl.uint8
    )
    q3 = tl.minimum(tl.maximum(tl.math.div_rn(vals3, s3) + z3 + 0.5, 0.0), 1.0).to(
        tl.uint8
    )
    q4 = tl.minimum(tl.maximum(tl.math.div_rn(vals4, s4) + z4 + 0.5, 0.0), 1.0).to(
        tl.uint8
    )
    q5 = tl.minimum(tl.maximum(tl.math.div_rn(vals5, s5) + z5 + 0.5, 0.0), 1.0).to(
        tl.uint8
    )
    q6 = tl.minimum(tl.maximum(tl.math.div_rn(vals6, s6) + z6 + 0.5, 0.0), 1.0).to(
        tl.uint8
    )
    q7 = tl.minimum(tl.maximum(tl.math.div_rn(vals7, s7) + z7 + 0.5, 0.0), 1.0).to(
        tl.uint8
    )
    packed = (
        q0
        | (q1 << 1)
        | (q2 << 2)
        | (q3 << 3)
        | (q4 << 4)
        | (q5 << 5)
        | (q6 << 6)
        | (q7 << 7)
    )

    dim_offs_o = tl.arange(0, BLOCK_OCTANT)
    cache_offset = (
        dst_quant_slot[:, None] * Q_STRIDE_LOC
        + head_idx * Q_STRIDE_HEAD
        + dim_offs_o[None, :] * Q_STRIDE_DIM
    )
    tl.store(quant_base + cache_offset, packed, mask=active[:, None])

    group_ids = tl.arange(0, NUM_GROUPS)
    sz_offset_base = dst_quant_slot[:, None] * SZ_STRIDE_LOC + head_idx * SZ_STRIDE_HEAD
    tl.store(
        sz_base + sz_offset_base + (group_ids[None, :] * 2) * SZ_STRIDE_DIM,
        scale,
        mask=active[:, None],
    )
    tl.store(
        sz_base + sz_offset_base + (group_ids[None, :] * 2 + 1) * SZ_STRIDE_DIM,
        zero,
        mask=active[:, None],
    )


@triton.jit
def _fused_flush_quant_kernel_int1(
    hp_k_ptrs_ptr,
    hp_v_ptrs_ptr,
    quant_k_ptrs_ptr,
    quant_v_ptrs_ptr,
    k_sz_ptrs_ptr,
    v_sz_ptrs_ptr,
    hp_k_sample_ptr,
    hp_v_sample_ptr,
    quant_k_sample_ptr,
    quant_v_sample_ptr,
    k_sz_sample_ptr,
    v_sz_sample_ptr,
    src_hp_slot_ptr,
    dst_quant_slot_ptr,
    valid_mask_ptr,
    num_flush_tokens,
    num_heads,
    num_layers,
    HP_K_STRIDE_LOC: tl.constexpr,
    HP_K_STRIDE_HEAD: tl.constexpr,
    HP_K_STRIDE_DIM: tl.constexpr,
    HP_V_STRIDE_LOC: tl.constexpr,
    HP_V_STRIDE_HEAD: tl.constexpr,
    HP_V_STRIDE_DIM: tl.constexpr,
    Q_K_STRIDE_LOC: tl.constexpr,
    Q_K_STRIDE_HEAD: tl.constexpr,
    Q_K_STRIDE_DIM: tl.constexpr,
    Q_V_STRIDE_LOC: tl.constexpr,
    Q_V_STRIDE_HEAD: tl.constexpr,
    Q_V_STRIDE_DIM: tl.constexpr,
    K_SZ_STRIDE_LOC: tl.constexpr,
    K_SZ_STRIDE_HEAD: tl.constexpr,
    K_SZ_STRIDE_DIM: tl.constexpr,
    V_SZ_STRIDE_LOC: tl.constexpr,
    V_SZ_STRIDE_HEAD: tl.constexpr,
    V_SZ_STRIDE_DIM: tl.constexpr,
    K_HEAD_DIM: tl.constexpr,
    K_BLOCK_OCTANT: tl.constexpr,
    K_NUM_GROUPS: tl.constexpr,
    K_GROUP_SIZE: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    V_BLOCK_OCTANT: tl.constexpr,
    V_NUM_GROUPS: tl.constexpr,
    V_GROUP_SIZE: tl.constexpr,
    BLOCK_TOK: tl.constexpr,
    K_CLIP_INDEX: tl.constexpr,
    V_CLIP_INDEX: tl.constexpr,
    K_BSEARCH_ITERS: tl.constexpr,
    V_BSEARCH_ITERS: tl.constexpr,
    LLOYD_MAX: tl.constexpr,
):
    pid_tok = tl.program_id(0)
    head = tl.program_id(1)
    layer = tl.program_id(2)
    if head >= num_heads or layer >= num_layers:
        return

    tok_offs = pid_tok * BLOCK_TOK + tl.arange(0, BLOCK_TOK)
    tok_mask = tok_offs < num_flush_tokens

    valid = tl.load(valid_mask_ptr + tok_offs, mask=tok_mask, other=0).to(tl.int32)
    if tl.max(valid, axis=0) == 0:
        return

    active = tok_mask & (valid != 0)
    src = tl.load(src_hp_slot_ptr + tok_offs, mask=tok_mask, other=0).to(tl.int64)
    dst = tl.load(dst_quant_slot_ptr + tok_offs, mask=tok_mask, other=0).to(tl.int64)
    head64 = head.to(tl.int64)

    hp_k_base = tl.load(hp_k_ptrs_ptr + layer).to(
        tl.pointer_type(hp_k_sample_ptr.dtype.element_ty)
    )
    q_k_base = tl.load(quant_k_ptrs_ptr + layer).to(
        tl.pointer_type(quant_k_sample_ptr.dtype.element_ty)
    )
    sz_k_base = tl.load(k_sz_ptrs_ptr + layer).to(
        tl.pointer_type(k_sz_sample_ptr.dtype.element_ty)
    )
    _fused_flush_quant_body_int1(
        hp_k_base,
        q_k_base,
        sz_k_base,
        src,
        dst,
        active,
        head64,
        HP_K_STRIDE_LOC,
        HP_K_STRIDE_HEAD,
        HP_K_STRIDE_DIM,
        Q_K_STRIDE_LOC,
        Q_K_STRIDE_HEAD,
        Q_K_STRIDE_DIM,
        K_SZ_STRIDE_LOC,
        K_SZ_STRIDE_HEAD,
        K_SZ_STRIDE_DIM,
        K_HEAD_DIM,
        K_BLOCK_OCTANT,
        K_NUM_GROUPS,
        K_GROUP_SIZE,
        BLOCK_TOK,
        K_CLIP_INDEX,
        K_BSEARCH_ITERS,
        LLOYD_MAX,
    )
    hp_v_base = tl.load(hp_v_ptrs_ptr + layer).to(
        tl.pointer_type(hp_v_sample_ptr.dtype.element_ty)
    )
    q_v_base = tl.load(quant_v_ptrs_ptr + layer).to(
        tl.pointer_type(quant_v_sample_ptr.dtype.element_ty)
    )
    sz_v_base = tl.load(v_sz_ptrs_ptr + layer).to(
        tl.pointer_type(v_sz_sample_ptr.dtype.element_ty)
    )
    _fused_flush_quant_body_int1(
        hp_v_base,
        q_v_base,
        sz_v_base,
        src,
        dst,
        active,
        head64,
        HP_V_STRIDE_LOC,
        HP_V_STRIDE_HEAD,
        HP_V_STRIDE_DIM,
        Q_V_STRIDE_LOC,
        Q_V_STRIDE_HEAD,
        Q_V_STRIDE_DIM,
        V_SZ_STRIDE_LOC,
        V_SZ_STRIDE_HEAD,
        V_SZ_STRIDE_DIM,
        V_HEAD_DIM,
        V_BLOCK_OCTANT,
        V_NUM_GROUPS,
        V_GROUP_SIZE,
        BLOCK_TOK,
        V_CLIP_INDEX,
        V_BSEARCH_ITERS,
        LLOYD_MAX,
    )


# ---------------------------------------------------------------------------
# Remap kernel — identical to INT2.
# ---------------------------------------------------------------------------


@triton.jit
def _flush_remap_kernel_int1(
    req_pool_indices_ptr,
    flush_pos_ptr,
    dst_quant_slot_ptr,
    valid_mask_ptr,
    req_to_token_ptr,
    rtt_stride_row,
    FLUSH_INTERVAL: tl.constexpr,
):
    i = tl.program_id(0)
    j = tl.program_id(1)
    out_idx = i * FLUSH_INTERVAL + j
    valid = tl.load(valid_mask_ptr + out_idx).to(tl.int32)
    if valid == 0:
        return
    req = tl.load(req_pool_indices_ptr + i).to(tl.int64)
    fp = tl.load(flush_pos_ptr + out_idx).to(tl.int64)
    dst = tl.load(dst_quant_slot_ptr + out_idx).to(tl.int32)
    tl.store(req_to_token_ptr + req * rtt_stride_row + fp, dst)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _resolve_kv_quant_config_int1(
    head_dim: int, num_scale_groups: int
) -> Tuple[int, int, int]:
    """Return (BLOCK_OCTANT, NUM_GROUPS, GROUP_SIZE) for the int1 quant kernel."""
    if head_dim % num_scale_groups != 0:
        raise ValueError(
            f"head_dim ({head_dim}) must be divisible by num_scale_groups "
            f"({num_scale_groups})"
        )
    if (head_dim & (head_dim - 1)) != 0:
        raise ValueError(
            f"head_dim ({head_dim}) must be a power of two for the int1 "
            f"flush quant kernel"
        )
    if head_dim % 8 != 0:
        raise ValueError(
            f"head_dim ({head_dim}) must be a multiple of 8 for int1 packing"
        )
    block_octant = head_dim // 8
    group_size = head_dim // num_scale_groups
    return block_octant, num_scale_groups, group_size


def _flush_clip_index_int1(clip_ratio: float, head_dim: int) -> int:
    if clip_ratio <= 0.0:
        return -1
    idx = int(clip_ratio * head_dim)
    if idx >= head_dim:
        idx = head_dim - 1
    if idx < 0:
        idx = 0
    return idx


def _flush_elements_per_thread_int1(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 8
    if dtype.is_floating_point and dtype.itemsize == 1:
        return 16
    raise AssertionError(
        f"flush quant kernel requires bf16 or fp8 HP dtype, got {dtype}"
    )


def _flush_block_tok_and_num_warps_int1(
    flush_interval: int, head_dim: int, elements_per_thread: int
) -> Tuple[int, int]:
    fi_pow2 = triton.next_power_of_2(max(1, int(flush_interval)))
    block_tok = 2
    while block_tok * head_dim < 32 * elements_per_thread:
        block_tok *= 2
    if block_tok > fi_pow2:
        block_tok = fi_pow2
    total_elems = block_tok * head_dim
    vectors_per_warp = 32 * elements_per_thread
    num_warps = triton.next_power_of_2(
        max(1, triton.cdiv(total_elems, vectors_per_warp))
    )
    return block_tok, num_warps


def gpu_flush_int1_plan(
    *,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    dst_quant_slots: torch.Tensor,
    req_to_token: torch.Tensor,
    flush_mask: torch.Tensor,
    hp_prefix_tokens: int,
    hp_recent_tokens: int,
    hp_global_offset: int,
    flush_interval: int,
):
    bs = int(seq_lens.shape[0])
    if bs == 0 or flush_interval <= 0:
        return None

    assert req_to_token.dtype == torch.int32
    assert seq_lens.dtype == torch.int32
    assert prefix_lens.dtype == torch.int32
    assert req_pool_indices.dtype == torch.int64
    assert dst_quant_slots.dtype == torch.int64
    assert dst_quant_slots.numel() == bs * flush_interval
    assert flush_mask.shape == (bs,), (
        f"flush_mask shape {tuple(flush_mask.shape)} != ({bs},)"
    )
    flush_mask_i8 = flush_mask.to(torch.int8)

    device = seq_lens.device
    total_flush_slots = bs * flush_interval
    returned_slot_ids = torch.empty(
        (total_flush_slots,), dtype=torch.int64, device=device
    )
    src_hp_slot = torch.empty((total_flush_slots,), dtype=torch.int64, device=device)
    flush_pos = torch.empty((total_flush_slots,), dtype=torch.int32, device=device)
    valid_mask = torch.empty((total_flush_slots,), dtype=torch.int8, device=device)

    rtt_stride_row = int(req_to_token.stride(0))

    _flush_plan_kernel_int1[(bs,)](
        seq_lens,
        prefix_lens,
        req_pool_indices,
        dst_quant_slots,
        req_to_token,
        flush_mask_i8,
        src_hp_slot,
        returned_slot_ids,
        flush_pos,
        valid_mask,
        int(req_to_token.shape[1]),
        rtt_stride_row,
        HP_PREFIX_TOKENS=int(hp_prefix_tokens),
        HP_RECENT_TOKENS=int(hp_recent_tokens),
        HP_OFFSET=int(hp_global_offset),
        FLUSH_INTERVAL=int(flush_interval),
        num_warps=1,
        num_stages=1,
    )

    return FlushPlanInt1(
        returned_slot_ids=returned_slot_ids,
        src_hp_slot=src_hp_slot,
        flush_pos=flush_pos,
        valid_mask=valid_mask,
        dst_quant_slots=dst_quant_slots,
        bs=bs,
        flush_interval=int(flush_interval),
    )


def gpu_flush_int1_apply(
    plan: FlushPlanInt1,
    *,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    hp_k_ptrs: torch.Tensor,
    hp_v_ptrs: torch.Tensor,
    quant_k_ptrs: torch.Tensor,
    quant_v_ptrs: torch.Tensor,
    k_sz_ptrs: torch.Tensor,
    v_sz_ptrs: torch.Tensor,
    hp_k_sample: torch.Tensor,
    hp_v_sample: torch.Tensor,
    quant_k_sample: torch.Tensor,
    quant_v_sample: torch.Tensor,
    k_sz_sample: torch.Tensor,
    v_sz_sample: torch.Tensor,
    hp_k_strides: Tuple[int, int, int],
    hp_v_strides: Tuple[int, int, int],
    quant_k_strides: Tuple[int, int, int],
    quant_v_strides: Tuple[int, int, int],
    k_sz_strides: Tuple[int, int, int],
    v_sz_strides: Tuple[int, int, int],
    num_heads: int,
    head_dim: int,
    v_head_dim: int,
    k_num_scale_groups: int,
    v_num_scale_groups: int,
    num_layers: int,
    k_clip_ratio: float = 0.0,
    v_clip_ratio: float = 0.0,
    lloyd_max: bool = False,
) -> None:
    bs = plan.bs
    flush_interval = plan.flush_interval
    total_flush_slots = bs * flush_interval

    safe_src_hp_slot = plan.src_hp_slot.clamp(min=0)

    k_block_octant, k_num_groups, k_group_size = _resolve_kv_quant_config_int1(
        head_dim, k_num_scale_groups
    )
    v_block_octant, v_num_groups, v_group_size = _resolve_kv_quant_config_int1(
        v_head_dim, v_num_scale_groups
    )

    k_clip_index = _flush_clip_index_int1(k_clip_ratio, head_dim)
    v_clip_index = _flush_clip_index_int1(v_clip_ratio, v_head_dim)

    elements_per_thread = _flush_elements_per_thread_int1(hp_k_sample.dtype)
    block_tok, num_warps = _flush_block_tok_and_num_warps_int1(
        flush_interval, head_dim, elements_per_thread
    )
    grid = (
        triton.cdiv(total_flush_slots, block_tok),
        num_heads,
        int(num_layers),
    )
    _fused_flush_quant_kernel_int1[grid](
        hp_k_ptrs,
        hp_v_ptrs,
        quant_k_ptrs,
        quant_v_ptrs,
        k_sz_ptrs,
        v_sz_ptrs,
        hp_k_sample,
        hp_v_sample,
        quant_k_sample,
        quant_v_sample,
        k_sz_sample,
        v_sz_sample,
        safe_src_hp_slot,
        plan.dst_quant_slots,
        plan.valid_mask,
        total_flush_slots,
        num_heads,
        int(num_layers),
        HP_K_STRIDE_LOC=hp_k_strides[0],
        HP_K_STRIDE_HEAD=hp_k_strides[1],
        HP_K_STRIDE_DIM=hp_k_strides[2],
        HP_V_STRIDE_LOC=hp_v_strides[0],
        HP_V_STRIDE_HEAD=hp_v_strides[1],
        HP_V_STRIDE_DIM=hp_v_strides[2],
        Q_K_STRIDE_LOC=quant_k_strides[0],
        Q_K_STRIDE_HEAD=quant_k_strides[1],
        Q_K_STRIDE_DIM=quant_k_strides[2],
        Q_V_STRIDE_LOC=quant_v_strides[0],
        Q_V_STRIDE_HEAD=quant_v_strides[1],
        Q_V_STRIDE_DIM=quant_v_strides[2],
        K_SZ_STRIDE_LOC=k_sz_strides[0],
        K_SZ_STRIDE_HEAD=k_sz_strides[1],
        K_SZ_STRIDE_DIM=k_sz_strides[2],
        V_SZ_STRIDE_LOC=v_sz_strides[0],
        V_SZ_STRIDE_HEAD=v_sz_strides[1],
        V_SZ_STRIDE_DIM=v_sz_strides[2],
        K_HEAD_DIM=int(head_dim),
        K_BLOCK_OCTANT=k_block_octant,
        K_NUM_GROUPS=k_num_groups,
        K_GROUP_SIZE=k_group_size,
        V_HEAD_DIM=int(v_head_dim),
        V_BLOCK_OCTANT=v_block_octant,
        V_NUM_GROUPS=v_num_groups,
        V_GROUP_SIZE=v_group_size,
        BLOCK_TOK=block_tok,
        K_CLIP_INDEX=k_clip_index,
        V_CLIP_INDEX=v_clip_index,
        K_BSEARCH_ITERS=(int(head_dim).bit_length() - 1) if head_dim >= 64 else 0,
        V_BSEARCH_ITERS=(int(v_head_dim).bit_length() - 1) if v_head_dim >= 64 else 0,
        LLOYD_MAX=bool(lloyd_max),
        num_warps=num_warps,
        num_stages=1,
    )

    rtt_stride_row = int(req_to_token.stride(0))
    _flush_remap_kernel_int1[(bs, flush_interval)](
        req_pool_indices,
        plan.flush_pos,
        plan.dst_quant_slots,
        plan.valid_mask,
        req_to_token,
        rtt_stride_row,
        FLUSH_INTERVAL=int(flush_interval),
        num_warps=1,
        num_stages=1,
    )


def gpu_flush_int1(
    *,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    dst_quant_slots: torch.Tensor,
    req_to_token: torch.Tensor,
    flush_mask: torch.Tensor,
    hp_k_ptrs: torch.Tensor,
    hp_v_ptrs: torch.Tensor,
    quant_k_ptrs: torch.Tensor,
    quant_v_ptrs: torch.Tensor,
    k_sz_ptrs: torch.Tensor,
    v_sz_ptrs: torch.Tensor,
    hp_k_sample: torch.Tensor,
    hp_v_sample: torch.Tensor,
    quant_k_sample: torch.Tensor,
    quant_v_sample: torch.Tensor,
    k_sz_sample: torch.Tensor,
    v_sz_sample: torch.Tensor,
    hp_k_strides: Tuple[int, int, int],
    hp_v_strides: Tuple[int, int, int],
    quant_k_strides: Tuple[int, int, int],
    quant_v_strides: Tuple[int, int, int],
    k_sz_strides: Tuple[int, int, int],
    v_sz_strides: Tuple[int, int, int],
    hp_prefix_tokens: int,
    hp_recent_tokens: int,
    hp_global_offset: int,
    num_heads: int,
    head_dim: int,
    v_head_dim: int,
    k_num_scale_groups: int,
    v_num_scale_groups: int,
    num_layers: int,
    flush_interval: int,
    k_clip_ratio: float = 0.0,
    v_clip_ratio: float = 0.0,
    lloyd_max: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    plan = gpu_flush_int1_plan(
        seq_lens=seq_lens,
        prefix_lens=prefix_lens,
        req_pool_indices=req_pool_indices,
        dst_quant_slots=dst_quant_slots,
        req_to_token=req_to_token,
        flush_mask=flush_mask,
        hp_prefix_tokens=hp_prefix_tokens,
        hp_recent_tokens=hp_recent_tokens,
        hp_global_offset=hp_global_offset,
        flush_interval=flush_interval,
    )
    if plan is None:
        device = seq_lens.device
        empty = torch.empty((0,), dtype=torch.int64, device=device)
        mask = torch.empty((0,), dtype=torch.int8, device=device)
        return empty, mask

    gpu_flush_int1_apply(
        plan,
        req_pool_indices=req_pool_indices,
        req_to_token=req_to_token,
        hp_k_ptrs=hp_k_ptrs,
        hp_v_ptrs=hp_v_ptrs,
        quant_k_ptrs=quant_k_ptrs,
        quant_v_ptrs=quant_v_ptrs,
        k_sz_ptrs=k_sz_ptrs,
        v_sz_ptrs=v_sz_ptrs,
        hp_k_sample=hp_k_sample,
        hp_v_sample=hp_v_sample,
        quant_k_sample=quant_k_sample,
        quant_v_sample=quant_v_sample,
        k_sz_sample=k_sz_sample,
        v_sz_sample=v_sz_sample,
        hp_k_strides=hp_k_strides,
        hp_v_strides=hp_v_strides,
        quant_k_strides=quant_k_strides,
        quant_v_strides=quant_v_strides,
        k_sz_strides=k_sz_strides,
        v_sz_strides=v_sz_strides,
        num_heads=num_heads,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        k_num_scale_groups=k_num_scale_groups,
        v_num_scale_groups=v_num_scale_groups,
        num_layers=num_layers,
        k_clip_ratio=k_clip_ratio,
        v_clip_ratio=v_clip_ratio,
        lloyd_max=lloyd_max,
    )

    return plan.returned_slot_ids, plan.valid_mask
