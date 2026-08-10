"""
Oscar-style per-row quantile clip + int1 KV cache pack kernels.

INT1 mirrors :mod:`sglang.QuantKernel.oscar_rotation_clip_int2_kv` but with
``max_q == 1`` (two quant levels {0, 1}) and an 8-way byte pack. Storage is
``head_dim // 8`` packed uint8 bytes (vs ``head_dim // 4`` for INT2). The
scale/zero layout is identical to INT2 (interleaved ``(scale, zero)`` pairs
per group), so the dequantize formula ``(q - zero) * scale`` is unchanged
except that ``q`` now comes from a 1-bit slot.

The octanted split inside the kernel uses three rounds of ``tl.split``:
``(BLOCK_TOK, BLOCK_OCTANT, 2, 2, 2)`` → 2 → 4 → 8 leaves of shape
``(BLOCK_TOK, BLOCK_OCTANT)``. Mirrors the INT2 "quartered split via
``tl.split×2``" idiom one level deeper.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.mem_cache.kv_quant_kernels import (
    _get_num_scale_groups,
    _is_power_of_two,
)


# ---------------------------------------------------------------------------
# Fused threshold + clip + int1 pack kernels
# ---------------------------------------------------------------------------


@triton.jit
def _pretransformed_int1_set_kv_clip_single_kernel(
    input_ptr,
    loc_ptr,
    cache_ptr,
    scales_zeros_ptr,
    num_tokens,
    num_heads,
    input_stride_token,
    input_stride_head,
    input_stride_dim,
    cache_stride_loc,
    cache_stride_head,
    cache_stride_dim,
    sz_stride_loc,
    sz_stride_head,
    sz_stride_dim,
    HP_OFFSET: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_OCTANT: tl.constexpr,
    BLOCK_TOK: tl.constexpr,
    CLIP_INDEX: tl.constexpr,
    LLOYD_MAX: tl.constexpr,
):
    """Multi-row fused threshold + single-scale clip + int1 pack.

    Mirrors the INT2 single-scale kernel but packs 8 quant slots per byte.
    ``BLOCK_OCTANT == HEAD_DIM // 8``.
    """
    pid_tok = tl.program_id(0)
    head_idx = tl.program_id(1)
    if head_idx >= num_heads:
        return

    tok_offs = pid_tok * BLOCK_TOK + tl.arange(0, BLOCK_TOK)
    tok_mask = tok_offs < num_tokens

    cache_loc = tl.load(loc_ptr + tok_offs, mask=tok_mask, other=0)
    if HP_OFFSET >= 0:
        active = tok_mask & (cache_loc < HP_OFFSET)
    else:
        active = tok_mask

    full_offs = tl.arange(0, HEAD_DIM)
    base = (
        tok_offs[:, None] * input_stride_token
        + head_idx * input_stride_head
        + full_offs[None, :] * input_stride_dim
    )
    rows = tl.load(
        input_ptr + base,
        mask=tok_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    if CLIP_INDEX >= 0:
        abs_rows = tl.abs(rows)
        sorted_rows = tl.sort(abs_rows)
        pick = (full_offs == CLIP_INDEX)[None, :]
        thr = tl.sum(tl.where(pick, sorted_rows, 0.0), axis=1)  # [BLOCK_TOK]
        rows = tl.minimum(
            tl.maximum(rows, -thr[:, None]),
            thr[:, None],
        )

    if LLOYD_MAX:
        LM_C_1BIT: tl.constexpr = 0.79788456  # sqrt(2/pi)
        mean = tl.sum(rows, axis=1) / HEAD_DIM
        diff = rows - mean[:, None]
        var = tl.sum(diff * diff, axis=1) / HEAD_DIM
        std = tl.sqrt(var + 1e-8)
        scale = tl.maximum(2.0 * LM_C_1BIT * std, 1e-8)
        zero = 0.5 - mean / scale
    else:
        val_min = tl.min(rows, axis=1)
        val_max = tl.max(rows, axis=1)
        val_range = tl.maximum(val_max - val_min, 1e-8)
        scale = val_range  # max_q = 1 → scale = range / 1
        zero = -val_min / scale

    # Octant split: reshape (BLOCK_TOK, 8*BLOCK_OCTANT) →
    # (BLOCK_TOK, BLOCK_OCTANT, 2, 2, 2) via reshape + permute, then split×3.
    rows_r = tl.reshape(rows, (BLOCK_TOK, 8, BLOCK_OCTANT))
    rows_p = tl.permute(rows_r, (0, 2, 1))
    rows_s = tl.reshape(rows_p, (BLOCK_TOK, BLOCK_OCTANT, 2, 2, 2))
    h_lo, h_hi = tl.split(rows_s)  # each [BLOCK_TOK, BLOCK_OCTANT, 2, 2]
    e0, e2 = tl.split(h_lo)  # each [BLOCK_TOK, BLOCK_OCTANT, 2]
    e1, e3 = tl.split(h_hi)
    vals0, vals4 = tl.split(e0)  # each [BLOCK_TOK, BLOCK_OCTANT]
    vals2, vals6 = tl.split(e2)
    vals1, vals5 = tl.split(e1)
    vals3, vals7 = tl.split(e3)

    # Lloyd-Max centroids do not bound tail values, so the rounded expression
    # can fall outside {0, 1}. Clamp before the uint8 cast; otherwise 2 or a
    # wrapped negative value spills into neighboring packed bits. This also
    # makes the decision boundary exactly ``value >= mean``, matching the
    # grouped prefill and decode-flush kernels.
    q0 = tl.minimum(
        tl.maximum(vals0 / scale[:, None] + zero[:, None] + 0.5, 0.0), 1.0
    ).to(tl.uint8)
    q1 = tl.minimum(
        tl.maximum(vals1 / scale[:, None] + zero[:, None] + 0.5, 0.0), 1.0
    ).to(tl.uint8)
    q2 = tl.minimum(
        tl.maximum(vals2 / scale[:, None] + zero[:, None] + 0.5, 0.0), 1.0
    ).to(tl.uint8)
    q3 = tl.minimum(
        tl.maximum(vals3 / scale[:, None] + zero[:, None] + 0.5, 0.0), 1.0
    ).to(tl.uint8)
    q4 = tl.minimum(
        tl.maximum(vals4 / scale[:, None] + zero[:, None] + 0.5, 0.0), 1.0
    ).to(tl.uint8)
    q5 = tl.minimum(
        tl.maximum(vals5 / scale[:, None] + zero[:, None] + 0.5, 0.0), 1.0
    ).to(tl.uint8)
    q6 = tl.minimum(
        tl.maximum(vals6 / scale[:, None] + zero[:, None] + 0.5, 0.0), 1.0
    ).to(tl.uint8)
    q7 = tl.minimum(
        tl.maximum(vals7 / scale[:, None] + zero[:, None] + 0.5, 0.0), 1.0
    ).to(tl.uint8)

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
        cache_loc[:, None] * cache_stride_loc
        + head_idx * cache_stride_head
        + dim_offs_o[None, :] * cache_stride_dim
    )
    tl.store(cache_ptr + cache_offset, packed, mask=active[:, None])

    sz_offset_base = cache_loc * sz_stride_loc + head_idx * sz_stride_head
    tl.store(
        scales_zeros_ptr + sz_offset_base + 0 * sz_stride_dim,
        scale,
        mask=active,
    )
    tl.store(
        scales_zeros_ptr + sz_offset_base + 1 * sz_stride_dim,
        zero,
        mask=active,
    )


@triton.jit
def _pretransformed_int1_set_kv_clip_grouped_kernel(
    input_ptr,
    loc_ptr,
    cache_ptr,
    scales_zeros_ptr,
    num_tokens,
    num_heads,
    input_stride_token,
    input_stride_head,
    input_stride_dim,
    cache_stride_loc,
    cache_stride_head,
    cache_stride_dim,
    sz_stride_loc,
    sz_stride_head,
    sz_stride_dim,
    HEAD_DIM: tl.constexpr,
    BLOCK_OCTANT: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HP_OFFSET: tl.constexpr,
    BLOCK_TOK: tl.constexpr,
    CLIP_INDEX: tl.constexpr,
    LLOYD_MAX: tl.constexpr,
):
    """Multi-row fused threshold + groupwise clip + int1 pack (8 per byte).

    When LLOYD_MAX=True: per-group LM binary quantizer (boundary=group mean,
    centroid=group_mean ± group_std*sqrt(2/pi)).
    """
    pid_tok = tl.program_id(0)
    head_idx = tl.program_id(1)
    if head_idx >= num_heads:
        return

    tok_offs = pid_tok * BLOCK_TOK + tl.arange(0, BLOCK_TOK)
    tok_mask = tok_offs < num_tokens

    cache_loc = tl.load(loc_ptr + tok_offs, mask=tok_mask, other=0)
    if HP_OFFSET >= 0:
        active = tok_mask & (cache_loc < HP_OFFSET)
    else:
        active = tok_mask

    full_offs = tl.arange(0, HEAD_DIM)
    base = (
        tok_offs[:, None] * input_stride_token
        + head_idx * input_stride_head
        + full_offs[None, :] * input_stride_dim
    )
    acc = tl.load(
        input_ptr + base,
        mask=tok_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    if CLIP_INDEX >= 0:
        abs_acc = tl.abs(acc)
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
        quant = (group_diff >= 0.0).to(tl.uint8)
    else:
        val_min = tl.min(grouped, axis=2)
        val_max = tl.max(grouped, axis=2)
        scale = tl.maximum(val_max - val_min, 1e-8)
        zero = tl.math.div_rn(-val_min, scale)
        quant = (
            tl.math.div_rn(grouped, scale[:, :, None]) + zero[:, :, None] + 0.5
        ).to(tl.uint8)
    quant_flat = tl.reshape(quant, (BLOCK_TOK, HEAD_DIM))
    quant_r = tl.reshape(quant_flat, (BLOCK_TOK, 8, BLOCK_OCTANT))
    quant_p = tl.permute(quant_r, (0, 2, 1))
    quant_s = tl.reshape(quant_p, (BLOCK_TOK, BLOCK_OCTANT, 2, 2, 2))
    h_lo, h_hi = tl.split(quant_s)
    e0, e2 = tl.split(h_lo)
    e1, e3 = tl.split(h_hi)
    q0, q4 = tl.split(e0)
    q2, q6 = tl.split(e2)
    q1, q5 = tl.split(e1)
    q3, q7 = tl.split(e3)

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
        cache_loc[:, None] * cache_stride_loc
        + head_idx * cache_stride_head
        + dim_offs_o[None, :] * cache_stride_dim
    )
    tl.store(cache_ptr + cache_offset, packed, mask=active[:, None])

    group_ids = tl.arange(0, NUM_GROUPS)
    sz_offset_base = cache_loc[:, None] * sz_stride_loc + head_idx * sz_stride_head
    tl.store(
        scales_zeros_ptr + sz_offset_base + (group_ids[None, :] * 2) * sz_stride_dim,
        scale,
        mask=active[:, None],
    )
    tl.store(
        scales_zeros_ptr
        + sz_offset_base
        + (group_ids[None, :] * 2 + 1) * sz_stride_dim,
        zero,
        mask=active[:, None],
    )


def _can_use_grouped_clip_kernel(
    head_dim: int, scales_zeros_buffer: torch.Tensor
) -> bool:
    num_groups = _get_num_scale_groups(scales_zeros_buffer)
    if num_groups == 1:
        return True
    if head_dim % num_groups != 0:
        return False
    group_size = head_dim // num_groups
    return _is_power_of_two(num_groups) and _is_power_of_two(group_size)


def _clip_index(clip_ratio: float, head_dim: int) -> int:
    if clip_ratio <= 0.0:
        return -1
    idx = int(clip_ratio * head_dim)
    if idx >= head_dim:
        idx = head_dim - 1
    if idx < 0:
        idx = 0
    return idx


def _vectorized_elems_per_thread(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 8
    if dtype.is_floating_point and dtype.itemsize == 1:
        return 16
    raise AssertionError(
        f"clip int1 kernel requires bf16 or fp8 input dtype, got {dtype}"
    )


def _pick_block_tok_and_num_warps(
    head_dim: int, elements_per_thread: int
) -> Tuple[int, int]:
    block_tok = 4
    while block_tok * head_dim < 32 * elements_per_thread:
        block_tok *= 2
    total_elems = block_tok * head_dim
    assert total_elems % (32 * elements_per_thread) == 0, (
        f"BLOCK_TOK={block_tok} head_dim={head_dim} epp={elements_per_thread}: "
        "tile size doesn't divide cleanly into 128-bit/thread loads"
    )
    num_warps = total_elems // (32 * elements_per_thread)
    return block_tok, num_warps


def _launch_single_clip_int1(
    data: torch.Tensor,
    loc: torch.Tensor,
    buf: torch.Tensor,
    sz_buf: torch.Tensor,
    clip_ratio: float,
    hp_global_offset=None,
    lloyd_max: bool = False,
) -> None:
    num_tokens, num_heads, head_dim = data.shape
    if num_tokens == 0:
        return
    assert _is_power_of_two(head_dim), (
        f"clip int1 kernel requires power-of-two head_dim, got {head_dim}"
    )
    assert head_dim % 8 == 0, (
        f"head_dim must be divisible by 8 for INT1, got {head_dim}"
    )
    elements_per_thread = _vectorized_elems_per_thread(data.dtype)
    block_tok, num_warps = _pick_block_tok_and_num_warps(head_dim, elements_per_thread)
    grid = (triton.cdiv(num_tokens, block_tok), num_heads)
    _pretransformed_int1_set_kv_clip_single_kernel[grid](
        data,
        loc,
        buf,
        sz_buf,
        num_tokens,
        num_heads,
        data.stride(0),
        data.stride(1),
        data.stride(2),
        buf.stride(0),
        buf.stride(1),
        buf.stride(2),
        sz_buf.stride(0),
        sz_buf.stride(1),
        sz_buf.stride(2),
        HP_OFFSET=-1 if hp_global_offset is None else int(hp_global_offset),
        HEAD_DIM=head_dim,
        BLOCK_OCTANT=head_dim // 8,
        BLOCK_TOK=block_tok,
        CLIP_INDEX=_clip_index(clip_ratio, head_dim),
        LLOYD_MAX=lloyd_max,
        num_warps=num_warps,
        num_stages=1,
    )


def _launch_grouped_clip_int1(
    data: torch.Tensor,
    loc: torch.Tensor,
    buf: torch.Tensor,
    sz_buf: torch.Tensor,
    clip_ratio: float,
    hp_global_offset=None,
    lloyd_max: bool = False,
) -> None:
    num_tokens, num_heads, head_dim = data.shape
    if num_tokens == 0:
        return
    num_groups = _get_num_scale_groups(sz_buf)

    group_size = head_dim // num_groups
    block_octant = triton.next_power_of_2(head_dim // 8)
    elements_per_thread = _vectorized_elems_per_thread(data.dtype)

    block_tok, num_warps = _pick_block_tok_and_num_warps(head_dim, elements_per_thread)

    assert _is_power_of_two(head_dim), (
        f"clip int1 kernel requires power-of-two head_dim, got {head_dim}"
    )
    assert head_dim % 8 == 0, (
        f"head_dim must be divisible by 8 for INT1, got {head_dim}"
    )
    assert _is_power_of_two(num_groups) and head_dim % num_groups == 0

    grid = (triton.cdiv(num_tokens, block_tok), num_heads)
    _pretransformed_int1_set_kv_clip_grouped_kernel[grid](
        data,
        loc,
        buf,
        sz_buf,
        num_tokens,
        num_heads,
        data.stride(0),
        data.stride(1),
        data.stride(2),
        buf.stride(0),
        buf.stride(1),
        buf.stride(2),
        sz_buf.stride(0),
        sz_buf.stride(1),
        sz_buf.stride(2),
        HEAD_DIM=head_dim,
        BLOCK_OCTANT=block_octant,
        NUM_GROUPS=num_groups,
        GROUP_SIZE=group_size,
        HP_OFFSET=-1 if hp_global_offset is None else int(hp_global_offset),
        BLOCK_TOK=block_tok,
        CLIP_INDEX=_clip_index(clip_ratio, head_dim),
        LLOYD_MAX=lloyd_max,
        num_warps=num_warps,
        num_stages=1,
    )


def quantized_set_kv_int1_pretransformed_clip_triton(
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    loc: torch.Tensor,
    k_cache_buffer: torch.Tensor,
    v_cache_buffer: torch.Tensor,
    k_scales_zeros_buffer: torch.Tensor,
    v_scales_zeros_buffer: torch.Tensor,
    clip_ratio_k: float,
    clip_ratio_v: float,
    hp_global_offset=None,
    lloyd_max: bool = False,
) -> None:
    """Fused threshold + clip + quantize + int1-pack for already-rotated K/V."""
    assert cache_k.shape[:2] == cache_v.shape[:2], (
        f"K/V shape mismatch in pretransformed_clip: {cache_k.shape} vs {cache_v.shape}"
    )
    num_tokens, _num_heads, k_head_dim = cache_k.shape
    v_head_dim = cache_v.shape[-1]
    assert k_head_dim % 8 == 0 and v_head_dim % 8 == 0, (
        "K/V head dims must be divisible by 8 for INT1, got "
        f"K={k_head_dim}, V={v_head_dim}"
    )

    if num_tokens == 0:
        return

    k_grouped_ok = _can_use_grouped_clip_kernel(k_head_dim, k_scales_zeros_buffer)
    v_grouped_ok = _can_use_grouped_clip_kernel(v_head_dim, v_scales_zeros_buffer)
    if not (k_grouped_ok and v_grouped_ok):
        raise NotImplementedError(
            f"pretransformed_clip int1 kernel requires power-of-two group configs "
            f"(k_head_dim={k_head_dim}, v_head_dim={v_head_dim}, "
            f"k_num_groups={_get_num_scale_groups(k_scales_zeros_buffer)}, "
            f"v_num_groups={_get_num_scale_groups(v_scales_zeros_buffer)})"
        )

    if _get_num_scale_groups(k_scales_zeros_buffer) == 1:
        _launch_single_clip_int1(
            cache_k,
            loc,
            k_cache_buffer,
            k_scales_zeros_buffer,
            clip_ratio_k,
            hp_global_offset,
            lloyd_max=lloyd_max,
        )
    else:
        _launch_grouped_clip_int1(
            cache_k,
            loc,
            k_cache_buffer,
            k_scales_zeros_buffer,
            clip_ratio_k,
            hp_global_offset,
            lloyd_max=lloyd_max,
        )

    if _get_num_scale_groups(v_scales_zeros_buffer) == 1:
        _launch_single_clip_int1(
            cache_v,
            loc,
            v_cache_buffer,
            v_scales_zeros_buffer,
            clip_ratio_v,
            hp_global_offset,
            lloyd_max=lloyd_max,
        )
    else:
        _launch_grouped_clip_int1(
            cache_v,
            loc,
            v_cache_buffer,
            v_scales_zeros_buffer,
            clip_ratio_v,
            hp_global_offset,
            lloyd_max=lloyd_max,
        )


# ---------------------------------------------------------------------------
# Fused rotate (K) + clip + quantize + int1 pack kernel for K and V
# ---------------------------------------------------------------------------


@triton.jit
def _kv_oscar_rotate_k_clip_single_kernel_int1(
    k_input_ptr,
    v_input_ptr,
    R_ptr,
    loc_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_sz_ptr,
    v_sz_ptr,
    num_tokens,
    num_heads,
    k_input_stride_token,
    k_input_stride_head,
    k_input_stride_dim,
    v_input_stride_token,
    v_input_stride_head,
    v_input_stride_dim,
    R_stride_in,
    R_stride_out,
    k_cache_stride_loc,
    k_cache_stride_head,
    k_cache_stride_dim,
    v_cache_stride_loc,
    v_cache_stride_head,
    v_cache_stride_dim,
    k_sz_stride_loc,
    k_sz_stride_head,
    k_sz_stride_dim,
    v_sz_stride_loc,
    v_sz_stride_head,
    v_sz_stride_dim,
    HP_OFFSET: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_OCTANT: tl.constexpr,
    BLOCK_TOK: tl.constexpr,
    K_CLIP_INDEX: tl.constexpr,
    V_CLIP_INDEX: tl.constexpr,
    BSEARCH_ITERS: tl.constexpr,
    LLOYD_MAX: tl.constexpr,
):
    """Single fused kernel: rotate(K) + clip(K) + quant(K) + pack(K), then
    clip(V) + quant(V) + pack(V) sharing the same token tile. INT1 variant.
    """
    pid_tok = tl.program_id(0)
    head_idx = tl.program_id(1)
    if head_idx >= num_heads:
        return

    tok_offs = pid_tok * BLOCK_TOK + tl.arange(0, BLOCK_TOK)
    tok_mask = tok_offs < num_tokens

    cache_loc = tl.load(loc_ptr + tok_offs, mask=tok_mask, other=0)
    if HP_OFFSET >= 0:
        active = tok_mask & (cache_loc < HP_OFFSET)
    else:
        active = tok_mask

    full_offs = tl.arange(0, HEAD_DIM)
    dim_offs_o = tl.arange(0, BLOCK_OCTANT)

    # ---------- K: load + rotate + clip + scale/zero + pack + write ----------
    k_base = (
        tok_offs[:, None] * k_input_stride_token
        + head_idx * k_input_stride_head
        + full_offs[None, :] * k_input_stride_dim
    )
    k_tile = tl.load(
        k_input_ptr + k_base,
        mask=tok_mask[:, None],
        other=0.0,
    )

    r_in = tl.arange(0, HEAD_DIM)
    r_out = tl.arange(0, HEAD_DIM)
    R_offs = r_in[:, None] * R_stride_in + r_out[None, :] * R_stride_out
    R_tile = tl.load(R_ptr + R_offs)

    k_rows = tl.dot(k_tile, R_tile, out_dtype=tl.float32)

    if K_CLIP_INDEX >= 0:
        abs_rows = tl.abs(k_rows)
        if BSEARCH_ITERS > 0:
            target_above = HEAD_DIM - K_CLIP_INDEX
            thr_lo = tl.zeros([BLOCK_TOK], dtype=tl.float32)
            thr_hi = tl.max(abs_rows, axis=1)
            for _ in tl.static_range(BSEARCH_ITERS):
                thr_mid = (thr_lo + thr_hi) * 0.5
                cnt_above = tl.sum((abs_rows > thr_mid[:, None]).to(tl.int32), axis=1)
                too_many = cnt_above > target_above
                thr_lo = tl.where(too_many, thr_mid, thr_lo)
                thr_hi = tl.where(too_many, thr_hi, thr_mid)
            thr = thr_hi
        else:
            sorted_rows = tl.sort(abs_rows)
            pick = (full_offs == K_CLIP_INDEX)[None, :]
            thr = tl.sum(tl.where(pick, sorted_rows, 0.0), axis=1)
        k_rows = tl.minimum(
            tl.maximum(k_rows, -thr[:, None]),
            thr[:, None],
        )

    if LLOYD_MAX:
        K_LM_C_1BIT: tl.constexpr = 0.79788456
        k_mean = tl.sum(k_rows, axis=1) / HEAD_DIM
        k_diff = k_rows - k_mean[:, None]
        k_std = tl.sqrt(tl.sum(k_diff * k_diff, axis=1) / HEAD_DIM + 1e-8)
        k_scale = tl.maximum(2.0 * K_LM_C_1BIT * k_std, 1e-8)
        k_zero = 0.5 - k_mean / k_scale
    else:
        k_min = tl.min(k_rows, axis=1)
        k_max = tl.max(k_rows, axis=1)
        k_scale = tl.maximum(k_max - k_min, 1e-8)
        k_zero = -k_min / k_scale

    k_r = tl.reshape(k_rows, (BLOCK_TOK, 8, BLOCK_OCTANT))
    k_p = tl.permute(k_r, (0, 2, 1))
    k_s = tl.reshape(k_p, (BLOCK_TOK, BLOCK_OCTANT, 2, 2, 2))
    k_lo, k_hi = tl.split(k_s)
    k_e0, k_e2 = tl.split(k_lo)
    k_e1, k_e3 = tl.split(k_hi)
    k_v0, k_v4 = tl.split(k_e0)
    k_v2, k_v6 = tl.split(k_e2)
    k_v1, k_v5 = tl.split(k_e1)
    k_v3, k_v7 = tl.split(k_e3)
    k_q0 = tl.minimum(
        tl.maximum(k_v0 / k_scale[:, None] + k_zero[:, None] + 0.5, 0.0), 1.0
    ).to(tl.uint8)
    k_q1 = tl.minimum(
        tl.maximum(k_v1 / k_scale[:, None] + k_zero[:, None] + 0.5, 0.0), 1.0
    ).to(tl.uint8)
    k_q2 = tl.minimum(
        tl.maximum(k_v2 / k_scale[:, None] + k_zero[:, None] + 0.5, 0.0), 1.0
    ).to(tl.uint8)
    k_q3 = tl.minimum(
        tl.maximum(k_v3 / k_scale[:, None] + k_zero[:, None] + 0.5, 0.0), 1.0
    ).to(tl.uint8)
    k_q4 = tl.minimum(
        tl.maximum(k_v4 / k_scale[:, None] + k_zero[:, None] + 0.5, 0.0), 1.0
    ).to(tl.uint8)
    k_q5 = tl.minimum(
        tl.maximum(k_v5 / k_scale[:, None] + k_zero[:, None] + 0.5, 0.0), 1.0
    ).to(tl.uint8)
    k_q6 = tl.minimum(
        tl.maximum(k_v6 / k_scale[:, None] + k_zero[:, None] + 0.5, 0.0), 1.0
    ).to(tl.uint8)
    k_q7 = tl.minimum(
        tl.maximum(k_v7 / k_scale[:, None] + k_zero[:, None] + 0.5, 0.0), 1.0
    ).to(tl.uint8)
    k_packed = (
        k_q0
        | (k_q1 << 1)
        | (k_q2 << 2)
        | (k_q3 << 3)
        | (k_q4 << 4)
        | (k_q5 << 5)
        | (k_q6 << 6)
        | (k_q7 << 7)
    )

    k_cache_offset = (
        cache_loc[:, None] * k_cache_stride_loc
        + head_idx * k_cache_stride_head
        + dim_offs_o[None, :] * k_cache_stride_dim
    )
    tl.store(k_cache_ptr + k_cache_offset, k_packed, mask=active[:, None])

    k_sz_base = cache_loc * k_sz_stride_loc + head_idx * k_sz_stride_head
    tl.store(k_sz_ptr + k_sz_base + 0 * k_sz_stride_dim, k_scale, mask=active)
    tl.store(k_sz_ptr + k_sz_base + 1 * k_sz_stride_dim, k_zero, mask=active)

    # ---------- V: load + clip + scale/zero + pack + write (no rotate) ------
    v_base = (
        tok_offs[:, None] * v_input_stride_token
        + head_idx * v_input_stride_head
        + full_offs[None, :] * v_input_stride_dim
    )
    v_rows = tl.load(
        v_input_ptr + v_base,
        mask=tok_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    if V_CLIP_INDEX >= 0:
        abs_rows = tl.abs(v_rows)
        if BSEARCH_ITERS > 0:
            target_above = HEAD_DIM - V_CLIP_INDEX
            thr_lo = tl.zeros([BLOCK_TOK], dtype=tl.float32)
            thr_hi = tl.max(abs_rows, axis=1)
            for _ in tl.static_range(BSEARCH_ITERS):
                thr_mid = (thr_lo + thr_hi) * 0.5
                cnt_above = tl.sum((abs_rows > thr_mid[:, None]).to(tl.int32), axis=1)
                too_many = cnt_above > target_above
                thr_lo = tl.where(too_many, thr_mid, thr_lo)
                thr_hi = tl.where(too_many, thr_hi, thr_mid)
            thr = thr_hi
        else:
            sorted_rows = tl.sort(abs_rows)
            pick = (full_offs == V_CLIP_INDEX)[None, :]
            thr = tl.sum(tl.where(pick, sorted_rows, 0.0), axis=1)
        v_rows = tl.minimum(
            tl.maximum(v_rows, -thr[:, None]),
            thr[:, None],
        )

    if LLOYD_MAX:
        V_LM_C_1BIT: tl.constexpr = 0.79788456
        v_mean = tl.sum(v_rows, axis=1) / HEAD_DIM
        v_diff = v_rows - v_mean[:, None]
        v_std = tl.sqrt(tl.sum(v_diff * v_diff, axis=1) / HEAD_DIM + 1e-8)
        v_scale = tl.maximum(2.0 * V_LM_C_1BIT * v_std, 1e-8)
        v_zero = 0.5 - v_mean / v_scale
    else:
        v_min = tl.min(v_rows, axis=1)
        v_max = tl.max(v_rows, axis=1)
        v_scale = tl.maximum(v_max - v_min, 1e-8)
        v_zero = -v_min / v_scale

    v_r = tl.reshape(v_rows, (BLOCK_TOK, 8, BLOCK_OCTANT))
    v_p = tl.permute(v_r, (0, 2, 1))
    v_s = tl.reshape(v_p, (BLOCK_TOK, BLOCK_OCTANT, 2, 2, 2))
    v_lo, v_hi = tl.split(v_s)
    v_e0, v_e2 = tl.split(v_lo)
    v_e1, v_e3 = tl.split(v_hi)
    v_v0, v_v4 = tl.split(v_e0)
    v_v2, v_v6 = tl.split(v_e2)
    v_v1, v_v5 = tl.split(v_e1)
    v_v3, v_v7 = tl.split(v_e3)
    v_q0 = tl.minimum(
        tl.maximum(v_v0 / v_scale[:, None] + v_zero[:, None] + 0.5, 0.0), 1.0
    ).to(tl.uint8)
    v_q1 = tl.minimum(
        tl.maximum(v_v1 / v_scale[:, None] + v_zero[:, None] + 0.5, 0.0), 1.0
    ).to(tl.uint8)
    v_q2 = tl.minimum(
        tl.maximum(v_v2 / v_scale[:, None] + v_zero[:, None] + 0.5, 0.0), 1.0
    ).to(tl.uint8)
    v_q3 = tl.minimum(
        tl.maximum(v_v3 / v_scale[:, None] + v_zero[:, None] + 0.5, 0.0), 1.0
    ).to(tl.uint8)
    v_q4 = tl.minimum(
        tl.maximum(v_v4 / v_scale[:, None] + v_zero[:, None] + 0.5, 0.0), 1.0
    ).to(tl.uint8)
    v_q5 = tl.minimum(
        tl.maximum(v_v5 / v_scale[:, None] + v_zero[:, None] + 0.5, 0.0), 1.0
    ).to(tl.uint8)
    v_q6 = tl.minimum(
        tl.maximum(v_v6 / v_scale[:, None] + v_zero[:, None] + 0.5, 0.0), 1.0
    ).to(tl.uint8)
    v_q7 = tl.minimum(
        tl.maximum(v_v7 / v_scale[:, None] + v_zero[:, None] + 0.5, 0.0), 1.0
    ).to(tl.uint8)
    v_packed = (
        v_q0
        | (v_q1 << 1)
        | (v_q2 << 2)
        | (v_q3 << 3)
        | (v_q4 << 4)
        | (v_q5 << 5)
        | (v_q6 << 6)
        | (v_q7 << 7)
    )

    v_cache_offset = (
        cache_loc[:, None] * v_cache_stride_loc
        + head_idx * v_cache_stride_head
        + dim_offs_o[None, :] * v_cache_stride_dim
    )
    tl.store(v_cache_ptr + v_cache_offset, v_packed, mask=active[:, None])

    v_sz_base = cache_loc * v_sz_stride_loc + head_idx * v_sz_stride_head
    tl.store(v_sz_ptr + v_sz_base + 0 * v_sz_stride_dim, v_scale, mask=active)
    tl.store(v_sz_ptr + v_sz_base + 1 * v_sz_stride_dim, v_zero, mask=active)


def _pick_block_tok_and_num_warps_for_dot(
    head_dim: int, elements_per_thread: int
) -> Tuple[int, int]:
    block_tok, num_warps = _pick_block_tok_and_num_warps(head_dim, elements_per_thread)
    if block_tok < 16:
        block_tok = 16
        total_elems = block_tok * head_dim
        assert total_elems % (32 * elements_per_thread) == 0, (
            f"BLOCK_TOK={block_tok} head_dim={head_dim} epp={elements_per_thread}: "
            "tile size doesn't divide cleanly into 128-bit/thread loads"
        )
        num_warps = total_elems // (32 * elements_per_thread)
    return block_tok, num_warps


def quantized_set_kv_int1_oscar_rotate_k_clip_triton(
    cache_k_unrotated: torch.Tensor,
    cache_v_rotated: torch.Tensor,
    R_k: torch.Tensor,
    loc: torch.Tensor,
    k_cache_buffer: torch.Tensor,
    v_cache_buffer: torch.Tensor,
    k_scales_zeros_buffer: torch.Tensor,
    v_scales_zeros_buffer: torch.Tensor,
    clip_ratio_k: float,
    clip_ratio_v: float,
    hp_global_offset=None,
    lloyd_max: bool = False,
) -> None:
    """Single-launch fused oscar K-rotation + clip + quantize + int1 pack."""
    assert cache_k_unrotated.shape == cache_v_rotated.shape, (
        "K/V shape mismatch in oscar_rotate_k_clip (int1 kernel requires identical "
        f"shapes incl. head_dim): {cache_k_unrotated.shape} vs {cache_v_rotated.shape}"
    )
    num_tokens, num_heads, head_dim = cache_k_unrotated.shape
    if num_tokens == 0:
        return

    assert head_dim % 8 == 0, (
        f"head_dim must be divisible by 8 for INT1, got {head_dim}"
    )
    assert _is_power_of_two(head_dim), (
        f"oscar rotate+clip int1 kernel requires power-of-two head_dim, got {head_dim}"
    )
    assert R_k.shape == (head_dim, head_dim), (
        f"R_k must be [head_dim, head_dim]={head_dim}x{head_dim}, "
        f"got {tuple(R_k.shape)}"
    )
    assert R_k.dtype == cache_k_unrotated.dtype, (
        f"R_k dtype ({R_k.dtype}) must match input dtype ({cache_k_unrotated.dtype})"
    )

    if _get_num_scale_groups(k_scales_zeros_buffer) != 1:
        raise NotImplementedError(
            "oscar rotate+clip+quant fused int1 kernel requires single-scale K layout "
            f"(got num_groups={_get_num_scale_groups(k_scales_zeros_buffer)})"
        )
    if _get_num_scale_groups(v_scales_zeros_buffer) != 1:
        raise NotImplementedError(
            "oscar rotate+clip+quant fused int1 kernel requires single-scale V layout "
            f"(got num_groups={_get_num_scale_groups(v_scales_zeros_buffer)})"
        )

    elements_per_thread = _vectorized_elems_per_thread(cache_k_unrotated.dtype)
    block_tok, num_warps = _pick_block_tok_and_num_warps_for_dot(
        head_dim, elements_per_thread
    )
    grid = (triton.cdiv(num_tokens, block_tok), num_heads)
    _kv_oscar_rotate_k_clip_single_kernel_int1[grid](
        cache_k_unrotated,
        cache_v_rotated,
        R_k,
        loc,
        k_cache_buffer,
        v_cache_buffer,
        k_scales_zeros_buffer,
        v_scales_zeros_buffer,
        num_tokens,
        num_heads,
        cache_k_unrotated.stride(0),
        cache_k_unrotated.stride(1),
        cache_k_unrotated.stride(2),
        cache_v_rotated.stride(0),
        cache_v_rotated.stride(1),
        cache_v_rotated.stride(2),
        R_k.stride(0),
        R_k.stride(1),
        k_cache_buffer.stride(0),
        k_cache_buffer.stride(1),
        k_cache_buffer.stride(2),
        v_cache_buffer.stride(0),
        v_cache_buffer.stride(1),
        v_cache_buffer.stride(2),
        k_scales_zeros_buffer.stride(0),
        k_scales_zeros_buffer.stride(1),
        k_scales_zeros_buffer.stride(2),
        v_scales_zeros_buffer.stride(0),
        v_scales_zeros_buffer.stride(1),
        v_scales_zeros_buffer.stride(2),
        HP_OFFSET=-1 if hp_global_offset is None else int(hp_global_offset),
        HEAD_DIM=head_dim,
        BLOCK_OCTANT=head_dim // 8,
        BLOCK_TOK=block_tok,
        K_CLIP_INDEX=_clip_index(clip_ratio_k, head_dim),
        V_CLIP_INDEX=_clip_index(clip_ratio_v, head_dim),
        BSEARCH_ITERS=(head_dim.bit_length() - 1) if head_dim >= 64 else 0,
        LLOYD_MAX=bool(lloyd_max),
        num_warps=num_warps,
        num_stages=1,
    )


# ---------------------------------------------------------------------------
# Non-clip pretransformed pack (used when clip ratios are 0). Mirrors the
# `quantized_set_kv_int2_pretransformed_triton` entry point from the
# `fused_hadamard_int2_kv` module so the unified pool can dispatch the same
# way when no clip is requested. The implementation just reuses the clip
# kernel with CLIP_INDEX = -1 (disable).
# ---------------------------------------------------------------------------


def quantized_set_kv_int1_pretransformed_triton(
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    loc: torch.Tensor,
    k_cache_buffer: torch.Tensor,
    v_cache_buffer: torch.Tensor,
    k_scales_zeros_buffer: torch.Tensor,
    v_scales_zeros_buffer: torch.Tensor,
    hp_global_offset=None,
) -> None:
    """No-clip int1 pretransformed pack (clip kernel with CLIP_INDEX=-1)."""
    quantized_set_kv_int1_pretransformed_clip_triton(
        cache_k,
        cache_v,
        loc,
        k_cache_buffer,
        v_cache_buffer,
        k_scales_zeros_buffer,
        v_scales_zeros_buffer,
        clip_ratio_k=0.0,
        clip_ratio_v=0.0,
        hp_global_offset=hp_global_offset,
    )
