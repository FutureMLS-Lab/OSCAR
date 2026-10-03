"""Prefix dequantization for product-quantized rows of the unified mixed
pool: HP slots are copied, quant slots are reconstructed from their codes
(plus the residual stage when present), in one launch per tensor."""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _mixed_prefix_pq_dequant_kernel(
    prefix_indices_ptr,
    codes_ptr,
    codebook_ptr,
    codes2_ptr,
    codebook2_ptr,
    hp_ptr,
    out_ptr,
    num_tokens,
    num_heads,
    codes_stride_token: tl.constexpr,
    codes_stride_head: tl.constexpr,
    codes_stride_sub: tl.constexpr,
    codes2_stride_token: tl.constexpr,
    codes2_stride_head: tl.constexpr,
    codes2_stride_sub: tl.constexpr,
    hp_stride_token: tl.constexpr,
    hp_stride_head: tl.constexpr,
    hp_stride_dim: tl.constexpr,
    out_stride_token: tl.constexpr,
    out_stride_head: tl.constexpr,
    out_stride_dim: tl.constexpr,
    HP_OFFSET: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SUB_DIM: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
    N_CENTROIDS2: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    HAS_STAGE2: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    offs = tl.arange(0, BLOCK_DIM)
    dim_mask = offs < HEAD_DIM

    slot = tl.load(prefix_indices_ptr + token_idx).to(tl.int64)
    is_hp = slot >= HP_OFFSET
    quant_slot = tl.where(is_hp, 0, slot)
    hp_slot = tl.where(is_hp, slot - HP_OFFSET, 0)

    sub_idx = offs // SUB_DIM
    sub_off = offs % SUB_DIM
    code = tl.load(
        codes_ptr
        + quant_slot * codes_stride_token
        + head_idx * codes_stride_head
        + sub_idx * codes_stride_sub,
        mask=(~is_hp) & dim_mask,
        other=0,
    ).to(tl.int64)
    quant_val = tl.load(
        codebook_ptr + (sub_idx * N_CENTROIDS + code) * SUB_DIM + sub_off,
        mask=(~is_hp) & dim_mask,
        other=0.0,
    ).to(tl.float32)
    if HAS_STAGE2:
        code2 = tl.load(
            codes2_ptr
            + quant_slot * codes2_stride_token
            + head_idx * codes2_stride_head
            + sub_idx * codes2_stride_sub,
            mask=(~is_hp) & dim_mask,
            other=0,
        ).to(tl.int64)
        quant_val += tl.load(
            codebook2_ptr + (sub_idx * N_CENTROIDS2 + code2) * SUB_DIM + sub_off,
            mask=(~is_hp) & dim_mask,
            other=0.0,
        ).to(tl.float32)

    hp_val = tl.load(
        hp_ptr + hp_slot * hp_stride_token + head_idx * hp_stride_head + offs * hp_stride_dim,
        mask=is_hp & dim_mask,
        other=0.0,
    ).to(tl.float32)
    out_val = tl.where(is_hp, hp_val, quant_val)
    tl.store(
        out_ptr + token_idx * out_stride_token + head_idx * out_stride_head + offs * out_stride_dim,
        out_val,
        mask=(token_idx < num_tokens) & (head_idx < num_heads) & dim_mask,
    )


def mixed_prefix_dequantize_pq(
    prefix_indices: torch.Tensor,
    codes: torch.Tensor,
    codebook: torch.Tensor,
    hp: torch.Tensor,
    hp_offset: int,
    model_dtype: torch.dtype,
    *,
    codes2: Optional[torch.Tensor] = None,
    codebook2: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dense ``[num_tokens, heads, head_dim]`` rows for a mixed HP / PQ prefix;
    ``out`` receives them when a caller owns the staging buffer."""
    num_tokens = prefix_indices.shape[0]
    num_heads = codes.shape[1]
    n_sub, n_centroids, sub_dim = codebook.shape
    head_dim = int(n_sub) * int(sub_dim)
    if out is None:
        out = torch.empty(
            (num_tokens, num_heads, head_dim), dtype=model_dtype, device=prefix_indices.device
        )
    else:
        assert out.shape == (num_tokens, num_heads, head_dim), (
            f"dequant out buffer {tuple(out.shape)} != {(num_tokens, num_heads, head_dim)}"
        )
        assert out.dtype == model_dtype
    if num_tokens == 0:
        return out
    has_stage2 = codes2 is not None
    if has_stage2:
        assert codebook2 is not None
        assert codebook2.shape[0] == n_sub and codebook2.shape[2] == sub_dim
        codes2_arg, codebook2_arg = codes2, codebook2
        n_centroids2 = int(codebook2.shape[1])
    else:
        # Triton still needs pointer arguments for the disabled constexpr branch.
        codes2_arg, codebook2_arg = codes, codebook
        n_centroids2 = int(n_centroids)
    grid = (num_tokens, num_heads)
    _mixed_prefix_pq_dequant_kernel[grid](
        prefix_indices,
        codes,
        codebook,
        codes2_arg,
        codebook2_arg,
        hp,
        out,
        num_tokens,
        num_heads,
        codes.stride(0),
        codes.stride(1),
        codes.stride(2),
        codes2_arg.stride(0),
        codes2_arg.stride(1),
        codes2_arg.stride(2),
        hp.stride(0),
        hp.stride(1),
        hp.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        HP_OFFSET=int(hp_offset),
        HEAD_DIM=head_dim,
        SUB_DIM=int(sub_dim),
        N_CENTROIDS=int(n_centroids),
        N_CENTROIDS2=n_centroids2,
        BLOCK_DIM=triton.next_power_of_2(head_dim),
        HAS_STAGE2=has_stage2,
        num_warps=4,
        num_stages=1,
    )
    return out
