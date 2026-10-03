"""Batched per-row rotation for the OSCAR decode path.

Decode rotates Q into the KV frame (``q @ R_k``) and the attention output back
(``o @ R_v^T``) on every layer. torch dispatches each of those tiny products to
a cuBLAS split-K GEMM (two kernels) plus a contiguous/copy kernel, so a
36-layer model spends about four launches per layer on them. One Triton
program per (token block, head) does the same bf16-in / fp32-accumulate /
bf16-out product in a single launch and reads the per-KV-head rotation
directly instead of materialising a repeat-interleaved copy of ``R``.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _rot_rows_kernel(
    X,  # [tokens, heads, D]
    R,  # [D, D] or [kv_heads, D, D]
    OUT,  # [tokens, heads, D]; may alias X (a program loads its tile before storing)
    n_tokens,
    stride_xt,
    stride_xh,
    stride_ot,
    stride_oh,
    KV_GROUP: tl.constexpr,  # query heads per KV head (per-head R only)
    PER_HEAD: tl.constexpr,
    TRANS_R: tl.constexpr,  # False: x @ R ; True: x @ R^T
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_t = tl.program_id(0)
    head = tl.program_id(1)
    toks = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    m = toks < n_tokens
    dk = tl.arange(0, D)
    x = tl.load(
        X + toks[:, None] * stride_xt + head * stride_xh + dk[None, :],
        mask=m[:, None],
        other=0.0,
    )
    if PER_HEAD:
        r_base = R + (head // KV_GROUP) * D * D
    else:
        r_base = R
    if TRANS_R:
        # (R^T)[k, j] = R[j, k]
        r = tl.load(r_base + dk[None, :] * D + dk[:, None])
    else:
        r = tl.load(r_base + dk[:, None] * D + dk[None, :])
    acc = tl.dot(x.to(r.dtype), r)  # fp32 accumulate, like the cuBLAS bf16 GEMM
    tl.store(
        OUT + toks[:, None] * stride_ot + head * stride_oh + dk[None, :],
        acc.to(OUT.dtype.element_ty),
        mask=m[:, None],
    )


def rotate_rows_supported(head_dim: int) -> bool:
    """``tl.dot`` wants a power-of-two contraction dim of at least 16; the
    rotation is loaded whole, so cap it where the tile still fits registers."""
    return 16 <= head_dim <= 256 and (head_dim & (head_dim - 1)) == 0


def fast_rotate_rows(
    x3: torch.Tensor,
    R: torch.Tensor,
    trans: bool = False,
    out: Optional[torch.Tensor] = None,
    kv_group_num: int = 1,
) -> torch.Tensor:
    """``out = x3 @ R`` (or ``x3 @ R^T``) along the last dim in one launch.

    ``x3`` is ``[tokens, heads, hd]``. ``R`` is ``[hd, hd]`` (one rotation for
    every head) or ``[kv_heads, hd, hd]`` (one per KV head; head ``h`` uses
    ``R[h // kv_group_num]``, which is how ``_apply_oscar_rotation`` lines up
    query heads with their KV head). Returns a new contiguous tensor in
    ``R.dtype`` -- the same contract as ``_apply_oscar_rotation`` -- unless
    ``out`` is given, in which case it is written in place in its own dtype
    (``out`` may be ``x3`` itself).
    """
    assert x3.dim() == 3, f"expected [tokens, heads, hd], got {tuple(x3.shape)}"
    n_tokens, n_heads, head_dim = x3.shape
    assert rotate_rows_supported(head_dim), head_dim
    if x3.stride(-1) != 1:
        x3 = x3.contiguous()
    if not R.is_contiguous():
        R = R.contiguous()
    per_head = R.dim() == 3
    if per_head:
        assert R.shape[0] * kv_group_num == n_heads, (
            f"R has {R.shape[0]} rotations for {n_heads} heads "
            f"(kv_group_num={kv_group_num})"
        )
    else:
        assert R.dim() == 2
    assert R.shape[-1] == head_dim and R.shape[-2] == head_dim, tuple(R.shape)
    if out is None:
        out = torch.empty((n_tokens, n_heads, head_dim), dtype=R.dtype, device=x3.device)
    else:
        assert out.shape == x3.shape and out.stride(-1) == 1, (
            tuple(out.shape),
            out.stride(),
        )
    if n_tokens == 0:
        return out
    BLOCK_T = 16
    _rot_rows_kernel[(triton.cdiv(n_tokens, BLOCK_T), n_heads)](
        x3,
        R,
        out,
        n_tokens,
        x3.stride(0),
        x3.stride(1),
        out.stride(0),
        out.stride(1),
        KV_GROUP=max(1, kv_group_num),
        PER_HEAD=per_head,
        TRANS_R=trans,
        D=head_dim,
        BLOCK_T=BLOCK_T,
        num_warps=4,
        num_stages=1,
    )
    return out
