"""Product-quantized KV rows: Triton encode/decode kernels and codebook helpers.

A row of ``head_dim`` values is split into ``n_sub`` sub-vectors of
``sub_dim`` channels; each sub-vector stores the uint8 index of its nearest
centroid in a per-(layer, sub-vector) codebook of at most 256 entries, so a
row costs ``n_sub`` bytes and carries no per-row scale. A residual (RVQ)
stage encodes ``row - decode(codes)`` with a second codebook into a second
code buffer of the same shape. Codebooks are fp16 ``[n_sub, n_centroids,
sub_dim]`` and must be contiguous: the kernels index them by arithmetic."""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


def pq_codebook_norms(codebook: torch.Tensor) -> torch.Tensor:
    """Squared centroid norms ``[n_sub, n_centroids]`` in fp32."""
    return (codebook.float() ** 2).sum(dim=-1).contiguous()


@triton.jit
def _pq_encode_kernel(
    rows_ptr,
    loc_ptr,
    codes_ptr,
    cb_ptr,
    cb_norm2_ptr,
    num_tokens,
    row_stride_tok,
    row_stride_head,
    codes_stride_loc,
    codes_stride_head,
    codes_stride_sub,
    N_SUB: tl.constexpr,
    SUB_DIM: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
    BLOCK_TOK: tl.constexpr,
    HP_OFFSET: tl.constexpr,
):
    pid_tok = tl.program_id(0)
    pid_head = tl.program_id(1)
    tok_range = pid_tok * BLOCK_TOK + tl.arange(0, BLOCK_TOK)
    active = tok_range < num_tokens
    cache_loc = tl.load(loc_ptr + tok_range, mask=active, other=0).to(tl.int64)
    if HP_OFFSET >= 0:
        active &= cache_loc < HP_OFFSET
    cent_range = tl.arange(0, N_CENTROIDS)

    for s in tl.static_range(N_SUB):
        # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x.c, accumulated one channel at a
        # time so the codebook column loads stay coalesced.
        x_norm2 = tl.zeros([BLOCK_TOK], dtype=tl.float32)
        dot = tl.zeros([BLOCK_TOK, N_CENTROIDS], dtype=tl.float32)
        for d in tl.static_range(SUB_DIM):
            x_d = tl.load(
                rows_ptr
                + tok_range * row_stride_tok
                + pid_head * row_stride_head
                + (s * SUB_DIM + d),
                mask=active,
                other=0.0,
            ).to(tl.float32)
            x_norm2 += x_d * x_d
            cb_d = tl.load(cb_ptr + (s * N_CENTROIDS + cent_range) * SUB_DIM + d).to(
                tl.float32
            )
            dot += x_d[:, None] * cb_d[None, :]
        cb_n2 = tl.load(cb_norm2_ptr + s * N_CENTROIDS + cent_range).to(tl.float32)
        d2 = x_norm2[:, None] + cb_n2[None, :] - 2.0 * dot
        code = tl.argmin(d2, axis=1).to(tl.uint8)
        tl.store(
            codes_ptr
            + cache_loc * codes_stride_loc
            + pid_head * codes_stride_head
            + s * codes_stride_sub,
            code,
            mask=active,
        )


@triton.jit
def _pq_decode_rows_kernel(
    codes_ptr,
    out_ptr,
    cb_ptr,
    num_tokens,
    codes_stride_loc,
    codes_stride_head,
    codes_stride_sub,
    out_stride_tok,
    out_stride_head,
    out_stride_dim,
    N_SUB: tl.constexpr,
    SUB_DIM: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
    BLOCK_TOK: tl.constexpr,
):
    pid_tok = tl.program_id(0)
    pid_head = tl.program_id(1)
    tok_range = pid_tok * BLOCK_TOK + tl.arange(0, BLOCK_TOK)
    active = tok_range < num_tokens
    sub_range = tl.arange(0, SUB_DIM)
    for s in tl.static_range(N_SUB):
        code = tl.load(
            codes_ptr
            + tok_range * codes_stride_loc
            + pid_head * codes_stride_head
            + s * codes_stride_sub,
            mask=active,
            other=0,
        ).to(tl.int32)
        cb_idx = (s * N_CENTROIDS + code[:, None]) * SUB_DIM + sub_range[None, :]
        recon = tl.load(cb_ptr + cb_idx, mask=active[:, None], other=0.0)
        out_off = (
            tok_range[:, None] * out_stride_tok
            + pid_head * out_stride_head
            + (s * SUB_DIM + sub_range[None, :]) * out_stride_dim
        )
        tl.store(out_ptr + out_off, recon, mask=active[:, None])


@triton.jit
def _pq_decode_at_locs_kernel(
    codes_ptr,
    loc_ptr,
    out_ptr,
    cb_ptr,
    num_tokens,
    codes_stride_loc,
    codes_stride_head,
    codes_stride_sub,
    out_stride_tok,
    out_stride_head,
    out_stride_dim,
    N_SUB: tl.constexpr,
    SUB_DIM: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
    BLOCK_TOK: tl.constexpr,
    HP_OFFSET: tl.constexpr,
):
    """Decode the cache rows at ``loc`` without data-dependent indexing on the
    host; HP locs produce zero rows."""
    pid_tok = tl.program_id(0)
    pid_head = tl.program_id(1)
    tok_range = pid_tok * BLOCK_TOK + tl.arange(0, BLOCK_TOK)
    token_mask = tok_range < num_tokens
    cache_loc = tl.load(loc_ptr + tok_range, mask=token_mask, other=0).to(tl.int64)
    active = token_mask
    if HP_OFFSET >= 0:
        active &= cache_loc < HP_OFFSET
    safe_loc = tl.where(active, cache_loc, 0)
    sub_range = tl.arange(0, SUB_DIM)
    for s in tl.static_range(N_SUB):
        code = tl.load(
            codes_ptr
            + safe_loc * codes_stride_loc
            + pid_head * codes_stride_head
            + s * codes_stride_sub,
            mask=active,
            other=0,
        ).to(tl.int32)
        cb_idx = (s * N_CENTROIDS + code[:, None]) * SUB_DIM + sub_range[None, :]
        recon = tl.load(cb_ptr + cb_idx, mask=active[:, None], other=0.0)
        out_off = (
            tok_range[:, None] * out_stride_tok
            + pid_head * out_stride_head
            + (s * SUB_DIM + sub_range[None, :]) * out_stride_dim
        )
        tl.store(out_ptr + out_off, recon, mask=token_mask[:, None])


def _check_codebook(codebook: torch.Tensor, head_dim: int) -> tuple[int, int, int]:
    n_sub, n_centroids, sub_dim = (int(x) for x in codebook.shape)
    if n_sub * sub_dim != head_dim:
        raise ValueError(
            f"PQ codebook reconstructs {n_sub}*{sub_dim} dims, rows have {head_dim}"
        )
    if n_centroids > 256:
        raise ValueError(f"PQ codebook has {n_centroids} centroids; uint8 codes hold 256")
    return n_sub, n_centroids, sub_dim


def pq_encode(
    rows: torch.Tensor,
    loc: torch.Tensor,
    codes_buffer: torch.Tensor,
    codebook: torch.Tensor,
    codebook_norms: torch.Tensor,
    *,
    hp_global_offset: Optional[int] = None,
    block_tok: int = 16,
) -> None:
    """Write the nearest-centroid codes of ``rows`` ``[n, heads, head_dim]``
    into ``codes_buffer[loc]``; locs at or past ``hp_global_offset`` are
    skipped (HP tier)."""
    num_tokens, num_heads, head_dim = rows.shape
    if num_tokens == 0:
        return
    n_sub, n_centroids, sub_dim = _check_codebook(codebook, head_dim)
    if codes_buffer.shape[-1] != n_sub:
        raise ValueError(
            f"code buffer holds {codes_buffer.shape[-1]} bytes per row, codebook needs {n_sub}"
        )
    rows_fp16 = rows.to(torch.float16).contiguous()
    cb = codebook.to(torch.float16).contiguous()
    grid = (triton.cdiv(num_tokens, block_tok), num_heads)
    _pq_encode_kernel[grid](
        rows_fp16,
        loc,
        codes_buffer,
        cb,
        codebook_norms.float().contiguous(),
        num_tokens,
        rows_fp16.stride(0),
        rows_fp16.stride(1),
        codes_buffer.stride(0),
        codes_buffer.stride(1),
        codes_buffer.stride(2),
        N_SUB=n_sub,
        SUB_DIM=sub_dim,
        N_CENTROIDS=n_centroids,
        BLOCK_TOK=block_tok,
        HP_OFFSET=-1 if hp_global_offset is None else int(hp_global_offset),
        num_warps=4,
        num_stages=2,
    )


def pq_decode_rows(
    codes: torch.Tensor, codebook: torch.Tensor, *, head_dim: int, block_tok: int = 16
) -> torch.Tensor:
    """Reconstruct ``codes`` ``[n, heads, n_sub]`` to fp16 ``[n, heads, head_dim]``."""
    num_tokens, num_heads, n_sub = codes.shape
    _, n_centroids, sub_dim = _check_codebook(codebook, head_dim)
    out = torch.empty(
        (num_tokens, num_heads, head_dim), dtype=torch.float16, device=codes.device
    )
    if num_tokens == 0:
        return out
    cb = codebook.to(torch.float16).contiguous()
    grid = (triton.cdiv(num_tokens, block_tok), num_heads)
    _pq_decode_rows_kernel[grid](
        codes,
        out,
        cb,
        num_tokens,
        codes.stride(0),
        codes.stride(1),
        codes.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        N_SUB=n_sub,
        SUB_DIM=sub_dim,
        N_CENTROIDS=n_centroids,
        BLOCK_TOK=block_tok,
        num_warps=4,
        num_stages=2,
    )
    return out


def pq_decode_at_locs(
    codes_buffer: torch.Tensor,
    loc: torch.Tensor,
    codebook: torch.Tensor,
    *,
    head_dim: int,
    hp_global_offset: Optional[int] = None,
    block_tok: int = 16,
) -> torch.Tensor:
    """Reconstruct the cache rows at ``loc`` to fp16 ``[len(loc), heads, head_dim]``."""
    num_tokens = int(loc.shape[0])
    _, num_heads, n_sub = codes_buffer.shape
    _, n_centroids, sub_dim = _check_codebook(codebook, head_dim)
    out = torch.empty(
        (num_tokens, num_heads, head_dim), dtype=torch.float16, device=codes_buffer.device
    )
    if num_tokens == 0:
        return out
    cb = codebook.to(torch.float16).contiguous()
    grid = (triton.cdiv(num_tokens, block_tok), num_heads)
    _pq_decode_at_locs_kernel[grid](
        codes_buffer,
        loc,
        out,
        cb,
        num_tokens,
        codes_buffer.stride(0),
        codes_buffer.stride(1),
        codes_buffer.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        N_SUB=n_sub,
        SUB_DIM=sub_dim,
        N_CENTROIDS=n_centroids,
        BLOCK_TOK=block_tok,
        HP_OFFSET=-1 if hp_global_offset is None else int(hp_global_offset),
        num_warps=4,
        num_stages=1,
    )
    return out


def pq_encode_decode_reference(rows: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """CPU reference: nearest-centroid encode then decode of ``[n, head_dim]`` rows."""
    n, head_dim = rows.shape
    n_sub, _, sub_dim = _check_codebook(codebook, head_dim)
    x = rows.float().reshape(n, n_sub, sub_dim)
    cb = codebook.float()
    pieces = []
    for s in range(n_sub):
        d2 = ((x[:, s, None, :] - cb[s][None, :, :]) ** 2).sum(dim=-1)
        pieces.append(cb[s][d2.argmin(dim=1)])
    return torch.cat(pieces, dim=1)


def build_pq_codebook(
    data: torch.Tensor,
    *,
    n_sub: int,
    n_centroids: int,
    sub_dim: int,
    n_iter: int = 30,
    seed: int = 42,
) -> torch.Tensor:
    """k-means a ``[samples, head_dim]`` matrix into an fp16 codebook
    ``[n_sub, n_centroids, sub_dim]`` (offline tooling; needs scipy)."""
    import numpy as np
    from scipy.cluster.vq import kmeans2

    samples, head_dim = data.shape
    if head_dim != n_sub * sub_dim:
        raise ValueError(f"head_dim {head_dim} != {n_sub} * {sub_dim}")
    sub_data = data.float().numpy().reshape(samples, n_sub, sub_dim)
    books = []
    for s in range(n_sub):
        centroids, _ = kmeans2(
            sub_data[:, s, :], n_centroids, minit="points", iter=n_iter, seed=seed
        )
        books.append(centroids)
    return torch.from_numpy(np.stack(books)).to(torch.float16)
