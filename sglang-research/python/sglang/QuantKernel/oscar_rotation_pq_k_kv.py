"""
Product Quantization (PQ) for K cache with INT2 LM fallback for V.

PQ K scheme:
  - K split into N_SUB=16 sub-vectors of SUB_DIM=8 channels each
  - Each sub-vector encoded to 1 byte (8-bit index into 256 centroids)
  - True 1.0 bpe for K (shared codebook, no per-token scale/zero)
  - SQNR on real Qwen3-4B-Thinking data: +9.09 dB K (vs INT2 LM +8.33 dB at 2.5 bpe)

Codebook format (fp16, stored on GPU):
  - K codebook: [N_SUB, N_CENTROIDS, SUB_DIM] = [16, 256, 8] fp16 = 64KB
  - Precomputed from calibration data via K-means
  - Saved to: <rot_dir>/k_pq_codebook_n16_c256_d8.pt

V is handled by existing INT2 LM kernel (SGLANG_LLOYD_MAX=1) at 2.5 bpe.

Combined BPE: K=1.0 bpe, V=2.5 bpe → avg 1.75 bpe
Combined SQNR: K=+9.09 dB, V=+8.34 dB → avg +8.7 dB (exceeds INT2 uniform)

Encode kernel:
  - grid: (cdiv(num_tokens, BLOCK_TOK), num_heads)
  - For each (tok_block, head): 16 sub-vectors × 8D
  - Per sub-vector: element-wise dot product loop over SUB_DIM dimensions
    (avoids tensor core padding complications for K=8)
  - L2² = ||k||² + ||c||² - 2·k·c, argmin over 256 centroids

Decode kernel:
  - grid: (cdiv(num_tokens, BLOCK_TOK), num_heads)
  - For each (tok_block, head): 16 sub-vector lookups
  - Gather centroid vectors from codebook and write to output
"""

from __future__ import annotations
import math, os
from typing import Optional

import torch
import triton
import triton.language as tl

N_SUB = 16  # number of sub-vectors per head
SUB_DIM = 8  # channels per sub-vector
N_CENTROIDS = 256  # 2^8, 1 byte per code

# ─── codebook helpers ─────────────────────────────────────────────────────────


def build_pq_codebook(
    data: torch.Tensor,  # [N_samples, head_dim]
    n_sub: int = N_SUB,
    n_centroids: int = N_CENTROIDS,
    sub_dim: int = SUB_DIM,
    n_iter: int = 30,
    seed: int = 42,
) -> torch.Tensor:
    """
    Train per-position PQ codebook using K-means on float32 data.
    Returns codebook [n_sub, n_centroids, sub_dim] as float16.
    """
    from scipy.cluster.vq import kmeans2
    import numpy as np

    N, D = data.shape
    assert D == n_sub * sub_dim
    sub_data = data.float().numpy().reshape(N, n_sub, sub_dim)

    books = []
    for s in range(n_sub):
        x = sub_data[:, s, :]  # [N, sub_dim]
        print(f"  sub {s}/{n_sub}: kmeans {n_centroids} centroids on {N} samples...")
        cb, _ = kmeans2(x, n_centroids, minit="points", iter=n_iter, seed=seed)
        books.append(cb)

    codebook = torch.from_numpy(np.stack(books)).to(torch.float16)
    return codebook  # [n_sub, n_centroids, sub_dim]


def save_pq_codebook(codebook: torch.Tensor, path: str) -> None:
    torch.save(codebook, path)


def load_pq_codebook(path: str, device="cuda") -> torch.Tensor:
    cb = torch.load(path, map_location="cpu").to(torch.float16)
    return cb.to(device).contiguous()  # [n_sub, n_centroids, sub_dim]


def pq_codebook_norms(codebook: torch.Tensor) -> torch.Tensor:
    """Precompute ||c||² for each centroid: [n_sub, n_centroids] fp32."""
    return (codebook.float() ** 2).sum(dim=-1)  # float32 for accumulation accuracy


# ─── Triton encode kernel ─────────────────────────────────────────────────────


@triton.jit
def _pq_encode_k_kernel(
    k_ptr,  # [num_tok, num_heads, HEAD_DIM] float16 input (row-major)
    loc_ptr,  # [num_tok] int32 — position in KV cache
    codes_ptr,  # [max_tok, num_heads, N_SUB] uint8 output (KV cache buffer)
    cb_ptr,  # [N_SUB, N_CENTROIDS, SUB_DIM] float16 codebook (contiguous)
    cb_norm2_ptr,  # [N_SUB, N_CENTROIDS] float32 ||centroid||²
    num_tokens,
    num_heads,
    k_stride_tok,
    k_stride_head,
    # k_stride_dim assumed = 1
    codes_stride_loc,
    codes_stride_head,
    codes_stride_sub,
    HEAD_DIM: tl.constexpr,  # 128
    N_SUB: tl.constexpr,  # 16
    SUB_DIM: tl.constexpr,  # 8
    N_CENTROIDS: tl.constexpr,  # 256
    BLOCK_TOK: tl.constexpr,  # e.g. 16
    HP_OFFSET: tl.constexpr,
):
    pid_tok = tl.program_id(0)
    pid_head = tl.program_id(1)

    tok_range = pid_tok * BLOCK_TOK + tl.arange(0, BLOCK_TOK)
    active = tok_range < num_tokens

    # cache location for each token in this block
    cache_loc = tl.load(loc_ptr + tok_range, mask=active, other=0)
    if HP_OFFSET >= 0:
        active &= cache_loc < HP_OFFSET

    cent_range = tl.arange(0, N_CENTROIDS)  # [N_CENTROIDS]

    for s in tl.static_range(N_SUB):
        # Accumulate: ||k_sub||² [BLOCK_TOK] and k·c [BLOCK_TOK, N_CENTROIDS]
        k_norm2 = tl.zeros([BLOCK_TOK], dtype=tl.float32)
        dot = tl.zeros([BLOCK_TOK, N_CENTROIDS], dtype=tl.float32)

        for d in tl.static_range(SUB_DIM):
            # Load k_d for this dimension: [BLOCK_TOK]
            k_d = tl.load(
                k_ptr
                + tok_range * k_stride_tok
                + pid_head * k_stride_head
                + (s * SUB_DIM + d),
                mask=active,
                other=0.0,
            ).to(tl.float32)

            k_norm2 += k_d * k_d

            # Load codebook column d for sub-vector s: [N_CENTROIDS]
            cb_d = tl.load(cb_ptr + (s * N_CENTROIDS + cent_range) * SUB_DIM + d).to(
                tl.float32
            )

            # Outer product accumulation: [BLOCK_TOK, N_CENTROIDS]
            dot += k_d[:, None] * cb_d[None, :]

        # Load precomputed ||c||²: [N_CENTROIDS]
        cb_n2 = tl.load(cb_norm2_ptr + s * N_CENTROIDS + cent_range).to(tl.float32)

        # L2² = ||k||² + ||c||² - 2·k·c
        d2 = k_norm2[:, None] + cb_n2[None, :] - 2.0 * dot  # [BLOCK_TOK, N_CENTROIDS]

        # Nearest centroid
        code = tl.argmin(d2, axis=1).to(tl.uint8)  # [BLOCK_TOK]

        # Store code to KV cache buffer
        out_off = (
            cache_loc * codes_stride_loc
            + pid_head * codes_stride_head
            + s * codes_stride_sub
        )
        tl.store(codes_ptr + out_off, code, mask=active)


# ─── Triton decode kernel ─────────────────────────────────────────────────────


@triton.jit
def _pq_decode_k_kernel(
    codes_ptr,  # [max_tok, num_heads, N_SUB] uint8
    out_ptr,  # [num_out_tok, num_heads, HEAD_DIM] float16 output
    cb_ptr,  # [N_SUB, N_CENTROIDS, SUB_DIM] float16
    num_tokens,
    num_heads,
    codes_stride_loc,
    codes_stride_head,
    codes_stride_sub,
    out_stride_tok,
    out_stride_head,
    out_stride_dim,
    HEAD_DIM: tl.constexpr,
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
        ).to(tl.int32)  # [BLOCK_TOK]

        # Gather centroid vectors: [BLOCK_TOK, SUB_DIM]
        # layout: cb[s, code, :] → cb_ptr + s*N_CENTROIDS*SUB_DIM + code*SUB_DIM + d
        cb_idx = (s * N_CENTROIDS + code[:, None]) * SUB_DIM + sub_range[None, :]
        recon = tl.load(
            cb_ptr + cb_idx, mask=active[:, None], other=0.0
        )  # [BLOCK_TOK, SUB_DIM] fp16

        # Write reconstructed sub-vector to output
        out_off = (
            tok_range[:, None] * out_stride_tok
            + pid_head * out_stride_head
            + (s * SUB_DIM + sub_range[None, :]) * out_stride_dim
        )
        tl.store(out_ptr + out_off, recon, mask=active[:, None])


@triton.jit
def _pq_decode_k_at_locs_kernel(
    codes_ptr,
    loc_ptr,
    out_ptr,
    cb_ptr,
    num_tokens,
    num_heads,
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
    """Decode cache rows selected by loc; HP locs produce zero rows."""
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

    for sub in tl.static_range(N_SUB):
        code = tl.load(
            codes_ptr
            + safe_loc * codes_stride_loc
            + pid_head * codes_stride_head
            + sub * codes_stride_sub,
            mask=active,
            other=0,
        ).to(tl.int32)
        cb_idx = (sub * N_CENTROIDS + code[:, None]) * SUB_DIM + sub_range[None, :]
        recon = tl.load(cb_ptr + cb_idx, mask=active[:, None], other=0.0)
        out_off = (
            tok_range[:, None] * out_stride_tok
            + pid_head * out_stride_head
            + (sub * SUB_DIM + sub_range[None, :]) * out_stride_dim
        )
        tl.store(out_ptr + out_off, recon, mask=token_mask[:, None])


# ─── Python launch wrappers ───────────────────────────────────────────────────


def pq_encode_k(
    cache_k: torch.Tensor,  # [num_tokens, num_heads, head_dim] bf16/fp16
    loc: torch.Tensor,  # [num_tokens] int32 — positions in KV cache buffer
    k_codes_buffer: torch.Tensor,  # [max_tokens, num_heads, N_SUB] uint8 output
    codebook: torch.Tensor,  # [N_SUB, N_CENTROIDS, SUB_DIM] fp16
    codebook_norms: torch.Tensor,  # [N_SUB, N_CENTROIDS] fp32
    block_tok: int = 16,
    hp_global_offset: Optional[int] = None,
) -> None:
    """Encode K tensor using product quantization, writing codes into the KV cache buffer."""
    num_tokens, num_heads, head_dim = cache_k.shape
    if num_tokens == 0:
        return

    n_sub, n_centroids, sub_dim = codebook.shape
    assert head_dim == n_sub * sub_dim, (
        f"head_dim {head_dim} != n_sub*sub_dim {n_sub * sub_dim}"
    )
    # Codes are stored as uint8, so any n_centroids <= 256 is valid (RVQ stage2 uses 16).
    assert n_centroids <= 256, f"n_centroids {n_centroids} must be <= 256 (uint8 codes)"

    k_fp16 = cache_k.to(torch.float16).contiguous()
    cb = codebook.to(torch.float16).contiguous()
    cb_n2 = codebook_norms.float().contiguous()

    grid = (triton.cdiv(num_tokens, block_tok), num_heads)
    _pq_encode_k_kernel[grid](
        k_fp16,
        loc,
        k_codes_buffer,
        cb,
        cb_n2,
        num_tokens,
        num_heads,
        k_fp16.stride(0),
        k_fp16.stride(1),
        k_codes_buffer.stride(0),
        k_codes_buffer.stride(1),
        k_codes_buffer.stride(2),
        HEAD_DIM=head_dim,
        N_SUB=n_sub,
        SUB_DIM=sub_dim,
        N_CENTROIDS=n_centroids,
        BLOCK_TOK=block_tok,
        HP_OFFSET=-1 if hp_global_offset is None else int(hp_global_offset),
        num_warps=4,
        num_stages=2,
    )


def pq_decode_k(
    k_codes_buffer: torch.Tensor,  # [max_tokens, num_heads, N_SUB] uint8
    codebook: torch.Tensor,  # [N_SUB, N_CENTROIDS, SUB_DIM] fp16
    num_decode_tokens: int,
    head_dim: int,
    block_tok: int = 16,
) -> torch.Tensor:
    """Decode PQ codes to reconstructed K tensor [num_decode_tokens, num_heads, head_dim] fp16."""
    _, num_heads, n_sub = k_codes_buffer.shape
    n_centroids, sub_dim = codebook.shape[1], codebook.shape[2]

    out = torch.empty(
        num_decode_tokens,
        num_heads,
        head_dim,
        dtype=torch.float16,
        device=k_codes_buffer.device,
    )
    cb = codebook.to(torch.float16).contiguous()

    grid = (triton.cdiv(num_decode_tokens, block_tok), num_heads)
    _pq_decode_k_kernel[grid](
        k_codes_buffer,
        out,
        cb,
        num_decode_tokens,
        num_heads,
        k_codes_buffer.stride(0),
        k_codes_buffer.stride(1),
        k_codes_buffer.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        HEAD_DIM=head_dim,
        N_SUB=n_sub,
        SUB_DIM=sub_dim,
        N_CENTROIDS=n_centroids,
        BLOCK_TOK=block_tok,
        num_warps=4,
        num_stages=2,
    )
    return out


def pq_decode_k_at_locs(
    k_codes_buffer: torch.Tensor,
    loc: torch.Tensor,
    codebook: torch.Tensor,
    head_dim: int,
    block_tok: int = 16,
    hp_global_offset: Optional[int] = None,
) -> torch.Tensor:
    """Decode cache rows selected by loc without data-dependent indexing."""
    num_tokens = int(loc.shape[0])
    _, num_heads, n_sub = k_codes_buffer.shape
    n_centroids, sub_dim = codebook.shape[1], codebook.shape[2]
    assert head_dim == n_sub * sub_dim
    out = torch.empty(
        num_tokens,
        num_heads,
        head_dim,
        dtype=torch.float16,
        device=k_codes_buffer.device,
    )
    if num_tokens == 0:
        return out
    cb = codebook.to(torch.float16).contiguous()
    grid = (triton.cdiv(num_tokens, block_tok), num_heads)
    _pq_decode_k_at_locs_kernel[grid](
        k_codes_buffer,
        loc,
        out,
        cb,
        num_tokens,
        num_heads,
        k_codes_buffer.stride(0),
        k_codes_buffer.stride(1),
        k_codes_buffer.stride(2),
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


# ─── Python reference (for SQNR validation) ──────────────────────────────────


def pq_encode_decode_python(
    k: torch.Tensor,  # [N, head_dim] fp32
    codebook: torch.Tensor,  # [N_SUB, N_CENTROIDS, SUB_DIM] fp32
) -> torch.Tensor:
    """Reference PQ encode+decode on CPU, returns reconstructed K [N, head_dim] fp32."""
    N, D = k.shape
    n_sub, n_centroids, sub_dim = codebook.shape
    k_np = k.float().numpy()
    cb_np = codebook.float().numpy()
    sub_data = k_np.reshape(N, n_sub, sub_dim)

    recon_subs = []
    for s in range(n_sub):
        x = sub_data[:, s, :]  # [N, sub_dim]
        cb = cb_np[s]  # [n_centroids, sub_dim]
        # L2 distance
        d2 = ((x[:, None, :] - cb[None, :, :]) ** 2).sum(axis=2)  # [N, n_centroids]
        codes = d2.argmin(axis=1)  # [N]
        recon_subs.append(cb[codes])  # [N, sub_dim]

    import numpy as np

    recon = np.concatenate(recon_subs, axis=1)  # [N, head_dim]
    return torch.from_numpy(recon)


# ─── codebook training ────────────────────────────────────────────────────────


def train_and_save_codebook_from_dumps(
    dump_base: str,
    rot_dir: str,
    save_path: str,
    n_sub: int = N_SUB,
    n_centroids: int = N_CENTROIDS,
    sub_dim: int = SUB_DIM,
    max_samples: int = 300_000,
):
    """Train PQ codebook from actual model K dumps after OSCAR rotation."""
    import glob
    import numpy as np

    rk_data = torch.load(f"{rot_dir}/k_rotation_qqt_r_h_pbr.pt", map_location="cpu")
    R_k = {v["layer_id"]: v["rotation"] for v in rk_data["layers"].values()}

    qkv_base = f"{dump_base}/qkv_dumps/gpqa"
    layers = sorted(
        int(d.split("_")[1])
        for d in os.listdir(qkv_base)
        if d.startswith("layer_") and os.path.isdir(f"{qkv_base}/{d}")
    )

    k_chunks = []
    for lid in layers:
        for kf in sorted(glob.glob(f"{qkv_base}/layer_{lid}/k/*.pt")):
            k = torch.load(kf, map_location="cpu").float() @ R_k[lid]
            k_chunks.append(k.reshape(-1, n_sub * sub_dim))

    k_all = torch.cat(k_chunks, dim=0)  # [N, head_dim]
    print(f"Total samples: {k_all.shape[0]}")

    rng = np.random.default_rng(42)
    idx = rng.choice(k_all.shape[0], min(max_samples, k_all.shape[0]), replace=False)
    k_train = k_all[idx]
    print(f"Training on {k_train.shape[0]} samples...")

    codebook = build_pq_codebook(k_train, n_sub, n_centroids, sub_dim)
    save_pq_codebook(codebook, save_path)
    print(f"Saved: {save_path}  shape={codebook.shape}")
    return codebook


# ─── standalone test ─────────────────────────────────────────────────────────


def _test_sqnr(n_tok=1024, n_heads=8, seed=0):
    """
    Quick correctness + SQNR test using synthetic Gaussian K data.

    NOTE: synthetic i.i.d. Gaussian data yields ~5 dB SQNR (scalar quantization limit).
    Real Qwen3 K data after OSCAR rotation has inter-channel correlations that PQ
    exploits, yielding ~9 dB (measured via sqnr_1bit_exploration.py).
    """
    import numpy as np

    torch.manual_seed(seed)
    head_dim = N_SUB * SUB_DIM  # 128
    device = "cuda"

    # Synthetic: N(0,1) with random per-token-head scale (mimics real K)
    k = torch.randn(n_tok, n_heads, head_dim, device=device)
    scale = torch.rand(n_tok, n_heads, 1, device=device) * 2 + 0.1
    k = k * scale

    # Train codebook from same data (upper bound; real use: separate calibration set)
    k_flat = k.reshape(-1, head_dim).cpu()
    print("Training codebook...")
    codebook_cpu = build_pq_codebook(k_flat, N_SUB, N_CENTROIDS, SUB_DIM, n_iter=20)
    codebook = codebook_cpu.to(device)
    cb_norms = pq_codebook_norms(codebook)

    # Encode via Triton
    loc = torch.arange(n_tok, device=device, dtype=torch.int32)
    codes_buf = torch.zeros(n_tok, n_heads, N_SUB, device=device, dtype=torch.uint8)
    pq_encode_k(k, loc, codes_buf, codebook, cb_norms)

    # Decode via Triton
    k_recon = pq_decode_k(codes_buf, codebook, n_tok, head_dim)

    # Triton SQNR
    k_f = k.float()
    r_f = k_recon.float()
    sqnr_db = 10 * torch.log10((k_f**2).mean() / ((k_f - r_f) ** 2).mean()).item()

    # Python reference on first 64 tokens (correctness check)
    n_check = 64
    k_py = k[:n_check].reshape(-1, head_dim)
    k_py_recon = pq_encode_decode_python(k_py.cpu(), codebook_cpu)
    py_sqnr = 10 * math.log10(
        (k_py.float() ** 2).mean().item()
        / ((k_py.float().cpu() - k_py_recon.float()) ** 2).mean().item()
    )

    # Code match: Triton codes vs Python codes
    k_sub = k[:n_check].cpu().float().numpy().reshape(-1, N_SUB, SUB_DIM)
    cb_np = codebook_cpu.float().numpy()
    py_codes = np.zeros((n_check * n_heads, N_SUB), dtype=np.uint8)
    for s in range(N_SUB):
        x = k_sub[:, s, :]
        d2 = ((x[:, None, :] - cb_np[s][None, :, :]) ** 2).sum(axis=2)
        py_codes[:, s] = d2.argmin(axis=1)
    triton_codes = codes_buf[:n_check].cpu().numpy().reshape(-1, N_SUB)
    code_match = (py_codes == triton_codes).mean() * 100

    print(f"Triton SQNR:      {sqnr_db:.2f} dB  (synthetic i.i.d. Gaussian)")
    print(f"Python ref SQNR:  {py_sqnr:.2f} dB  (same data, fp32 codebook)")
    print(
        f"Code match:       {code_match:.2f}%  (Triton vs Python, first {n_check} tokens)"
    )
    print(f"  [Real Qwen3 K data yields ~+9 dB from inter-channel correlations]")
    return sqnr_db, code_match


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        if len(sys.argv) != 5:
            raise SystemExit(
                "usage: oscar_rotation_pq_k_kv.py train "
                "<dump_dir> <rotation_dir> <save_path>"
            )
        train_and_save_codebook_from_dumps(*sys.argv[2:5])
    else:
        sqnr, code_match = _test_sqnr()
        ok = sqnr > 4.5 and code_match > 95.0
        print("PASS" if ok else "FAIL")
