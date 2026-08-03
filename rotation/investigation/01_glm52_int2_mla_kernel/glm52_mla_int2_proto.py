#!/usr/bin/env python3
"""REAL INT2 MLA-latent KV prototype for GLM-5.2 (pack + fused decode kernel).

The shipped NSAInt2HPKVPool is *fake-quant* (quant->dequant, stored BF16 —
accuracy only). This prototype implements the real thing:

  storage : rotated c_kv packed to 2-bit codes [T, 128B] + per-group
            (scale, bias) fp32 [T, 4, 2]  (group=128, kv_lora_rank=512)
            k_pe [T, 64] stays BF16.       -> c_kv bytes: 1024B -> 160B (6.4x)
  kernels : (1) pack_int2   — LM bucketize + bit-pack (matches
                _fake_quant_int2_groupwise(lloyd_max=True) EXACTLY)
            (2) mla_decode_int2 — MQA absorb decode: scores = qL.c_kv + qpe.k_pe,
                online softmax, out = sum p*c_kv, dequant INLINE from codes.
  checks  : (1) kernel dequant == pool fake-quant; (2) kernel attention ==
            torch reference, on REAL dumped c_kv + the fitted rotation.
  bench   : int2 kernel vs identical-structure bf16-latent kernel.

Run (single GPU):
  python glm52_mla_int2_proto.py --dump /shared/glm52_fp8_c_kv_dump \
      --rot .../GLM-5.2-FP8/GPQA/_rcov_phblock/rotations --layer 0 --ctx 8192
"""
import argparse
import math
import os

import torch
import triton
import triton.language as tl

# ── Lloyd-Max constants (MUST match mla_int2_kv_pool.py) ─────────────────────
LM_T0, LM_T1, LM_T2 = -0.9810652732849121, 0.0, 0.9810652732849121
LM_C0 = -1.5095585584640503
LM_SPAN = 3.0191171169281006          # C3 - C0
LM_RATIO = 1.16
GROUP = 128


# ── kernel 1: LM quantize + pack (one program per token) ─────────────────────
@triton.jit
def pack_int2_kernel(x_ptr, codes_ptr, sb_ptr, D: tl.constexpr, G: tl.constexpr,
                     T0: tl.constexpr, T1: tl.constexpr, T2: tl.constexpr,
                     C0: tl.constexpr, SPAN3: tl.constexpr, RATIO: tl.constexpr):
    t = tl.program_id(0)
    n_g: tl.constexpr = D // G
    for g in tl.static_range(n_g):
        offs = t * D + g * G + tl.arange(0, G)
        x = tl.load(x_ptr + offs).to(tl.float32)
        mean = tl.sum(x, axis=0) / G
        diff = x - mean
        std = tl.sqrt(tl.sum(diff * diff, axis=0) / G + 1e-8)
        z = diff / std
        q = (z >= T0).to(tl.uint8) + (z >= T1).to(tl.uint8) + (z >= T2).to(tl.uint8)
        scale = SPAN3 * RATIO * std
        bias = mean + C0 * RATIO * std
        tl.store(sb_ptr + (t * n_g + g) * 2 + 0, scale)
        tl.store(sb_ptr + (t * n_g + g) * 2 + 1, bias)
        # pack 4 crumbs/byte: byte b holds dims 4b..4b+3 (crumb i at bits 2i..2i+1)
        qb = tl.reshape(q, (G // 4, 4))
        sh = tl.arange(0, 4) * 2
        packed = tl.sum(qb.to(tl.int32) << sh[None, :], axis=1).to(tl.uint8)
        tl.store(codes_ptr + t * (D // 4) + g * (G // 4) + tl.arange(0, G // 4), packed)


# ── kernel 2: fused MQA absorb decode with inline INT2 dequant ───────────────
# One program per (head, split). Whole-row dequant once per t-block; the
# dequantized tile feeds BOTH the score dot and the p@V accumulate — identical
# structure to the bf16 baseline below, only the load path differs.
@triton.jit
def mla_decode_int2_kernel(qL_ptr, qpe_ptr, codes_ptr, sb_ptr, kpe_ptr,
                           m_ptr, l_ptr, acc_ptr,
                           T, SM_SCALE,
                           D: tl.constexpr, DP: tl.constexpr, G: tl.constexpr,
                           BT: tl.constexpr, SPLITS: tl.constexpr):
    h = tl.program_id(0)
    s = tl.program_id(1)
    n_g: tl.constexpr = D // G
    per = tl.cdiv(T, SPLITS)
    t_lo = s * per
    t_hi = tl.minimum(t_lo + per, T)

    offs_d = tl.arange(0, D)
    byte_idx = offs_d // 4
    shift = (offs_d % 4) * 2
    qL = tl.load(qL_ptr + h * D + offs_d).to(tl.float32)
    qpe = tl.load(qpe_ptr + h * DP + tl.arange(0, DP)).to(tl.float32)

    m_i = -1e30
    l_i = 0.0
    acc = tl.zeros((D,), dtype=tl.float32)

    for t0 in range(t_lo, t_hi, BT):
        offs_t = t0 + tl.arange(0, BT)
        mask_t = offs_t < t_hi
        raw = tl.load(codes_ptr + offs_t[:, None] * (D // 4) + byte_idx[None, :],
                      mask=mask_t[:, None], other=0)
        q = ((raw >> shift[None, :]) & 3).to(tl.float32)
        # per-group scale/bias -> full-width via static select
        scale_f = tl.zeros((BT, D), dtype=tl.float32)
        bias_f = tl.zeros((BT, D), dtype=tl.float32)
        for g in tl.static_range(n_g):
            sc_g = tl.load(sb_ptr + (offs_t * n_g + g) * 2 + 0, mask=mask_t, other=1.0)
            bi_g = tl.load(sb_ptr + (offs_t * n_g + g) * 2 + 1, mask=mask_t, other=0.0)
            in_g = (offs_d[None, :] // G) == g
            scale_f = tl.where(in_g, sc_g[:, None], scale_f)
            bias_f = tl.where(in_g, bi_g[:, None], bias_f)
        c = q * scale_f + bias_f                                     # (BT, D)

        kpe = tl.load(kpe_ptr + offs_t[:, None] * DP + tl.arange(0, DP)[None, :],
                      mask=mask_t[:, None], other=0.0).to(tl.float32)
        sc = tl.sum(c * qL[None, :], axis=1) + tl.sum(kpe * qpe[None, :], axis=1)
        sc = sc * SM_SCALE
        sc = tl.where(mask_t, sc, -1e30)

        m_new = tl.maximum(m_i, tl.max(sc, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(sc - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * c, axis=0)
        m_i = m_new

    tl.store(m_ptr + h * SPLITS + s, m_i)
    tl.store(l_ptr + h * SPLITS + s, l_i)
    tl.store(acc_ptr + (h * SPLITS + s) * D + tl.arange(0, D), acc)


# ── bf16 baseline kernel (identical structure, reads bf16 latent) ────────────
@triton.jit
def mla_decode_bf16_kernel(qL_ptr, qpe_ptr, ckv_ptr, kpe_ptr,
                           m_ptr, l_ptr, acc_ptr, T, SM_SCALE,
                           D: tl.constexpr, DP: tl.constexpr,
                           BT: tl.constexpr, SPLITS: tl.constexpr):
    h = tl.program_id(0)
    s = tl.program_id(1)
    per = tl.cdiv(T, SPLITS)
    t_lo = s * per
    t_hi = tl.minimum(t_lo + per, T)
    offs_d = tl.arange(0, D)
    qL = tl.load(qL_ptr + h * D + offs_d).to(tl.float32)
    qpe = tl.load(qpe_ptr + h * DP + tl.arange(0, DP)).to(tl.float32)
    m_i = -1e30
    l_i = 0.0
    acc = tl.zeros((D,), dtype=tl.float32)
    for t0 in range(t_lo, t_hi, BT):
        offs_t = t0 + tl.arange(0, BT)
        mask_t = offs_t < t_hi
        c = tl.load(ckv_ptr + offs_t[:, None] * D + offs_d[None, :],
                    mask=mask_t[:, None], other=0.0).to(tl.float32)
        kpe = tl.load(kpe_ptr + offs_t[:, None] * DP + tl.arange(0, DP)[None, :],
                      mask=mask_t[:, None], other=0.0).to(tl.float32)
        sc = tl.sum(c * qL[None, :], axis=1) + tl.sum(kpe * qpe[None, :], axis=1)
        sc = sc * SM_SCALE
        sc = tl.where(mask_t, sc, -1e30)
        m_new = tl.maximum(m_i, tl.max(sc, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(sc - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * c, axis=0)
        m_i = m_new
    tl.store(m_ptr + h * SPLITS + s, m_i)
    tl.store(l_ptr + h * SPLITS + s, l_i)
    tl.store(acc_ptr + (h * SPLITS + s) * D + tl.arange(0, D), acc)


def combine_splits(m, l, acc):
    """standard flash-decode split reduction (fp32, torch-side)."""
    m_g = m.max(dim=1, keepdim=True).values                     # [H,1]
    w = (m - m_g).exp()                                         # [H,S]
    out = (acc * w.unsqueeze(-1)).sum(1) / (l * w).sum(1, keepdim=True)
    return out                                                  # [H,D]


def fake_quant_ref(x, group=GROUP):
    """EXACT copy of pool _fake_quant_int2_groupwise(lloyd_max=True)."""
    orig = x.shape
    x = x.reshape(-1, group).to(torch.float32)
    mean = x.mean(-1, keepdim=True)
    diff = x - mean
    std = (diff.pow(2).mean(-1, keepdim=True) + 1e-8).sqrt()
    z = diff / std
    q = ((z >= LM_T0).to(torch.uint8) + (z >= LM_T1).to(torch.uint8)
         + (z >= LM_T2).to(torch.uint8))
    us = (LM_SPAN / 3.0) * LM_RATIO * std
    uz = -LM_C0 / (LM_SPAN / 3.0) - mean / us
    return ((q.to(torch.float32) - uz) * us).reshape(orig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="/shared/glm52_fp8_c_kv_dump")
    ap.add_argument("--rot", default="/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/oscar/rotation/GLM-5.2-FP8/GPQA/_rcov_phblock/rotations")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--heads", type=int, default=64)
    ap.add_argument("--splits", type=int, default=8)
    ap.add_argument("--bt", type=int, default=16)
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    D, DP = 512, 64

    # real data: dumped c_kv + fitted rotation
    ckv = torch.load(os.path.join(args.dump, f"layer_{args.layer}.pt"),
                     map_location=dev)[: args.ctx].to(torch.float32)
    T = ckv.shape[0]
    R = torch.load(os.path.join(args.rot, f"layer_{args.layer}.pt"),
                   map_location=dev).to(torch.float32)
    ckv_rot = ckv @ R                                   # rotated latent (what gets stored)
    kpe = torch.randn(T, DP, device=dev, dtype=torch.float32) * 0.5
    qL = torch.randn(args.heads, D, device=dev, dtype=torch.float32) * 0.1
    qpe = torch.randn(args.heads, DP, device=dev, dtype=torch.float32) * 0.1
    sm = 1.0 / math.sqrt(D + DP)

    # ── pack ──
    n_g = D // GROUP
    codes = torch.empty(T, D // 4, dtype=torch.uint8, device=dev)
    sb = torch.empty(T, n_g, 2, dtype=torch.float32, device=dev)
    pack_int2_kernel[(T,)](ckv_rot, codes, sb, D, GROUP,
                           LM_T0, LM_T1, LM_T2, LM_C0, LM_SPAN / 3.0, LM_RATIO)
    torch.cuda.synchronize()

    # check 1: kernel pack+dequant == pool fake-quant (same math)
    q_unpacked = torch.stack([(codes >> (2 * i)) & 3 for i in range(4)], -1).reshape(T, D)
    deq_kernel = (q_unpacked.float().reshape(T, n_g, GROUP)
                  * sb[:, :, 0:1] + sb[:, :, 1:2]).reshape(T, D)
    deq_pool = fake_quant_ref(ckv_rot)
    e1 = (deq_kernel - deq_pool).abs().max().item()
    rel1 = ((deq_kernel - deq_pool).norm() / deq_pool.norm()).item()
    print(f"[check1] pack+dequant vs pool fake-quant: max_abs={e1:.3e} rel_l2={rel1:.3e}")

    # ── reference attention (fp32 on dequantized latent) ──
    scores = (deq_pool @ qL.T + kpe @ qpe.T) * sm       # [T,H]
    p = torch.softmax(scores, dim=0)
    ref = p.T @ deq_pool                                # [H,D]

    # ── int2 kernel ──
    S = args.splits
    m = torch.empty(args.heads, S, device=dev); l = torch.empty_like(m)
    acc = torch.empty(args.heads, S, D, device=dev)
    mla_decode_int2_kernel[(args.heads, S)](qL, qpe, codes, sb.reshape(-1), kpe,
                                            m, l, acc.reshape(-1), T, sm,
                                            D, DP, GROUP, args.bt, S)
    torch.cuda.synchronize()
    out = combine_splits(m, l, acc)
    rel = ((out - ref).norm() / ref.norm()).item()
    cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    print(f"[check2] int2 kernel vs torch ref: rel_l2={rel:.3e} cos={cos:.6f}")

    # ── bf16 baseline kernel (identical structure) ──
    ckv_bf = ckv_rot.to(torch.bfloat16).contiguous()
    kpe_c = kpe.contiguous()
    m2 = torch.empty_like(m); l2 = torch.empty_like(l); acc2 = torch.empty_like(acc)
    mla_decode_bf16_kernel[(args.heads, S)](qL, qpe, ckv_bf, kpe_c,
                                            m2, l2, acc2.reshape(-1), T, sm,
                                            D, DP, args.bt, S)
    torch.cuda.synchronize()
    out_bf = combine_splits(m2, l2, acc2)
    # sanity: bf16 kernel vs its own torch ref
    sc_bf = (ckv_bf.float() @ qL.T + kpe @ qpe.T) * sm
    ref_bf = torch.softmax(sc_bf, 0).T @ ckv_bf.float()
    rel_bf = ((out_bf - ref_bf).norm() / ref_bf.norm()).item()
    print(f"[check3] bf16 kernel vs torch ref: rel_l2={rel_bf:.3e}")

    # ── microbench ──
    def bench(fn, iters=50):
        for _ in range(5): fn()
        torch.cuda.synchronize()
        s0 = torch.cuda.Event(True); e0 = torch.cuda.Event(True)
        s0.record()
        for _ in range(iters): fn()
        e0.record(); torch.cuda.synchronize()
        return s0.elapsed_time(e0) / iters

    t_i2 = bench(lambda: mla_decode_int2_kernel[(args.heads, S)](
        qL, qpe, codes, sb.reshape(-1), kpe, m, l, acc.reshape(-1), T, sm, D, DP, GROUP, args.bt, S))
    t_bf = bench(lambda: mla_decode_bf16_kernel[(args.heads, S)](
        qL, qpe, ckv_bf, kpe_c, m2, l2, acc2.reshape(-1), T, sm, D, DP, args.bt, S))
    t_pk = bench(lambda: pack_int2_kernel[(T,)](
        ckv_rot, codes, sb, D, GROUP, LM_T0, LM_T1, LM_T2, LM_C0, LM_SPAN / 3.0, LM_RATIO))

    bytes_i2 = T * (D // 4 + n_g * 8)          # codes + (scale,bias) fp32
    bytes_bf = T * D * 2
    print(f"[bench] ctx={T} heads={args.heads} splits={S} bt={args.bt}")
    print(f"  decode  int2: {t_i2:.3f} ms | bf16: {t_bf:.3f} ms | speedup x{t_bf/max(t_i2,1e-9):.2f}")
    print(f"  pack(quant-write): {t_pk:.3f} ms / {T} tokens")
    print(f"  c_kv bytes/token: int2 {bytes_i2//T} vs bf16 {bytes_bf//T} ({bytes_bf/bytes_i2:.2f}x smaller)")


if __name__ == "__main__":
    main()
