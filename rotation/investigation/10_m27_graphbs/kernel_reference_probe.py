#!/usr/bin/env python3
"""Check ``decode_attention_fwd_int2_unified`` against a dense reference.

The graph-vs-eager probes in this directory only prove the two *paths* agree.
They cannot see a defect that is present in both. This probe adds the missing
reference: it dequantises the INT2 tier in torch, concatenates it with the BF16
HP tier, runs ordinary softmax attention, and compares against the fused Triton
kernel -- at every batch size, so a batch-size-dependent correctness bug shows
up as a row that stops matching.

Geometry defaults to MiniMax-M2.7 at TP=4 (12 q heads / 2 kv heads / 128 dim,
quant group 128, 8 HP splits + 8 quant splits).
"""
import argparse
import os
import sys

import torch

p = argparse.ArgumentParser()
p.add_argument("--tree", default=os.environ.get("TREE", "/tmp/tree"))
p.add_argument("--heads", type=int, default=12)
p.add_argument("--kv-heads", type=int, default=2)
p.add_argument("--head-dim", type=int, default=128)
p.add_argument("--hp-splits", type=int, default=8)
p.add_argument("--quant-splits", type=int, default=8)
p.add_argument("--hp-len", type=int, default=64 + 263)
p.add_argument("--seq-min", type=int, default=4000)
p.add_argument("--seq-max", type=int, default=33000)
p.add_argument("--seed", type=int, default=0)
p.add_argument(
    "--batch-sizes", default="1,2,4,8,12,13,14,15,16,17,20,24,32",
)
args = p.parse_args()

sys.path.insert(0, os.path.join(args.tree, "sglang-research", "python"))
from sglang.srt.layers.attention.triton_ops.decode_attention import (  # noqa: E402
    decode_attention_fwd_int2_unified,
)

dev = "cuda"
torch.manual_seed(args.seed)

H, KVH, D = args.heads, args.kv_heads, args.head_dim
GROUP = H // KVH
HP_S, Q_S = args.hp_splits, args.quant_splits
TOTAL = HP_S + Q_S
SM = D ** -0.5
BS_LIST = [int(x) for x in args.batch_sizes.split(",")]
MAXBS = max(BS_LIST)
HP_LEN = args.hp_len

HP_SLOTS = MAXBS * (HP_LEN + 8) + 16
Q_SLOTS = MAXBS * args.seq_max + 16

print(f"[ref] H={H} KVH={KVH} D={D} hp_splits={HP_S} quant_splits={Q_S} "
      f"hp_len={HP_LEN}", flush=True)

hp_k = (torch.randn(HP_SLOTS, KVH, D, device=dev) * 0.5).to(torch.bfloat16)
hp_v = (torch.randn(HP_SLOTS, KVH, D, device=dev) * 0.5).to(torch.bfloat16)
qk = torch.randint(0, 256, (Q_SLOTS, KVH, D // 4), dtype=torch.uint8, device=dev)
qv = torch.randint(0, 256, (Q_SLOTS, KVH, D // 4), dtype=torch.uint8, device=dev)
# [scale, zero] per (slot, kv_head): one group over the whole head dim, which is
# what quant group 128 == head_dim gives.
ksz = torch.stack(
    [torch.rand(Q_SLOTS, KVH, device=dev) * 0.3 + 0.05,
     torch.rand(Q_SLOTS, KVH, device=dev) * 2.0], dim=-1
).contiguous()
vsz = torch.stack(
    [torch.rand(Q_SLOTS, KVH, device=dev) * 0.3 + 0.05,
     torch.rand(Q_SLOTS, KVH, device=dev) * 2.0], dim=-1
).contiguous()


def dequant(packed, sz, idx):
    """packed[idx] : [L, KVH, D//4] uint8 -> [L, KVH, D] float32."""
    b = packed[idx].to(torch.int32)                    # [L, KVH, D/4]
    crumbs = torch.stack(
        [(b >> s) & 0x3 for s in (0, 2, 4, 6)], dim=-1  # [L, KVH, D/4, 4]
    ).to(torch.float32)
    L = b.shape[0]
    # The kernel treats quarter q of the packed byte as head-dim block q, i.e.
    # element (j, q) of the unpacked row maps to output index q*(D//4)+j.
    crumbs = crumbs.permute(0, 1, 3, 2).reshape(L, KVH, D)
    scale = sz[idx, :, 0:1]
    zero = sz[idx, :, 1:2]
    return (crumbs - zero) * scale


def reference(q, hp_idx_list, q_idx_list):
    bs = q.shape[0]
    out = torch.empty(bs, H, D, dtype=torch.float32, device=dev)
    for i in range(bs):
        hi, qi = hp_idx_list[i], q_idx_list[i]
        k_hp = hp_k[hi].to(torch.float32)               # [Lh, KVH, D]
        v_hp = hp_v[hi].to(torch.float32)
        k_q = dequant(qk, ksz, qi)                      # [Lq, KVH, D]
        v_q = dequant(qv, vsz, qi)
        k = torch.cat([k_hp, k_q], dim=0)               # [L, KVH, D]
        v = torch.cat([v_hp, v_q], dim=0)
        for h in range(H):
            kh = k[:, h // GROUP, :]
            vh = v[:, h // GROUP, :]
            s = (kh @ q[i, h].to(torch.float32)) * SM
            pr = torch.softmax(s, dim=0)
            out[i, h] = pr @ vh
    return out


import numpy as np  # noqa: E402

gen = np.random.default_rng(args.seed)
print()
print(f"{'bs':>4} {'seq_len range':>18} {'max|out-ref|':>14} {'rel_l2':>11}  verdict")
print("-" * 70)

worst_bs, worst_rel = None, 0.0
for bs in BS_LIST:
    seq = [int(gen.integers(args.seq_min, args.seq_max)) for _ in range(bs)]
    hp_idx_list, q_idx_list = [], []
    hp_cum, q_cum = [0], [0]
    for i, s in enumerate(seq):
        base_hp = i * (HP_LEN + 8)
        hp_idx_list.append(torch.arange(base_hp, base_hp + HP_LEN,
                                        dtype=torch.int64, device=dev))
        base_q = i * args.seq_max
        q_idx_list.append(torch.arange(base_q, base_q + (s - HP_LEN),
                                       dtype=torch.int64, device=dev))
        hp_cum.append(hp_cum[-1] + HP_LEN)
        q_cum.append(q_cum[-1] + s - HP_LEN)

    hp_indptr = torch.tensor(hp_cum, dtype=torch.int32, device=dev)
    q_indptr = torch.tensor(q_cum, dtype=torch.int32, device=dev)
    hp_idx = torch.cat(hp_idx_list)
    q_idx = torch.cat(q_idx_list)

    q = (torch.randn(bs, H, D, device=dev) * 0.6).to(torch.bfloat16)
    o = torch.empty(bs, H, D, dtype=torch.bfloat16, device=dev)
    logits = torch.empty(bs, H, TOTAL, D, dtype=torch.float32, device=dev)
    lse = torch.full((bs, H, TOTAL), float("-inf"), dtype=torch.float32, device=dev)
    hp_sp = torch.full((bs,), HP_S, dtype=torch.int32, device=dev)
    # Production planner saturates at max_kv_splits for every decode-length
    # sequence on this geometry; use the cap directly.
    q_sp = torch.full((bs,), Q_S, dtype=torch.int32, device=dev)

    decode_attention_fwd_int2_unified(
        q, hp_k, hp_v, qk, qv, ksz, vsz, o,
        hp_indptr, hp_idx, q_indptr, q_idx,
        logits, lse, hp_sp, q_sp, HP_S, Q_S, SM,
    )
    torch.cuda.synchronize()

    ref = reference(q, hp_idx_list, q_idx_list)
    d = (o.float() - ref).abs()
    rel = (d.norm() / ref.norm().clamp_min(1e-9)).item()
    if rel > worst_rel:
        worst_rel, worst_bs = rel, bs
    ok = rel < 5e-3
    print(f"{bs:>4} {min(seq):>8}..{max(seq):<8} {d.max().item():>14.6g} "
          f"{rel:>11.3e}  {'OK' if ok else '*** WRONG ***'}")

print()
print(f"[ref] worst rel_l2 = {worst_rel:.3e} at bs={worst_bs}")
print("[ref] bf16 output vs fp32 reference: ~1e-3 is expected rounding; "
      "a jump of 10x or more at one batch size is a real defect.")
