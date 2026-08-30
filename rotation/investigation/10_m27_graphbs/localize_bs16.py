#!/usr/bin/env python3
"""Localise the bs>=16 error in the INT2 mixed-KV decode path.

``kernel_reference_probe.py`` showed the fused kernel matches a dense
reference to 1.4e-3 for bs<=15 and to only 1.3e-2 for bs>=16. This narrows it:

* one fixed 32-request payload, so per-request reference output does not depend
  on the batch size at all;
* run the kernel on the first ``bs`` requests for a range of ``bs`` and report
  the per-request error, so "request i changes when other requests join the
  batch" becomes directly visible;
* tier ablations (HP only / quant only) to say which stage-1 is responsible;
* a split-count sweep, to say whether the boundary tracks the batch size or the
  number of splits.
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
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

sys.path.insert(0, os.path.join(args.tree, "sglang-research", "python"))
from sglang.srt.layers.attention.triton_ops.decode_attention import (  # noqa: E402
    decode_attention_fwd_int2_unified,
)

dev = "cuda"
torch.manual_seed(args.seed)
H, KVH, D = args.heads, args.kv_heads, args.head_dim
GROUP = H // KVH
SM = D ** -0.5
HP_LEN = args.hp_len
NREQ = 32
SEQ_MAX = 33000

HP_SLOTS = NREQ * (HP_LEN + 8) + 16
Q_SLOTS = NREQ * SEQ_MAX + 16

hp_k = (torch.randn(HP_SLOTS, KVH, D, device=dev) * 0.5).to(torch.bfloat16)
hp_v = (torch.randn(HP_SLOTS, KVH, D, device=dev) * 0.5).to(torch.bfloat16)
qk = torch.randint(0, 256, (Q_SLOTS, KVH, D // 4), dtype=torch.uint8, device=dev)
qv = torch.randint(0, 256, (Q_SLOTS, KVH, D // 4), dtype=torch.uint8, device=dev)
ksz = torch.stack([torch.rand(Q_SLOTS, KVH, device=dev) * 0.3 + 0.05,
                   torch.rand(Q_SLOTS, KVH, device=dev) * 2.0], dim=-1).contiguous()
vsz = torch.stack([torch.rand(Q_SLOTS, KVH, device=dev) * 0.3 + 0.05,
                   torch.rand(Q_SLOTS, KVH, device=dev) * 2.0], dim=-1).contiguous()

import numpy as np  # noqa: E402

gen = np.random.default_rng(args.seed)
SEQ = [int(gen.integers(4000, SEQ_MAX)) for _ in range(NREQ)]
Q_ALL = (torch.randn(NREQ, H, D, device=dev) * 0.6).to(torch.bfloat16)

HP_IDX = [torch.arange(i * (HP_LEN + 8), i * (HP_LEN + 8) + HP_LEN,
                       dtype=torch.int64, device=dev) for i in range(NREQ)]
Q_IDX = [torch.arange(i * SEQ_MAX, i * SEQ_MAX + SEQ[i] - HP_LEN,
                      dtype=torch.int64, device=dev) for i in range(NREQ)]


def dequant(packed, sz, idx):
    b = packed[idx].to(torch.int32)
    crumbs = torch.stack([(b >> s) & 0x3 for s in (0, 2, 4, 6)], dim=-1).to(torch.float32)
    L = b.shape[0]
    crumbs = crumbs.permute(0, 1, 3, 2).reshape(L, KVH, D)
    return (crumbs - sz[idx, :, 1:2]) * sz[idx, :, 0:1]


def ref_one(i, use_hp=True, use_quant=True):
    ks, vs = [], []
    if use_hp:
        ks.append(hp_k[HP_IDX[i]].float()); vs.append(hp_v[HP_IDX[i]].float())
    if use_quant:
        ks.append(dequant(qk, ksz, Q_IDX[i])); vs.append(dequant(qv, vsz, Q_IDX[i]))
    k, v = torch.cat(ks), torch.cat(vs)
    out = torch.empty(H, D, dtype=torch.float32, device=dev)
    for h in range(H):
        s = (k[:, h // GROUP, :] @ Q_ALL[i, h].float()) * SM
        out[h] = torch.softmax(s, 0) @ v[:, h // GROUP, :]
    return out


def run(bs, hp_splits, quant_splits, use_hp=True, use_quant=True):
    hp_cum, q_cum = [0], [0]
    for i in range(bs):
        hp_cum.append(hp_cum[-1] + (HP_LEN if use_hp else 0))
        q_cum.append(q_cum[-1] + ((SEQ[i] - HP_LEN) if use_quant else 0))
    hp_indptr = torch.tensor(hp_cum, dtype=torch.int32, device=dev)
    q_indptr = torch.tensor(q_cum, dtype=torch.int32, device=dev)
    hp_idx = (torch.cat(HP_IDX[:bs]) if use_hp
              else torch.zeros(1, dtype=torch.int64, device=dev))
    q_idx = (torch.cat(Q_IDX[:bs]) if use_quant
             else torch.zeros(1, dtype=torch.int64, device=dev))
    total = hp_splits + quant_splits
    o = torch.empty(bs, H, D, dtype=torch.bfloat16, device=dev)
    logits = torch.empty(bs, H, total, D, dtype=torch.float32, device=dev)
    lse = torch.full((bs, H, total), float("-inf"), dtype=torch.float32, device=dev)
    decode_attention_fwd_int2_unified(
        Q_ALL[:bs], hp_k, hp_v, qk, qv, ksz, vsz, o,
        hp_indptr, hp_idx, q_indptr, q_idx, logits, lse,
        torch.full((bs,), hp_splits, dtype=torch.int32, device=dev),
        torch.full((bs,), quant_splits, dtype=torch.int32, device=dev),
        hp_splits, quant_splits, SM,
    )
    torch.cuda.synchronize()
    return o


def relerr(o, refs):
    return [((o[i].float() - refs[i]).norm() / refs[i].norm()).item()
            for i in range(o.shape[0])]


print("=== per-request rel err, both tiers, hp=%d quant=%d splits ==="
      % (args.hp_splits, args.quant_splits), flush=True)
REFS = [ref_one(i) for i in range(NREQ)]
for bs in (8, 12, 14, 15, 16, 17, 20, 32):
    e = relerr(run(bs, args.hp_splits, args.quant_splits), REFS)
    bad = [i for i, x in enumerate(e) if x > 5e-3]
    print(f"bs={bs:>3} max={max(e):.3e} mean={sum(e)/len(e):.3e} "
          f"n_bad={len(bad)} bad_reqs={bad[:12]}")

print()
print("=== tier ablation at bs=15 vs 16 ===", flush=True)
for use_hp, use_quant, name in ((True, False, "HP only"), (False, True, "quant only")):
    refs = [ref_one(i, use_hp, use_quant) for i in range(NREQ)]
    for bs in (15, 16):
        e = relerr(run(bs, args.hp_splits, args.quant_splits, use_hp, use_quant), refs)
        print(f"{name:<12} bs={bs:>3} max={max(e):.3e} "
              f"n_bad={sum(1 for x in e if x > 5e-3)}")

print()
print("=== split-count sweep (both tiers) ===", flush=True)
for hs, qs in ((8, 8), (8, 16), (16, 8), (4, 4), (1, 1), (8, 4), (4, 8)):
    row = []
    for bs in (8, 15, 16, 32):
        e = relerr(run(bs, hs, qs), REFS)
        row.append(f"bs{bs}={max(e):.2e}")
    print(f"hp_splits={hs:>3} quant_splits={qs:>3} total={hs+qs:>3}  " + "  ".join(row))
