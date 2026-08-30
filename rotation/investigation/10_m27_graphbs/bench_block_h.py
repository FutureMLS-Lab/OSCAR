#!/usr/bin/env python3
"""Cost of the BLOCK_H clamp on the grouped INT2 decode kernel.

The batch-size tile heuristic picked BLOCK_H=4 at batch >= 16 for register
pressure; the correctness clamp raises it back to 8 whenever it would straddle
a KV head. This measures what that costs on the M2.7 geometry, by driving the
same kernel through the ``SGL_INT2_BLOCK_H`` override (the clamp applies to the
override too, so BLOCK_H=4 is only reachable on an unpatched tree -- run this
before and after the fix to compare).
"""
import argparse
import os
import sys
import time

import torch

p = argparse.ArgumentParser()
p.add_argument("--tree", default=os.environ.get("TREE", "/tmp/tree"))
p.add_argument("--heads", type=int, default=12)
p.add_argument("--kv-heads", type=int, default=2)
p.add_argument("--head-dim", type=int, default=128)
p.add_argument("--seq", type=int, default=16384)
p.add_argument("--iters", type=int, default=50)
args = p.parse_args()

sys.path.insert(0, os.path.join(args.tree, "sglang-research", "python"))
from sglang.srt.layers.attention.triton_ops.decode_attention import (  # noqa: E402
    decode_attention_fwd_int2_unified,
)

dev = "cuda"
H, KVH, D = args.heads, args.kv_heads, args.head_dim
HP_LEN, HP_S, Q_S = 327, 8, 8
TOTAL = HP_S + Q_S
SM = D ** -0.5
QLEN = args.seq - HP_LEN

for bs in (8, 16, 24, 32):
    hp_k = (torch.randn(bs * HP_LEN, KVH, D, device=dev) * 0.5).to(torch.bfloat16)
    hp_v = (torch.randn(bs * HP_LEN, KVH, D, device=dev) * 0.5).to(torch.bfloat16)
    qk = torch.randint(0, 256, (bs * QLEN, KVH, D // 4), dtype=torch.uint8, device=dev)
    qv = torch.randint(0, 256, (bs * QLEN, KVH, D // 4), dtype=torch.uint8, device=dev)
    ksz = torch.rand(bs * QLEN, KVH, 2, device=dev)
    vsz = torch.rand(bs * QLEN, KVH, 2, device=dev)
    q = (torch.randn(bs, H, D, device=dev) * 0.6).to(torch.bfloat16)
    o = torch.empty(bs, H, D, dtype=torch.bfloat16, device=dev)
    logits = torch.empty(bs, H, TOTAL, D, dtype=torch.float32, device=dev)
    lse = torch.full((bs, H, TOTAL), float("-inf"), dtype=torch.float32, device=dev)
    hp_indptr = torch.arange(0, (bs + 1) * HP_LEN, HP_LEN, dtype=torch.int32, device=dev)
    q_indptr = torch.arange(0, (bs + 1) * QLEN, QLEN, dtype=torch.int32, device=dev)
    hp_idx = torch.arange(bs * HP_LEN, dtype=torch.int64, device=dev)
    q_idx = torch.arange(bs * QLEN, dtype=torch.int64, device=dev)
    hp_sp = torch.full((bs,), HP_S, dtype=torch.int32, device=dev)
    q_sp = torch.full((bs,), Q_S, dtype=torch.int32, device=dev)

    def once():
        decode_attention_fwd_int2_unified(
            q, hp_k, hp_v, qk, qv, ksz, vsz, o, hp_indptr, hp_idx,
            q_indptr, q_idx, logits, lse, hp_sp, q_sp, HP_S, Q_S, SM,
        )

    for _ in range(10):
        once()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        once()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / args.iters * 1e3
    print(f"bs={bs:>3} seq={args.seq}  BLOCK_H_env={os.environ.get('SGL_INT2_BLOCK_H','auto')}"
          f"  {ms:.3f} ms/call", flush=True)
