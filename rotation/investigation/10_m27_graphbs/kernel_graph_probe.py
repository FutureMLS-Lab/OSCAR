#!/usr/bin/env python3
"""Offline eager-vs-CUDA-graph probe for the mixed-KV INT2 decode path.

Reproduces, without a model, the exact buffer discipline the CUDA-graph path
uses in ``TritonAttnBackend``:

  * persistent scratch  ``mixed_attn_logits`` / ``mixed_attn_lse`` sized for
    ``max_bs`` and shared by every captured graph size,
  * persistent per-tier indptr / indices buffers rebuilt eagerly before each
    replay,
  * persistent ``mixed_{hp,quant}_num_kv_splits``,
  * graphs captured in sglang's order (descending bs) into ONE shared
    ``torch.cuda.graph_pool_handle()``,
  * capture-time ``seq_lens`` = 1 (sglang's ``seq_len_fill_value``).

Then, for a fixed synthetic decode payload, it compares the graph replay output
against the eager path (fresh -inf-filled scratch, exactly-sized indices) for
every captured batch size.

Usage: python graph_probe.py [--heads 12 --kv-heads 2 --head-dim 128]
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
p.add_argument("--quant-splits", type=int, default=16)
p.add_argument("--hp-prefix", type=int, default=64)
p.add_argument("--hp-recent-ring", type=int, default=263)
p.add_argument("--maxctx", type=int, default=20000)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

sys.path.insert(0, os.path.join(args.tree, "sglang-research", "python"))
from sglang.srt.layers.attention.triton_ops.decode_attention import (  # noqa: E402
    decode_attention_fwd_int2_unified,
)
import sglang  # noqa: E402

print(f"[probe] sglang from {sglang.__file__}", flush=True)
print(f"[probe] torch {torch.__version__} cuda {torch.version.cuda}", flush=True)

dev = "cuda"
torch.manual_seed(args.seed)

H, KVH, D = args.heads, args.kv_heads, args.head_dim
HP_S, Q_S = args.hp_splits, args.quant_splits
TOTAL = HP_S + Q_S
CAPTURE_BS = [1, 2, 4, 8, 12, 16, 24, 32]
MAXBS = max(CAPTURE_BS)
MAXCTX = args.maxctx
HP_LEN = args.hp_prefix + args.hp_recent_ring          # per-request HP tier length
SM = D ** -0.5

HP_SLOTS = MAXBS * (HP_LEN + 8) + 16
Q_SLOTS = MAXBS * MAXCTX + 16

# ---------------------------------------------------------------- KV "pool"
hp_k = (torch.randn(HP_SLOTS, KVH, D, device=dev) * 0.5).to(torch.bfloat16)
hp_v = (torch.randn(HP_SLOTS, KVH, D, device=dev) * 0.5).to(torch.bfloat16)
qk = torch.randint(0, 256, (Q_SLOTS, KVH, D // 4), dtype=torch.uint8, device=dev)
qv = torch.randint(0, 256, (Q_SLOTS, KVH, D // 4), dtype=torch.uint8, device=dev)
ksz = torch.stack(
    [torch.rand(Q_SLOTS, KVH, device=dev) * 0.4 + 0.05,
     torch.rand(Q_SLOTS, KVH, device=dev) * 2.0], dim=-1
).contiguous()
vsz = torch.stack(
    [torch.rand(Q_SLOTS, KVH, device=dev) * 0.4 + 0.05,
     torch.rand(Q_SLOTS, KVH, device=dev) * 2.0], dim=-1
).contiguous()

# ------------------------------------------------------- persistent (graph)
g_q = torch.zeros(MAXBS, H, D, dtype=torch.bfloat16, device=dev)
g_o = torch.zeros(MAXBS, H, D, dtype=torch.bfloat16, device=dev)
g_logits = torch.zeros(MAXBS, H, TOTAL, D, dtype=torch.float32, device=dev)
g_lse = torch.full((MAXBS, H, TOTAL), float("-inf"), dtype=torch.float32, device=dev)
g_hp_indptr = torch.zeros(MAXBS + 1, dtype=torch.int32, device=dev)
g_q_indptr = torch.zeros(MAXBS + 1, dtype=torch.int32, device=dev)
g_hp_idx = torch.zeros(MAXBS * (HP_LEN + 8), dtype=torch.int64, device=dev)
g_q_idx = torch.zeros(MAXBS * MAXCTX, dtype=torch.int64, device=dev)
g_hp_splits = torch.full((MAXBS,), HP_S, dtype=torch.int32, device=dev)
g_q_splits = torch.zeros(MAXBS, dtype=torch.int32, device=dev)


def call(bs, q, o, logits, lse, hp_indptr, hp_idx, q_indptr, q_idx,
         hp_splits, q_splits):
    decode_attention_fwd_int2_unified(
        q, hp_k, hp_v, qk, qv, ksz, vsz, o,
        hp_indptr, hp_idx, q_indptr, q_idx,
        logits, lse, hp_splits, q_splits, HP_S, Q_S, SM,
    )


# ------------------------------------------------------------- num_kv_splits
def num_kv_splits(seq_lens, max_kv_splits, num_head, num_kv_head, core_count=132):
    """Faithful CPU port of get_num_kv_splits_triton (num_group == 1)."""
    import math
    n = len(seq_lens)
    max_seq = max(seq_lens)
    min_seq = min(seq_lens)
    if max_seq * 8 < min_seq * 10:
        min_seq = max_seq
    mks1 = min(-(-max_seq // min_seq), max_kv_splits)
    chunk1 = -(-max_seq // mks1)
    ext_seq = max_seq / 64.0
    ext_cores = int(core_count * max(math.log2(ext_seq), 1.0))
    block_h, kvg = 16, num_head // num_kv_head
    if kvg == 1:
        token_grid = n * num_head
    else:
        block_h = min(block_h, kvg)
        token_grid = n * (-(-num_head // block_h))
    mks2 = min(max(-(-ext_cores // token_grid), 1), max_kv_splits)
    chunk2 = -(-max_seq // mks2)
    return [max(-(-s // chunk1), -(-s // chunk2)) for s in seq_lens]


# ---------------------------------------------------------------- workloads
def make_payload(bs, gen):
    """Per-request (seq_len, hp slot ids, quant slot ids)."""
    seq = [int(gen.integers(4000, MAXCTX)) for _ in range(bs)]
    hp_ids, q_ids = [], []
    for i, s in enumerate(seq):
        base_hp = i * (HP_LEN + 8)
        hp_ids.append(torch.arange(base_hp, base_hp + HP_LEN, dtype=torch.int64))
        qlen = s - HP_LEN
        base_q = i * MAXCTX
        q_ids.append(torch.arange(base_q, base_q + qlen, dtype=torch.int64))
    return seq, hp_ids, q_ids


def load_into_graph_buffers(bs, seq, hp_ids, q_ids, q_vals):
    hp_cum, q_cum = [0], [0]
    for i in range(bs):
        hp_cum.append(hp_cum[-1] + len(hp_ids[i]))
        q_cum.append(q_cum[-1] + len(q_ids[i]))
    g_hp_indptr[: bs + 1] = torch.tensor(hp_cum, dtype=torch.int32, device=dev)
    g_q_indptr[: bs + 1] = torch.tensor(q_cum, dtype=torch.int32, device=dev)
    g_hp_idx[: hp_cum[-1]] = torch.cat(hp_ids).to(dev)
    g_q_idx[: q_cum[-1]] = torch.cat(q_ids).to(dev)
    g_q_splits[:bs] = torch.tensor(
        num_kv_splits(seq, Q_S, H, KVH), dtype=torch.int32, device=dev
    )
    g_hp_splits[:bs] = HP_S
    g_q[:bs] = q_vals
    return hp_cum, q_cum


def eager_run(bs, seq, hp_ids, q_ids, q_vals):
    """Exactly what init_forward_metadata does: fresh, exactly sized."""
    hp_cum, q_cum = [0], [0]
    for i in range(bs):
        hp_cum.append(hp_cum[-1] + len(hp_ids[i]))
        q_cum.append(q_cum[-1] + len(q_ids[i]))
    hp_indptr = torch.tensor(hp_cum, dtype=torch.int32, device=dev)
    q_indptr = torch.tensor(q_cum, dtype=torch.int32, device=dev)
    hp_idx = torch.cat(hp_ids).to(dev)
    q_idx = torch.cat(q_ids).to(dev)
    logits = torch.empty(bs, H, TOTAL, D, dtype=torch.float32, device=dev)
    lse = torch.full((bs, H, TOTAL), float("-inf"), dtype=torch.float32, device=dev)
    hp_sp = torch.full((bs,), HP_S, dtype=torch.int32, device=dev)
    q_sp = torch.tensor(num_kv_splits(seq, Q_S, H, KVH), dtype=torch.int32, device=dev)
    o = torch.empty(bs, H, D, dtype=torch.bfloat16, device=dev)
    call(bs, q_vals, o, logits, lse, hp_indptr, hp_idx, q_indptr, q_idx, hp_sp, q_sp)
    torch.cuda.synchronize()
    return o, q_sp.tolist()


# ------------------------------------------------------------------ capture
graphs = {}
pool = torch.cuda.graph_pool_handle()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())

# sglang captures with seq_lens == seq_len_fill_value (1) everywhere.
for bs in sorted(CAPTURE_BS, reverse=True):
    g_hp_indptr.zero_()
    g_q_indptr.zero_()
    g_hp_splits[:bs] = HP_S
    g_q_splits[:bs] = 1
    with torch.cuda.stream(s):
        for _ in range(2):
            call(bs, g_q[:bs], g_o[:bs], g_logits[:bs], g_lse[:bs],
                 g_hp_indptr[: bs + 1], g_hp_idx, g_q_indptr[: bs + 1], g_q_idx,
                 g_hp_splits[:bs], g_q_splits[:bs])
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=pool):
        call(bs, g_q[:bs], g_o[:bs], g_logits[:bs], g_lse[:bs],
             g_hp_indptr[: bs + 1], g_hp_idx, g_q_indptr[: bs + 1], g_q_idx,
             g_hp_splits[:bs], g_q_splits[:bs])
    graphs[bs] = g
    print(f"[probe] captured bs={bs}", flush=True)

torch.cuda.synchronize()

# ------------------------------------------------------------------- compare
import numpy as np  # noqa: E402

gen = np.random.default_rng(args.seed)
print()
print(f"{'bs':>4} {'quant splits (min..max)':>24} {'max|graph-eager|':>18} "
      f"{'rel_l2':>10}  verdict")
print("-" * 78)

worst = 0.0
for bs in CAPTURE_BS:
    seq, hp_ids, q_ids = make_payload(bs, gen)
    q_vals = (torch.randn(bs, H, D, device=dev) * 0.6).to(torch.bfloat16)

    o_eager, qsp = eager_run(bs, seq, hp_ids, q_ids, q_vals)

    # A smaller graph first, to leave stale state in the shared scratch --
    # this is what actually happens when a request finishes and the batch
    # shrinks, then grows again.
    small = 4 if bs != 4 else 2
    ss, sh, sq = make_payload(small, gen)
    load_into_graph_buffers(small, ss, sh, sq,
                            (torch.randn(small, H, D, device=dev) * 0.6).to(torch.bfloat16))
    graphs[small].replay()
    torch.cuda.synchronize()

    load_into_graph_buffers(bs, seq, hp_ids, q_ids, q_vals)
    graphs[bs].replay()
    torch.cuda.synchronize()
    o_graph = g_o[:bs].clone()

    d = (o_graph.float() - o_eager.float()).abs()
    md = d.max().item()
    rel = (d.norm() / o_eager.float().norm().clamp_min(1e-9)).item()
    worst = max(worst, rel)
    print(f"{bs:>4} {min(qsp):>10}..{max(qsp):<11} {md:>18.6g} {rel:>10.3e}  "
          f"{'OK' if rel < 2e-2 else '*** MISMATCH ***'}")

print()
print(f"[probe] worst rel_l2 across captured sizes: {worst:.3e}")
print("[probe] NOTE: bf16 output => rel_l2 ~1e-3 is numerical noise from a "
      "different split schedule; >1e-2 means the graph path is wrong.")
