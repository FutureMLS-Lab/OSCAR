#!/usr/bin/env python3
"""Diff every derived quantity the mixed-KV decode path builds, eager vs replay.

Drives the REAL ``TritonAttnBackend`` (stubbed ModelRunner) through

    init_forward_metadata(forward_batch)                       # eager
    init_forward_metadata_capture_cuda_graph(bs, ...)          # capture
    init_forward_metadata_replay_cuda_graph(bs, ...)           # replay

for the same synthetic decode state at each captured batch size, and compares
the tensors the decode kernels actually consume:

    mixed_hp_kv_indptr / mixed_hp_kv_indices
    mixed_quant_kv_indptr / mixed_quant_kv_indices
    mixed_hp_num_kv_splits / mixed_quant_num_kv_splits
    mixed_attn_lse (the -inf prefill state entering stage-2)

This is the cheap version of "compare eager and replay for the same batch size
and diff the derived quantities": no model, no weights, one GPU, seconds.
"""
import os
import sys
import types

import torch

TREE = os.environ.get("TREE", "/tmp/tree")
sys.path.insert(0, os.path.join(TREE, "sglang-research", "python"))

import sglang.srt.layers.attention.triton_backend as tb  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402

# The backend asks the distributed layer for the attention TP size; this probe
# runs single-process, so answer 1 without initialising a process group.
tb.get_attention_tp_size = lambda: 1

DEV = "cuda"
CAPTURE_BS = [1, 2, 4, 8, 12, 16, 24, 32]
MAX_BS = 32
MAX_CTX = 40960
Q_HEADS, KV_HEADS, HEAD_DIM = 12, 2, 128     # MiniMax-M2.7 at TP=4
HP_PREFIX, HP_RECENT, N_Q = 64, 256, 8
RING = HP_RECENT + N_Q - 1                   # 263
HP_PREFIX_SLOTS = 32776
NUM_QUANT_SLOTS = 1291320
HP_OFFSET = NUM_QUANT_SLOTS


class StubPool:
    dtype = "int2"
    v_head_dim = HEAD_DIM
    hp_prefix_tokens = HP_PREFIX
    hp_recent_tokens = HP_RECENT
    hp_global_offset = HP_OFFSET

    def mixed_kv_enabled(self):
        return True

    def get_value_buffer(self, i):
        return torch.zeros(1, KV_HEADS, HEAD_DIM, device=DEV)


def make_runner():
    req_to_token = torch.zeros(MAX_BS, MAX_CTX, dtype=torch.int32, device=DEV)
    sa = types.SimpleNamespace(
        triton_attention_num_kv_splits=8,
        triton_attention_split_tile_size=None,
        speculative_num_draft_tokens=None,
        speculative_num_steps=None,
        disable_cuda_graph=False,
        chunked_prefill_size=8192,
        enable_deterministic_inference=False,
    )
    mc = types.SimpleNamespace(
        context_len=MAX_CTX,
        attention_arch=None,
        num_attention_heads=Q_HEADS,
        v_head_dim=HEAD_DIM,
        swa_v_head_dim=HEAD_DIM,
        is_encoder_decoder=False,
        get_num_kv_heads=lambda tp: KV_HEADS,
    )
    return types.SimpleNamespace(
        req_to_token_pool=types.SimpleNamespace(size=MAX_BS, req_to_token=req_to_token),
        token_to_kv_pool=StubPool(),
        token_to_kv_pool_allocator=None,
        server_args=sa,
        model_config=mc,
        sliding_window_size=None,
        hybrid_gdn_config=None,
        kimi_linear_config=None,
        linear_attn_model_spec=None,
        dtype=torch.bfloat16,
        device=DEV,
        gpu_id=0,
    )


runner = make_runner()
# ``AttentionArch.MLA`` comparison must be False; the stub sets attention_arch
# to None which is never equal to the enum member.
backend = tb.TritonAttnBackend(runner)
print(f"[meta] max_kv_splits={backend.max_kv_splits} "
      f"max_hp_kv_splits={backend.max_hp_kv_splits} "
      f"enable_mixed_kv={backend.enable_mixed_kv} "
      f"num_head={backend.num_head} num_kv_head={backend.num_kv_head} "
      f"v_head_dim={backend.v_head_dim}", flush=True)

backend.init_cuda_graph_state(MAX_BS, MAX_BS)


def build_state(bs, gen):
    """Lay out a plausible mid-decode req_to_token for ``bs`` requests."""
    seq_lens = torch.tensor(
        [int(gen.integers(4000, 33000)) for _ in range(bs)],
        dtype=torch.int64, device=DEV,
    )
    req_pool_indices = torch.arange(bs, dtype=torch.int64, device=DEV)
    rtt = runner.req_to_token_pool.req_to_token
    rtt.zero_()
    for i in range(bs):
        s = int(seq_lens[i])
        row = torch.empty(s, dtype=torch.int32, device=DEV)
        # [0, HP_PREFIX)              -> shared HP-prefix pool
        row[:HP_PREFIX] = torch.arange(
            HP_OFFSET + i * HP_PREFIX, HP_OFFSET + (i + 1) * HP_PREFIX,
            dtype=torch.int32, device=DEV,
        )
        # [HP_PREFIX, s - RING)       -> packed int2 bulk
        bulk = s - HP_PREFIX - RING
        row[HP_PREFIX:HP_PREFIX + bulk] = torch.arange(
            i * 40000, i * 40000 + bulk, dtype=torch.int32, device=DEV
        )
        # tail                        -> this request's HP-recent ring slab
        ring_base = HP_OFFSET + HP_PREFIX_SLOTS + i * RING
        row[HP_PREFIX + bulk:] = torch.arange(
            ring_base, ring_base + RING, dtype=torch.int32, device=DEV
        )
        rtt[i, :s] = row
    return req_pool_indices, seq_lens


def snapshot(md, bs):
    keys = [
        "mixed_hp_kv_indptr", "mixed_quant_kv_indptr",
        "mixed_hp_num_kv_splits", "mixed_quant_num_kv_splits",
    ]
    out = {}
    for k in keys:
        t = getattr(md, k)
        n = bs + 1 if k.endswith("indptr") else bs
        out[k] = t[:n].clone()
    hp_n = int(out["mixed_hp_kv_indptr"][bs])
    q_n = int(out["mixed_quant_kv_indptr"][bs])
    out["mixed_hp_kv_indices"] = md.mixed_hp_kv_indices[:hp_n].clone()
    out["mixed_quant_kv_indices"] = md.mixed_quant_kv_indices[:q_n].clone()
    out["mixed_attn_lse"] = md.mixed_attn_lse[:bs].clone()
    out["_logits_shape"] = tuple(md.mixed_attn_logits.shape)
    out["_lse_shape"] = tuple(md.mixed_attn_lse.shape)
    return out


import numpy as np  # noqa: E402

gen = np.random.default_rng(0)

# Capture all sizes first, exactly as sglang does (descending).
for bs in sorted(CAPTURE_BS, reverse=True):
    rpi, sl = build_state(bs, gen)
    backend.init_forward_metadata_capture_cuda_graph(
        bs, bs, rpi, sl, None, ForwardMode.DECODE, None
    )
print("[meta] captured all sizes", flush=True)

print()
print(f"{'bs':>4}  {'quant splits':>14}  {'hp splits':>10}  {'differing tensors'}")
print("-" * 78)

bad = 0
for bs in CAPTURE_BS:
    rpi, sl = build_state(bs, gen)

    # --- eager ---------------------------------------------------------
    fb = types.SimpleNamespace(
        batch_size=bs,
        forward_mode=ForwardMode.DECODE,
        spec_info=None,
        seq_lens=sl,
        seq_lens_sum=int(sl.sum()),
        req_pool_indices=rpi,
    )
    backend.init_forward_metadata(fb)
    eager = snapshot(backend.forward_metadata, bs)

    # --- replay --------------------------------------------------------
    backend.init_forward_metadata_replay_cuda_graph(
        bs, rpi, sl, int(sl.sum()), None, ForwardMode.DECODE, None, seq_lens_cpu=None
    )
    # replay path does not rebuild ForwardMetadata; read the graph buffers
    md = types.SimpleNamespace(
        mixed_hp_kv_indptr=backend.cuda_graph_mixed_hp_kv_indptr,
        mixed_quant_kv_indptr=backend.cuda_graph_mixed_quant_kv_indptr,
        mixed_hp_kv_indices=backend.cuda_graph_mixed_hp_kv_indices,
        mixed_quant_kv_indices=backend.cuda_graph_mixed_quant_kv_indices,
        mixed_hp_num_kv_splits=backend.cuda_graph_mixed_hp_num_kv_splits,
        mixed_quant_num_kv_splits=backend.cuda_graph_mixed_quant_num_kv_splits,
        mixed_attn_lse=backend.cuda_graph_mixed_attn_lse,
        mixed_attn_logits=backend.cuda_graph_mixed_attn_logits,
    )
    replay = snapshot(md, bs)

    diffs = []
    for k in eager:
        if k.startswith("_"):
            if eager[k][1:] != replay[k][1:]:
                diffs.append(f"{k}{eager[k]}vs{replay[k]}")
            continue
        a, b = eager[k], replay[k]
        if a.shape != b.shape:
            diffs.append(f"{k}(shape {tuple(a.shape)} vs {tuple(b.shape)})")
        elif not torch.equal(a, b):
            n = int((a != b).sum())
            diffs.append(f"{k}({n} elems)")
    if diffs:
        bad += 1
    qs = replay["mixed_quant_num_kv_splits"]
    hs = replay["mixed_hp_num_kv_splits"]
    print(f"{bs:>4}  {int(qs.min()):>6}..{int(qs.max()):<6}  "
          f"{int(hs.min()):>4}..{int(hs.max()):<4}  "
          f"{', '.join(diffs) if diffs else 'identical'}")

print()
print(f"[meta] batch sizes with any eager/replay metadata difference: {bad}/{len(CAPTURE_BS)}")
