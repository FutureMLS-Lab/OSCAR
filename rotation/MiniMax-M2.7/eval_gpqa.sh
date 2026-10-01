#!/usr/bin/env bash
# GPQA eval wrapper for MiniMaxAI/MiniMax-M2.7.
#
# This used to pin CUDA_GRAPH_MAX_BS=1 and NUM_WORKERS=1. That was adopted in
# May 2026 as a way to "effectively disable CUDA graphs for bs>1" after graphs
# were blamed for garbled INT2 output -- a misdiagnosis. The real cause was the
# CUDA-graph padding bug (padded replays wrote a dummy token's K/V into
# HP-prefix page 0, which the allocator handed to a live request), now fixed by
# reserving that page.
#
# max-bs 1 is not "graphs on at batch 1", it is graphs *off* above batch 1:
# CudaGraphRunner.can_run gates on cuda_graph_bs <= max_bs, so a batch of 2+
# cannot replay and runs eager. Measured on this model: max-bs 1 at concurrency
# 4 gives `cuda graph: False` on every decode step and 0 padded replays. That is
# what the old "115 tok/s vs 8 at two workers" cliff actually was -- eager
# fallback, not graph contention -- and it means M2.7 was never evaluated with
# graphs active at any batch size. max-bs 32 captures
# [1,2,4,8,12,16,24,32] and serves concurrent requests normally.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-MiniMaxAI/MiniMax-M2.7}"
export ROT_DIR="${ROT_DIR:-${SCRIPT_DIR}/rotations}"
export RUN_DIR="${RUN_DIR:-$(dirname "${ROT_DIR}")/_eval_gpqa_oscar}"
export TP_SIZE="${TP_SIZE:-4}"
export GROUP_SIZE="${GROUP_SIZE:-128}"
export K_CLIP="${K_CLIP:-0.96}"
export V_CLIP="${V_CLIP:-0.92}"
# The bs >= 16 defect this used to guard against was never about CUDA graphs.
# It was `_decode_grouped_att_m_fwd_quant_int2` picking BLOCK_H = 4 at batch
# >= 16, which cannot tile this model's kv_group_num = 6 (48 q heads / 8 KV
# heads): q heads 6 and 7 then read KV head 0's cache for the whole INT2 tier.
# Fixed in decode_attention.py::_safe_block_h. Graphs looked responsible only
# because at client concurrency 15 the decode batch pads to 16, so turning
# graphs off (which never pads) appeared to cure it.
#
# Note what capping at 12 actually did: `CudaGraphRunner.can_run` gates on
# cuda_graph_bs <= max_bs, so it stops batches above 12 from *replaying*, not
# from *existing*. At concurrency 16 the decode batch is still 16, now running
# eagerly through the same kernel with the same BLOCK_H = 4 -- equally wrong
# and ~5x slower (143 vs 700-800 tok/s measured here). It only ever helped
# because every arm that used it also ran at concurrency <= 15.
export CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-32}"
export NAME="${NAME:-gpqa_oscar_minimax_m27}"
# At matched concurrency INT2 KV is at PARITY with BF16 on GPQA here -- +6/+4/-4
# correct over ~97 matched questions, 3 seeds, both arms at concurrency 7. The
# -21.05 pp that the M2.7 re-check reported was two things compounding, neither
# of them quantization: the bs>=16 graph defect above, and the fact that every
# INT2 arm ran concurrency 15 while every BF16 arm ran concurrency 7.
#
# So: never compare arms at different concurrency on this model. That asymmetry
# is what produced the fictitious published +2.0, and it produced the -21.05
# just as easily in the other direction.
#
# Windows stay at the validated 64/256. Raising them to 256/1024 measured
# +7.75 pp -- but entirely at concurrency 15, inside the defect, where a bigger
# BF16 window simply exposes less KV to the corruption. It is masking, not
# fixing, and it costs 2.79 -> 3.65 bits/element. Do not adopt it.
export SGLANG_MIXED_KV_PREFIX_TOKENS="${SGLANG_MIXED_KV_PREFIX_TOKENS:-64}"
export SGLANG_MIXED_KV_RECENT_TOKENS="${SGLANG_MIXED_KV_RECENT_TOKENS:-256}"
# M2.7 is a long-thinking model, so generation still needs a budget. Note its
# BF16 KV pool is 6.4x smaller than INT2's (201,769 vs 1,291,320 tokens
# measured at the same mem-fraction), so a BF16 arm at a 95K budget holds only
# ~2 requests and retracts continuously -- do not read its truncation tax as
# an INT2 gain (that is where the implausible published AIME +13.3 came from).
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-95000}"
# Concurrency is no longer forced to 1: that was the other half of the same
# workaround. Keep it >=4 so decode actually reaches batch sizes that exercise
# the graph path.
export NUM_WORKERS="${NUM_WORKERS:-8}"
exec bash "${SCRIPT_DIR}/../eval_oscar_gpqa.sh"
