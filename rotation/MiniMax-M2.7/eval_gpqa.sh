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
export CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-32}"
export NAME="${NAME:-gpqa_oscar_minimax_m27}"
# WARNING: CUDA_GRAPH_MAX_BS=32 above is currently UNSAFE for INT2 on this
# model. Measured control, same session, same rotation, same 64/256 windows,
# identical client concurrency 15, varying ONLY --cuda-graph-max-bs, matched
# question-by-question over the same ~69 GPQA items, 3 seeds:
#
#   max-bs 32 (100% padded graph replay)   38 / 38 / 40 correct
#   max-bs 1  (100% eager, 0 padded)       56 / 57 / 56 correct
#   2x2: eager gains 19-20 and loses 1-3   -> McNemar p < 0.001, 3/3 seeds
#   capped 26/20/22 -> 6/5/6 ; generation length drops to 0.63-0.73x
#
# Eager decode recovers INT2 to roughly BF16 parity on the matched subset, so
# the large INT2 deficit measured at max-bs 32 is predominantly a defect on the
# CUDA-graph replay path, NOT 2-bit quality. This also reconciles the good
# historical numbers -- SWE-bench 70.8, LCB v6 68.4, AIME25 90.0, GPQA 80.3 all
# ran max-bs 1 and single-stream, i.e. never on the broken path.
#
# It is NOT the known page-0 padding bug; that is fixed here (page 0 reserved,
# hp_prefix_page0_ever_allocated false, min allocated page 1) and the pre-fix
# twin still shows page 0. Something else on the replay path is wrong.
#
# Until that is root-caused, serve INT2 with CUDA_GRAPH_MAX_BS=1, and do not
# read any max-bs 32 INT2 number as a quantization result.
#
# Windows stay at the validated 64/256. Raising them to 256/1024 measured
# +7.75 pp (57.91 -> 65.66, 3 seeds) -- but that was measured entirely at
# max-bs 32, so it is most likely masking the replay defect rather than fixing
# a quality problem, and it costs 2.79 -> 3.65 bits/element amortized. Do not
# adopt it as a recipe on that evidence.
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
