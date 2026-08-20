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
