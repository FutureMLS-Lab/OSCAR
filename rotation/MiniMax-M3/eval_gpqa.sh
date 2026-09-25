#!/usr/bin/env bash
# GPQA eval wrapper for MiniMaxAI/MiniMax-M3, CUDA graph on (the configuration all published
# OSCAR GPQA numbers were measured under). Override CUDA_GRAPH_MAX_BS to change it.
#
# WARNING about CUDA_GRAPH_MAX_BS=8 below. `CudaGraphRunner.can_run` tests
# `cuda_graph_bs <= self.max_bs`, so at max_bs=8 any decode batch above 8
# cannot replay a graph at all and silently runs eager -- "graph on" in the
# flags is not the same as graphs actually replaying. This is the same shape of
# mistake as MiniMax-M2.7's `--cuda-graph-max-bs 1`, which meant graphs were off
# above batch 1 for months without anyone noticing. If you serve M3 above 8
# concurrent requests, raise this (32 gives the captured set
# [1,2,4,8,12,16,24,32]) and verify from server.log that decode batches actually
# print `cuda graph: True` -- do not infer it from the flag.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-MiniMaxAI/MiniMax-M3}"
export ROT_DIR="${ROT_DIR:-${SCRIPT_DIR}/rotations}"
export RUN_DIR="${RUN_DIR:-$(dirname "${ROT_DIR}")/_eval_gpqa_oscar}"
export TP_SIZE="${TP_SIZE:-16}"
export GROUP_SIZE="${GROUP_SIZE:-128}"
export K_CLIP="${K_CLIP:-0.96}"
export V_CLIP="${V_CLIP:-0.92}"
export CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-8}"
export NAME="${NAME:-gpqa_oscar_minimax_m3}"
# 428B across 2 nodes; set NNODES/NODE_RANK/DIST_ADDR per node.
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
# M3's text_config has no torch_dtype, so sglang falls back to fp16 unless the
# dtype is stated explicitly, and the shared launcher never passes --dtype.
# Every published M3 number is bf16; fp16 is not a crash, it just silently runs
# a different numeric format on a 428B MoE.
export EXTRA_SERVER_ARGS="--dtype bfloat16 ${EXTRA_SERVER_ARGS:-}"
exec bash "${SCRIPT_DIR}/../eval_oscar_gpqa.sh"
