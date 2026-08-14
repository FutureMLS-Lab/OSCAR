#!/usr/bin/env bash
# GPQA eval wrapper for MiniMaxAI/MiniMax-M2.7, CUDA graph on (the configuration all published
# OSCAR GPQA numbers were measured under). Override CUDA_GRAPH_MAX_BS to change it.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-MiniMaxAI/MiniMax-M2.7}"
export ROT_DIR="${ROT_DIR:-${SCRIPT_DIR}/rotations}"
export RUN_DIR="${RUN_DIR:-$(dirname "${ROT_DIR}")/_eval_gpqa_oscar}"
export TP_SIZE="${TP_SIZE:-4}"
export GROUP_SIZE="${GROUP_SIZE:-128}"
export K_CLIP="${K_CLIP:-0.96}"
export V_CLIP="${V_CLIP:-0.92}"
export CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-1}"
export NAME="${NAME:-gpqa_oscar_minimax_m27}"
# M2.7 is a long-thinking model: budget generation, and with a single captured
# graph shape the eval client must be single-threaded (115 tok/s vs 8 at two).
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-95000}"
export NUM_WORKERS="${NUM_WORKERS:-1}"
exec bash "${SCRIPT_DIR}/../eval_oscar_gpqa.sh"
