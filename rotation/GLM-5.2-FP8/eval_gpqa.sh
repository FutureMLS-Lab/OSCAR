#!/usr/bin/env bash
# GPQA eval wrapper for zai-org/GLM-5.2-FP8, CUDA graph on (the configuration all published
# OSCAR GPQA numbers were measured under). Override CUDA_GRAPH_MAX_BS to change it.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-zai-org/GLM-5.2-FP8}"
export ROT_DIR="${ROT_DIR:-${SCRIPT_DIR}/rotations}"
export RUN_DIR="${RUN_DIR:-$(dirname "${ROT_DIR}")/_eval_gpqa_oscar}"
export TP_SIZE="${TP_SIZE:-16}"
export GROUP_SIZE="${GROUP_SIZE:-128}"
export K_CLIP="${K_CLIP:-0.96}"
export V_CLIP="${V_CLIP:-0.92}"
export CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-32}"
export NAME="${NAME:-gpqa_oscar_glm_5_2}"
# MLA latent path: rotations are per-layer c_kv files, not k/v pairs.
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
exec bash "${SCRIPT_DIR}/../eval_oscar_gpqa.sh"
