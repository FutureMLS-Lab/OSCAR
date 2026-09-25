#!/usr/bin/env bash
# GPQA eval wrapper for zai-org/GLM-5.2-FP8, CUDA graph on (the configuration all published
# OSCAR GPQA numbers were measured under). Override CUDA_GRAPH_MAX_BS to change it.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-zai-org/GLM-5.2-FP8}"
# MLA: the rotation is per-layer latent files (layer_*.pt), not a k/v pair, and
# is passed through MLA_ROT_PATH -- ROT_DIR/K_ROT_FILENAME do not apply here.
export MLA_ROT_PATH="${MLA_ROT_PATH:-${SCRIPT_DIR}/GPQA/_rcov_phblock/rotations}"
export ROT_DIR="${ROT_DIR:-${MLA_ROT_PATH}}"
export RUN_DIR="${RUN_DIR:-$(dirname "${ROT_DIR}")/_eval_gpqa_oscar}"
export TP_SIZE="${TP_SIZE:-16}"
export GROUP_SIZE="${GROUP_SIZE:-128}"
export K_CLIP="${K_CLIP:-0.96}"
export V_CLIP="${V_CLIP:-0.92}"
export CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-32}"
export LLOYD_MAX="${LLOYD_MAX:-1}"   # GLM-5.2 recipe uses Lloyd-Max
# 64/512 is GLM-5.2's best measured window and the only one statistically
# indistinguishable from BF16 (80.30, McNemar p=0.48). The shared launcher
# defaults to 256, which reproduces the weaker archived arm (76.77, p=0.019).
# The window is NON-MONOTONE -- 1024 falls back to 75.76 -- so do not assume
# bigger is better if you override this.
export SGLANG_MIXED_KV_RECENT_TOKENS="${SGLANG_MIXED_KV_RECENT_TOKENS:-512}"
export NAME="${NAME:-gpqa_oscar_glm_5_2}"
# MLA latent path: rotations are per-layer c_kv files, not k/v pairs.
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
exec bash "${SCRIPT_DIR}/../eval_oscar_gpqa.sh"
