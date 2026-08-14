#!/usr/bin/env bash
# GPQA eval wrapper for GLM-4.7-FP8 (92-layer MoE).
# Per-model defaults validated for OSCAR GLM-4.7-FP8 INT2 KV-cache.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-zai-org/GLM-4.7-FP8}"
export ROT_DIR="${ROT_DIR:-${SCRIPT_DIR}/rotations}"
export RUN_DIR="${RUN_DIR:-$(dirname "${ROT_DIR}")/_eval_gpqa_oscar}"
export TP_SIZE="${TP_SIZE:-8}"
export GROUP_SIZE="${GROUP_SIZE:-128}"
export K_CLIP="${K_CLIP:-0.96}"
export V_CLIP="${V_CLIP:-0.92}"
export NAME="${NAME:-gpqa_oscar_glm_4_7}"

exec bash "${SCRIPT_DIR}/../eval_oscar_gpqa.sh"
