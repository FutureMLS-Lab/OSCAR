#!/usr/bin/env bash
# GPQA eval wrapper for Qwen/Qwen3-32B (base hybrid).
# Per-model defaults validated for OSCAR Qwen3-32B INT2 KV-cache.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-Qwen/Qwen3-32B}"
export ROT_DIR="${ROT_DIR:-${SCRIPT_DIR}/rotations}"
export RUN_DIR="${RUN_DIR:-$(dirname "${ROT_DIR}")/_eval_gpqa_oscar}"
export TP_SIZE="${TP_SIZE:-4}"
export GROUP_SIZE="${GROUP_SIZE:-128}"
export K_CLIP="${K_CLIP:-0.96}"
export V_CLIP="${V_CLIP:-0.96}"                 # V_CLIP=0.96 (not 0.92) — model-specific tuning
export NAME="${NAME:-gpqa_oscar_qwen3_32b}"

exec bash "${SCRIPT_DIR}/../eval_oscar_gpqa.sh"
