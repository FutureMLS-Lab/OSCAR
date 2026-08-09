#!/usr/bin/env bash
# Rotations for Qwen3.5-35B-A3B. Hybrid linear-attn: 10 full-attention layers,
# head_dim 256. Everything else is the shared driver.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HEAD_DIM="${HEAD_DIM:-256}" \
NUM_LAYERS="${NUM_LAYERS:-10}" \
LAYER_IDS="${LAYER_IDS:-3,7,11,15,19,23,27,31,35,39}" \
CALIB_ROOT="${CALIB_ROOT:-${SCRIPT_DIR}}" \
  exec bash "${SCRIPT_DIR}/../qwen3-8B/compute_rotation.sh" "$@"
