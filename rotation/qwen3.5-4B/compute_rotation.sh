#!/usr/bin/env bash
# Rotations for Qwen3.5-4B. Hybrid linear-attn: only 8 of 36 layers hold a KV
# cache, and head_dim is 256. Everything else is the shared driver.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HEAD_DIM="${HEAD_DIM:-256}" \
NUM_LAYERS="${NUM_LAYERS:-8}" \
LAYER_IDS="${LAYER_IDS:-3,7,11,15,19,23,27,31}" \
CALIB_ROOT="${CALIB_ROOT:-${SCRIPT_DIR}}" \
  exec bash "${SCRIPT_DIR}/../qwen3-8B/compute_rotation.sh" "$@"
