#!/usr/bin/env bash
# Shared per-layer rotations (V1) for Qwen3-30B-A3B.
#
# For comparison only. On this model a shared rotation costs +16.6 % PPL and
# degenerates long generations regardless of how well it is calibrated -- see
# README.md. Use fit_perhead.sh to serve.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${DATASET:-GPQA}"
if [[ -z "${CALIB_DIR:-}" ]]; then
  # earlier calibrations were written under the lower-case sibling dir;
  # look there too so an existing dump is not silently ignored
  CALIB_DIR="$(ls -1dt "${SCRIPT_DIR}/${DATASET}"/*/ "${SCRIPT_DIR}/../qwen3-30b-a3b/${DATASET}"/*/ 2>/dev/null | head -1 | sed 's:/$::')"
fi
DUMP_PATH="${DUMP_PATH:-${CALIB_DIR}/qkv_dumps/gpqa}" \
OUTPUT_DIR="${OUTPUT_DIR:-${CALIB_DIR}/rotations}" \
HEAD_DIM=128 NUM_LAYERS="${NUM_LAYERS:-48}" METHOD="${METHOD:-qqt_sst}" \
COMPOSITION="${COMPOSITION:-r_h_pbr}" \
  bash "${SCRIPT_DIR}/../qwen3-8B/compute_rotation.sh"
