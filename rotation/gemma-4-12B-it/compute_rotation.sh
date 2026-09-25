#!/usr/bin/env bash
# Phase 2 — fit OSCAR rotations for gemma-4-12B-it from the Phase-1 dump.
#
# compute_kv_rotation.py infers head_dim PER LAYER from the dumped K tensor
# (sliding=256, full=512) and builds a matching Hadamard, so each layer gets a
# correctly-sized rotation. Produces:
#   rotations/k_rotation_qqt_r_h_pbr.pt   (K, from Q-covariance qqt)
#   rotations/v_rotation_sst_r_h_pbr.pt   (V, from score-weighted V-cov sst)
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PY="${PY:?Set PY (any python with torch); CPU eigendecomposition}"

# Pick the calibration dir: explicit CALIB_DIR, else newest seq*_prompt*_group*, else latest
if [[ -n "${CALIB_DIR:-}" ]]; then
    DUMP_PATH="${CALIB_DIR}/qkv_dumps/gpqa"
else
    CALIB_DIR=$(ls -1d "${SCRIPT_DIR}"/GPQA/seq*_prompt*_group* 2>/dev/null | tail -1 || true)
    [[ -z "${CALIB_DIR}" ]] && CALIB_DIR="${SCRIPT_DIR}/GPQA/latest"
    DUMP_PATH="${CALIB_DIR}/qkv_dumps/gpqa"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${CALIB_DIR}/rotations}"
mkdir -p "${OUTPUT_DIR}"

echo "[compute_rotation] dump=${DUMP_PATH} out=${OUTPUT_DIR}"
"${PY}" "${REPO_ROOT}/rotation/compute_kv_rotation.py" \
    --dump-path "${DUMP_PATH}" \
    --method qqt_sst --composition r_h_pbr \
    --chunk-id all \
    --output-dir "${OUTPUT_DIR}"
echo "[compute_rotation] done:"; ls -la "${OUTPUT_DIR}"
