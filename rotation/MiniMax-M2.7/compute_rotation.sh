#!/usr/bin/env bash
# Fit the shared per-layer K/V rotation for MiniMaxAI/MiniMax-M2.7.
#
#   METHOD=qqt_sst (default)  calibrated from a Q/K/V dump (DUMP_PATH)
#   METHOD=hadamard           data-free, no dump needed
#
# Outputs into OUTPUT_DIR:
#   qqt_sst  -> k_rotation_qqt_r_h_pbr.pt, v_rotation_sst_r_h_pbr.pt
#   hadamard -> k_rotation_hadamard.pt,    v_rotation_hadamard.pt
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMPUTE_SCRIPT="${SCRIPT_DIR}/../compute_kv_rotation.py"

METHOD="${METHOD:-qqt_sst}"
COMPOSITION="${COMPOSITION:-r_h_pbr}"
HEAD_DIM="${HEAD_DIM:-128}"
NUM_LAYERS="${NUM_LAYERS:-62}"   # MiniMax-M2.7

DATASET="${DATASET:-GPQA}"
CALIB_DIR="${CALIB_DIR:-${SCRIPT_DIR}/${DATASET}/latest}"
DUMP_PATH="${DUMP_PATH:-${CALIB_DIR}/qkv_dumps_merged}"
OUTPUT_DIR="${OUTPUT_DIR:-${CALIB_DIR}/rotations}"
mkdir -p "${OUTPUT_DIR}"

if [[ "${METHOD}" == "qqt_sst" && ! -d "${DUMP_PATH}" ]]; then
    echo "no dump at ${DUMP_PATH} -- run save_qkv_m27.sh first" >&2
    exit 1
fi

python3 "${COMPUTE_SCRIPT}" \
    --method "${METHOD}" \
    --composition "${COMPOSITION}" \
    --dump "${DUMP_PATH}" \
    --layers "${NUM_LAYERS}" \
    --head-dim "${HEAD_DIM}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"

echo "rotations -> ${OUTPUT_DIR}"
ls -la "${OUTPUT_DIR}"
