#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
: "${DUMP_PATH:?Set DUMP_PATH to the QKV dump directory}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the rotation output directory}"
: "${MODEL_CONFIG:?Set MODEL_CONFIG to the Kimi-K3 config.json path}"
PY="${PY:-python3}"

mkdir -p "${OUTPUT_DIR}"
"${PY}" "${REPO_ROOT}/rotation/compute_kv_rotation.py" \
  --dump-path "${DUMP_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --chunk-id all \
  --method qqt_sst \
  --composition r_h_pbr \
  --k-head-dim 192 \
  --v-head-dim 128

"${PY}" "${SCRIPT_DIR}/validate_rotations.py" \
  --config "${MODEL_CONFIG}" \
  --k-rotation "${OUTPUT_DIR}/k_rotation_qqt_r_h_pbr.pt" \
  --v-rotation "${OUTPUT_DIR}/v_rotation_sst_r_h_pbr.pt" \
  | tee "${OUTPUT_DIR}/validation.json"
