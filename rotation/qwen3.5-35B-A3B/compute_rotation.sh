#!/usr/bin/env bash
# Compute K/V rotation checkpoints for Qwen3.5-35B-A3B.
#   HEAD_DIM=256 (full-attention layers)
#   NUM_LAYERS=10 (count of full_attention layers, not all 40)
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMPUTE_SCRIPT="${SCRIPT_DIR}/../compute_kv_rotation.py"

METHOD="${METHOD:-qqt_sst}"
HEAD_DIM="${HEAD_DIM:-256}"
NUM_LAYERS="${NUM_LAYERS:-10}"   # 10 full-attention layers in Qwen3.5-35B-A3B
# Global layer ids of full-attention layers in Qwen3.5-35B-A3B. The runtime
# OSCAR pool looks up rotations by global id, so we key the file the same way.
LAYER_IDS="${LAYER_IDS:-3,7,11,15,19,23,27,31,35,39}"
COMPOSITION="${COMPOSITION:-r_h_pbr}"
CHUNK_ID="${CHUNK_ID:-all}"
DATASET="${DATASET:-GPQA}"
if [[ -z "${CALIB_DIR:-}" ]]; then
    CALIB_DIR="$(ls -1dt "${SCRIPT_DIR}/${DATASET}"/seq*_prompt*_group*/ 2>/dev/null | head -1 | sed 's:/$::' || true)"
fi
DUMP_PATH="${DUMP_PATH:-${CALIB_DIR}/qkv_dumps/gpqa}"
OUTPUT_DIR="${OUTPUT_DIR:-${CALIB_DIR}/rotations}"
export DUMP_PATH
echo "[compute_rotation] calib_dir=${CALIB_DIR}"
echo "[compute_rotation] dump_path=${DUMP_PATH}"
echo "[compute_rotation] output_dir=${OUTPUT_DIR}"

if [[ -z "${PY:-}" ]]; then
    for candidate in \
        ${HOME}/miniconda3/envs/oscar/bin/python3 \
        ${HOME}/anaconda3/envs/oscar/bin/python3 \
        "$(command -v python3 || true)"
    do
        if [[ -x "${candidate}" ]]; then
            PY="${candidate}"
            break
        fi
    done
fi
if [[ -z "${PY:-}" ]]; then
    echo "[compute_rotation] no python3 found; set PY=/path/to/python3" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "[compute_rotation] method=${METHOD} head_dim=${HEAD_DIM} num_layers=${NUM_LAYERS} output_dir=${OUTPUT_DIR}"

case "${METHOD}" in
    hadamard)
        "${PY}" "${COMPUTE_SCRIPT}" \
            --method hadamard \
            --head-dim "${HEAD_DIM}" \
            --layer-ids "${LAYER_IDS}" \
            --output-dir "${OUTPUT_DIR}"
        ;;
    qqt_sst|ktk_vtv|qqt|sst|ktk|vtv|uresidual)
        if [[ -z "${DUMP_PATH:-}" ]]; then
            echo "[compute_rotation] METHOD=${METHOD} requires DUMP_PATH" >&2
            exit 1
        fi
        extra_args=()
        if [[ "${METHOD}" == "uresidual" ]]; then
            : "${REF_K_ROTATION:?REF_K_ROTATION required for uresidual}"
            : "${REF_V_ROTATION:?REF_V_ROTATION required for uresidual}"
            extra_args+=(
                --ref-k-rotation "${REF_K_ROTATION}"
                --ref-v-rotation "${REF_V_ROTATION}"
            )
        fi
        "${PY}" "${COMPUTE_SCRIPT}" \
            --dump-path "${DUMP_PATH}" \
            --output-dir "${OUTPUT_DIR}" \
            --head-dim "${HEAD_DIM}" \
            --chunk-id "${CHUNK_ID}" \
            --method "${METHOD}" \
            --composition "${COMPOSITION}" \
            "${extra_args[@]}"
        ;;
    *)
        echo "[compute_rotation] unknown METHOD=${METHOD}" >&2
        exit 1
        ;;
esac

echo "[compute_rotation] done. files:"
ls -la "${OUTPUT_DIR}" | grep -E "rotation.*\.pt" || true
