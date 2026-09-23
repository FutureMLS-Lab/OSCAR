#!/usr/bin/env bash
# Compute K/V rotation checkpoints for Qwen3-32B.
#
# Two modes, switched via METHOD env (default qqt_sst — the calibrated recipe):
#
#   METHOD=qqt_sst (default)  Calibrated rotation from a Q/K/V dump.
#                             Requires DUMP_PATH pointing at save_qkv output.
#                             Composition r_h_pbr (validated at gpqa 62%).
#
#   METHOD=hadamard           Data-free fixed Hadamard rotation per layer.
#                             No dump needed.
#
# Outputs in OUTPUT_DIR:
#   qqt_sst   -> k_rotation_qqt_r_h_pbr.pt, v_rotation_sst_r_h_pbr.pt
#   hadamard  -> k_rotation_hadamard.pt,    v_rotation_hadamard.pt

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMPUTE_SCRIPT="${SCRIPT_DIR}/../compute_kv_rotation.py"

METHOD="${METHOD:-qqt_sst}"
HEAD_DIM="${HEAD_DIM:-128}"
NUM_LAYERS="${NUM_LAYERS:-64}"   # Qwen3-32B has 64 layers
COMPOSITION="${COMPOSITION:-r_h_pbr}"
CHUNK_ID="${CHUNK_ID:-all}"
# Auto-find newest calibration produced by save_qkv (newest mtime); override
# with DUMP_PATH or CALIB_DIR for a specific calibration.
DATASET="${DATASET:-GPQA}"
if [[ -z "${CALIB_DIR:-}" ]]; then
    CALIB_DIR="$(ls -1dt "${SCRIPT_DIR}/${DATASET}"/seq*_prompt*_group*/ 2>/dev/null | head -1 | sed 's:/$::')"
fi
DUMP_PATH="${DUMP_PATH:-${CALIB_DIR}/qkv_dumps/gpqa}"
OUTPUT_DIR="${OUTPUT_DIR:-${CALIB_DIR}/rotations}"
export DUMP_PATH
echo "[compute_rotation] calib_dir=${CALIB_DIR}"
echo "[compute_rotation] dump_path=${DUMP_PATH}"
echo "[compute_rotation] output_dir=${OUTPUT_DIR}"

# Pick a python with torch. Override with PY=/path/to/python3 if needed.
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

echo "[compute_rotation] method=${METHOD} head_dim=${HEAD_DIM} output_dir=${OUTPUT_DIR}"

case "${METHOD}" in
    hadamard)
        "${PY}" "${COMPUTE_SCRIPT}" \
            --method hadamard \
            --head-dim "${HEAD_DIM}" \
            --num-layers "${NUM_LAYERS}" \
            --output-dir "${OUTPUT_DIR}"
        ;;
    qqt_sst|ktk_vtv|qqt|sst|ktk|vtv|uresidual)
        if [[ -z "${DUMP_PATH:-}" ]]; then
            echo "[compute_rotation] METHOD=${METHOD} requires DUMP_PATH" >&2
            echo "  e.g. DUMP_PATH=${SCRIPT_DIR}/qkv_dumps/gpqa $0" >&2
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
        echo "  valid: qqt_sst (default) | hadamard | ktk_vtv | qqt | sst | ktk | vtv | uresidual" >&2
        exit 1
        ;;
esac

echo "[compute_rotation] done. files:"
ls -la "${OUTPUT_DIR}" | grep -E "rotation.*\.pt" || true
