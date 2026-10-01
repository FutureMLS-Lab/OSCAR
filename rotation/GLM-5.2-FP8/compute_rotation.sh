#!/usr/bin/env bash
# Fit the per-layer MLA latent rotation (Rcov . P . Hblock) for GLM-5.2-FP8.
#
# Input is a c_kv dump: one tensor per layer of the 512-d compressed latent,
# captured while serving. Point DUMP_PATH at it. Unlike the MHA models there is
# no per-head structure here -- the latent is shared across heads.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMPUTE_SCRIPT="${SCRIPT_DIR}/../compute_mla_oscar_rotation.py"

DUMP_PATH="${DUMP_PATH:?set DUMP_PATH to the c_kv dump directory}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/GPQA/_rcov_phblock/rotations}"
KV_LORA_RANK="${KV_LORA_RANK:-512}"
GROUP_SIZE="${GROUP_SIZE:-128}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
mkdir -p "${OUTPUT_DIR}"

python3 "${COMPUTE_SCRIPT}" \
    --dump-path "${DUMP_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --kv-lora-rank "${KV_LORA_RANK}" \
    --group-size "${GROUP_SIZE}" \
    --max-tokens "${MAX_TOKENS}" \
    "$@"

echo "rotations -> ${OUTPUT_DIR}"
ls "${OUTPUT_DIR}" | head
