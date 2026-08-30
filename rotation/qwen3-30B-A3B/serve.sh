#!/usr/bin/env bash
# Prefix caching is ON by default: the two defects that forced it off are
# fixed, and cache-ON measured indistinguishable from cache-OFF across four
# paired arms. Set DISABLE_RADIX=1 to turn it back off for an A/B.
# Serve Qwen3-30B-A3B with INT2 KV + per-head rotations.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export HF_HOME="${HF_HOME:-/shared/huggingface}"
export PYTHONPATH="${REPO_ROOT}/sglang-research/python:${PYTHONPATH:-}"

DATASET="${DATASET:-GPQA}"
if [[ -z "${CAL:-}" ]]; then
  CAL="$(ls -1dt "${SCRIPT_DIR}/${DATASET}"/*/rotations "${SCRIPT_DIR}/../qwen3-30b-a3b/${DATASET}"/*/rotations 2>/dev/null | head -1)"
fi
K_ROT="${K_ROT:-${CAL}/k_perhead.pt}"
V_ROT="${V_ROT:-${CAL}/v_perhead.pt}"
for f in "${K_ROT}" "${V_ROT}"; do
  [[ -f "$f" ]] || { echo "missing $f -- run fit_perhead.sh" >&2; exit 1; }
done

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
TP_SIZE="${TP_SIZE:-2}"
PORT="${PORT:-31760}"
PY="${PY:-${HOME}/miniconda3/envs/oscar/bin/python3}"

env \
  SGLANG_ENABLE_MIXED_KV_WINDOWS=1 \
  SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  SGLANG_MIXED_KV_PREFIX_TOKENS="${PREFIX_TOKENS:-64}" \
  SGLANG_MIXED_KV_RECENT_TOKENS="${RECENT_TOKENS:-256}" \
  SGLANG_MIXED_KV_HP_DTYPE=bfloat16 \
  SGLANG_MIXED_KV_SCALE_DTYPE=float32 \
  SGLANG_MIXED_KV_HP_MAX_SPLITS=8 \
  SGLANG_OSCAR_ABSORB_V_ROTATION=0 \
  SGLANG_OSCAR_K_CLIP_RATIO=0.96 \
  SGLANG_OSCAR_V_CLIP_RATIO=0.92 \
  SGLANG_LLOYD_MAX="${LLOYD_MAX:-0}" \
  SGLANG_OSCAR_K_ROTATION_PATH="${K_ROT}" \
  SGLANG_OSCAR_V_ROTATION_PATH="${V_ROT}" \
  "${PY}" -m sglang.launch_server \
    --model-path "${MODEL}" --tensor-parallel-size "${TP_SIZE}" \
    --kv-cache-dtype int2 --kv-cache-quant-group-size 128 \
    --prefill-attention-backend fa3 --decode-attention-backend triton \
    --mem-fraction-static "${MEM_FRACTION_STATIC:-0.75}" \
    ${DISABLE_RADIX:+--disable-radix-cache} --trust-remote-code \
    --host 127.0.0.1 --port "${PORT}" "$@"
