#!/usr/bin/env bash
# Phase 3 — GPQA eval for gemma-4-12B-it. Defaults to INT2 OSCAR; set MODE=bf16 for
# the BF16 baseline. Uses this fork's gemma4_unified support (transformers>=5.5 env).
#
# NOTE: INT2 OSCAR on gemma requires the two-geometry-group UnifiedInt2HPKVPool
# (see ../../.RUD/gemma4-12b/PLAN.md P4). BF16 mode works today (baseline 62.63%).
set -euo pipefail
export HF_HOME="${HF_HOME:-/shared/huggingface}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SGLANG_RESEARCH_DIR="${SGLANG_RESEARCH_DIR:-${REPO_ROOT}/sglang-research}"

MODEL="${MODEL:-google/gemma-4-12B-it}"
PY="${PY:?Set PY to a python with transformers>=5.5 + sglang/flashinfer/sgl_kernel}"
MODE="${MODE:-int2}"                     # int2 | bf16
GPU="${GPU:-0}"
PORT="${PORT:-31075}"; DIST_PORT="${DIST_PORT:-41075}"
MEM_FRAC="${MEM_FRACTION_STATIC:-0.85}"
GROUP_SIZE="${GROUP_SIZE:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
NUM_EXAMPLES="${NUM_EXAMPLES:-}"         # empty = full 198
NUM_WORKERS="${NUM_WORKERS:-24}"
ROT_DIR="${ROT_DIR:-${SCRIPT_DIR}/GPQA/latest/rotations}"
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/GPQA/latest/_eval_${MODE}}"
mkdir -p "${RUN_DIR}"

export PYTHONPATH="${SGLANG_RESEARCH_DIR}/python:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${GPU}"

SERVER_ARGS=(
    --model-path "${MODEL}" --tensor-parallel-size 1
    --context-length 16384 --mem-fraction-static "${MEM_FRAC}"
    --prefill-attention-backend triton --decode-attention-backend triton
    --disable-cuda-graph --disable-piecewise-cuda-graph --trust-remote-code
    --host 127.0.0.1 --port "${PORT}" --dist-init-addr "127.0.0.1:${DIST_PORT}"
)
declare -a OSCAR_ENV=()
if [[ "${MODE}" == "int2" ]]; then
    # gemma's 8-head sliding-layer HP arena is ~8x heavier per token than uniform
    # models, so cap concurrency/tokens + the HP-prefix pool to fit one H100.
    SERVER_ARGS+=(
        --kv-cache-dtype int2 --kv-cache-quant-group-size "${GROUP_SIZE}"
        --max-running-requests "${MAX_RUNNING:-32}"
        --max-total-tokens "${MAX_TOTAL_TOKENS:-262144}"
    )
    OSCAR_ENV=(
        SGLANG_ENABLE_MIXED_KV_WINDOWS=1 SGLANG_OSCAR_ABSORB_V_ROTATION=1
        SGLANG_OSCAR_K_ROTATION_PATH="${ROT_DIR}/k_rotation_qqt_r_h_pbr.pt"
        SGLANG_OSCAR_V_ROTATION_PATH="${ROT_DIR}/v_rotation_sst_r_h_pbr.pt"
        SGLANG_OSCAR_K_CLIP_RATIO="${K_CLIP:-0.96}" SGLANG_OSCAR_V_CLIP_RATIO="${V_CLIP:-0.92}"
        SGLANG_MIXED_KV_PREFIX_TOKENS="${SGLANG_MIXED_KV_PREFIX_TOKENS:-64}"
        SGLANG_MIXED_KV_RECENT_TOKENS="${SGLANG_MIXED_KV_RECENT_TOKENS:-256}"
        SGLANG_MIXED_KV_HP_MAX_SPLITS=8 SGLANG_MIXED_KV_HP_DTYPE=bfloat16 SGLANG_MIXED_KV_SCALE_DTYPE=float32
        SGLANG_MIXED_KV_HP_PREFIX_POOL_TOKENS="${SGLANG_MIXED_KV_HP_PREFIX_POOL_TOKENS:-8192}"
        SGLANG_LLOYD_MAX="${SGLANG_LLOYD_MAX:-0}"
    )
fi

LOG="${RUN_DIR}/server.log"; : > "${LOG}"
cleanup() { for pid in $(nvidia-smi -i "${GPU}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do kill -TERM "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

echo "[eval] mode=${MODE} model=${MODEL} rot=${ROT_DIR}"
env "${OSCAR_ENV[@]}" "${PY}" -m sglang.launch_server "${SERVER_ARGS[@]}" >> "${LOG}" 2>&1 &
for _ in $(seq 1 240); do
    curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { echo "[eval] server ready"; break; }
    sleep 5
done
curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 || { echo "[eval] server not ready"; tail -60 "${LOG}"; exit 1; }

"${PY}" "${REPO_ROOT}/rotation/_eval_runner/run_simple_eval.py" \
    --task gpqa --model "${MODEL}" --base-url "http://127.0.0.1:${PORT}/v1" \
    --max-tokens "${MAX_NEW_TOKENS}" --temperature 1.0 --top-p 0.95 --top-k 40 \
    --num-threads "${NUM_WORKERS}" ${NUM_EXAMPLES:+--num-examples ${NUM_EXAMPLES}} \
    --output-dir "${RUN_DIR}"
echo "[eval] score:"; grep -iE "gpqa/score" "${RUN_DIR}/eval.log" | tail -2
