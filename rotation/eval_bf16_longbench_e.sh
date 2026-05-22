#!/usr/bin/env bash
# Plain BF16 LongBench-E eval driver (Phase 1c C1 baseline).
# Same client/eval pipeline as eval_oscar_longbench_e.sh — but the SGLang
# server runs WITHOUT --kv-cache-dtype int2, WITHOUT any OSCAR rotation env
# vars, and WITHOUT --disable-radix-cache. This establishes our eval client's
# own ceiling (target ~49.56, OScaR-KV-Quant's published BF16 result).
#
# Required env:
#   MODEL          HuggingFace model id (e.g. Qwen/Qwen3-8B)
#   RUN_DIR        Output dir
#
# Optional env:
#   TP_SIZE        Tensor-parallel size (default 4)
#   GPUS           CUDA_VISIBLE_DEVICES list (default 0,1,2,3)
#   PORT           HTTP port (default 31220)
#   DIST_PORT      Dist-init port (default 41220)
#   MEM_FRAC       --mem-fraction-static (default 0.85)
#   MAX_RUNNING    max-running-requests (default 16)
#   CUDA_GRAPH_MAX_BS (default 16)
#   MAX_INPUT_LEN  --max-input-len for the client (default 32768)
#   NUM_WORKERS    client thread pool (default 8)
#   USE_CHAT       chat-mode flag for the client (default true)
#   DATASETS       Space-separated subset names; defaults to all 13.

set -euo pipefail
export HF_HOME="${HF_HOME:-/shared/huggingface}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

: "${MODEL:?MODEL is required}"
: "${RUN_DIR:?RUN_DIR is required}"

SGLANG_RESEARCH_DIR="${SGLANG_RESEARCH_DIR:-${REPO_ROOT}/sglang-research}"
TP_SIZE="${TP_SIZE:-4}"
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
PORT="${PORT:-31220}"
DIST_PORT="${DIST_PORT:-41220}"
MEM_FRAC="${MEM_FRAC:-0.85}"
MAX_RUNNING="${MAX_RUNNING:-16}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-16}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-32768}"
NUM_WORKERS="${NUM_WORKERS:-8}"
USE_CHAT="${USE_CHAT:-true}"

CONDA_BASE="${CONDA_BASE:-${HOME}/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-oscar}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

export PATH="${CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/rotation/_triton_per_rank:${SGLANG_RESEARCH_DIR}/python:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "${RUN_DIR}"
LOG_SERVER="${RUN_DIR}/server.log"
LOG_CLIENT="${RUN_DIR}/eval.log"
: > "${LOG_SERVER}"
: > "${LOG_CLIENT}"

export OSCAR_TRITON_PER_RANK_BASE="${OSCAR_TRITON_PER_RANK_BASE:-${RUN_DIR}/triton_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${OSCAR_TRITON_PER_RANK_BASE}/main}"
mkdir -p "${OSCAR_TRITON_PER_RANK_BASE}" "${TRITON_CACHE_DIR}"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
        pkill -TERM -P "${SERVER_PID}" 2>/dev/null || true
        sleep 2
        kill -KILL "${SERVER_PID}" 2>/dev/null || true
        pkill -KILL -P "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# Plain BF16 server — no INT2, no rotation, no mixed-KV. Keep
# --disable-radix-cache to match the LongBench-E run shape (unique
# long-context prompts).
SERVER_ARGS=(
    --model-path "${MODEL}"
    --tensor-parallel-size "${TP_SIZE}"
    --prefill-attention-backend fa3
    --decode-attention-backend triton
    --mem-fraction-static "${MEM_FRAC}"
    --max-running-requests "${MAX_RUNNING}"
    --enable-cache-report
    --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}"
    --host 127.0.0.1
    --port "${PORT}"
    --dist-init-addr "127.0.0.1:${DIST_PORT}"
    --trust-remote-code
    --disable-radix-cache
)

echo "[lb-bf16] model=${MODEL} tp=${TP_SIZE} gpus=${GPUS} out=${RUN_DIR}"
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
CUDA_VISIBLE_DEVICES="${GPUS}" \
python -m sglang.launch_server "${SERVER_ARGS[@]}" >> "${LOG_SERVER}" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 240); do
    if curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "[lb-bf16] server ready"
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[lb-bf16] server died"
        tail -100 "${LOG_SERVER}" || true
        exit 1
    fi
    sleep 5
done

if ! curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "[lb-bf16] server not ready after 20 min"
    tail -100 "${LOG_SERVER}" || true
    exit 1
fi

CLIENT_ARGS=(
    --base-url "http://127.0.0.1:${PORT}/v1"
    --model "${MODEL}"
    --output-dir "${RUN_DIR}"
    --max-input-len "${MAX_INPUT_LEN}"
    --num-workers "${NUM_WORKERS}"
    --use-chat "${USE_CHAT}"
)
if [[ -n "${DATASETS:-}" ]]; then
    # shellcheck disable=SC2206
    DS_ARR=(${DATASETS})
    CLIENT_ARGS+=(--datasets "${DS_ARR[@]}")
fi

echo "[lb-bf16] launching eval_longbench_e.py (use_chat=${USE_CHAT})"
python "${REPO_ROOT}/rotation/eval_longbench_e.py" "${CLIENT_ARGS[@]}" 2>&1 | tee "${LOG_CLIENT}"

echo "[lb-bf16] done. summary:"
python -c "import json; d=json.load(open('${RUN_DIR}/result.json')); print(json.dumps(d, indent=2))" || true
