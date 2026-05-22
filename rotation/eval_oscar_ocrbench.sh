#!/usr/bin/env bash
# Generic OCRBench eval driver for OSCAR INT2 KV cache on a Qwen3-VL server.
#
# Required env:
#   MODEL          HuggingFace model id (e.g. Qwen/Qwen3-VL-8B-Instruct)
#   ROT_DIR        Folder containing {k,v}_rotation_*.pt
#                  (use Hadamard/ for identity rotation = online-Hadamard only)
#   RUN_DIR        Output dir (logs + eval results)
#
# Optional env (mirror eval_oscar_longbench_e.sh):
#   TP_SIZE, GPUS, PORT, DIST_PORT, MEM_FRAC, MAX_RUNNING, CUDA_GRAPH_MAX_BS,
#   GROUP_SIZE, NUM_WORKERS, K_CLIP, V_CLIP, K_ROT_FILENAME, V_ROT_FILENAME.

set -euo pipefail
export HF_HOME="${HF_HOME:-/shared/huggingface}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

: "${MODEL:?MODEL is required}"
: "${ROT_DIR:?ROT_DIR is required}"
: "${RUN_DIR:?RUN_DIR is required}"

SGLANG_RESEARCH_DIR="${SGLANG_RESEARCH_DIR:-${REPO_ROOT}/sglang-research}"
TP_SIZE="${TP_SIZE:-4}"
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
PORT="${PORT:-31300}"
DIST_PORT="${DIST_PORT:-41300}"
MEM_FRAC="${MEM_FRAC:-0.85}"
MAX_RUNNING="${MAX_RUNNING:-16}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-16}"
GROUP_SIZE="${GROUP_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"

CONDA_BASE="${CONDA_BASE:-${HOME}/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-oscar}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

export PATH="${CONDA_PREFIX}/bin:${PATH}"
# Prepend the conda env's site-packages so transformers (and any other oscar-
# specific deps) are picked up from the conda env rather than from a stale
# user-local install at ~/.local/lib/python3.12/site-packages — other pods in
# the namespace mutate that shared dir mid-run, which corrupted transformers
# and caused 'No module named transformers.model_debugging_utils' on first try.
CONDA_SITE="${CONDA_PREFIX}/lib/python3.12/site-packages"
export PYTHONPATH="${CONDA_SITE}:${REPO_ROOT}/rotation/_triton_per_rank:${SGLANG_RESEARCH_DIR}/python:${PYTHONPATH:-}"
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

SERVER_ARGS=(
    --model-path "${MODEL}"
    --tensor-parallel-size "${TP_SIZE}"
    --prefill-attention-backend fa3
    --decode-attention-backend triton
    --kv-cache-dtype int2
    --kv-cache-quant-group-size "${GROUP_SIZE}"
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
if [[ "${DISABLE_CUDA_GRAPH:-0}" == "1" ]]; then
    SERVER_ARGS+=(--disable-cuda-graph)
fi

echo "[ocr-oscar] model=${MODEL} tp=${TP_SIZE} gpus=${GPUS} rot=${ROT_DIR} out=${RUN_DIR}"
SGLANG_ENABLE_MIXED_KV_WINDOWS=1 \
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
SGLANG_OSCAR_ABSORB_V_ROTATION=1 \
SGLANG_MIXED_KV_HP_MAX_SPLITS=8 \
SGLANG_MIXED_KV_PREFIX_TOKENS=${SGLANG_MIXED_KV_PREFIX_TOKENS:-64} \
SGLANG_MIXED_KV_RECENT_TOKENS=${SGLANG_MIXED_KV_RECENT_TOKENS:-256} \
SGLANG_MIXED_KV_HP_DTYPE=bfloat16 \
SGLANG_MIXED_KV_SCALE_DTYPE=float32 \
SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 \
SGLANG_OSCAR_K_ROTATION_PATH="${ROT_DIR}/${K_ROT_FILENAME:-k_rotation_qqt_r_h_pbr.pt}" \
SGLANG_OSCAR_V_ROTATION_PATH="${ROT_DIR}/${V_ROT_FILENAME:-v_rotation_sst_r_h_pbr.pt}" \
SGLANG_OSCAR_K_CLIP_RATIO="${K_CLIP:-1.0}" \
SGLANG_OSCAR_V_CLIP_RATIO="${V_CLIP:-1.0}" \
CUDA_VISIBLE_DEVICES="${GPUS}" \
python -m sglang.launch_server "${SERVER_ARGS[@]}" >> "${LOG_SERVER}" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 240); do
    if curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "[ocr-oscar] server ready"
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[ocr-oscar] server died"
        tail -100 "${LOG_SERVER}" || true
        exit 1
    fi
    sleep 5
done

if ! curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "[ocr-oscar] server not ready after 20 min"
    tail -100 "${LOG_SERVER}" || true
    exit 1
fi

CLIENT_ARGS=(
    --base-url "http://127.0.0.1:${PORT}/v1"
    --model "${MODEL}"
    --output-dir "${RUN_DIR}"
    --num-workers "${NUM_WORKERS}"
)

echo "[ocr-oscar] launching eval_ocrbench.py"
python "${REPO_ROOT}/rotation/eval_ocrbench.py" "${CLIENT_ARGS[@]}" 2>&1 | tee "${LOG_CLIENT}"

echo "[ocr-oscar] done. summary:"
python -c "import json; d=json.load(open('${RUN_DIR}/results.json')); print('OCRBench score = %d/1000' % d['score'])" || true
