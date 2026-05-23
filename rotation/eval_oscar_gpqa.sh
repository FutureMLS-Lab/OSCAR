#!/usr/bin/env bash
# Generic GPQA eval driver for INT2 KV cache + OSCAR rotation.
#
# Required env:
#   MODEL          HuggingFace model id (e.g. Qwen/Qwen3-8B)
#   ROT_DIR        Folder containing {k,v}_rotation_qqt_r_h_pbr.pt
#   RUN_DIR        Output dir (logs + eval results)
#
# Optional env:
#   TP_SIZE        Tensor-parallel size for the eval server (default 4)
#   GPUS           CUDA_VISIBLE_DEVICES list (default 0,1,2,3)
#   PORT           HTTP port (default 31057)
#   DIST_PORT      Dist-init port (default 41057)
#   MEM_FRAC       --mem-fraction-static (default 0.8)
#   MAX_RUNNING    max-running-requests (default 64)
#   CUDA_GRAPH_MAX_BS (default 32)
#   GROUP_SIZE     int2 quant group size (default 128 — validated)
#   MAX_NEW_TOKENS (default 32768)
#   NUM_WORKERS    simple-evals client workers (default 32)
#   N_REPEATS      (default 1)
#   PRE_ROPE_FA3   set to 1 to force prefill fa3 + decode triton (default 1)

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
PORT="${PORT:-31057}"
DIST_PORT="${DIST_PORT:-41057}"
MEM_FRAC="${MEM_FRAC:-0.8}"
MAX_RUNNING="${MAX_RUNNING:-64}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-32}"
GROUP_SIZE="${GROUP_SIZE:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
NUM_WORKERS="${NUM_WORKERS:-32}"
N_REPEATS="${N_REPEATS:-1}"
NAME="${NAME:-gpqa_oscar}"

CONDA_BASE="${CONDA_BASE:-${HOME}/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-oscar}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

export PATH="${CONDA_PREFIX}/bin:${PATH}"
# Prepend per-rank Triton cache redirector so TP workers don't race on shared
# launcher .so / metadata files in TRITON_CACHE_DIR.
export PYTHONPATH="${REPO_ROOT}/rotation/_triton_per_rank:${SGLANG_RESEARCH_DIR}/python:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "${RUN_DIR}"
LOG_SERVER="${RUN_DIR}/server.log"
LOG_RUNNER="${RUN_DIR}/runner.log"   # streaming stdout from the eval runner
# run_simple_eval.py writes the canonical pretty-table eval.log to ${RUN_DIR}/.
: > "${LOG_SERVER}"

# Per-run Triton cache to avoid races when multiple eval servers compile the
# same kernel name into the shared default cache (~/.triton/cache).
# OSCAR_TRITON_PER_RANK_BASE is read by sitecustomize.py to route each TP
# rank into its own subdir (rank0/, rank1/, ...) — breaks intra-job races.
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
)
if [[ -n "${REASONING_PARSER:-}" ]]; then
    SERVER_ARGS+=(--reasoning-parser "${REASONING_PARSER}")
fi

echo "[eval-oscar] model=${MODEL} tp=${TP_SIZE} gpus=${GPUS} rot=${ROT_DIR} out=${RUN_DIR}"
SGLANG_ENABLE_MIXED_KV_WINDOWS=1 \
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
SGLANG_OSCAR_ABSORB_V_ROTATION=1 \
SGLANG_MIXED_KV_HP_MAX_SPLITS=8 \
SGLANG_MIXED_KV_PREFIX_TOKENS=${SGLANG_MIXED_KV_PREFIX_TOKENS:-64} \
SGLANG_MIXED_KV_RECENT_TOKENS=${SGLANG_MIXED_KV_RECENT_TOKENS:-256} \
SGLANG_MIXED_KV_HP_DTYPE=bfloat16 \
SGLANG_MIXED_KV_SCALE_DTYPE=float32 \
SGLANG_OSCAR_K_ROTATION_PATH="${ROT_DIR}/${K_ROT_FILENAME:-k_rotation_qqt_r_h_pbr.pt}" \
SGLANG_OSCAR_V_ROTATION_PATH="${ROT_DIR}/${V_ROT_FILENAME:-v_rotation_sst_r_h_pbr.pt}" \
SGLANG_OSCAR_K_CLIP_RATIO="${K_CLIP:-0.96}" \
SGLANG_OSCAR_V_CLIP_RATIO="${V_CLIP:-0.92}" \
SGLANG_LLOYD_MAX="${SGLANG_LLOYD_MAX:-0}" \
CUDA_VISIBLE_DEVICES="${GPUS}" \
python -m sglang.launch_server "${SERVER_ARGS[@]}" >> "${LOG_SERVER}" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 240); do
    if curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "[eval-oscar] server ready"
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[eval-oscar] server died"
        tail -100 "${LOG_SERVER}" || true
        exit 1
    fi
    sleep 5
done

if ! curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "[eval-oscar] server not ready after 20 min"
    tail -100 "${LOG_SERVER}" || true
    exit 1
fi

echo "[eval-oscar] launching eval via simple_evals (vendored at third_party/simple_evals)"
RUNNER="${REPO_ROOT}/rotation/_eval_runner/run_simple_eval.py"
python "${RUNNER}" \
    --task gpqa \
    --model "${MODEL}" \
    --base-url "http://127.0.0.1:${PORT}/v1" \
    --max-tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE:-1.0}" \
    --top-p "${TOP_P:-0.95}" \
    --top-k "${TOP_K:-40}" \
    --n-repeats "${N_REPEATS}" \
    --num-threads "${NUM_WORKERS:-32}" \
    ${NUM_EXAMPLES:+--num-examples ${NUM_EXAMPLES}} \
    --output-dir "${RUN_DIR}" \
    2>&1 | tee "${LOG_RUNNER}"
echo "[eval-oscar] done. score:"
grep -iE "gpqa/score|gpqa/chars" "${RUN_DIR}/eval.log" | tail -10 || true
