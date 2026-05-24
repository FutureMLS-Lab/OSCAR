#!/usr/bin/env bash
# GPQA eval for Qwen/Qwen3.5-35B-A3B (hybrid MoE: 10 full-attn / 40 layers, head_dim=256).
#
# Usage: MODE=<mode> bash eval_gpqa.sh
#   MODE=bf16        (default) BF16 baseline
#   MODE=hadamard    OSCAR INT2 + Hadamard rotation
#   MODE=calibrated  OSCAR INT2 + calibrated qqt_sst rotation
#
# Rotation checkpoints are at rotation/qwen3.5-35B-A3B/rotations/{hadamard,calibrated}/.
# Set SGLANG_RESEARCH_DIR to override the sglang checkout (must include Qwen3.5
# OSCAR patches: HybridLinearKVPool + UnifiedInt2HPKVPool wiring).
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

MODE="${MODE:-bf16}"
export MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B}"
export TP_SIZE="${TP_SIZE:-4}"
export K_CLIP="${K_CLIP:-0.96}"
export V_CLIP="${V_CLIP:-0.92}"
export GROUP_SIZE="${GROUP_SIZE:-128}"
export NAME="${NAME:-gpqa_qwen35_35b_a3b}"
export SGLANG_RESEARCH_DIR="${SGLANG_RESEARCH_DIR:-${REPO_ROOT}/sglang-research}"

if [[ "${MODE}" == "hadamard" || "${MODE}" == "calibrated" ]]; then
    ROT_BASE="${SCRIPT_DIR}/rotations/${MODE}"
    export SGLANG_OSCAR_K_ROTATION_PATH="${ROT_BASE}/k_rotation.pt"
    export SGLANG_OSCAR_V_ROTATION_PATH="${ROT_BASE}/v_rotation.pt"
    export SGLANG_OSCAR_K_CLIP_RATIO="${K_CLIP}"
    export SGLANG_OSCAR_V_CLIP_RATIO="${V_CLIP}"
    export SGLANG_OSCAR_ABSORB_V_ROTATION=1
    EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-} --disable-radix-cache"
    export EXTRA_SERVER_ARGS
    export RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/_eval_gpqa_${MODE}}"
    exec bash "${SCRIPT_DIR}/../eval_oscar_gpqa.sh"
elif [[ "${MODE}" != "bf16" ]]; then
    echo "Unknown MODE=${MODE}. Use bf16, hadamard, or calibrated." >&2
    exit 1
fi

# ---------- BF16 baseline ----------
export HF_HOME="${HF_HOME:-/shared/huggingface}"
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/_eval_gpqa_bf16}"
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
PORT="${PORT:-31059}"
DIST_PORT="${DIST_PORT:-41059}"
MEM_FRAC="${MEM_FRAC:-0.85}"
MAX_RUNNING="${MAX_RUNNING:-64}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
NUM_WORKERS="${NUM_WORKERS:-32}"
N_REPEATS="${N_REPEATS:-1}"

CONDA_BASE="${CONDA_BASE:-${HOME}/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-oscar}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/rotation/_triton_per_rank:${SGLANG_RESEARCH_DIR}/python:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "${RUN_DIR}"
LOG_SERVER="${RUN_DIR}/server.log"
LOG_RUNNER="${RUN_DIR}/runner.log"
: > "${LOG_SERVER}"

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
    --kv-cache-dtype auto
    --prefill-attention-backend triton
    --decode-attention-backend triton
    --mem-fraction-static "${MEM_FRAC}"
    --max-running-requests "${MAX_RUNNING}"
    --enable-cache-report
    --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}"
    --host 127.0.0.1
    --port "${PORT}"
    --dist-init-addr "127.0.0.1:${DIST_PORT}"
    --trust-remote-code
)
if [[ -n "${EXTRA_SERVER_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    SERVER_ARGS+=(${EXTRA_SERVER_ARGS})
fi

echo "[bf16] model=${MODEL} tp=${TP_SIZE} gpus=${GPUS} out=${RUN_DIR}"
CUDA_VISIBLE_DEVICES="${GPUS}" \
    python -m sglang.launch_server "${SERVER_ARGS[@]}" >> "${LOG_SERVER}" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 240); do
    if curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "[bf16] server ready"
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[bf16] server died"
        tail -100 "${LOG_SERVER}" || true
        exit 1
    fi
    sleep 5
done

if ! curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "[bf16] server not ready after 20 min"
    tail -100 "${LOG_SERVER}" || true
    exit 1
fi

echo "[bf16] launching eval"
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
echo "[bf16] done. score:"
grep -iE "gpqa/score|gpqa/chars" "${RUN_DIR}/eval.log" | tail -10 || true
