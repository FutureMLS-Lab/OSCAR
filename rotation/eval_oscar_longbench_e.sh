#!/usr/bin/env bash
# Generic LongBench-E eval driver for INT2 KV cache + OSCAR rotation.
#
# Mirrors eval_oscar_gpqa.sh exactly for the server side; only swaps the eval
# client from simple-evals to rotation/eval_longbench_e.py.
#
# Required env:
#   MODEL          HuggingFace model id (e.g. Qwen/Qwen3-8B)
#   ROT_DIR        Folder containing {k,v}_rotation_*.pt
#   RUN_DIR        Output dir (logs + eval results)
#
# Optional env (same defaults as eval_oscar_gpqa.sh unless noted):
#   TP_SIZE        Tensor-parallel size (default 4)
#   GPUS           CUDA_VISIBLE_DEVICES list (default 0,1,2,3)
#   PORT           HTTP port (default 31200)
#   DIST_PORT      Dist-init port (default 41200)
#   MEM_FRAC       --mem-fraction-static (default 0.8)
#   MAX_RUNNING    max-running-requests (default 64)
#   CUDA_GRAPH_MAX_BS (default 32)
#   GROUP_SIZE     int2 quant group size (default 128)
#   MAX_INPUT_LEN  --max-input-len for the client (default 32768)
#   NUM_WORKERS    client thread pool (default 16)
#   K_ROT_FILENAME (default k_rotation_qqt_r_h_pbr.pt)
#   V_ROT_FILENAME (default v_rotation_sst_r_h_pbr.pt)
#   K_CLIP         (default 0.96)
#   V_CLIP         (default 0.92)
#   DATASETS       Space-separated subset names; defaults to all 13.

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
PORT="${PORT:-31200}"
DIST_PORT="${DIST_PORT:-41200}"
MEM_FRAC="${MEM_FRAC:-0.85}"
# Cap server batching well below the LongBench client's NUM_WORKERS so the
# mixed-KV pool isn't asked to hold every long context concurrently
# (gov_report/multi_news have 512-token outputs which thrash the HP-recent ring).
MAX_RUNNING="${MAX_RUNNING:-16}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-16}"
GROUP_SIZE="${GROUP_SIZE:-128}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-32768}"
NUM_WORKERS="${NUM_WORKERS:-8}"

CONDA_BASE="${CONDA_BASE:-${HOME}/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-oscar}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

export PATH="${CONDA_PREFIX}/bin:${PATH}"
# Prepend conda env site-packages so transformers comes from the oscar env
# rather than from a possibly-corrupted user-site (concurrent pods in the
# same namespace mutate ~/.local mid-run — see eval_oscar_ocrbench.sh).
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
    # LongBench has unique long-context prompts per row — prefix caching
    # would just pin pages in the HP-prefix BF16 pool until it OOMs.
    --disable-radix-cache
)

echo "[lb-oscar] model=${MODEL} tp=${TP_SIZE} gpus=${GPUS} rot=${ROT_DIR} out=${RUN_DIR}"
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
SGLANG_OSCAR_K_CLIP_RATIO="${K_CLIP:-0.96}" \
SGLANG_OSCAR_V_CLIP_RATIO="${V_CLIP:-0.92}" \
CUDA_VISIBLE_DEVICES="${GPUS}" \
python -m sglang.launch_server "${SERVER_ARGS[@]}" >> "${LOG_SERVER}" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 240); do
    if curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "[lb-oscar] server ready"
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[lb-oscar] server died"
        tail -100 "${LOG_SERVER}" || true
        exit 1
    fi
    sleep 5
done

if ! curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "[lb-oscar] server not ready after 20 min"
    tail -100 "${LOG_SERVER}" || true
    exit 1
fi

CLIENT_ARGS=(
    --base-url "http://127.0.0.1:${PORT}/v1"
    --model "${MODEL}"
    --output-dir "${RUN_DIR}"
    --max-input-len "${MAX_INPUT_LEN}"
    --num-workers "${NUM_WORKERS}"
    --use-chat "${USE_CHAT:-true}"
)
if [[ -n "${DATASETS:-}" ]]; then
    # shellcheck disable=SC2206
    DS_ARR=(${DATASETS})
    CLIENT_ARGS+=(--datasets "${DS_ARR[@]}")
fi

echo "[lb-oscar] launching eval_longbench_e.py"
python "${REPO_ROOT}/rotation/eval_longbench_e.py" "${CLIENT_ARGS[@]}" 2>&1 | tee "${LOG_CLIENT}"

echo "[lb-oscar] done. summary:"
python -c "import json; d=json.load(open('${RUN_DIR}/result.json')); print(json.dumps(d, indent=2))" || true
