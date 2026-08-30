#!/usr/bin/env bash
# Dump post-RoPE Q/K/V tensors for GPQA calibration on MiniMax-M2.7 (dense MHA).
set -euo pipefail
export HF_HOME="${HF_HOME:-/shared/huggingface}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SGLANG_DUMP_DIR="${SGLANG_DUMP_DIR:-${REPO_ROOT}/sglang-dump-qkv}"

MODEL="${MODEL:-MiniMaxAI/MiniMax-M2.7}"
TP_SIZE="${TP_SIZE:-4}"
PORT="${PORT:-31050}"
DIST_PORT="${DIST_PORT:-41050}"
GPU="${GPU:-0,1,2,3,4,5,6,7}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.8}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
MAX_QUEUED_REQUESTS="${MAX_QUEUED_REQUESTS:-64}"
MAX_WAIT_SECS="${MAX_WAIT_SECS:-7200}"

export DUMP_KVCACHE="${DUMP_KVCACHE:-true}"
export DUMP_KVCACHE_TOKENS="${DUMP_KVCACHE_TOKENS:-30000}"

# Calibration layout:
#   <model>/<DATASET>/seq<TOKENS>_prompt<N>_group<GROUP>/{qkv_dumps,rotations}
# Dump writes to <DATASET>/latest/, then renames to the final tag once N is known.
# Eval scripts find the right calibration by passing ROT_DIR explicitly.
DATASET="${DATASET:-GPQA}"
GROUP_SIZE="${GROUP_SIZE:-128}"
CALIB_DIR="${SCRIPT_DIR}/${DATASET}/latest"
export DUMP_KVCACHE_DIR="${DUMP_KVCACHE_DIR:-${CALIB_DIR}/qkv_dumps/gpqa}"
mkdir -p "${DUMP_KVCACHE_DIR}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-oscar}"
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python3" ]]; then
    CONDA_ENV_DIR="${CONDA_ENV_DIR:-${CONDA_PREFIX}}"
else
    CONDA_BASE="${CONDA_BASE:-${HOME}/anaconda3}"
    CONDA_ENV_DIR="${CONDA_ENV_DIR:-${CONDA_BASE}/envs/${CONDA_ENV_NAME}}"
    if [[ ! -x "${CONDA_ENV_DIR}/bin/python3" && -x "${HOME}/miniconda3/envs/${CONDA_ENV_NAME}/bin/python3" ]]; then
        CONDA_ENV_DIR="${HOME}/miniconda3/envs/${CONDA_ENV_NAME}"
    fi
fi
PY="${PY:-${CONDA_ENV_DIR}/bin/python3}"
PY_EVAL="${PY_EVAL:-${PY}}"
export PATH="${CONDA_ENV_DIR}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU}}"
export PYTHONUNBUFFERED=1

LOCAL_PYTHONPATH="${REPO_ROOT}/rotation/_dump_compat:${SGLANG_DUMP_DIR}/python"
if [[ -n "${PYTHONPATH:-}" ]]; then
    LOCAL_PYTHONPATH="${LOCAL_PYTHONPATH}:${PYTHONPATH}"
fi
export PYTHONPATH="${LOCAL_PYTHONPATH}"

SERVER_LOG="${DUMP_KVCACHE_DIR}/server.log"
DUMP_RUNNER_LOG="${DUMP_KVCACHE_DIR}/dump_runner.log"
: > "${SERVER_LOG}"

log() { echo "[$(date '+%F %T')] $*"; }

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        log "Stopping server PID ${SERVER_PID}"
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
    --max-running-requests "${MAX_RUNNING_REQUESTS}"
    --max-queued-requests "${MAX_QUEUED_REQUESTS}"
    --page-size 128
    --chunked-prefill-size 4096
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --pp-max-micro-batch-size 32
    --kv-cache-dtype auto
    --prefill-attention-backend triton
    --decode-attention-backend triton
    --sampling-backend flashinfer
    --host 127.0.0.1
    --port "${PORT}"
    --dist-init-addr "127.0.0.1:${DIST_PORT}"
    --trust-remote-code
    --disable-custom-all-reduce
    --disable-cuda-graph
    --watchdog-timeout 1800
)
if [[ -n "${EXTRA_SERVER_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    SERVER_ARGS+=(${EXTRA_SERVER_ARGS})
fi

log "Starting sglang server for QKV dump"
log "  sglang_dump=${SGLANG_DUMP_DIR}"
log "  model=${MODEL}"
log "  dump_dir=${DUMP_KVCACHE_DIR}"
log "  dump_tokens=${DUMP_KVCACHE_TOKENS}"

PYTHONPATH="${LOCAL_PYTHONPATH}" \
    "${PY}" -m sglang.launch_server "${SERVER_ARGS[@]}" >> "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
log "Server PID ${SERVER_PID}; log=${SERVER_LOG}"

elapsed=0
while [[ "${elapsed}" -lt "${MAX_WAIT_SECS}" ]]; do
    if curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        log "Server ready after ${elapsed}s"
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        log "Server died. Last log lines:"
        tail -80 "${SERVER_LOG}" || true
        exit 1
    fi
    sleep 5
    elapsed=$((elapsed + 5))
done
if [[ "${elapsed}" -ge "${MAX_WAIT_SECS}" ]]; then
    log "Server start timed out. Last log lines:"
    tail -80 "${SERVER_LOG}" || true
    exit 1
fi

log "Sending GPQA prompts (max_tokens=1) to trigger the Q/K/V dump hook"
"${PY_EVAL}" ${REPO_ROOT}/rotation/_eval_runner/dump_gpqa_prompts.py \
    --model "${MODEL}" \
    --base-url "http://127.0.0.1:${PORT}/v1" \
    --num-prompts 198 \
    --num-threads "${NUM_WORKERS:-32}" \
    --temperature 0.6 --top-p 0.95 --top-k 40 \
    --max-tokens 1 \
    2>&1 | tee "${DUMP_RUNNER_LOG}"

log "Dump complete"
log "  dump_dir=${DUMP_KVCACHE_DIR}"
log "  server_log=${SERVER_LOG}"
log "  dump_runner_log=${DUMP_RUNNER_LOG}"
if [[ -d "${DUMP_KVCACHE_DIR}/layer_0/q" ]]; then
    log "layer_0 q chunks:"
    ls "${DUMP_KVCACHE_DIR}/layer_0/q" | head -20
else
    log "Warning: ${DUMP_KVCACHE_DIR}/layer_0/q was not created"
    exit 0
fi

# Post-process: count prompts captured, rename `latest` to seq<T>_prompt<N>_group<G>.
N_PROMPTS=$("${PY}" - "${DUMP_KVCACHE_DIR}/layer_0/seq_lens" <<'PYEOF'
import os, sys, torch
seq_dir = sys.argv[1]
total = 0
for f in sorted(os.listdir(seq_dir), key=lambda x: int(x.split('.')[0])):
    s = torch.load(os.path.join(seq_dir, f), weights_only=True, map_location='cpu')
    total += len(s.tolist())
print(total)
PYEOF
)
log "  prompts_captured=${N_PROMPTS}"

FINAL_TAG="seq${DUMP_KVCACHE_TOKENS}_prompt${N_PROMPTS}_group${GROUP_SIZE}"
FINAL_DIR="${SCRIPT_DIR}/${DATASET}/${FINAL_TAG}"
if [[ "${CALIB_DIR}" != "${FINAL_DIR}" ]]; then
    rm -rf "${FINAL_DIR}"
    mv "${CALIB_DIR}" "${FINAL_DIR}"
fi
log "  final_dir=${FINAL_DIR}"
