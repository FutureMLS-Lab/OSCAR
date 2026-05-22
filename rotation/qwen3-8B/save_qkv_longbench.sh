#!/usr/bin/env bash
# Phase 1d: Dump post-RoPE Q/K/V tensors using LongBench-style long-context
# prompts (8K-28K tokens from gov_report_e, 2wikimqa_e, hotpotqa_e,
# multifieldqa_en_e). Target ~150K tokens of activations for a more
# data-matched rotation than the GPQA calibration (~30K tokens, 500-token
# MCQ prompts).
set -euo pipefail
export HF_HOME="${HF_HOME:-/shared/huggingface}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SGLANG_DUMP_DIR="${SGLANG_DUMP_DIR:-${REPO_ROOT}/sglang-dump-qkv}"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
TP_SIZE="${TP_SIZE:-1}"
PORT="${PORT:-31050}"
DIST_PORT="${DIST_PORT:-41050}"
GPU="${GPU:-0}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
# LongBench prompts are 8K-28K tokens. Lower concurrency than GPQA to
# avoid running out of activation memory during the long prefill pass.
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-4}"
MAX_QUEUED_REQUESTS="${MAX_QUEUED_REQUESTS:-32}"
MAX_WAIT_SECS="${MAX_WAIT_SECS:-1800}"

# Budget enough activations to cover long-context distribution.
export DUMP_KVCACHE="${DUMP_KVCACHE:-true}"
export DUMP_KVCACHE_TOKENS="${DUMP_KVCACHE_TOKENS:-150000}"

# Calibration layout: <model>/<DATASET>/seq<TOK>_group<G>/{qkv_dumps,rotations}
# CALIB_BASE_DIR defaults to the SCRIPT_DIR (worktree). Override (e.g. in
# the K8S job) to the main-checkout rotation/ path so the dumped tensors and
# computed rotations persist outside the worktree.
DATASET="${DATASET:-LongBench}"
GROUP_SIZE="${GROUP_SIZE:-128}"
CALIB_BASE_DIR="${CALIB_BASE_DIR:-${SCRIPT_DIR}}"
CALIB_DIR="${CALIB_BASE_DIR}/${DATASET}/latest"
export DUMP_KVCACHE_DIR="${DUMP_KVCACHE_DIR:-${CALIB_DIR}/qkv_dumps/longbench}"
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
    --watchdog-timeout 3600
)
if [[ -n "${EXTRA_SERVER_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    SERVER_ARGS+=(${EXTRA_SERVER_ARGS})
fi

log "Starting sglang server for QKV dump (LongBench calibration)"
log "  sglang_dump=${SGLANG_DUMP_DIR}"
log "  model=${MODEL}"
log "  dump_dir=${DUMP_KVCACHE_DIR}"
log "  dump_tokens=${DUMP_KVCACHE_TOKENS}"

PYTHONPATH="${LOCAL_PYTHONPATH}" \
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
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

NUM_PROMPTS="${NUM_PROMPTS:-25}"
NUM_WORKERS="${NUM_WORKERS:-4}"
log "Sending ${NUM_PROMPTS} long-context LongBench prompts (max_tokens=1)"
"${PY_EVAL}" ${REPO_ROOT}/rotation/_eval_runner/dump_longbench_prompts.py \
    --model "${MODEL}" \
    --base-url "http://127.0.0.1:${PORT}/v1" \
    --num-prompts "${NUM_PROMPTS}" \
    --num-threads "${NUM_WORKERS}" \
    --max-input-len 28000 \
    --min-context-len 8000 \
    --max-tokens 1 \
    --use-chat true \
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
fi

# Post-process: count prompts, rename calib dir to seq<T>_prompt<N>_group<G>
if [[ -d "${DUMP_KVCACHE_DIR}/layer_0/q" ]]; then
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
    FINAL_DIR="${CALIB_BASE_DIR}/${DATASET}/${FINAL_TAG}"
    if [[ "${CALIB_DIR}" != "${FINAL_DIR}" ]]; then
        rm -rf "${FINAL_DIR}"
        mv "${CALIB_DIR}" "${FINAL_DIR}"
    fi
    log "  final_dir=${FINAL_DIR}"

    # Auto-compute rotation right after dump (no second job needed).
    if [[ "${AUTO_COMPUTE_ROTATION:-1}" == "1" ]]; then
        ROT_OUT="${FINAL_DIR}/rotations"
        mkdir -p "${ROT_OUT}"
        log "Computing OSCAR rotation (qqt_sst, composition r_h_pbr) ..."
        "${PY}" "${REPO_ROOT}/rotation/compute_kv_rotation.py" \
            --dump-path "${FINAL_DIR}/qkv_dumps/longbench" \
            --output-dir "${ROT_OUT}" \
            --method qqt_sst \
            --composition r_h_pbr \
            --head-dim 128 \
            2>&1 | tee "${ROT_OUT}/compute.log"
        log "  rotation_dir=${ROT_OUT}"
    fi
fi
