#!/usr/bin/env bash
# Same-session BF16 paired arm for the Qwen3.5 INT2 runs.
#
# Matched to run_arm.sh's INT2 arm on everything a BF16 server can be matched
# on -- same seeds, same client concurrency, same max-running-requests, same
# mem fraction, same CUDA-graph capture list, prefix cache OFF, overlap
# scheduler left at its default (ON) in BOTH arms, same attention backends
# (fa3 prefill / triton decode).
# What cannot be matched is intrinsic to the KV format: no rotation, no
# HP windows, and page_size 1 instead of the 8 the INT2 unified pool requires.
#
# Required env: ARM MODEL RUN_DIR TP_SIZE PORT DIST_PORT
set -uo pipefail

: "${ARM:?ARM required}"
: "${MODEL:?MODEL required}"
: "${RUN_DIR:?RUN_DIR required}"

W="${W:?W required (workspace CoQuant checkout)}"
export HF_HOME=/shared/huggingface
export HF_DATASETS_CACHE=/shared/huggingface/datasets
SGLANG_RESEARCH_DIR="${SGLANG_RESEARCH_DIR:-${W}/sglang-research}"

SEEDS="${SEEDS:-0 1 2}"
TP_SIZE="${TP_SIZE:-1}"
GPUS="${GPUS:-0}"
MEM_FRAC="${MEM_FRAC:-0.80}"
MAX_RUNNING="${MAX_RUNNING:-32}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-32}"
NUM_WORKERS="${NUM_WORKERS:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"

CONDA_BASE="${CONDA_BASE:-/home/charlie/miniconda3}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-oscar}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${W}/rotation/_triton_per_rank:${SGLANG_RESEARCH_DIR}/python:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "${RUN_DIR}"
SUMMARY="${RUN_DIR}/arm_summary.txt"
: > "${SUMMARY}"
log() { echo "[arm:${ARM}] $*" | tee -a "${SUMMARY}"; }

log "model=${MODEL} tp=${TP_SIZE} BF16 baseline, prefix cache OFF, seeds=${SEEDS}"
log "commit=$(git -C "${W}" rev-parse HEAD)"

ARM_START=$(date +%s)
RC_ALL=0

for SEED in ${SEEDS}; do
    SRD="${RUN_DIR}/seed${SEED}"
    [[ -f "${SRD}/metrics.json" ]] && { log "seed=${SEED} already done"; continue; }
    mkdir -p "${SRD}"
    LOG_SERVER="${SRD}/server.log"; : > "${LOG_SERVER}"
    export OSCAR_TRITON_PER_RANK_BASE="${SRD}/triton_cache"
    export TRITON_CACHE_DIR="${OSCAR_TRITON_PER_RANK_BASE}/main"
    mkdir -p "${TRITON_CACHE_DIR}"

    log "=== seed=${SEED} start ==="
    START=$(date +%s)

    SERVER_ARGS=(
        --model-path "${MODEL}"
        --tensor-parallel-size "${TP_SIZE}"
        --kv-cache-dtype auto
        --attention-backend fa3
        --prefill-attention-backend fa3
        --decode-attention-backend triton
        --mem-fraction-static "${MEM_FRAC}"
        --max-running-requests "${MAX_RUNNING}"
        --disable-radix-cache
        --enable-cache-report
        --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}"
        --host 127.0.0.1 --port "${PORT}"
        --dist-init-addr "127.0.0.1:${DIST_PORT}"
        --trust-remote-code
    )
    CUDA_VISIBLE_DEVICES="${GPUS}" \
        python -m sglang.launch_server "${SERVER_ARGS[@]}" >> "${LOG_SERVER}" 2>&1 &
    SERVER_PID=$!

    READY=0
    for _ in $(seq 1 300); do
        if curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then READY=1; break; fi
        kill -0 "${SERVER_PID}" 2>/dev/null || break
        sleep 5
    done
    if [[ "${READY}" != "1" ]]; then
        log "seed=${SEED} server never came up"
        tail -40 "${LOG_SERVER}" | sed "s#^#[arm:${ARM}] srv| #" | tee -a "${SUMMARY}"
        RC_ALL=1
        kill -KILL "${SERVER_PID}" 2>/dev/null; pkill -KILL -P "${SERVER_PID}" 2>/dev/null
        continue
    fi

    GOT=$(grep -o "disable_radix_cache=\(True\|False\)" "${LOG_SERVER}" | head -1 | cut -d= -f2)
    log "seed=${SEED} server reports disable_radix_cache=${GOT} (intended True)"
    if [[ "${GOT}" != "True" ]]; then
        log "FATAL: BF16 control arm has the prefix cache ON -- aborting"
        kill -KILL "${SERVER_PID}" 2>/dev/null; pkill -KILL -P "${SERVER_PID}" 2>/dev/null
        exit 3
    fi

    python "${W}/rotation/_eval_runner/run_simple_eval.py" \
        --task gpqa --model "${MODEL}" \
        --base-url "http://127.0.0.1:${PORT}/v1" \
        --max-tokens "${MAX_NEW_TOKENS}" \
        --temperature 1.0 --top-p 0.95 --top-k 40 \
        --n-repeats 1 --seed "${SEED}" \
        --num-threads "${NUM_WORKERS}" \
        --output-dir "${SRD}" > "${SRD}/runner.log" 2>&1
    RC=$?

    kill -TERM "${SERVER_PID}" 2>/dev/null; pkill -TERM -P "${SERVER_PID}" 2>/dev/null
    sleep 5
    kill -KILL "${SERVER_PID}" 2>/dev/null; pkill -KILL -P "${SERVER_PID}" 2>/dev/null

    END=$(date +%s)
    echo $((END - START)) > "${SRD}/wall_seconds"
    log "seed=${SEED} rc=${RC} wall=$((END - START))s score=$(grep 'gpqa/score ' "${SRD}/eval.log" 2>/dev/null | head -1 | awk -F'|' '{print $3}' | tr -d ' ')"
    [[ "${RC}" == "0" ]] || RC_ALL="${RC}"

    grep -o "kv_cache_dtype='[^']*'\|cuda_graph_max_bs=[0-9]*\|max_running_requests=[0-9]*\|mem_fraction_static=[0-9.]*\|disable_radix_cache=[A-Za-z]*\|disable_cuda_graph=[A-Za-z]*\|disable_overlap_schedule=[A-Za-z]*\|page_size=[0-9]*" \
        "${LOG_SERVER}" 2>/dev/null | sort -u | tr '\n' ' ' \
        | sed "s#^#[arm:${ARM}] seed${SEED} cfg| #" | tee -a "${SUMMARY}"
    echo | tee -a "${SUMMARY}"
done

log "arm finished rc=${RC_ALL} wall=$(( $(date +%s) - ARM_START ))s"
exit "${RC_ALL}"
