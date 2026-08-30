#!/usr/bin/env bash
# One INT2 arm of the Qwen3.5-35B-A3B prefix-cache A/B (CUDA graph on).
#
# Required env: ARM MODEL ROT_DIR RUN_DIR DISABLE_RADIX TP_SIZE PORT DIST_PORT
# Optional:     LLOYD_MAX GROUP_SIZE SEEDS NUM_EXAMPLES MIXED_KV_AUDIT NUM_WORKERS
#               EXTRA_SERVER_ARGS MEM_FRAC MAX_RUNNING CUDA_GRAPH_MAX_BS
#
# Carried over from investigation/06_qwen35_radix_cache/run_arm.sh (the 4B A/B).
# Two reasons this exists rather than calling eval_oscar_gpqa.sh directly:
#
# 1. Every setting that silently yields a plausible-but-wrong number is read
#    back out of the server's own log. Above all the prefix-cache state, which
#    is the whole point of the A/B -- eval_oscar_gpqa.sh defaults it ON now, so
#    inferring it from the script default is exactly the mistake that would
#    make both arms identical. On disagreement the seed is RELABELLED, not
#    aborted: the server's report is the truth about what ran, and tearing down
#    a server that has already loaded 67 GB of weights costs far more than
#    relabelling.
#
# 2. A fresh server per seed. Sharing one server across seeds would let the
#    second and third seed hit the *first* seed's cached copies of the very
#    same 198 prompts, so the hit rate would climb far above what this workload
#    can actually reach and stop being comparable to anything.
#
# The cache-ON arm needs --mamba-scheduler-strategy extra_buffer (via
# EXTRA_SERVER_ARGS): MambaRadixCache v1 asserts page_size == 1 while the INT2
# unified pool needs page_size == N_Q == 8. extra_buffer also raises ValueError
# together with --disable-radix-cache, so the cache-OFF control necessarily
# runs no_buffer. That asymmetry is forced by the server, not chosen here, and
# is reported in the table.
set -uo pipefail

: "${ARM:?ARM required}"
: "${MODEL:?MODEL required}"
: "${ROT_DIR:?ROT_DIR required}"
: "${RUN_DIR:?RUN_DIR required}"
: "${DISABLE_RADIX:?DISABLE_RADIX required (0=cache on, 1=cache off)}"

W="${W:?W required (workspace CoQuant checkout)}"
export HF_HOME=/shared/huggingface
export HF_DATASETS_CACHE=/shared/huggingface/datasets
export SGLANG_RESEARCH_DIR="${W}/sglang-research"

SEEDS="${SEEDS:-0 1 2}"
WANT_RADIX="False"; [[ "${DISABLE_RADIX}" == "1" ]] && WANT_RADIX="True"

mkdir -p "${RUN_DIR}"
SUMMARY="${RUN_DIR}/arm_summary.txt"
: > "${SUMMARY}"
log() { echo "[arm:${ARM}] $*" | tee -a "${SUMMARY}"; }

log "model=${MODEL} tp=${TP_SIZE} disable_radix=${DISABLE_RADIX} rot=${ROT_DIR}"
log "group=${GROUP_SIZE:-256} lloyd_max=${LLOYD_MAX:-0} absorb_v=${ABSORB_V:-0} audit=${MIXED_KV_AUDIT:-0}"
# Overlap schedule is left at its default (ON) in BOTH INT2 arms. sglang
# force-disables it for a hybrid mamba model only under the no_buffer strategy
# that the radix cache used to require; extra_buffer keeps it on, so the pair
# does not differ there and the wall-clock column stays meaningful.
log "extra_server_args=${EXTRA_SERVER_ARGS:-<none>} seeds=${SEEDS}"
log "commit=$(git -C "${W}" rev-parse HEAD) dirty=$(git -C "${W}" status --porcelain | wc -l)"

ARM_START=$(date +%s)
RC_ALL=0

for SEED in ${SEEDS}; do
    SRD="${RUN_DIR}/seed${SEED}"
    if [[ -f "${SRD}/metrics.json" ]]; then
        log "seed=${SEED} already has metrics.json -- skipping"
        continue
    fi
    mkdir -p "${SRD}"
    log "=== seed=${SEED} start ==="
    START=$(date +%s)

    env \
      MODEL="${MODEL}" ROT_DIR="${ROT_DIR}" RUN_DIR="${SRD}" \
      TP_SIZE="${TP_SIZE}" GPUS="${GPUS:-0}" \
      PORT="${PORT}" DIST_PORT="${DIST_PORT}" \
      DISABLE_RADIX="${DISABLE_RADIX}" \
      GROUP_SIZE="${GROUP_SIZE:-256}" \
      LLOYD_MAX="${LLOYD_MAX:-0}" \
      ABSORB_V="${ABSORB_V:-0}" \
      MEM_FRAC="${MEM_FRAC:-0.80}" \
      MAX_RUNNING="${MAX_RUNNING:-32}" \
      CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-32}" \
      NUM_WORKERS="${NUM_WORKERS:-32}" \
      MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}" \
      MIXED_KV_AUDIT="${MIXED_KV_AUDIT:-0}" \
      MIXED_KV_AUDIT_EVERY="${MIXED_KV_AUDIT_EVERY:-25}" \
      SEED="${SEED}" \
      CONDA_ENV_NAME="${CONDA_ENV_NAME:-oscar}" \
      CONDA_BASE="${CONDA_BASE:-/home/charlie/miniconda3}" \
      bash "${W}/rotation/eval_oscar_gpqa.sh" &
    CHILD=$!

    # ---- read the config back out of the server, relabel on disagreement ----
    CONFIRMED=0
    for _ in $(seq 1 480); do
        if grep -q "disable_radix_cache=" "${SRD}/server.log" 2>/dev/null; then
            GOT=$(grep -o "disable_radix_cache=\(True\|False\)" "${SRD}/server.log" | head -1 | cut -d= -f2)
            log "seed=${SEED} server reports disable_radix_cache=${GOT} (intended ${WANT_RADIX})"
            echo "${GOT}" > "${SRD}/observed_disable_radix_cache"
            if [[ "${GOT}" != "${WANT_RADIX}" ]]; then
                log "RELABEL seed=${SEED}: server says disable_radix_cache=${GOT}, intent was ${WANT_RADIX}; continuing under the server's label"
            fi
            CONFIRMED=1
            break
        fi
        kill -0 "${CHILD}" 2>/dev/null || break
        sleep 5
    done
    [[ "${CONFIRMED}" == "1" ]] || log "WARN seed=${SEED}: never saw disable_radix_cache= in server.log"

    # Rotation provenance: which file, which layers. A missing rotation does not
    # error, it serves at Hadamard quality, so the filename has to be seen.
    for _ in $(seq 1 240); do
        if grep -q "Loaded Oscar rotation from" "${SRD}/server.log" 2>/dev/null; then
            grep -h "Loaded Oscar rotation from" "${SRD}/server.log" | head -4 \
                | sed "s#^#[arm:${ARM}] seed${SEED} rot| #" | tee -a "${SUMMARY}"
            break
        fi
        kill -0 "${CHILD}" 2>/dev/null || break
        sleep 5
    done

    wait "${CHILD}"; RC=$?
    END=$(date +%s)
    echo $((END - START)) > "${SRD}/wall_seconds"
    log "seed=${SEED} rc=${RC} wall=$((END - START))s score=$(grep 'gpqa/score ' "${SRD}/eval.log" 2>/dev/null | head -1 | awk -F'|' '{print $3}' | tr -d ' ')"
    [[ "${RC}" == "0" ]] || RC_ALL="${RC}"

    grep -h "\[mixed-kv-prefix\]" "${SRD}/server.log" 2>/dev/null | head -6 \
        | sed "s#^#[arm:${ARM}] seed${SEED} mixin| #" | tee -a "${SUMMARY}"

    grep -o "Capture cuda graph bs \[[0-9, ]*\]" "${SRD}/server.log" 2>/dev/null | head -1 \
        | sed "s#^#[arm:${ARM}] seed${SEED} capt| #" | tee -a "${SUMMARY}"

    grep -o "kv_cache_quant_group_size=[0-9]*\|kv_cache_dtype='[^']*'\|cuda_graph_max_bs=[0-9]*\|max_running_requests=[0-9]*\|mem_fraction_static=[0-9.]*\|disable_radix_cache=[A-Za-z]*\|disable_cuda_graph=[A-Za-z]*\|disable_overlap_schedule=[A-Za-z]*\|page_size=[0-9]*\|mamba_scheduler_strategy='[^']*'" \
        "${SRD}/server.log" 2>/dev/null | sort -u | tr '\n' ' ' \
        | sed "s#^#[arm:${ARM}] seed${SEED} cfg| #" | tee -a "${SUMMARY}"
    echo | tee -a "${SUMMARY}"
done

for SEED in ${SEEDS}; do
    SRD="${RUN_DIR}/seed${SEED}"
    [[ -d "${SRD}" ]] || continue
    N=$(grep -c "\[mixed-kv-audit\]" "${SRD}/server.log" 2>/dev/null || echo 0)
    V=$(grep -o "\[mixed-kv-audit\] [A-Z_]*" "${SRD}/server.log" 2>/dev/null | grep -vc "HP_PREFIX_" || echo 0)
    E=$(grep -c "Scheduler hit an exception" "${SRD}/server.log" 2>/dev/null || echo 0)
    log "seed=${SEED} audit_lines=${N} audit_violations=${V} scheduler_exceptions=${E}"
    if [[ "${E}" != "0" ]]; then
        grep -A25 "Scheduler hit an exception" "${SRD}/server.log" | head -30 \
            | sed "s#^#[arm:${ARM}] seed${SEED} exc| #" | tee -a "${SUMMARY}"
    fi
done
log "arm finished rc=${RC_ALL} wall=$(( $(date +%s) - ARM_START ))s"
exit "${RC_ALL}"
