#!/usr/bin/env bash
# Phase 1 — dump post-norm/post-RoPE Q/K/V for OSCAR calibration on gemma-4-12B-it.
#
# gemma-4-12B-it is `model_type: gemma4_unified` with heterogeneous per-layer KV
# geometry (40 sliding layers @ 8 KV heads x head_dim 256; 8 full layers @ 1 KV
# head x head_dim 512). It is served via this fork's `Gemma4UnifiedForConditionalGeneration`
# shim. Unlike the other OSCAR models, the calibration dump runs in the EVAL fork
# (sglang-research) using the DUMP_KVCACHE hook ported into triton_backend.py
# forward_extend — the legacy dump fork has no gemma4 at all.
#
# ENV PREREQUISITE: transformers >= 5.5 (has Gemma4TextConfig; the env's default
# 5.3.0 cannot import gemma4). Point PY at a python with transformers 5.5-5.9.
set -euo pipefail
export HF_HOME="${HF_HOME:-/shared/huggingface}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SGLANG_RESEARCH_DIR="${SGLANG_RESEARCH_DIR:-${REPO_ROOT}/sglang-research}"

MODEL="${MODEL:-google/gemma-4-12B-it}"
PY="${PY:?Set PY to a python with transformers>=5.5 + sglang deps (e.g. the oscar-g4 venv)}"
GPU="${GPU:-0}"
PORT="${PORT:-31060}"
DIST_PORT="${DIST_PORT:-41060}"
MEM_FRAC="${MEM_FRACTION_STATIC:-0.85}"
DUMP_KVCACHE_TOKENS="${DUMP_KVCACHE_TOKENS:-30000}"
GROUP_SIZE="${GROUP_SIZE:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-198}"
MAX_WAIT_SECS="${MAX_WAIT_SECS:-900}"

DATASET="${DATASET:-GPQA}"
CALIB_DIR="${SCRIPT_DIR}/${DATASET}/latest"
DUMP_DIR="${CALIB_DIR}/qkv_dumps/gpqa"
mkdir -p "${DUMP_DIR}"

export PYTHONPATH="${SGLANG_RESEARCH_DIR}/python:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${GPU}"
export DUMP_KVCACHE=1
export DUMP_KVCACHE_TOKENS
export DUMP_KVCACHE_DIR="${DUMP_DIR}"

SERVER_LOG="${DUMP_DIR}/server.log"
: > "${SERVER_LOG}"
log() { echo "[$(date '+%F %T')] $*"; }

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        for pid in $(nvidia-smi -i "${GPU}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
            kill -TERM "$pid" 2>/dev/null || true
        done
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

log "Launching gemma-4-12B-it dump server (triton prefill, DUMP_KVCACHE=1)"
"${PY}" -m sglang.launch_server \
    --model-path "${MODEL}" \
    --tensor-parallel-size 1 \
    --context-length 8192 \
    --mem-fraction-static "${MEM_FRAC}" \
    --prefill-attention-backend triton \
    --decode-attention-backend triton \
    --disable-cuda-graph --disable-piecewise-cuda-graph \
    --trust-remote-code \
    --host 127.0.0.1 --port "${PORT}" \
    --dist-init-addr "127.0.0.1:${DIST_PORT}" >> "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

elapsed=0
while [[ "${elapsed}" -lt "${MAX_WAIT_SECS}" ]]; do
    curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { log "Server ready (${elapsed}s)"; break; }
    kill -0 "${SERVER_PID}" 2>/dev/null || { log "Server died:"; tail -40 "${SERVER_LOG}"; exit 1; }
    sleep 5; elapsed=$((elapsed + 5))
done

log "Sending ${NUM_PROMPTS} GPQA prompts (max_tokens=1, prefill-only)"
"${PY}" "${REPO_ROOT}/rotation/_eval_runner/dump_gpqa_prompts.py" \
    --model "${MODEL}" --base-url "http://127.0.0.1:${PORT}/v1" \
    --num-prompts "${NUM_PROMPTS}" --num-threads 24 --max-tokens 1

# Rename latest -> seq<T>_prompt<N>_group<G>
if [[ -d "${DUMP_DIR}/layer_0/q" ]]; then
    N=$("${PY}" - "${DUMP_DIR}/layer_0/seq_lens" <<'PYEOF'
import os, sys, torch
d = sys.argv[1]; t = 0
for f in os.listdir(d):
    t += len(torch.load(os.path.join(d, f), weights_only=True, map_location="cpu").tolist())
print(t)
PYEOF
)
    FINAL="${SCRIPT_DIR}/${DATASET}/seq${DUMP_KVCACHE_TOKENS}_prompt${N}_group${GROUP_SIZE}"
    rm -rf "${FINAL}"; mv "${CALIB_DIR}" "${FINAL}"
    log "Dump complete -> ${FINAL}"
fi
