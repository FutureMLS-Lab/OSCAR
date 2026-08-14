#!/usr/bin/env bash
# Dump post-RoPE Q/K/V for GPQA calibration on MiniMax-M3.
#
# Unlike the MHA models, M3 dumps through a hook in the model file itself
# (DUMP_M3_QKV_DIR in minimax_m3.py), not through the separate sglang-dump-qkv
# build -- so run this against the research checkout.
#
# EVERY TP rank writes its own shard: layer_<id>/rank_<r>/{q,k,v}/. A rank-0
# dump sees 4 of 64 query heads and 1 of 4 KV heads and fits a rotation no
# better than Hadamard, silently. Run this on all nodes, then merge with
# merge_qkv_allhead.py before fitting.
#
# Two nodes, TP=16. Set NODE_RANK and DIST_ADDR per node.
set -euo pipefail
export HF_HOME="${HF_HOME:-/shared/huggingface}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

MODEL="${MODEL:-MiniMaxAI/MiniMax-M3}"
REVISION="${REVISION:-}"          # pin it; see README on the refs/main hazard
TP_SIZE="${TP_SIZE:-16}"
NNODES="${NNODES:-2}"
NODE_RANK="${NODE_RANK:?set NODE_RANK (0 on the head, 1 on the worker)}"
DIST_ADDR="${DIST_ADDR:?set DIST_ADDR to <head-ip>:<port>}"
PORT="${PORT:-31150}"
NUM_PROMPTS="${NUM_PROMPTS:-45}"
NUM_THREADS="${NUM_THREADS:-16}"
MAX_WAIT_SECS="${MAX_WAIT_SECS:-3600}"

DATASET="${DATASET:-GPQA}"
CALIB_DIR="${CALIB_DIR:-${SCRIPT_DIR}/${DATASET}/latest}"
export DUMP_M3_QKV_DIR="${DUMP_M3_QKV_DIR:-${CALIB_DIR}/qkv_dumps_perrank}"
mkdir -p "${DUMP_M3_QKV_DIR}"

export PYTHONPATH="${REPO_ROOT}/sglang-research/python:${PYTHONPATH:-}"
LOG="${DUMP_M3_QKV_DIR}/server_rank${NODE_RANK}.log"

REV_ARG=()
[[ -n "${REVISION}" ]] && REV_ARG=(--revision "${REVISION}")

echo "[dump] node_rank=${NODE_RANK} tp=${TP_SIZE} dir=${DUMP_M3_QKV_DIR}"
python -m sglang.launch_server --model-path "${MODEL}" "${REV_ARG[@]}" \
    --tensor-parallel-size "${TP_SIZE}" --nnodes "${NNODES}" \
    --node-rank "${NODE_RANK}" --dist-init-addr "${DIST_ADDR}" \
    --prefill-attention-backend fa3 --decode-attention-backend triton \
    --disable-cuda-graph --context-length 40960 \
    --mem-fraction-static 0.85 --max-running-requests 16 \
    --trust-remote-code --host 0.0.0.0 --port "${PORT}" > "${LOG}" 2>&1 &
SERVER_PID=$!

# Only the head drives the prompts; workers just serve their shard.
if [[ "${NODE_RANK}" != "0" ]]; then
    wait ${SERVER_PID}
    exit 0
fi

for _ in $(seq 1 $((MAX_WAIT_SECS / 5))); do
    curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
    kill -0 ${SERVER_PID} 2>/dev/null || { echo "[dump] server died"; tail -40 "${LOG}"; exit 1; }
    sleep 5
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 || {
    echo "[dump] server never became healthy"; tail -40 "${LOG}"; exit 1; }

# max_tokens=1: the dump happens during prefill, generation is not needed.
python3 "${REPO_ROOT}/rotation/_eval_runner/dump_gpqa_prompts.py" \
    --port "${PORT}" --num-prompts "${NUM_PROMPTS}" --num-threads "${NUM_THREADS}" \
    2>&1 | tee "${DUMP_M3_QKV_DIR}/dump_runner.log"

kill ${SERVER_PID} 2>/dev/null || true

echo "[dump] ranks present for layer_3: $(ls -d ${DUMP_M3_QKV_DIR}/layer_3/rank_* 2>/dev/null | wc -l) / ${TP_SIZE}"
echo "[dump] next: merge_qkv_allhead.py --tp ${TP_SIZE} --num-q-heads 64 --num-kv-heads 4"
