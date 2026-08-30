#!/usr/bin/env bash
# Dump post-RoPE Q/K/V on GPQA for Qwen3-30B-A3B calibration.
#
# The default server limits are too small for 30k-token dumps on this model:
# the scheduler hits "alloc_req_slots runs out of memory", drops every prompt
# but the first, and the dump silently ends up with a dozen tokens. The fits
# below then succeed and serve garbage. Hence the explicit context/queue caps
# and the ok=/err= assertion at the end.
set -euo pipefail
export HF_HOME="${HF_HOME:-/shared/huggingface}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SGLANG_DUMP_DIR="${SGLANG_DUMP_DIR:-${REPO_ROOT}/sglang-dump-qkv}"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
TP_SIZE="${TP_SIZE:-2}"
PORT="${PORT:-31060}"
GROUP_SIZE="${GROUP_SIZE:-128}"
DATASET="${DATASET:-GPQA}"

# --- the caps that make the dump complete ---
CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-8}"
MAX_QUEUED_REQUESTS="${MAX_QUEUED_REQUESTS:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"          # must be <= MAX_RUNNING_REQUESTS
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"

export DUMP_KVCACHE=true
export DUMP_KVCACHE_TOKENS="${DUMP_KVCACHE_TOKENS:-30000}"
CALIB_DIR="${CALIB_DIR:-${SCRIPT_DIR}/${DATASET}/latest}"
export DUMP_KVCACHE_DIR="${CALIB_DIR}/qkv_dumps/gpqa"
mkdir -p "${DUMP_KVCACHE_DIR}"

PY="${PY:-${HOME}/miniconda3/envs/oscar/bin/python3}"
export PYTHONPATH="${REPO_ROOT}/rotation/_dump_compat:${SGLANG_DUMP_DIR}/python:${PYTHONPATH:-}"

echo "[dump] model=${MODEL} tp=${TP_SIZE} out=${DUMP_KVCACHE_DIR}"
"${PY}" -m sglang.launch_server --model-path "${MODEL}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --context-length "${CONTEXT_LENGTH}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --max-queued-requests "${MAX_QUEUED_REQUESTS}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --host 127.0.0.1 --port "${PORT}" --trust-remote-code \
  > "${CALIB_DIR}/dump_server.log" 2>&1 &
SP=$!
trap 'kill -KILL ${SP} 2>/dev/null || true' EXIT

for _ in $(seq 1 240); do
  curl -s "localhost:${PORT}/health" >/dev/null 2>&1 && break
  kill -0 ${SP} 2>/dev/null || { echo "[dump] server died"; tail -40 "${CALIB_DIR}/dump_server.log"; exit 1; }
  sleep 5
done

"${PY}" "${SCRIPT_DIR}/../_eval_runner/dump_gpqa_prompts.py" \
  --model "${MODEL}" --base-url "http://127.0.0.1:${PORT}/v1" \
  --num-prompts 198 --num-threads "${NUM_WORKERS}" \
  --temperature 0.6 --top-p 0.95 --top-k 40 --max-tokens 1 \
  2>&1 | tee "${CALIB_DIR}/dump_client.log"

# A partial dump is the failure mode that costs days. Refuse to hand it on.
OK="$(grep -oE 'ok=[0-9]+' "${CALIB_DIR}/dump_client.log" | tail -1 | cut -d= -f2)"
ERR="$(grep -oE 'err=[0-9]+' "${CALIB_DIR}/dump_client.log" | tail -1 | cut -d= -f2)"
echo "[dump] ok=${OK:-?} err=${ERR:-?}"
if [[ "${ERR:-1}" != "0" ]]; then
  echo "[dump] ABORT: ${ERR} prompts failed; the dump is partial. Lower NUM_WORKERS" >&2
  echo "       or CONTEXT_LENGTH and re-run before fitting." >&2
  exit 1
fi
echo "[dump] tokens captured:"
"${PY}" - <<PY
import glob, torch
fs = sorted(glob.glob("${DUMP_KVCACHE_DIR}/layer_0/k/*.pt"))
n = 0
for f in fs:
    t = torch.load(f, map_location="cpu")
    if isinstance(t, dict):
        t = t.get("k", next(iter(t.values())))
    n += t.shape[0]
print(f"  layer_0 k chunks={len(fs)} tokens={n}")
assert n > 5000, (
    f"only {n} tokens captured -- a partial dump fits and serves garbage. "
    "See the caps in this script's header."
)
PY
