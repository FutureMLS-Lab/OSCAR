#!/usr/bin/env bash
# Start ONE sglang server (bf16 | calibrated-OSCAR) and run a matrix of
# {benchmarks} x {seeds} against it, writing metrics to
#   ${OUT_BASE}/<bench>/seed<seed>/metrics.json
#
# Required env: MODE (bf16|calibrated), MODEL, OUT_BASE
# Optional: TP_SIZE, GPUS, ROT_DIR (calibrated), BENCHES, SEEDS,
#   MAX_NEW_TOKENS, NUM_EXAMPLES (smoke), NUM_WORKERS, PORT, DIST_PORT
set -uo pipefail
export HF_HOME="${HF_HOME:-/shared/huggingface}"
# Force the real datasets cache location (AIME loads via HF datasets). The
# manifest may pass HF_DATASETS_CACHE=/shared/huggingface, but the cached
# datasets actually live under .../datasets/ — point there explicitly.
export HF_DATASETS_CACHE=/shared/huggingface/datasets

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SGLANG_RESEARCH_DIR="${SGLANG_RESEARCH_DIR:-${REPO_ROOT}/sglang-research}"

: "${MODE:?MODE=bf16|calibrated}"; : "${MODEL:?MODEL required}"; : "${OUT_BASE:?OUT_BASE required}"
TP_SIZE="${TP_SIZE:-1}"
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
PORT="${PORT:-31080}"; DIST_PORT="${DIST_PORT:-41080}"
BENCHES="${BENCHES:-gpqa humaneval aime25 math500}"
SEEDS="${SEEDS:-0 1 2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
NUM_WORKERS="${NUM_WORKERS:-32}"
MEM_FRAC="${MEM_FRAC:-0.85}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-32}"

CONDA_BASE="${CONDA_BASE:-${HOME}/miniconda3}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"; conda activate "${CONDA_ENV_NAME:-oscar}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/rotation/_triton_per_rank:${SGLANG_RESEARCH_DIR}/python:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "${OUT_BASE}"
LOG_SERVER="${OUT_BASE}/server.log"; : > "${LOG_SERVER}"
export OSCAR_TRITON_PER_RANK_BASE="${OUT_BASE}/triton_cache"
export TRITON_CACHE_DIR="${OSCAR_TRITON_PER_RANK_BASE}/main"
mkdir -p "${TRITON_CACHE_DIR}"

cleanup(){ if [[ -n "${SERVER_PID:-}" ]]; then kill -TERM "${SERVER_PID}" 2>/dev/null||true; pkill -TERM -P "${SERVER_PID}" 2>/dev/null||true; sleep 2; kill -KILL "${SERVER_PID}" 2>/dev/null||true; pkill -KILL -P "${SERVER_PID}" 2>/dev/null||true; fi; }
trap cleanup EXIT INT TERM

COMMON_ARGS=( --model-path "${MODEL}" --tensor-parallel-size "${TP_SIZE}"
    --mem-fraction-static "${MEM_FRAC}" --max-running-requests 32 --enable-cache-report
    --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}" --host 127.0.0.1 --port "${PORT}"
    --dist-init-addr "127.0.0.1:${DIST_PORT}" --trust-remote-code )

if [[ "${MODE}" == "bf16" ]]; then
    SERVER_ARGS=( "${COMMON_ARGS[@]}" --kv-cache-dtype auto
        --prefill-attention-backend triton --decode-attention-backend triton )
    echo "[matrix:bf16] model=${MODEL} tp=${TP_SIZE} out=${OUT_BASE}"
    CUDA_VISIBLE_DEVICES="${GPUS}" python -m sglang.launch_server "${SERVER_ARGS[@]}" >>"${LOG_SERVER}" 2>&1 &
elif [[ "${MODE}" == "calibrated" ]]; then
    : "${ROT_DIR:?ROT_DIR required for calibrated}"
    SERVER_ARGS=( "${COMMON_ARGS[@]}" --kv-cache-dtype int2 --kv-cache-quant-group-size 128
        --prefill-attention-backend fa3 --decode-attention-backend triton --disable-radix-cache )
    echo "[matrix:oscar] model=${MODEL} tp=${TP_SIZE} rot=${ROT_DIR} out=${OUT_BASE}"
    SGLANG_ENABLE_MIXED_KV_WINDOWS=1 SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
      SGLANG_MIXED_KV_HP_MAX_SPLITS=8 SGLANG_MIXED_KV_PREFIX_TOKENS="${SGLANG_MIXED_KV_PREFIX_TOKENS:-64}" \
      SGLANG_MIXED_KV_RECENT_TOKENS="${SGLANG_MIXED_KV_RECENT_TOKENS:-256}" \
      SGLANG_MIXED_KV_HP_DTYPE=bfloat16 SGLANG_MIXED_KV_SCALE_DTYPE=float32 \
      SGLANG_OSCAR_K_ROTATION_PATH="${ROT_DIR}/${K_ROT_FILENAME:-k_rotation.pt}" \
      SGLANG_OSCAR_V_ROTATION_PATH="${ROT_DIR}/${V_ROT_FILENAME:-v_rotation.pt}" \
      SGLANG_OSCAR_K_CLIP_RATIO="${K_CLIP:-0.96}" SGLANG_OSCAR_V_CLIP_RATIO="${V_CLIP:-0.92}" \
      SGLANG_OSCAR_ABSORB_V_ROTATION="${SGLANG_OSCAR_ABSORB_V_ROTATION:-1}" SGLANG_LLOYD_MAX="${SGLANG_LLOYD_MAX:-0}" \
      CUDA_VISIBLE_DEVICES="${GPUS}" python -m sglang.launch_server "${SERVER_ARGS[@]}" >>"${LOG_SERVER}" 2>&1 &
else
    echo "bad MODE=${MODE}" >&2; exit 1
fi
SERVER_PID=$!

for _ in $(seq 1 300); do
    curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { echo "[matrix] server ready"; break; }
    kill -0 "${SERVER_PID}" 2>/dev/null || { echo "[matrix] server died"; tail -80 "${LOG_SERVER}"; exit 1; }
    sleep 5
done
curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 || { echo "[matrix] server not ready"; tail -80 "${LOG_SERVER}"; exit 1; }

RUNNER="${REPO_ROOT}/rotation/_eval_runner/run_simple_eval.py"
for bench in ${BENCHES}; do
  for seed in ${SEEDS}; do
    RD="${OUT_BASE}/${bench}/seed${seed}"; mkdir -p "${RD}"
    if [[ -f "${RD}/metrics.json" ]]; then echo "[matrix] skip existing ${bench}/seed${seed}"; continue; fi
    echo "[matrix] === ${bench} seed=${seed} ==="
    python "${RUNNER}" --task "${bench}" --model "${MODEL}" \
        --base-url "http://127.0.0.1:${PORT}/v1" --max-tokens "${MAX_NEW_TOKENS}" \
        --temperature 1.0 --top-p 0.95 --top-k 40 --n-repeats 1 --seed "${seed}" \
        --num-threads "${NUM_WORKERS}" ${NUM_EXAMPLES:+--num-examples ${NUM_EXAMPLES}} \
        --output-dir "${RD}" >"${RD}/runner.log" 2>&1 \
      && echo "[matrix] ${bench}/seed${seed} score=$(python -c "import json;print(json.load(open('${RD}/metrics.json')).get('score'))" 2>/dev/null)" \
      || echo "[matrix] ${bench}/seed${seed} FAILED (see runner.log)"
  done
done

echo "[matrix] ===== SUMMARY (${MODE}) ====="
for bench in ${BENCHES}; do
  for seed in ${SEEDS}; do
    m="${OUT_BASE}/${bench}/seed${seed}/metrics.json"
    [[ -f "$m" ]] && echo "  ${bench} seed${seed}: $(python -c "import json;print('%.4f'%json.load(open('$m')).get('score',-1))" 2>/dev/null)" || echo "  ${bench} seed${seed}: MISSING"
  done
done
echo "[matrix] done."
