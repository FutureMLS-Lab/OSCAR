#!/usr/bin/env bash
# Loop the 4-benchmark suite x seeds against an already-running server.
# Env: BASE_URL, OUT_BASE, MODEL, SEEDS(default "0 1 2"), BENCHES(default 4),
#      MAX_NEW_TOKENS(default 32768), NUM_WORKERS(default 16),
#      MATH500_MAX (cap MATH500 examples on this expensive 2-node model; default 200).
set -uo pipefail
: "${BASE_URL:?}"; : "${OUT_BASE:?}"; : "${MODEL:?}"
SEEDS="${SEEDS:-0 1 2}"; BENCHES="${BENCHES:-gpqa humaneval aime25 math500}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"; NUM_WORKERS="${NUM_WORKERS:-16}"; MATH500_MAX="${MATH500_MAX:-200}"
RUNNER="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/run_simple_eval.py"
for bench in ${BENCHES}; do
  ne=""; [ "$bench" = "math500" ] && [ -n "$MATH500_MAX" ] && ne="--num-examples ${MATH500_MAX}"
  mt="${MAX_NEW_TOKENS}"; [ "$bench" = "humaneval" ] && mt=$(( MAX_NEW_TOKENS < 16384 ? MAX_NEW_TOKENS : 16384 ))
  for seed in ${SEEDS}; do
    RD="${OUT_BASE}/${bench}/seed${seed}"; mkdir -p "$RD"
    [ -f "${RD}/metrics.json" ] && { echo "[glm-mb] skip ${bench}/seed${seed}"; continue; }
    echo "[glm-mb] === ${bench} seed=${seed} ==="
    python "${RUNNER}" --task "${bench}" --model "${MODEL}" --base-url "${BASE_URL}" \
      --max-tokens "${mt}" --temperature 1.0 --top-p 0.95 --top-k 40 \
      --n-repeats 1 --seed "${seed}" --num-threads "${NUM_WORKERS}" ${ne} \
      --output-dir "${RD}" >"${RD}/runner.log" 2>&1 \
      && echo "[glm-mb] ${bench}/seed${seed} score=$(python -c "import json;print(json.load(open('${RD}/metrics.json')).get('score'))" 2>/dev/null)" \
      || echo "[glm-mb] ${bench}/seed${seed} FAILED"
  done
done
echo "[glm-mb] done"
