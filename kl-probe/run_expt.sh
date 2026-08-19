#!/usr/bin/env bash
# Smoke experiment runner for a reference-vs-candidate pair.
#
#   ./run_expt.sh concurrent    # both servers side by side on DISJOINT GPU
#                               # sets, one `probe run` end-to-end
#   ./run_expt.sh serial        # pair too big to serve at once: servers one
#                               # at a time on a (possibly shared) GPU set
#                               #   phase 1: ref up -> prompts, generate, score ref -> down
#                               #   phase 2: cand up -> score cand -> down
#                               #   phase 3: metrics + report (no servers)
#
# GPU placement comes from the config's runtime.ref/cand.gpus. NOTE: the
# default M3 config ships with overlapping GPU sets (serial layout) — edit its
# runtime section to disjoint sets before running concurrent.
#
#   CONFIG=configs/foo.yaml ./run_expt.sh serial     # different pair
#   N=512 MAX_NEW_TOKENS=256 ./run_expt.sh serial    # full-size
#   FORCE=1 ./run_expt.sh concurrent                 # recompute existing artifacts
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-}"
if [[ "$MODE" != "concurrent" && "$MODE" != "serial" ]]; then
  echo "usage: $0 concurrent|serial" >&2
  exit 2
fi

CONFIG="${CONFIG:-configs/minimax-m3-wildchat.yaml}"
N="${N:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
export HF_HOME="${HF_HOME:-/data/huggingface}"

ARGS=(-c "$CONFIG" -n "$N" --max-new-tokens "$MAX_NEW_TOKENS")
[[ "${FORCE:-0}" == "1" ]] && ARGS+=(--force)

# whatever happens, don't leave servers squatting on the GPUs
cleanup() {
  echo "==> cleanup: stopping any running servers"
  uv run probe servers down -c "$CONFIG" >/dev/null 2>&1 || true
}
trap cleanup EXIT

if [[ "$MODE" == "concurrent" ]]; then
  uv run probe run "${ARGS[@]}" --down
else
  echo "==> phase 1/3: reference server (generate + score ref)"
  uv run probe servers up -c "$CONFIG" --only ref
  uv run probe prompts  "${ARGS[@]}"
  uv run probe generate "${ARGS[@]}"
  uv run probe score --model ref "${ARGS[@]}"
  uv run probe servers down -c "$CONFIG" --only ref

  echo "==> phase 2/3: candidate server (score cand)"
  uv run probe servers up -c "$CONFIG" --only cand
  uv run probe score --model cand "${ARGS[@]}"
  uv run probe servers down -c "$CONFIG" --only cand

  echo "==> phase 3/3: metrics + report"
  uv run probe metrics "${ARGS[@]}"
  uv run probe report  "${ARGS[@]}"
fi

trap - EXIT
echo "==> done"
