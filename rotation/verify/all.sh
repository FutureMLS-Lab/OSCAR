#!/usr/bin/env bash
# Every supported model, one long-lived pod, radix cache and cuda graph ON.
#
# Verdicts append to $V after EACH model. Three earlier sweeps were killed
# partway through the heavy arms and each time discarded verdicts already
# earned, and exec stdout has lost results twice on pod death -- so the file on
# the volume is the record, not this script's output.
#
# Run one model instead of the sweep with:  all.sh qwen3-8b
set -uo pipefail
D=$(cd "$(dirname "$0")" && pwd)
export OSCAR_ROTATIONS=${OSCAR_ROTATIONS:-/oscar/rotations}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export KEEP_WEIGHTS=${KEEP_WEIGHTS:-1}
export OUT=${OUT:-./verify-out}
V=${V:-$OUT/verdicts.txt}
ONLY=${1:-}
mkdir -p "$OUT" "$HF_HOME"

# Kimi-K3 is deliberately NOT in this table. It needs 16 GPUs across two nodes
# (tp 8 x pp 2, 1.4 TB of MXFP4 weights) and this harness is one pod with eight,
# so the row could only ever report FAIL(no-serve) -- which reads as "the model
# is broken" when it means "the harness cannot host it". Verify K3 with its own
# two-node job (rotation/run/kimi-k3.sh) and record that verdict separately.
# name | kind | repo | tp | group-or-rotdir | rot | sink | recent | memfrac
MODELS="
qwen3-4b-think|mha|Qwen/Qwen3-4B-Thinking-2507|1|128|Qwen3-4B-Thinking-2507/seq20000_prompt83_group128|64|256|0.55
qwen3-8b|mha|Qwen/Qwen3-8B|1|128|Qwen3-8B/seq20000_prompt83_group128|64|512|0.55
qwen3-32b|mha|Qwen/Qwen3-32B|2|128|Qwen3-32B/seq16000_prompt69_group128|64|256|0.55
qwen3-30b-a3b|mha|Qwen/Qwen3-30B-A3B|2|128|Qwen3-30B-A3B|64|256|0.60
qwen35-4b|mha|Qwen/Qwen3.5-4B|1|256|Qwen3.5-4B|64|256|0.55
gemma4-12b|mha|google/gemma-4-12B-it|4|128|Gemma4-12B|64|512|0.45
minimax-m27|mha|MiniMaxAI/MiniMax-M2.7|4|128|MiniMax-M2.7|64|256|0.80
qwen35-35b|mha|Qwen/Qwen3.5-35B-A3B|4|256|Qwen3.5-35B-A3B|64|256|0.85
minimax-m3|mha|MiniMaxAI/MiniMax-M3|8|128|MiniMax-M3|64|256|0.90
glm52|mla|zai-org/GLM-5.2-FP8|8|glm52-rotations|-|-|-|0.90
glm53|mla|zai-org/GLM-5.3|8|glm53-rotations|-|-|-|0.90
"

echo "### sweep $(date -u +%FT%TZ)  image=$(cat /oscar/IMAGE_TAG 2>/dev/null)  ${ONLY:+only=$ONLY}" >> "$V"
# Not `echo "$MODELS" | while read`. Two things go wrong with that shape and
# both were observed in one sweep: the loop body inherits the pipe as stdin, so
# a child that reads stdin -- the server launcher does -- swallows the model
# lines the loop has not read yet, and models silently never run. qwen3-30b-a3b
# was skipped outright that way, and qwen35-35b's verdict landed under
# minimax-m27's header. A here-string keeps stdin free, and `< /dev/null` on
# each launch makes sure nothing downstream can eat the list either.
while IFS='|' read -r name kind repo tp a rot sink recent mf; do
  [ -n "$name" ] || continue
  if [ -n "${ONLY}" ] && [ "$name" != "$ONLY" ]; then
    continue
  fi
  if [ "${RESUME:-1}" = "1" ] && grep -aq "^RESULT=$name " "$V" 2>/dev/null; then
    echo "  $name: already has a verdict, skipping (RESUME=0 to force)"
    continue
  fi
  echo; echo "======================= $name ($kind) ======================="
  if [ "$kind" = "mha" ]; then
    case "$name" in
      qwen35-4b|qwen35-35b) export MAMBA_STRATEGY=extra_buffer ;;
      *)                    unset MAMBA_STRATEGY ;;
    esac
    export NO_AUTOTUNE=1
    bash "$D/mha.sh" "$name" "$repo" "$tp" "$a" "$rot" "$sink" "$recent" "$mf" \
      < /dev/null 2>&1 | tee "$OUT/$name.out" | tail -22
  else
    bash "$D/mla.sh" "$name" "$repo" "$a" "$tp" "$mf" \
      < /dev/null 2>&1 | tee "$OUT/$name.out" | tail -22
  fi
  grep -a "^RESULT=" "$OUT/$name.out" >> "$V" || echo "RESULT=$name FAIL(no verdict line)" >> "$V"
done <<EOF
$(printf '%s\n' "$MODELS" | grep -av '^$')
EOF
echo "### done $(date -u +%FT%TZ)" >> "$V"
echo; echo "=== verdicts ==="; grep -a "^RESULT=\|^###" "$V" | tail -30
