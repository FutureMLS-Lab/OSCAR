#!/bin/bash
# Kimi-K3 prefix-cache-ON GPQA arm, head (node-rank 0), b200 / charlie.
#
# WHAT THIS RUN IS FOR
# The K3 headline (93/168 = 55.4%) was measured with the prefix cache OFF,
# because head.sh's cache gate could not self-certify cache-ON: it probed with
# INPUT LOGPROBS, which makes sglang recompute the prompt and therefore
# suppresses reuse, so the probe read cached_tokens=0 and the script downgraded
# the arm.  Every other model in the sweep reports a cache-ON number, so K3 is
# the only inconsistent row.  k3_gate.py v2 fixes the gate (plain generation
# requests + the server's own cumulative #cached-token counters); this script
# runs the arm that gate now certifies.
#
# Inherited from head2.sh, and not to be simplified away:
#   * the worker_alive barrier (the queue admits one pod at a time, and rank 0
#     parked alone in dist-init got SIGKILLed after ~10 min, costing a whole arm)
#   * the RUNTOKEN-stamped arm handshake (a stale arm file once brought up
#     rank 0 as BF16 and rank 1 as INT2, which sglang serves SILENTLY)
#   * mem-fraction is bumped to buy the concurrency, never the other way round
#   * every result file is named after the FULL arm config, cache mode included
set -uo pipefail
RUN=${RUN:-on1}
ARMS=${ARMS:-"int2on"}
MAXREQ=${MAXREQ:-8}                 # == cuda-graph-max-bs; every bs 1..MAXREQ captured
THREADS=${THREADS:-6}               # requests in flight; MUST match the BF16 arm
MEMFRAC=${MEMFRAC:-0.68}
MEMFRAC_RETRY=${MEMFRAC_RETRY:-0.74}   # server refused to start at all
MEMFRAC_MAX=${MEMFRAC_MAX:-0.74}       # started, but the KV pool was too small
MEMFRAC_FALLBACK=${MEMFRAC_FALLBACK:-0.72}
MEMFRAC0=$MEMFRAC
GEN_CAP=${GEN_CAP:-30720}           # 32K total budget; identical to the reference
ARM_DEADLINE=${ARM_DEADLINE:-18000} # stop dispatching new questions after this
START=${START:-0}; END=${END:-198}
KEEP=${KEEP:-1800}
PORT=31255
. /shared/oscar-venv/bin/activate
W=/shared/oscar
export PYTHONPATH=$W/sglang-research/python:${PYTHONPATH:-}
ROT=/shared/kimi-k3-rotations
OUT=/shared/kimi3_out/$RUN; mkdir -p $OUT $OUT/gpqa
note() { echo "[$(date -u +%H:%M:%S)] $*" >> $OUT/progress.log; sync; echo "[k3] $*"; }

. /scripts/k3_args4.sh

MODEL=$(resolve_model) || { note "FATAL: no Kimi-K3 snapshot on the HF PVC"; exit 1; }
NODERANK=0
MY_IP=$(hostname -I | awk '{print $1}'); MASTER=$MY_IP
rm -f /shared/kimi3_master_ip.$RUN; echo "$MY_IP" > /shared/kimi3_master_ip.$RUN
echo "$ARMS" > /shared/kimi3_arms.$RUN

RUNTOKEN="$(date -u +%s)-$$"
# Clear the worker's announcement BEFORE publishing the token, never after: the
# worker only announces once it has read the token, so clearing first makes it
# impossible to delete an announcement that has already been made.
rm -f /shared/kimi3_arm.$RUN.* /shared/kimi3_memfrac.$RUN.* /shared/kimi3_worker_alive.$RUN
sync
echo "$RUNTOKEN" > /shared/kimi3_token.$RUN
sync
publish_arm() {   # $1 = GEN, $2 = arm, $3 = mem-fraction
  printf '%s %s %s\n' "$RUNTOKEN" "$2" "$3" > /shared/kimi3_arm.$RUN.$1
  sync
}

note "head start run=$RUN arms='$ARMS' maxreq=$MAXREQ threads=$THREADS memfrac=$MEMFRAC gencap=$GEN_CAP arm_deadline=${ARM_DEADLINE}s"
note "model=$MODEL"
nvidia-smi -L | head -2
echo "[k3] HEAD=$(cd $W && git log --oneline -1)"
python -c "import torch,transformers,sgl_kernel;print('[k3] torch',torch.__version__,'tf',transformers.__version__,'sgl_kernel ok')" || exit 1
echo "[k3] shards=$(ls $MODEL | grep -c 'safetensors$') index=$([ -f $MODEL/model.safetensors.index.json ] && echo yes || echo NO)"

wait_healthy() {   # $1 = log ; 0 healthy, 1 dead/timeout
  local log=$1 i
  for i in $(seq 1 2880); do          # 4 h: the two ranks are admitted one at a time
    curl -s http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { note "server READY after $((i*5))s"; return 0; }
    kill -0 $SP 2>/dev/null || { note "SERVER DIED (rank0 gone)"; tail -80 "$log"; return 1; }
    [ $((i % 24)) -eq 0 ] && echo "[k3] waiting $((i*5))s: $(tail -1 "$log" | cut -c1-160)"
    sleep 5
  done
  note "TIMEOUT waiting for health"; tail -80 "$log"; return 1
}

# Both ranks must agree on the KV precision and the cache mode.  This is not
# defensive programming for its own sake: on 2026-08-22 a stale handshake file
# brought up rank 0 as BF16 and rank 1 as INT2, and because sglang gives every
# PP stage its own KV pool it served that model happily -- 47 layers in one
# precision, 46 in the other, no error, a clean-looking and entirely wrong
# number.  Diff the ranks before trusting anything.
#
# Returns 0 = agree, 1 = a real MISMATCH (skip the arm), 2 = could not read one
# of the logs.  The distinction matters: rank 1's log is written by the other pod
# onto the PVC, so it can lag by seconds, and treating "not readable yet" as a
# mismatch would throw away a good arm and a 30-minute weight load.  Only a value
# that is present on BOTH ranks and DIFFERS is fatal.
ranks_agree() {   # $1 = rank0 log, $2 = rank1 log
  local a b k rc=0 missing=0 i
  # Give the worker's log up to 2 min to appear on the PVC before judging it.
  for i in $(seq 1 24); do
    [ -s "$2" ] && grep -qa "kv_cache_dtype=" "$2" 2>/dev/null && break
    sleep 5
  done
  for k in kv_cache_dtype disable_radix_cache mamba_scheduler_strategy page_size kv_cache_quant_group_size max_total_num_tokens; do
    a=$(grep -aoE "$k=[^,)]*" "$1" 2>/dev/null | head -1)
    b=$(grep -aoE "$k=[^,)]*" "$2" 2>/dev/null | head -1)
    if [ -z "$a" ] || [ -z "$b" ]; then
      missing=1
      printf '[k3] rank-diff %-28s rank0=%-34s rank1=%-34s UNREADABLE\n' "$k" "${a:-<absent>}" "${b:-<absent>}"
    elif [ "$a" != "$b" ]; then
      rc=1
      printf '[k3] rank-diff %-28s rank0=%-34s rank1=%-34s MISMATCH\n' "$k" "$a" "$b"
    else
      printf '[k3] rank-diff %-28s rank0=%-34s rank1=%-34s same\n' "$k" "$a" "$b"
    fi
  done
  [ $rc = 1 ] && return 1
  [ $missing = 1 ] && return 2
  return 0
}

# Cumulative prefix-cache hit rate that the SERVER reported about itself.
hit_rate() {   # $1 = server log
  python - "$1" <<'PY'
import re, sys
try:
    t = open(sys.argv[1], errors="replace").read()
except Exception:
    print("cached=? new=? rate=?"); raise SystemExit
h = sum(int(x) for x in re.findall(r"#cached-token:\s*(\d+)", t))
n = sum(int(x) for x in re.findall(r"#new-token:\s*(\d+)", t))
print("cached=%d new=%d rate=%.3f%%" % (h, n, 100.0 * h / max(1, h + n)))
PY
}

note "waiting for the worker pod to announce itself before launching rank 0"
WOK=0
for i in $(seq 1 720); do
  [ -s /shared/kimi3_worker_alive.$RUN ] && { WOK=1; note "worker alive after $((i*5))s: $(cat /shared/kimi3_worker_alive.$RUN)"; break; }
  [ $((i % 60)) -eq 0 ] && echo "[k3] still waiting for worker: $((i*5))s"
  sleep 5
done
[ "$WOK" = 1 ] || { note "FATAL: worker never announced itself; not burning the arms on a solo rank"; exit 1; }

GEN=0
LAST_RES=""
for ARM in $ARMS; do
  GEN=$((GEN+1))
  RES=$OUT/gpqa/results.$ARM.jsonl
  LAST_RES=$RES
  MEMFRAC=$MEMFRAC0
  publish_arm "$GEN" "$ARM" "$MEMFRAC"
  note "================ arm $GEN = $ARM (memfrac $MEMFRAC) ================"
  LOG=$OUT/server.$GEN-$ARM.log
  WLOG=$OUT/worker.$GEN-$ARM.log
  build_launch "$ARM"
  note "arm $ARM expects cache mode: $ARM_CACHE_MODE"
  env "${ENVV[@]}" python -m sglang.launch_server "${ARGS[@]}" > "$LOG" 2>&1 &
  SP=$!
  if ! wait_healthy "$LOG"; then
    # These two point in OPPOSITE directions and have been confused before:
    # "102 Killed" in the process log is HOST OOM (raise the pod's
    # requests.memory), while "Not enough memory ... increase
    # --mem-fraction-static" is the GPU-side KV pool being too small.
    if grep -qa "increase --mem-fraction-static\|Not enough memory" "$LOG"; then
      note "arm $ARM: GPU KV pool too small at $MEMFRAC -> retrying at $MEMFRAC_RETRY"
      kill -KILL $SP 2>/dev/null; sleep 30
      MEMFRAC=$MEMFRAC_RETRY
      GEN=$((GEN+1))
      publish_arm "$GEN" "$ARM" "$MEMFRAC"
      LOG=$OUT/server.$GEN-$ARM.log; WLOG=$OUT/worker.$GEN-$ARM.log
      build_launch "$ARM"
      env "${ENVV[@]}" python -m sglang.launch_server "${ARGS[@]}" > "$LOG" 2>&1 &
      SP=$!
      wait_healthy "$LOG" || { note "arm $ARM FAILED TO START, skipping"; kill -KILL $SP 2>/dev/null; sleep 30; continue; }
    else
      note "arm $ARM FAILED TO START, skipping"; kill -KILL $SP 2>/dev/null; sleep 30; continue
    fi
  fi

  # Concurrency is the variable to hold fixed across arms; mem-fraction is only
  # the budget that buys it.  A pool too small to hold THREADS x (cap+prompt)
  # makes requests QUEUE, which silently lowers this arm's decode batch relative
  # to the arm it is compared against -- the asymmetry that made MiniMax-M2.7
  # look like a -21 pp quantization disaster.
  NEED=$(( (GEN_CAP + 3072) * THREADS ))
  POOL=$(grep -aoE "max_total_num_tokens=[0-9]+" "$LOG" | head -1 | grep -oE "[0-9]+$")
  note "arm $ARM pool=${POOL:-unknown} tokens, need ~$NEED for $THREADS x (${GEN_CAP}+prompt) in flight"
  if [ -n "${POOL:-}" ] && [ "$POOL" -lt "$NEED" ] && [ "$MEMFRAC" != "$MEMFRAC_MAX" ]; then
    note "arm $ARM: pool $POOL < $NEED -> relaunching at mem-fraction $MEMFRAC_MAX (weights are warm now)"
    kill -KILL $SP 2>/dev/null; sleep 45
    MEMFRAC=$MEMFRAC_MAX
    GEN=$((GEN+1))
    publish_arm "$GEN" "$ARM" "$MEMFRAC"
    LOG=$OUT/server.$GEN-$ARM.log; WLOG=$OUT/worker.$GEN-$ARM.log
    build_launch "$ARM"
    env "${ENVV[@]}" python -m sglang.launch_server "${ARGS[@]}" > "$LOG" 2>&1 &
    SP=$!
    if wait_healthy "$LOG"; then
      POOL=$(grep -aoE "max_total_num_tokens=[0-9]+" "$LOG" | head -1 | grep -oE "[0-9]+$")
      note "arm $ARM pool after bump = ${POOL:-unknown} (need $NEED)"
    else
      note "arm $ARM: bump to $MEMFRAC_MAX failed to come up; falling back"
      kill -KILL $SP 2>/dev/null; sleep 30
      MEMFRAC=$MEMFRAC_FALLBACK
      GEN=$((GEN+1))
      publish_arm "$GEN" "$ARM" "$MEMFRAC"
      LOG=$OUT/server.$GEN-$ARM.log; WLOG=$OUT/worker.$GEN-$ARM.log
      build_launch "$ARM"
      env "${ENVV[@]}" python -m sglang.launch_server "${ARGS[@]}" > "$LOG" 2>&1 &
      SP=$!
      wait_healthy "$LOG" || { note "arm $ARM FAILED after fallback, skipping"; kill -KILL $SP 2>/dev/null; sleep 30; continue; }
    fi
  fi

  echo "[k3] ---- served config ($ARM), read from the LIVE server ----"
  python /scripts/k3_cfg.py $PORT || true
  echo "[k3] ---- both ranks must agree ----"
  ranks_agree "$LOG" "$WLOG"; RA=$?
  case $RA in
    0) note "arm $ARM: rank 0 and rank 1 agree on KV dtype, cache mode, page size, quant grouping and pool size" ;;
    2) note "arm $ARM: WARNING could not read one rank's config off the PVC; proceeding, but this arm's rank agreement is UNVERIFIED" ;;
    *) note "arm $ARM: RANK CONFIG MISMATCH -> refusing to score this arm (half-precision-per-PP-stage is served silently)"
       kill -KILL $SP 2>/dev/null; sleep 30; continue ;;
  esac
  echo "[k3] ---- pool size / capture ----"
  grep -aoE "max_total_num_tokens=[0-9]+" "$LOG" | head -1
  grep -aoE "Capture cuda graph bs \[[^]]*\]" "$LOG" | tail -1

  echo "[k3] ---- smoke ($ARM) ----"
  python /scripts/judge.py $PORT $OUT "$ARM" 2>&1 | tail -12

  # The gate.  rc=2 (inconclusive) is NOT fatal -- a 30-minute weight load must
  # not be thrown away because a probe could not prove something -- but the arm
  # only earns the cache-ON label from measurement.
  LABEL=unlabelled
  if [ "$ARM_CACHE_MODE" = on ]; then
    echo "[k3] ---- prefix-cache gate ($ARM) ----"
    GOUT=$(python /scripts/k3_gate.py $PORT "$LOG" 2>&1); GRC=$?
    echo "$GOUT" | tail -30
    LABEL=$(echo "$GOUT" | grep -oE "GATE_VERDICT label=[a-z-]+" | tail -1 | cut -d= -f2)
    case $GRC in
      0) note "GATE PASS arm=$ARM label=${LABEL:-?} (reuse proven from the server's own counters)" ;;
      2) note "GATE INCONCLUSIVE arm=$ARM label=${LABEL:-?} -> keeping the server, but the arm is NOT labelled cache-ON on intent alone" ;;
      *) note "GATE FAIL arm=$ARM label=${LABEL:-?} -> the restored state is wrong; NOT scoring this arm"
         kill -KILL $SP 2>/dev/null; sleep 30; continue ;;
    esac
    echo "${LABEL:-unlabelled}" > $OUT/gate_label.$ARM.txt
  fi

  note "---- GPQA dispatch arm=$ARM -> $RES ----"
  python /scripts/k3_gpqa.py --out "$RES" --port $PORT \
    --start $START --end $END --threads $THREADS --max-tokens $GEN_CAP \
    --deadline-s $ARM_DEADLINE --system "You are a helpful assistant." 2>&1 | tee -a $OUT/eval.$ARM.log
  note "---- GPQA dispatch DONE arm=$ARM ----"

  echo "[k3] ---- CONTIGUOUS-PREFIX SCORE ($ARM) ----"
  python /scripts/k3_prefix_score.py "$RES" --labels "$ARM" --growth 2>&1
  echo "[k3] ---- decode batch histogram ($ARM); every value must be in the captured list ----"
  grep -aoE "Decode batch[^,]*, #running-req: [0-9]+" "$LOG" | grep -oE "[0-9]+$" | sort -n | uniq -c
  echo -n "[k3] decode steps with cuda graph False (want 0): "
  grep -a "Decode batch" "$LOG" | grep -c "cuda graph: False"
  echo -n "[k3] requests retracted/queued (pool pressure): "
  grep -acE "retract|Retract" "$LOG"
  # Final label: cache-ON is claimed only when the live server had the cache
  # ENABLED *and* a non-zero hit rate actually accumulated over the scoring run.
  HR=$(hit_rate "$LOG")
  echo "[k3] ---- prefix cache over the SCORED run ($ARM): $HR ----"
  RATE=$(echo "$HR" | grep -oE "rate=[0-9.]+" | cut -d= -f2)
  FINAL=cache-off
  if [ "$ARM_CACHE_MODE" = on ]; then
    if [ -n "${RATE:-}" ] && [ "$(python -c "print(1 if float('${RATE:-0}')>0 else 0)")" = 1 ]; then
      FINAL=cache-on
    else
      FINAL=cache-on-unproven
    fi
  fi
  note "arm $ARM FINAL LABEL=$FINAL ($HR) pool=${POOL:-?} memfrac=$MEMFRAC threads=$THREADS"
  echo "$FINAL $HR pool=${POOL:-?} memfrac=$MEMFRAC threads=$THREADS" > $OUT/final_label.$ARM.txt

  kill -KILL $SP 2>/dev/null
  note "arm $ARM server down; 90s settle before the next arm"
  sleep 90
done

publish_arm "$((GEN+1))" DONE "$MEMFRAC"
note "======== ALL ARMS DONE ========"
# Pair against the BF16 control from the ctl1 run, at matched scope, if present.
BF=/shared/kimi3_out/ctl1/gpqa/results.bf16.jsonl
if [ -s "$BF" ] && [ -s "$LAST_RES" ]; then
  python /scripts/k3_prefix_score.py "$BF" "$LAST_RES" --labels BF16,INT2-cacheON 2>&1 | tee $OUT/COMPARISON.txt
fi
RC=0
[ -s "$LAST_RES" ] || { note "NO ARM PRODUCED RESULTS -> exiting non-zero so the watchdog relaunches"; RC=1; }
note "HEAD_DONE rc=$RC (idling ${KEEP}s so the logs can be read)"
sleep $KEEP
exit $RC
