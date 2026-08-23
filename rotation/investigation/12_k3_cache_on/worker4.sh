#!/bin/bash
# Kimi-K3 prefix-cache-ON GPQA arm, worker (node-rank 1).
#
# Identical in structure to worker2.sh -- it follows the arm the head publishes
# per generation so a head-side mem-fraction retry relaunches this rank with
# matching args -- except that it sources k3_args4.sh, where the prefix-cache
# mode is part of the arm name.  Both ranks build their launch line from that
# same file, which is the only reason the two nodes cannot drift.
set -uo pipefail
RUN=${RUN:-on1}
MAXREQ=${MAXREQ:-8}
MEMFRAC=${MEMFRAC:-0.68}
PORT=31255
. /shared/oscar-venv/bin/activate
W=/shared/oscar
export PYTHONPATH=$W/sglang-research/python:${PYTHONPATH:-}
ROT=/shared/kimi-k3-rotations
OUT=/shared/kimi3_out/$RUN; mkdir -p $OUT
nvidia-smi -L | head -2

. /scripts/k3_args4.sh
MODEL=$(resolve_model) || { echo "[k3-w] FATAL: no Kimi-K3 snapshot on the HF PVC"; exit 1; }
NODERANK=1

MASTER=""
for i in $(seq 1 480); do
  [ -s /shared/kimi3_master_ip.$RUN ] && MASTER=$(cat /shared/kimi3_master_ip.$RUN) && break
  sleep 5
done
[ -z "$MASTER" ] && { echo "[k3-w] no master ip file"; exit 1; }
echo "[k3-w] master=$MASTER run=$RUN model=$MODEL"

# Only accept generation files stamped with THIS head's token.  A previous run of
# the same $RUN leaves kimi3_arm.$RUN.* behind, and reading one of those is not a
# theoretical risk: it once brought up an INT2 rank 1 against a BF16 rank 0, a
# model with one PP stage in each precision, which sglang serves silently.
TOKEN=""
for i in $(seq 1 480); do
  [ -s /shared/kimi3_token.$RUN ] && TOKEN=$(cat /shared/kimi3_token.$RUN) && break
  sleep 5
done
[ -z "$TOKEN" ] && { echo "[k3-w] no run token published by the head"; exit 1; }
echo "[k3-w] run token=$TOKEN"

# Tell the head this pod exists, so it does not launch rank 0 into a gap where
# rank 1 has not even been scheduled (that cost a whole arm on 2026-08-22).
printf '%s pod=%s host=%s at=%s\n' "$TOKEN" "${HOSTNAME:-?}" "$(hostname -I | awk '{print $1}')" "$(date -u +%FT%TZ)" \
  > /shared/kimi3_worker_alive.$RUN
sync
echo "[k3-w] announced worker_alive for token $TOKEN"

GEN=0
while true; do
  GEN=$((GEN+1))
  ARM=""; TMF=""
  for i in $(seq 1 360); do
    if [ -s /shared/kimi3_arm.$RUN.$GEN ]; then
      read -r tk a mf < /shared/kimi3_arm.$RUN.$GEN || true
      if [ "$tk" = "$TOKEN" ]; then ARM=$a; TMF=$mf; break; fi
    fi
    sleep 10
  done
  [ -z "$ARM" ] && { echo "[k3-w] no gen $GEN for token $TOKEN within 3600s, head is done"; break; }
  [ "$ARM" = "DONE" ] && { echo "[k3-w] head signalled DONE"; break; }
  [ -n "$TMF" ] && MEMFRAC=$TMF
  LOG=$OUT/worker.$GEN-$ARM.log
  echo "[k3-w] ======== gen=$GEN arm=$ARM memfrac=$MEMFRAC -> $LOG ========"
  build_launch "$ARM"
  echo "[k3-w] cache mode for this arm: $ARM_CACHE_MODE"
  env "${ENVV[@]}" python -m sglang.launch_server "${ARGS[@]}" > "$LOG" 2>&1
  echo "[k3-w] gen=$GEN arm=$ARM worker exited rc=$?"
  # The real traceback for a dead rank lives in this file and has been lost to
  # pod reclamation twice, so always flush a generous tail into the job log too.
  tail -80 "$LOG"
done
echo "[k3-w] WORKER_DONE"
