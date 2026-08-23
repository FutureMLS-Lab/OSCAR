#!/usr/bin/env bash
# Bounded watchdog: recreate a vanished Job, up to a limit.
#
# b200 runs priority-1e9 CI and has also seen *external bulk deletions* --
# graceful `Killing`, no `Preempted` event, three owners hit in the same
# ten-minute window. A 5-hour arm that disappears at hour 4 with no retry is
# the single most expensive failure mode here, and `backoffLimit: 0` (which is
# right: a crashed arm should not silently re-run into the same crash) means
# the Job does not recreate its own pod.
#
# Bounded to 3 recreations on purpose. If someone really is clearing the
# namespace, an unbounded watchdog is a fight, not a fix.
#
# Never `pkill -f watch13.sh`: the parent shell's cmdline contains that string
# too, so it kills the shell that runs it. Start with
#   setsid nohup bash ./watch13.sh <job> <yaml> > watch13.<job>.log 2>&1 &
# and stop it with the recorded PID.
set -uo pipefail
JOB="${1:?job name}"
YAML="${2:?manifest}"
CTX="${CTX:-b200}"
NS="${NS:-charlie}"
MAX="${MAX:-3}"
SLEEP="${SLEEP:-120}"
n=0
while :; do
  st=$(kubectl --context="$CTX" -n "$NS" get job "$JOB" -o jsonpath='{.status.succeeded} {.status.failed}' 2>/dev/null)
  if [[ -z "${st// /}" ]]; then
    if ! kubectl --context="$CTX" -n "$NS" get job "$JOB" >/dev/null 2>&1; then
      if (( n >= MAX )); then
        echo "$(date -u +%FT%TZ) $JOB gone, recreation budget ($MAX) spent -- stopping"
        exit 1
      fi
      n=$((n + 1))
      echo "$(date -u +%FT%TZ) $JOB vanished (no Failed/Succeeded status, no object) -- recreate $n/$MAX"
      kubectl --context="$CTX" -n "$NS" apply -f "$YAML"
    fi
  else
    succ=${st%% *}; fail=${st##* }
    if [[ "${succ:-0}" != "0" && -n "${succ:-}" ]]; then
      echo "$(date -u +%FT%TZ) $JOB succeeded"; exit 0
    fi
    if [[ "${fail:-0}" != "0" && -n "${fail:-}" ]]; then
      echo "$(date -u +%FT%TZ) $JOB failed -- leaving it for inspection, not retrying into the same crash"
      exit 1
    fi
  fi
  sleep "$SLEEP"
done
