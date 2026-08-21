#!/usr/bin/env bash
# Watchdog for the GLM-5.2 window sweep on b200.
#
# Three failure modes, and only the first is recoverable by waiting:
#
#   decision=blocked_by_head  -- normal FIFO queueing behind someone else's
#     head pod. Waiting is correct; recreating would forfeit queue position.
#   decision=timeout          -- the gpu-queue.xp/admit webhook gave up after
#     3600 s. This state is TERMINAL: the pod keeps SchedulingGated forever and
#     no amount of waiting admits it. Recreating the Job is the only escape.
#   Job disappears entirely  -- observed on a Running arm 84 min in (see the
#     inline note below). Nothing to wait for; the arm is simply gone.
#
# Conflating these is how an arm silently never runs, or runs and is lost. This
# script waits on the first and recreates on the other two, each with a hard cap
# so a genuinely full cluster -- or someone deliberately clearing the namespace
# -- does not turn into an infinite resubmit loop.
set -uo pipefail
CTX=b200
NS=charlie
YAML="$(dirname "$0")/sweep09_live.yaml"
LOG="$(dirname "$0")/watch09.log"
JOBS=(zz-glm52-w256-r5 zz-glm52-w512-r4 zz-glm52-w1024-r4)
MAX_RECREATE=3
declare -A recreated
for j in "${JOBS[@]}"; do recreated[$j]=0; done

say() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOG"; }
say "watchdog start; jobs=${JOBS[*]}"

while :; do
    alive=0
    for j in "${JOBS[@]}"; do
        st=$(kubectl --context=$CTX -n $NS get job "$j" \
             -o jsonpath='{.status.succeeded}/{.status.failed}/{.status.active}' 2>/dev/null)
        if [[ -z "$st" ]]; then
            # OBSERVED 2026-08-21 06:26: zz-glm52-w256-r4 was Running normally
            # for 84 min and then its Job disappeared -- graceful
            # "Killing/Stopping container", no Preempted event, and two kimi3
            # pods went in the same 10-minute window. That is an external bulk
            # delete, not preemption and not this watchdog (which only deletes
            # on decision=timeout). Losing an arm silently costs ~6 h, so a
            # vanished Job is recreated rather than merely logged.
            #
            # Bounded, because if someone is deliberately clearing the
            # namespace an unbounded resubmit loop would fight them.
            if [[ ${recreated[$j]} -lt $MAX_RECREATE ]]; then
                recreated[$j]=$(( recreated[$j] + 1 ))
                say "$j: Job VANISHED (external delete?) -> recreate #${recreated[$j]}"
                kubectl --context=$CTX -n $NS apply -f "$YAML" >>"$LOG" 2>&1
                alive=1
            else
                say "$j: Job vanished but recreate cap ($MAX_RECREATE) reached; giving up"
            fi
            continue
        fi
        succ=${st%%/*}; rest=${st#*/}; fail=${rest%%/*}; act=${rest#*/}
        if [[ "${succ:-0}" == "1" ]]; then say "$j: COMPLETE"; continue; fi
        if [[ "${fail:-0}" -ge 1 ]]; then say "$j: FAILED (read \$RUN_DIR/pod.log)"; continue; fi
        alive=1

        pod=$(kubectl --context=$CTX -n $NS get pod -l job-name="$j" \
              -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
        [[ -z "$pod" ]] && { say "$j: no pod yet"; continue; }
        phase=$(kubectl --context=$CTX -n $NS get pod "$pod" \
                -o jsonpath='{.status.phase}' 2>/dev/null)
        dec=$(kubectl --context=$CTX -n $NS get pod "$pod" \
              -o jsonpath='{.metadata.annotations.gpu-queue\.xp/decision}' 2>/dev/null)

        if [[ "$dec" == "timeout" ]]; then
            if [[ ${recreated[$j]} -lt $MAX_RECREATE ]]; then
                recreated[$j]=$(( recreated[$j] + 1 ))
                say "$j: gate decision=timeout (TERMINAL) -> recreate #${recreated[$j]}"
                # Delete by EXACT name only. Never a field-selector sweep --
                # that has destroyed another user's pod in this namespace.
                kubectl --context=$CTX -n $NS delete job "$j" --wait=true >/dev/null 2>&1
                # Re-apply the whole file; already-present Jobs are unchanged.
                kubectl --context=$CTX -n $NS apply -f "$YAML" >>"$LOG" 2>&1
            else
                say "$j: gate timeout but recreate cap ($MAX_RECREATE) reached; leaving it"
            fi
        else
            say "$j: phase=$phase gate=${dec:-none}"
        fi
    done
    [[ $alive -eq 0 ]] && { say "watchdog exit: no active jobs"; break; }
    sleep 300
done
