#!/usr/bin/env bash
# Shared verification: bring a server up with radix cache AND cuda graph ON, then
# decide PASS/FAIL on four criteria. Sourced by mha.sh and mla.sh, which differ
# only in how they launch -- the probe, the judge and the verdict are identical
# and used to be copy-pasted in both.
#
# Each criterion below encodes a wrong verdict this harness has already produced:
#
#   cuda graph     capture lines in the log; absence is silent otherwise
#   prefix cache   the SAME long prompt sent twice must report cached tokens. A
#                  SHORT probe reports 0 and reads as "the cache is off" when the
#                  prefix is merely below block granularity.
#   not garbled    four SHAPE checks. A letter-ratio judge gave four false
#                  verdicts -- error strings are mostly letters, so they passed.
#   0 traceback    counted BEFORE teardown: the gloo shutdown path emits its own
#                  errors and once failed a healthy 30B-A3B.
#
# A CLEAN judgement means "did not collapse", NOT "answered correctly" -- a
# fluent, off-task answer passes. Scores come from the eval scripts, not here.
set -uo pipefail

OUT=${OUT:-/scratch/v}; mkdir -p "$OUT"

# Frees the GPUs from the previous model before this one is sized. Without it a
# leftover allocation is charged to the next model: it once made a 12B report
# 79 GiB in use and sent the diagnosis down the wrong path.
verify_drain_gpus() {
  pkill -f "sglang.launch_server" 2>/dev/null
  sleep 15
  local u=
  for _ in $(seq 1 40); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1)
    [ -z "$u" ] && break
    [ "$u" -lt 2000 ] && break
    sleep 5
  done
  echo "  gpu drained (max used ${u:-?} MiB)"
}

# Other checkpoints are removed unless KEEP_WEIGHTS=1: the pod was evicted once
# for "ephemeral local storage usage exceeds the total limit" after seven models
# piled up. On a large volume the purge is pure cost, so it is opt-out.
verify_prune_weights() {
  local repo=$1 keep
  if [ "${KEEP_WEIGHTS:-0}" = "1" ]; then
    echo "  KEEP_WEIGHTS=1: other checkpoints left in place"
    return
  fi
  keep=$(echo "$repo" | sed "s|/|--|")
  for d in "$HF_HOME"/hub/models--*; do
    case "$d" in *"$keep"*) : ;; *) rm -rf "$d" ;; esac
  done
}

# Returns the weight volume's page cache to the kernel. See the note above
# verify_download for why a long-lived pod needs this.
verify_release_page_cache() {
  sync
  python3 - "${HF_HOME:-/scratch/hf}" <<'PYCACHE' 2>/dev/null || true
import os, sys
root = sys.argv[1]
freed = 0
for dp, _, fns in os.walk(root):
    for fn in fns:
        path = os.path.join(dp, fn)
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            freed += 1
        except (OSError, AttributeError):
            pass
        finally:
            os.close(fd)
print(f"  page cache released for {freed} files", flush=True)
PYCACHE
}

verify_download() {
  timeout 3000 python3 -c "
from huggingface_hub import snapshot_download
import os, sys
print(snapshot_download(sys.argv[1], max_workers=8, token=os.environ.get('HF_TOKEN')))" "$1" 2>/dev/null
}

# Waits for /health_generate. $1 = server pid, $2 = port, $3 = attempts.
verify_wait_serve() {
  local srv=$1 port=$2 tries=${3:-200}
  for _ in $(seq 1 "$tries"); do
    kill -0 "$srv" 2>/dev/null || { echo "  server exited early"; return 1; }
    curl -sS -m 5 "http://127.0.0.1:$port/health_generate" >/dev/null 2>&1 && return 0
    sleep 10
  done
  return 1
}

verify_gen() {  # port prompt max_new -> raw JSON
  local port=$1 prompt=$2 n=${3:-48}
  local body
  body=$(python3 -c "
import json, sys
print(json.dumps({'text': sys.argv[1],
                  'sampling_params': {'max_new_tokens': int(sys.argv[2]),
                                      'temperature': 0.7, 'top_p': 0.95}}))" "$prompt" "$n")
  curl -sS -m 240 -X POST "http://127.0.0.1:$port/generate" \
    -H 'Content-Type: application/json' -d "$body" 2>/dev/null
}

# Four shape checks. Broken low-bit output has signatures a letter ratio cannot
# see: digit soup, a collapsed vocabulary, and word shapes no language produces.
verify_judge() {
  python3 - "$1" <<'PY'
import json, re, sys
from collections import Counter
try:
    d = json.loads(sys.argv[1])
    t = (d[0] if isinstance(d, list) else d).get("text", "")
except Exception:
    print("BAD-RESPONSE"); raise SystemExit
if len(t) < 8:
    print("EMPTY"); raise SystemExit
w = re.findall(r"[A-Za-z']+", t)
digits = sum(c.isdigit() for c in t) / len(t)
bad = []
if not w:
    # No alphabetic tokens: shape ratios are undefined. A numeric answer is
    # legitimate here, so look for character-level collapse instead.
    body = re.sub(r"\s+", "", t)
    top = Counter(body).most_common(1)[0][1] / len(body) if body else 1.0
    if digits > 0.90:  bad.append(f"digit-soup {digits:.2f}")
    if top > 0.60:     bad.append(f"char-repeat {top:.2f}")
else:
    rep = Counter(w).most_common(1)[0][1] / len(w)
    longw = sum(len(x) > 18 for x in w) / len(w)
    alpha = len(w) / max(1, len(t.split()))
    # The probe asks for numbers, so digits are expected; soup means almost nothing else.
    if digits > 0.60:             bad.append(f"digit-soup {digits:.2f}")
    if rep > 0.50 and len(w) > 6: bad.append(f"repetition {rep:.2f}")
    if longw > 0.25:              bad.append(f"word-shape {longw:.2f}")
    if alpha < 0.30:              bad.append(f"non-word {alpha:.2f}")
print("; ".join(bad) if bad else "CLEAN")
PY
}

# Probes a live server and prints the verdict. $1 name, $2 port, $3 logfile.
verify_probe_and_verdict() {
  local name=$1 port=$2 log=$3
  local pre r r2 garb tb cg rx pool

  # 24 repeats: long enough to exceed the cache's block granularity.
  pre=$(python3 -c "print('The following is a reference passage about European geography. ' * 24)")
  verify_gen "$port" "$pre What is the capital of France? Answer in one sentence." 32 >/dev/null
  r=$(verify_gen "$port" "$pre What is the capital of France? Answer in one sentence." 32)
  echo "  probe : $(echo "$r" | head -c 130)"
  r2=$(verify_gen "$port" "List the first five prime numbers." 48)
  echo "  probe2: $(echo "$r2" | head -c 130)"

  garb=$(verify_judge "$r2")
  tb=$(grep -ac Traceback "$log")          # BEFORE teardown
  cg=$(grep -acE "Capture cuda graph|Capturing cuda graph|cuda graph" "$log")
  rx=$(grep -aoE "#cached-token: [0-9]+" "$log" | awk '{s+=$2} END{print s+0}')
  pool=$(grep -aoE "\[MLAPacked\][^\"]{0,70}" "$log" | sort -u | head -1)

  [ -n "$pool" ] && echo "  pool  : $pool"
  echo "  cuda-graph=$cg  cached-tokens=$rx  garbling=$garb  tb=$tb"

  local v=PASS
  [ "$tb"   != "0"     ] && v=FAIL
  [ "$cg"   = "0"      ] && v="FAIL(no-cuda-graph)"
  [ "$rx"   = "0"      ] && v="FAIL(no-prefix-cache)"
  [ "$garb" != "CLEAN" ] && v="FAIL(garbled: $garb)"
  echo "RESULT=$name $v"
}
