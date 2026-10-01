#!/usr/bin/env bash
# Does the garbling judge actually flag garbling?
#
# It did not, for the failure that mattered. Kimi-K3 produced 68-73 KB
# responses that reasoned coherently for thousands of characters and then
# collapsed into `!!!!` to the token limit. Whole-text ratios flagged 1 of 34
# real collapses: a long healthy prefix dilutes every ratio below threshold.
# A judge that cannot fail is indistinguishable from a model that cannot break,
# so this pins both directions -- the collapse shapes must FAIL, and ordinary
# prose must PASS.
#
#   bash rotation/verify/test_garbling_judge.sh
set -uo pipefail
D=$(cd "$(dirname "$0")" && pwd)
source "$D/_common.sh"

pass=0; fail=0
check() {  # name expect(FAIL|CLEAN) text
  local name=$1 expect=$2 text=$3 verdict
  verdict=$(verify_judge "$(python3 -c '
import json,sys
print(json.dumps({"text": sys.stdin.read()}))' <<< "$text")")
  local got=CLEAN
  [ "$verdict" != "CLEAN" ] && got=FAIL
  if [ "$got" = "$expect" ]; then
    pass=$((pass+1)); printf "  ok    %-34s %s\n" "$name" "$verdict"
  else
    fail=$((fail+1)); printf "  FAIL  %-34s expected %s, got %s (%s)\n" "$name" "$expect" "$got" "$verdict"
  fi
}

PROSE=$(python3 -c "
import random
random.seed(7)
v='the a of to in and for with on by is are was were from that this these those'.split()
n='engine piston valve stroke cycle mixture spark exhaust intake compression'.split()
print(' '.join(random.choice(v+n) for _ in range(900)))")

# Healthy long-form prose must not trip anything.
check "ordinary long prose"        CLEAN "$PROSE"
check "short clean answer"         CLEAN "Paris is the capital of France."
check "numeric answer"             CLEAN "The first five primes are 2, 3, 5, 7, and 11."

# The real shape: coherent prefix, then collapse. Prefix length is the point --
# this is exactly what the ratio checks could not see.
check "coherent prefix then !!!!"  FAIL  "$PROSE$(python3 -c "print('!'*4000)")"
check "collapse after 8k of prose" FAIL  "$PROSE$PROSE$(python3 -c "print('!'*2000)")"
# Ordinary repetition is NOT the failure this gate exists for. A model that
# repeats a word when pushed to 2048 tokens is verbose, not broken, and
# flagging it cost two healthy models a PASS. Only a character run -- the shape
# every one of the 34 real collapses had, and no healthy response came within
# three orders of magnitude of -- is treated as garbling.
check "ordinary word repetition"   CLEAN "$PROSE$(python3 -c "print(' wait'*60)")"
check "verbose tail"               CLEAN "$PROSE$(python3 -c "print(' hmm'*500)")"

# Formatting that legitimately repeats. An 82-character markdown rule failed a
# healthy answer when the run threshold was 40.
check "markdown rule in prose"     CLEAN "$PROSE$(python3 -c "print('\n' + '-'*82 + '\n')")$PROSE"
check "table separator row"        CLEAN "$PROSE$(python3 -c "print('|' + '-'*60 + '|')")"

# Shapes the original checks already caught; keep them caught.
check "digit soup"                 FAIL  "$(python3 -c "
import random; random.seed(1); print(''.join(random.choice('0123456789 ') for _ in range(600)))")"

# A server refusal must NOT be reported as garbling. It was: raising the probe
# budget past the sweep's --context-length made the server return an error, the
# judge saw no "text", printed EMPTY, and every model came back
# FAIL(garbled: EMPTY) -- the harness blaming the model for its own budget.
raw_check() {  # name expect-prefix raw-json
  local name=$1 want=$2 raw=$3 verdict
  verdict=$(verify_judge "$raw")
  case "$verdict" in
    "$want"*) pass=$((pass+1)); printf "  ok    %-34s %s\n" "$name" "${verdict:0:60}" ;;
    *)        fail=$((fail+1)); printf "  FAIL  %-34s expected %s*, got %s\n" "$name" "$want" "$verdict" ;;
  esac
}
raw_check "server error is not garbling" "SERVER-ERROR" \
  '{"error":{"message":"Requested token count exceeds the model'"'"'s maximum context length of 2048 tokens."}}'
raw_check "unparseable payload"          "BAD-RESPONSE" 'not json at all'

echo
echo "garbling judge: $pass passed, $fail failed"
[ "$fail" -eq 0 ] || exit 1
