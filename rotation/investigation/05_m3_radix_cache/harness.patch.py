#!/usr/bin/env python3
"""Driver-only patch applied to the run copy of rotation/eval_oscar_gpqa.sh.

Nothing here touches server or kernel code; the code under test stays at
4b06c0593e. Three changes, all forced by MiniMax-M3:

1. HEALTH_WAIT_STEPS. The harness waits 240*5s = 20 min for /health. M3's
   weights alone take ~21.5 min to load at TP=16 (measured: `Load weight end.
   elapsed=1292 s`), so every arm was killed mid-load with "server not ready
   after 20 min". Now env-tunable.

2. SEEDS. The harness scores one run per server lifetime. Three seeds would mean
   three ~25 min model loads per arm; instead loop the *client* over SEEDS
   inside one server lifetime, writing ${RUN_DIR}/seed<N>/. Each seed is an
   explicit --seed, so the draws are reproducible rather than relying on the
   server's random startup seed.

3. KV_MODE=bf16. Selects a plain BF16 KV cache for the paired reference arm
   (drops --kv-cache-dtype int2 and the group-size arg, which sglang only
   accepts alongside int2). INT2 arms are untouched.
"""
import re
import sys

path = sys.argv[1]
s = open(path).read()
orig = s

# 1. health wait
s = s.replace(
    "for _ in $(seq 1 240); do",
    'for _ in $(seq 1 "${HEALTH_WAIT_STEPS:-240}"); do',
)
s = s.replace(
    '    echo "[eval-oscar] server not ready after 20 min"',
    '    echo "[eval-oscar] server not ready after $(( ${HEALTH_WAIT_STEPS:-240} * 5 / 60 )) min"',
)

# 3. bf16 KV reference arm
s = s.replace(
    """else
    KV_DTYPE_ARGS=(--kv-cache-dtype int2)
    GROUP_SIZE_ARGS=(--kv-cache-quant-group-size "${GROUP_SIZE}")
fi""",
    """elif [[ "${KV_MODE:-int2}" == "bf16" ]]; then
    # Paired same-session BF16 reference arm: no int2, no mixed-KV tiering.
    KV_DTYPE_ARGS=(--kv-cache-dtype bfloat16)
    GROUP_SIZE_ARGS=()
else
    KV_DTYPE_ARGS=(--kv-cache-dtype int2)
    GROUP_SIZE_ARGS=(--kv-cache-quant-group-size "${GROUP_SIZE}")
fi""",
)

# 2. seed loop
old_run = """python "${RUNNER}" \\
    --task gpqa \\
    --model "${MODEL}" \\
    --base-url "http://127.0.0.1:${PORT}/v1" \\
    --max-tokens "${MAX_NEW_TOKENS}" \\
    --temperature "${TEMPERATURE:-1.0}" \\
    --top-p "${TOP_P:-0.95}" \\
    --top-k "${TOP_K:-40}" \\
    --n-repeats "${N_REPEATS}" \\
    --num-threads "${NUM_WORKERS:-32}" \\
    ${NUM_EXAMPLES:+--num-examples ${NUM_EXAMPLES}} \\
    --output-dir "${RUN_DIR}" \\
    2>&1 | tee "${LOG_RUNNER}"
echo "[eval-oscar] done. score:"
grep -iE "gpqa/score|gpqa/chars" "${RUN_DIR}/eval.log" | tail -10 || true"""

new_run = """for _SEED in ${SEEDS:-__none__}; do
    if [[ "${_SEED}" == "__none__" ]]; then
        _OUT="${RUN_DIR}"; _SEED_ARG=()
    else
        _OUT="${RUN_DIR}/seed${_SEED}"; _SEED_ARG=(--seed "${_SEED}")
        mkdir -p "${_OUT}"
    fi
    echo "[eval-oscar] === seed=${_SEED} -> ${_OUT} ==="
    python "${RUNNER}" \\
        --task gpqa \\
        --model "${MODEL}" \\
        --base-url "http://127.0.0.1:${PORT}/v1" \\
        --max-tokens "${MAX_NEW_TOKENS}" \\
        --temperature "${TEMPERATURE:-1.0}" \\
        --top-p "${TOP_P:-0.95}" \\
        --top-k "${TOP_K:-40}" \\
        --n-repeats "${N_REPEATS}" \\
        --num-threads "${NUM_WORKERS:-32}" \\
        "${_SEED_ARG[@]}" \\
        ${NUM_EXAMPLES:+--num-examples ${NUM_EXAMPLES}} \\
        --output-dir "${_OUT}" \\
        2>&1 | tee "${_OUT}/runner.log"
    echo "[eval-oscar] done seed=${_SEED}. score:"
    grep -iE "gpqa/score|gpqa/chars" "${_OUT}/eval.log" | tail -10 || true
done"""

assert old_run in s, "runner invocation block not found -- harness changed"
s = s.replace(old_run, new_run)

assert s != orig
assert 'HEALTH_WAIT_STEPS' in s and 'KV_MODE' in s and '_SEED_ARG' in s
open(path, "w").write(s)
print("patched", path)
