#!/usr/bin/env bash
# DECODE speed at a 64K context: INT2 OSCAR against a BF16 control.
#
#   examples/bench_decode64k.sh qwen3-8b
#
# Both modes run through rotation/run/<model>.sh, so the model, TP, attention
# backends, radix cache, cuda-graph batch size and memory fraction are identical
# and the only difference is the KV path. That is what makes the ratio a
# statement about the KV path rather than about two different harnesses.
#
# The headline is DECODE, measured after the 64K prefill has already happened:
# what it costs to read a 64K KV cache once per generated token, which is the
# thing a 2-bit cache exists to make cheaper. Prefill is still reported, but it
# measures something else -- on Blackwell FA3 does not support the int2 path
# (use_sdpa = head_dim > 256 or not _is_fa3_supported()), so INT2 prefill falls
# back to SDPA and its ratio is a statement about that fallback. The two ratios
# point in opposite directions, so they are kept separate rather than merged.
set -uo pipefail
MODEL_KEY=${1:?usage: bench_decode64k.sh <model-key>   (see rotation/run/)}
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${OSCAR_SRC:-$(cd -- "$HERE/.." && pwd)}"
RUN="$ROOT/rotation/run/${MODEL_KEY}.sh"
[ -x "$RUN" ] || { echo "no run script for '$MODEL_KEY'"; exit 1; }

TOK=${BENCH_PREFILL_TOKENS:-65536}
BASE=${BENCH_BASE:-${RUN_DIR:-/tmp}/bench64k/$MODEL_KEY}
mkdir -p "$BASE"
export BENCH_PREFILL=1 BENCH_PREFILL_TOKENS="$TOK"
# The context window must admit the prompt plus the generation. Models whose
# config says less than 64K need the override, which is why the eval path
# already exports SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1.
export EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-} --context-length $((TOK + 2048))"

for MODE in int2 bf16; do
  echo "=================== $MODEL_KEY / $MODE / ${TOK} prefill ==================="
  ( export KV_MODE="$MODE" RUN_DIR="$BASE/$MODE"; mkdir -p "$RUN_DIR"; bash "$RUN" )
done

python3 - "$BASE" "$MODEL_KEY" "$TOK" <<'PY'
import json, os, sys
base, key, tok = sys.argv[1], sys.argv[2], int(sys.argv[3])
def load(mode):
    p = os.path.join(base, mode, f"bench_{mode}.json")
    try:
        return json.load(open(p))
    except Exception:
        return None
i, b = load("int2"), load("bf16")
if not i or not b:
    print(f"BENCH {key}: incomplete (int2={'ok' if i else 'MISSING'} bf16={'ok' if b else 'MISSING'})")
    sys.exit(1)
# Slower-is-bigger, so the ratio reads directly as "how much slower".
r_ttft = i["ttft_s_median"] / b["ttft_s_median"]
r_dec  = i["decode_s_per_token_median"] / b["decode_s_per_token_median"]
out = {"model": key, "context_tokens": tok,
       "int2_decode_ms": i["decode_s_per_token_median"] * 1000,
       "bf16_decode_ms": b["decode_s_per_token_median"] * 1000,
       "int2_decode_tps": i["decode_tok_per_s_median"],
       "bf16_decode_tps": b["decode_tok_per_s_median"],
       "decode_ratio_int2_over_bf16": r_dec,
       "int2_ttft_s": i["ttft_s_median"], "bf16_ttft_s": b["ttft_s_median"],
       "prefill_ratio_int2_over_bf16": r_ttft}
json.dump(out, open(os.path.join(base, "ratio.json"), "w"), indent=2)
print(f"BENCH {key} @{tok}ctx: DECODE {i['decode_s_per_token_median']*1000:.2f} vs "
      f"{b['decode_s_per_token_median']*1000:.2f} ms/tok = {r_dec:.2f}x "
      f"({i['decode_tok_per_s_median']:.1f} vs {b['decode_tok_per_s_median']:.1f} tok/s) "
      f"| prefill {r_ttft:.2f}x")
PY
