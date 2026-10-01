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
ROOT="${OSCAR_SRC:-$(cd -- "$HERE/../.." && pwd)}"
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

# Three arms, not two.
#
#   int2            our KV path, on the backends it actually uses
#   bf16            BF16 KV on the SAME backends -- isolates the KV path
#   bf16_flashinfer BF16 KV on the FlashInfer-family decode backend, i.e. what
#                   a standard sglang deployment would run
#
# The first ratio answers "what does our KV path cost against BF16, all else
# equal". The second answers "how do we compare to what people actually run",
# which is a different question and the one a reader usually means. Reporting
# only the first invites the reply that the baseline was hobbled.
#
# FlashInfer's MLA kernels ship as trtllm_mla; plain flashinfer cannot serve an
# MLA model at all (it asserts head_dim_qk == head_dim_vo, which is 192 vs 128
# on Kimi-K3). So the baseline backend is per-geometry, and a model that the
# baseline genuinely cannot serve is reported as such rather than skipped
# silently.
# The baseline backend is per model family, and "FlashInfer family" means the
# backend upstream sglang actually deploys that model on:
#   GLM-5.x   DSA sparse attention. Its MLA is qk 256 / v 256 (qk_nope 192 +
#             rope 64, v 256), which plain trtllm_mla rejects outright ("only
#             support deepseek r1 192/128 or 128/128"); upstream serves GLM-5
#             with --attention-backend dsa (named `nsa` in this fork) whose
#             prefill/decode kernels are flashmla_sparse / trtllm.
#   Kimi-K3   trtllm_mla, as in upstream's day-0 K3 support. Its MLA is the
#             DeepSeek-R1 192/128 shape that kernel is built for; what broke
#             here before was the hybrid wrapper not exposing .data_type.
#   others    flashinfer (Gemma-4's 512-wide global heads are the one MHA
#             geometry it has no tile for, and that is reported as such).
case "$MODEL_KEY" in
  glm-5*)  FI_BACKEND="${FI_BACKEND:-dsa}" ;;
  kimi-k3) FI_BACKEND="${FI_BACKEND:-trtllm_mla}" ;;
  *)       if grep -qE 'MLA_ROT_PATH|MLA_PACKED' "$RUN" 2>/dev/null; then FI_BACKEND="${FI_BACKEND:-trtllm_mla}"; else FI_BACKEND="${FI_BACKEND:-flashinfer}"; fi ;;
esac
echo "baseline decode backend for $MODEL_KEY: $FI_BACKEND"

for MODE in int2 bf16 bf16_flashinfer; do
  # Idempotent per arm: an arm whose measurement already exists is not re-run.
  # Re-measuring the K3 baseline after a one-line backend fix should cost the
  # one arm that failed, not two more hours of 16-GPU weight loading for the
  # two arms that did not.
  if ls "$BASE/$MODE"/bench_*.json >/dev/null 2>&1; then
    echo "=================== $MODEL_KEY / $MODE: already measured, skipping ==================="
    continue
  fi
  echo "=================== $MODEL_KEY / $MODE / ${TOK} prefill ==================="
  (
    if [ "$MODE" = "bf16_flashinfer" ]; then
      export KV_MODE=bf16 RUN_DIR="$BASE/$MODE"
      export ATTN_BACKEND="$FI_BACKEND" PREFILL_BACKEND="$FI_BACKEND" DECODE_BACKEND="$FI_BACKEND"
      [ -n "${FI_KV_DTYPE:-}" ] && export MLA_KV_CACHE_DTYPE="$FI_KV_DTYPE"
      # Autotune is what a real flashinfer deployment runs; leaving it off
      # would understate the baseline.
      unset NO_AUTOTUNE
    else
      export KV_MODE="$MODE" RUN_DIR="$BASE/$MODE"
    fi
    mkdir -p "$RUN_DIR"; bash "$RUN"
  )
done

python3 - "$BASE" "$MODEL_KEY" "$TOK" <<'PY'
import json, os, sys
base, key, tok = sys.argv[1], sys.argv[2], int(sys.argv[3])
def load(mode, kv=None):
    p = os.path.join(base, mode, f"bench_{kv or mode}.json")
    try:
        return json.load(open(p))
    except Exception:
        return None
# The flashinfer arm runs with KV_MODE=bf16, so the runner names its file
# bench_bf16.json inside the bf16_flashinfer directory. Looking for
# bench_bf16_flashinfer.json found nothing and every model printed "baseline
# did not serve this model" while the baseline had served all of them.
i, b, f = load("int2"), load("bf16"), load("bf16_flashinfer", kv="bf16")
if not i or not b:
    print(f"BENCH {key}: incomplete (int2={'ok' if i else 'MISSING'} bf16={'ok' if b else 'MISSING'})")
    sys.exit(1)
# Slower-is-bigger, so the ratio reads directly as "how much slower".
r_ttft = i["ttft_s_median"] / b["ttft_s_median"]
r_dec  = i["decode_s_per_token_median"] / b["decode_s_per_token_median"]
fi_ms = f["decode_s_per_token_median"] * 1000 if f else None
fi_tps = f["decode_tok_per_s_median"] if f else None
r_fi = (i["decode_s_per_token_median"] / f["decode_s_per_token_median"]) if f else None
out = {"model": key, "context_tokens": tok,
       "flashinfer_decode_ms": fi_ms, "flashinfer_decode_tps": fi_tps,
       "decode_ratio_int2_over_flashinfer": r_fi,
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
if f:
    print(f"BENCH {key} @{tok}ctx vs FLASHINFER: {i['decode_s_per_token_median']*1000:.2f} vs "
          f"{fi_ms:.2f} ms/tok = {r_fi:.2f}x "
          f"({i['decode_tok_per_s_median']:.1f} vs {fi_tps:.1f} tok/s)")
else:
    print(f"BENCH {key} @{tok}ctx vs FLASHINFER: baseline did not serve this model")
PY
