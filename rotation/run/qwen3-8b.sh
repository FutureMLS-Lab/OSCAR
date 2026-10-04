#!/usr/bin/env bash
# Qwen3-8B -- dense MHA. Verified serving INT2 with correct answers.
source "$(dirname "$0")/_common.sh"
ROT_DIR="${ROT_DIR:-$ZOO/Qwen3-8B/seq20000_prompt83_group128}"; need_rot "$ROT_DIR"
export ROT_DIR MODEL="${MODEL:-Qwen/Qwen3-8B}" TP_SIZE="${TP_SIZE:-1}"
export GROUP_SIZE="${GROUP_SIZE:-128}"
# 64 / 512, not the 64/256 the other Qwen3 rows use.
export SGLANG_MIXED_KV_PREFIX_TOKENS="${SGLANG_MIXED_KV_PREFIX_TOKENS:-128}"
# 128/2048. GPQA against a BF16 baseline of 58.08:
#     64/512    55.05   (old default)
#     128/1024  55.56
#     128/2048  58.08   -- exactly the baseline, delta 0.00
# The 3 pp previously charged to 2-bit KV was the window. Note this model needs
# a LARGER window than Qwen3-30B-A3B, which saturates at 128/1024: the setting
# does not transfer between models and has to be swept per model.
# 2026-10-04: recent window 512 (was 2048, the GPQA-neutral setting of the
# sweep above) so the INT2 tier carries more of the context; the 128/512
# numbers are re-measured on the 64K PPL metric and GPQA.
export SGLANG_MIXED_KV_RECENT_TOKENS="${SGLANG_MIXED_KV_RECENT_TOKENS:-512}"
# Uniform per-row min-max levels (LLOYD_MAX=0). On the clean 32K PPL metric
# (tokens past 20480, 16K INT2 keys) the Lloyd-Max std-loaded levels cost
# 1.3-3 points that uniform does not: shared basis 9.265 vs 8.989 against
# BF16 8.991; GPQA@64K agrees in direction on four paired seeds (+0.5 to
# +5.6 pp, none significant alone).
export LLOYD_MAX="${LLOYD_MAX:-0}"
export DISABLE_RADIX="${DISABLE_RADIX:-0}"   # prefix cache ON by default
export ATTN_BACKEND="${ATTN_BACKEND:-triton}" PREFILL_BACKEND="${PREFILL_BACKEND:-triton}"
# Qwen3 (not 3.5, not the 2507 -Thinking) is trained to 40960 positions. A 64K
# generation budget, or a 64K prompt, does not fit: the server rejects every
# request with 400 and -- before run_simple_eval learned to abort on it -- the
# eval scored those rejections as 198 wrong answers, metrics.json said 0.0, and
# it looked like a real number. When the requested context exceeds the native
# window, enable Qwen's documented long-context recipe (YaRN x4 over the 32768
# training length, 131072 max) and widen the window. Applied identically to
# the INT2 and BF16 arms, so their comparison is unaffected.
_req_ctx=$(( ${MAX_NEW_TOKENS:-32768} > ${BENCH_PREFILL_TOKENS:-0} ? ${MAX_NEW_TOKENS:-32768} : ${BENCH_PREFILL_TOKENS:-0} ))
if (( _req_ctx > 32768 )); then
  # Three keys, not one. The first attempt set only the v4 key `rope_scaling`
  # and it never reached the model: under transformers 5 sglang reads
  # `config.rope_parameters` (get_rope_config), so the dense models silently
  # kept plain RoPE and extrapolated past 40960 -- the server warned
  # "context_length (81920) greater than derived (40960)" four times and the
  # 64K scores it produced were from a misconfigured model. The MoE crashed
  # instead: get_rope_config indexes rope_parameters["rope_theta"], which a
  # bare yarn dict does not carry. And get_context_length forces the factor
  # to 1 when original_max_position_embeddings is present, so the derived
  # window only grows if max_position_embeddings itself is raised.
  #   rope_parameters   v5 key the model reads; includes rope_theta (1e6 for Qwen3)
  #   rope_scaling      v4 mirror get_context_length reads
  #   max_position_embeddings  32768 x 4 = 131072, Qwen's published YaRN limit
  # Verify on the server log: NO "greater than the derived context_length".
  _yarn='{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768,"rope_theta":1000000.0}'
  export EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-} --context-length $(( _req_ctx + 16384 )) --json-model-override-args {\"rope_parameters\":${_yarn},\"rope_scaling\":${_yarn},\"max_position_embeddings\":131072}"
  echo "[run] requested context ${_req_ctx} > native 40960: YaRN x4 enabled (rope_parameters+rope_scaling+max_position_embeddings=131072), --context-length $(( _req_ctx + 16384 ))"
fi

launch
