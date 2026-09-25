#!/usr/bin/env bash
# Qwen3.5-4B -- HYBRID linear attention (32 layers = 24 linear + 8 full).
#
# The prefix cache stays ON. A hybrid model selects MambaRadixCache, which
# asserts page_size==1 while the INT2 KV path uses page 8:
#     AssertionError: Page size must be 1 for MambaRadixCache v1, got 8
# That assertion is guarded by `if not self.enable_mamba_extra_buffer`, so
# --mamba-scheduler-strategy extra_buffer lifts it and the cache (worth ~4.7%
# here) keeps working. An earlier version of this file disabled the cache and
# called that mandatory -- a workaround written before the guard was read.
#
# The failure is worth knowing because it lies: it surfaces asynchronously as
# "CUDA error: an illegal memory access" inside attn_lse.fill_(), which cost two
# wrong diagnoses (group size, then a rotation mismatch) before
# CUDA_LAUNCH_BLOCKING=1 exposed the real assert.
#
# The rotation covers exactly the 8 full-attention layers (ids 3,7,11,...), which
# is correct, not a truncated file.
source "$(dirname "$0")/_common.sh"
ROT_DIR="${ROT_DIR:-$ZOO/Qwen3.5-4B}"; need_rot "$ROT_DIR"
# Qwen/Qwen3.5-4B, the native model. This defaulted to
# togethercomputer/reducto-qwen3.5-4b, which is a different checkpoint with no
# model card, and the whole row was therefore reporting a fine-tune's score
# against a table of native models: measured BF16 41.4 against the published
# 76.2 for Qwen3.5-4B. INT2 was never the problem there -- INT2 43.4 vs BF16
# 41.4 on that checkpoint -- but the number was not about the model it named.
export ROT_DIR MODEL="${MODEL:-Qwen/Qwen3.5-4B}" TP_SIZE="${TP_SIZE:-1}"
export GROUP_SIZE="${GROUP_SIZE:-256}"          # g256, per the recipe table
export SGLANG_MIXED_KV_PREFIX_TOKENS="${SGLANG_MIXED_KV_PREFIX_TOKENS:-64}"
export SGLANG_MIXED_KV_RECENT_TOKENS="${SGLANG_MIXED_KV_RECENT_TOKENS:-256}"
export DISABLE_RADIX="${DISABLE_RADIX:-0}"
# A hybrid model selects MambaRadixCache, whose page_size==1 assertion the
# INT2 page-8 layout violates -- but that assertion is guarded by
#   if not self.enable_mamba_extra_buffer
# so the extra buffer lifts it and the prefix cache (worth ~4.7% here) stays
# on. An earlier version of this file hardcoded DISABLE_RADIX=1 and called it
# mandatory; that was a workaround written before the guard was read.
export MAMBA_SCHEDULER_STRATEGY="${MAMBA_SCHEDULER_STRATEGY:-extra_buffer}"
export ATTN_BACKEND="${ATTN_BACKEND:-triton}" PREFILL_BACKEND="${PREFILL_BACKEND:-triton}"
# Qwen3.5 thinking-mode sampling, from the model's own documentation:
# temperature 1.0 / top_p 0.95 / top_k 20 / presence_penalty 1.5. The eval
# harness defaults to top_k 40 with no presence penalty, which is not any
# model's recommended configuration, and Qwen3.5 documents the penalty as the
# knob that "reduces endless repetitions" -- exactly the failure seen here.
export TOP_K="${TOP_K:-20}" PRESENCE_PENALTY="${PRESENCE_PENALTY:-1.5}"
launch
