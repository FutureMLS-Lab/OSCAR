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
export SGLANG_MIXED_KV_RECENT_TOKENS="${SGLANG_MIXED_KV_RECENT_TOKENS:-2048}"
export LLOYD_MAX="${LLOYD_MAX:-1}"
export DISABLE_RADIX="${DISABLE_RADIX:-0}"   # prefix cache ON by default
export ATTN_BACKEND="${ATTN_BACKEND:-triton}" PREFILL_BACKEND="${PREFILL_BACKEND:-triton}"
launch
