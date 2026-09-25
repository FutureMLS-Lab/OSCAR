#!/usr/bin/env bash
# GLM-4.7-FP8 -- dense MHA INT2 (paper configuration).
source "$(dirname "$0")/_common.sh"
ROT_DIR="${ROT_DIR:-$ZOO/GLM-4.7-FP8/seq10000_prompt43_group128}"; need_rot "$ROT_DIR"
export ROT_DIR MODEL="${MODEL:-zai-org/GLM-4.7-FP8}" TP_SIZE="${TP_SIZE:-8}"
export GROUP_SIZE="${GROUP_SIZE:-128}"
export SGLANG_MIXED_KV_PREFIX_TOKENS="${SGLANG_MIXED_KV_PREFIX_TOKENS:-64}"
export SGLANG_MIXED_KV_RECENT_TOKENS="${SGLANG_MIXED_KV_RECENT_TOKENS:-256}"
export DISABLE_RADIX="${DISABLE_RADIX:-0}"   # prefix cache ON by default
export ATTN_BACKEND="${ATTN_BACKEND:-triton}" PREFILL_BACKEND="${PREFILL_BACKEND:-triton}"
launch
