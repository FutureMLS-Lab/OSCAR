#!/usr/bin/env bash
# MiniMax-M3 -- VL-MoE, 428B, TP=8.
#
# Its rotations are the PUBLISHED Hadamard set (k_rotation_hadamard.pt), not a
# full-rank calibration like the other models, so the filename differs from the
# qqt_r_h_pbr form. A harness that hardcodes the usual name reports "no rotation"
# and the model reads as unsupported when the files were there all along.
#
# Uniform codebooks, no Lloyd-Max.
source "$(dirname "$0")/_common.sh"
ROT_DIR="${ROT_DIR:-$ZOO/MiniMax-M3}"; need_rot "$ROT_DIR"
export ROT_DIR MODEL="${MODEL:-MiniMaxAI/MiniMax-M3}" TP_SIZE="${TP_SIZE:-8}"
export GROUP_SIZE="${GROUP_SIZE:-128}"
export SGLANG_MIXED_KV_PREFIX_TOKENS="${SGLANG_MIXED_KV_PREFIX_TOKENS:-64}"
export SGLANG_MIXED_KV_RECENT_TOKENS="${SGLANG_MIXED_KV_RECENT_TOKENS:-256}"
export DISABLE_RADIX="${DISABLE_RADIX:-0}"
export ATTN_BACKEND="${ATTN_BACKEND:-triton}" PREFILL_BACKEND="${PREFILL_BACKEND:-triton}"
launch
