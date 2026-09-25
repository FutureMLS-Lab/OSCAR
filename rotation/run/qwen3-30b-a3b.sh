#!/usr/bin/env bash
# Qwen3-30B-A3B -- MoE with only 4 KV heads.
#
# This model is the reason PER-HEAD rotations exist. A single shared per-layer
# rotation collapses it (GPQA 43.9 vs 58.6, HumanEval 25.8 vs 89.0) because with
# 4 KV heads the heads are near-orthogonal (mean |diag(R0^T R1)| ~= 0.07), so one
# basis cannot whiten all of them. The zoo ships BOTH formats in one directory:
#
#   k_rotation_perhead.pt        format_version 2, [4, 128, 128] per layer  <- use
#   k_rotation_qqt_r_h_pbr.pt    format_version 1, one shared [128, 128]    <- comparison only
#
# The pool logs "[per-head: 4 kv heads]" when V2 is active. Under TP the V2
# checkpoint ships every KV head and each rank slices its own -- UnifiedInt2HPKVPool
# does that already; MHATokenToKVPool did not until 2026-09-04.
#
# No Lloyd-Max here: this model uses uniform codebooks (see the recipe table).
source "$(dirname "$0")/_common.sh"
# The names must be set BEFORE need_rot: the directory ships both formats, and
# need_rot refuses to guess between them. Exporting them after the call left the
# variables empty at the moment it looked, so it aborted with "holds several
# k_rotation_*.pt" on the one model that most needs the explicit name.
export K_ROTATION_FILE="${K_ROTATION_FILE:-k_rotation_perhead.pt}"
export V_ROTATION_FILE="${V_ROTATION_FILE:-v_rotation_perhead.pt}"
ROT_DIR="${ROT_DIR:-$ZOO/Qwen3-30B-A3B}"; need_rot "$ROT_DIR"
export ROT_DIR MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}" TP_SIZE="${TP_SIZE:-2}"
export GROUP_SIZE="${GROUP_SIZE:-128}"
# 128/1024, not 64/256. This model holds only 4 KV heads, and the window is the
# dominant term in its accuracy -- GPQA rises monotonically with it:
#   64/256   55.56      (the old default)
#   64/512   60.61
#   128/1024 62.12      against a BF16 baseline of 61.01
# The 5.5 pp that used to be recorded as a quantization loss was the window
# being too small; at 128/1024 INT2 is 1.1 pp AHEAD of BF16.
export SGLANG_MIXED_KV_PREFIX_TOKENS="${SGLANG_MIXED_KV_PREFIX_TOKENS:-128}"
export SGLANG_MIXED_KV_RECENT_TOKENS="${SGLANG_MIXED_KV_RECENT_TOKENS:-1024}"
export DISABLE_RADIX="${DISABLE_RADIX:-0}"
export ATTN_BACKEND="${ATTN_BACKEND:-triton}" PREFILL_BACKEND="${PREFILL_BACKEND:-triton}"
launch
