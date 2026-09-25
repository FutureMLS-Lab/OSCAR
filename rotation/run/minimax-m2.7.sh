#!/usr/bin/env bash
# MiniMax-M2.7 -- dense MHA (48 Q / 8 KV, head_dim 128), FP8 block-quantized MoE.
#
# TP_SIZE MUST keep the MoE shard divisible by the FP8 block width. The
# checkpoint has intermediate_size 1536 and weight_block_size [128, 128], so the
# per-rank gate/up output is 1536/TP and that has to be a multiple of 128:
#
#     TP=2 -> 768 (ok)   TP=4 -> 384 (ok)   TP=8 -> 192  ->
#     ValueError: The output_size of gate's and up's weight = 192
#                 is not divisible by weight quantization block_n = 128
#
# TP=8 is what an eight-GPU node invites, and it is exactly the value that fails.
# I first read that error as "wrong checkpoint variant"; the repo is correct --
# MiniMaxAI/MiniMax-M2.7 matches the recorded 48Q/8KV dense-MHA shape -- and the
# defect was the parallelism I chose.
source "$(dirname "$0")/_common.sh"
ROT_DIR="${ROT_DIR:-$ZOO/MiniMax-M2.7}"; need_rot "$ROT_DIR"
export ROT_DIR MODEL="${MODEL:-MiniMaxAI/MiniMax-M2.7}" TP_SIZE="${TP_SIZE:-4}"
case "$(( 1536 % (TP_SIZE * 128) ))" in
  0) : ;;
  *) echo "FATAL: TP_SIZE=$TP_SIZE gives a gate/up shard of $((1536 / TP_SIZE)), not a multiple of 128"; exit 1 ;;
esac
export GROUP_SIZE="${GROUP_SIZE:-128}"
export SGLANG_MIXED_KV_PREFIX_TOKENS="${SGLANG_MIXED_KV_PREFIX_TOKENS:-128}"
# 128/1024. GPQA against a BF16 baseline of 83.33:
#     64/256    80.30   (old default)
#     128/1024  83.33   -- exactly the baseline, delta 0.00
# The 3 pp previously charged to 2-bit KV was the window. Each model saturates
# at its own size -- Qwen3-30B-A3B at 128/1024, Qwen3-8B only at 128/2048 -- so
# this is swept per model rather than copied.
export SGLANG_MIXED_KV_RECENT_TOKENS="${SGLANG_MIXED_KV_RECENT_TOKENS:-1024}"
export LLOYD_MAX="${LLOYD_MAX:-1}"
export DISABLE_RADIX="${DISABLE_RADIX:-0}"   # prefix cache ON by default
export ATTN_BACKEND="${ATTN_BACKEND:-triton}" PREFILL_BACKEND="${PREFILL_BACKEND:-triton}"
launch
