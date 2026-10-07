#!/usr/bin/env bash
# Qwen3.8-27B -- dense hybrid, 64 layers = 16 x (3 GDN + 1 gated attention).
# Only the 16 full-attention layers (3, 7, ..., 63) hold a KV cache: 24 Q / 4 KV
# heads at head_dim 256, so the rotation covers exactly those layers. Same model
# class as Qwen3.5 (Qwen3_5ForConditionalGeneration, VL checkpoint served
# text-only); native context 262,144, so 128K windows need no YaRN.
# Prefix cache ON via extra_buffer, same as the Qwen3.5 rows: the page_size==1
# assertion is guarded by `if not self.enable_mamba_extra_buffer`.
source "$(dirname "$0")/_common.sh"
ROT_DIR="${ROT_DIR:-$ZOO/Qwen3.8-27B}"; need_rot "$ROT_DIR"
export ROT_DIR MODEL="${MODEL:-Qwen/Qwen3.8-27B}" TP_SIZE="${TP_SIZE:-2}"
export GROUP_SIZE="${GROUP_SIZE:-256}"
export SGLANG_MIXED_KV_PREFIX_TOKENS="${SGLANG_MIXED_KV_PREFIX_TOKENS:-64}"
export SGLANG_MIXED_KV_RECENT_TOKENS="${SGLANG_MIXED_KV_RECENT_TOKENS:-256}"
export LLOYD_MAX="${LLOYD_MAX:-0}"
export DISABLE_RADIX="${DISABLE_RADIX:-0}"
export MAMBA_SCHEDULER_STRATEGY="${MAMBA_SCHEDULER_STRATEGY:-extra_buffer}"
export ATTN_BACKEND="${ATTN_BACKEND:-triton}" PREFILL_BACKEND="${PREFILL_BACKEND:-triton}"
# Thinking-mode sampling from the model card: temperature 1.0 / top_p 0.95 /
# top_k 20 / presence_penalty 0.0. Unlike Qwen3.5, the card reserves the 1.5
# presence penalty for the non-thinking mode.
export TOP_K="${TOP_K:-20}" PRESENCE_PENALTY="${PRESENCE_PENALTY:-0}"
# The checkpoint is written for transformers >= 5.8 (config transformers_version
# 5.8.0.dev0); the image's 5.16.1 serves it. Do not prepend the Qwen3.5-35B
# /oscar/tf53 overlay (transformers 5.3.0) here.
launch
