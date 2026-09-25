#!/usr/bin/env bash
# Qwen3.5-35B-A3B -- hybrid linear attention, shared rotation + Lloyd-Max, g256.
# Prefix cache ON via extra_buffer, same as the 4B: the page_size==1 assertion
# is guarded by `if not self.enable_mamba_extra_buffer`.
source "$(dirname "$0")/_common.sh"
ROT_DIR="${ROT_DIR:-$ZOO/Qwen3.5-35B-A3B}"; need_rot "$ROT_DIR"
export ROT_DIR MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B}" TP_SIZE="${TP_SIZE:-4}"
export GROUP_SIZE="${GROUP_SIZE:-256}"
export SGLANG_MIXED_KV_PREFIX_TOKENS="${SGLANG_MIXED_KV_PREFIX_TOKENS:-64}"
export SGLANG_MIXED_KV_RECENT_TOKENS="${SGLANG_MIXED_KV_RECENT_TOKENS:-256}"
export LLOYD_MAX="${LLOYD_MAX:-1}"
export DISABLE_RADIX="${DISABLE_RADIX:-0}"
export MAMBA_SCHEDULER_STRATEGY="${MAMBA_SCHEDULER_STRATEGY:-extra_buffer}"
export ATTN_BACKEND="${ATTN_BACKEND:-triton}" PREFILL_BACKEND="${PREFILL_BACKEND:-triton}"
# Qwen3.5 thinking-mode sampling, from the model's own documentation:
# temperature 1.0 / top_p 0.95 / top_k 20 / presence_penalty 1.5. The eval
# harness defaults to top_k 40 with no presence penalty, which is not any
# model's recommended configuration, and Qwen3.5 documents the penalty as the
# knob that "reduces endless repetitions" -- exactly the failure seen here.
export TOP_K="${TOP_K:-20}" PRESENCE_PENALTY="${PRESENCE_PENALTY:-1.5}"
# THIS MODEL REQUIRES transformers 5.3.0. sglang-research's own pyproject.toml
# pins that version; the image ships 5.16.1 because Gemma-4 needs >= 5.5 for
# Gemma4UnifiedForConditionalGeneration. On 5.16.1 this model never stops
# reasoning on GPQA -- </think> closes in 1-4 of 198 responses and the score is
# 0.0, in BF16 as well as INT2, so it reads like a quantization collapse and is
# not one. Measured, same harness, BF16, n=48:
#     5.16.1  </think> 1-4/198   0.0
#     5.5.0   </think> 2/24      still looping
#     5.4.0   </think> 48/48     83.33
#     5.3.0   </think> 48/48     91.67
# The regression enters at 5.5.0 -- exactly where Gemma-4's support begins -- so
# no single version serves both. Everything comparable between the two is
# IDENTICAL: config fields, the 93-entry layer_types, the chat template rendered
# through both AutoTokenizer and AutoProcessor, token ids, eos, and all 111
# ServerArgs. The root cause is still open.
#
# /oscar/tf53 is a 112 MB overlay holding transformers 5.3.0 and the
# tokenizers 0.22.2 it pins. Prepending it to PYTHONPATH gives this one model
# the versions it needs without a second venv, without a run-time pip install,
# and without disturbing the other eleven models in the same container.
TF_OVERLAY="${TF_OVERLAY:-/oscar/tf53}"
if [ -d "$TF_OVERLAY" ]; then
  export PYTHONPATH="$TF_OVERLAY:${PYTHONPATH:-}"
  echo "[run] transformers overlay: $TF_OVERLAY ($(PYTHONPATH=$TF_OVERLAY python3 -c 'import transformers;print(transformers.__version__)' 2>/dev/null))"
else
  echo "[run] WARNING: $TF_OVERLAY missing; this model needs transformers 5.3.0 and will loop on 5.5+"
fi
launch
