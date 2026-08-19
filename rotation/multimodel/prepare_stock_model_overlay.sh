#!/usr/bin/env bash
# Keep the experimental LUT-B runtime while taking new-model support from the
# exact SGLang image used by the Kubernetes jobs.
set -euo pipefail

TARGET_ROOT="${1:?usage: $0 TARGET_COQUANT_ROOT}"
STOCK_ROOT="${STOCK_SGLANG_ROOT:-/sgl-workspace/sglang/python/sglang}"
TARGET_PY="${TARGET_ROOT}/sglang-research/python/sglang"

files=(
  srt/configs/qwen3_5.py
  srt/models/qwen3_5.py
  srt/models/qwen3_vl.py
  srt/models/gemma4_mm.py
  srt/models/gemma4_causal.py
  srt/models/gemma4_audio.py
  srt/models/gemma4_vision.py
  srt/layers/layernorm.py
  srt/multimodal/processors/gemma4.py
  srt/utils/hf_transformers/__init__.py
  srt/utils/hf_transformers/common.py
  srt/utils/hf_transformers/config.py
  srt/utils/hf_transformers/mistral_utils.py
  srt/utils/hf_transformers/processor.py
  srt/utils/hf_transformers/tokenizer.py
  srt/utils/hf_transformers_patches.py
  srt/utils/hf_transformers_utils.py
)

for relative in "${files[@]}"; do
  install -D -m 0644 "${STOCK_ROOT}/${relative}" "${TARGET_PY}/${relative}"
done

printf '%s\n' "stock_image_commit=0b3bb0c" >"${TARGET_ROOT}/.stock-model-overlay"
