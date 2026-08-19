#!/usr/bin/env bash
# Build a runtime from the SGLang image (for current model support), then
# overlay only the LUT-B research modules from this checkout.
set -euo pipefail

CUSTOM_ROOT="${1:?usage: $0 CUSTOM_COQUANT_ROOT TARGET_RUNTIME_ROOT}"
TARGET_ROOT="${2:?usage: $0 CUSTOM_COQUANT_ROOT TARGET_RUNTIME_ROOT}"
STOCK_PY="${STOCK_SGLANG_PYTHON:-/sgl-workspace/sglang/python}"
CUSTOM_PY="${CUSTOM_ROOT}/sglang-research/python"

mkdir -p "${TARGET_ROOT}/python"
cp -a "${STOCK_PY}/sglang" "${TARGET_ROOT}/python/"
cp -a "${CUSTOM_PY}/sglang/QuantKernel" "${TARGET_ROOT}/python/sglang/"

files=(
  sglang/srt/server_args.py
  sglang/srt/mem_cache/memory_pool.py
  sglang/srt/mem_cache/kv_quant_kernels.py
  sglang/srt/mem_cache/swa_memory_pool.py
  sglang/srt/mem_cache/lutb_format.py
  sglang/srt/mem_cache/lutb_kv_pool.py
  sglang/srt/model_executor/model_runner_kv_cache_mixin.py
  sglang/srt/layers/attention/quantized_kv_prefill.py
  sglang/srt/layers/attention/triton_backend.py
  sglang/srt/layers/attention/triton_ops/decode_attention.py
  sglang/srt/layers/attention/triton_ops/extend_attention.py
)

for relative in "${files[@]}"; do
  install -D -m 0644 "${CUSTOM_PY}/${relative}" "${TARGET_ROOT}/python/${relative}"
done

printf '%s\n' "stock_image_commit=0b3bb0c" >"${TARGET_ROOT}/BUILD_INFO"
