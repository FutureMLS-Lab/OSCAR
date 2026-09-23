#!/usr/bin/env bash
# Gemma-4-12B-it -- heterogeneous SWA (40 sliding + 8 full attention).
#
# REQUIRES the :tf55 image tag. On the 5.3.0 baseline this model dies with
#   ValueError: 'Gemma4UnifiedForConditionalGeneration' is not a registered model
# because sglang resolves the architecture through transformers, which needs
# >= 5.5. The upgrade is kept on a separate tag so the nine models verified
# against 5.3.0 are not put at risk by one model's requirement.
source "$(dirname "$0")/_common.sh"
python3 - <<'PY' || exit 1
import transformers, sys
v = tuple(int(x) for x in transformers.__version__.split(".")[:2])
if v < (5, 5):
    print(f"FATAL: transformers {transformers.__version__} < 5.5 -- use a runtime image built with transformers >= 5.5")
    sys.exit(1)
print("transformers", transformers.__version__, "ok")
PY
ROT_DIR="${ROT_DIR:-$ZOO/Gemma4-12B}"; need_rot "$ROT_DIR"
export ROT_DIR MODEL="${MODEL:-google/gemma-4-12B-it}" TP_SIZE="${TP_SIZE:-1}"
export GROUP_SIZE="${GROUP_SIZE:-128}"
export SGLANG_MIXED_KV_PREFIX_TOKENS="${SGLANG_MIXED_KV_PREFIX_TOKENS:-64}"
export SGLANG_MIXED_KV_RECENT_TOKENS="${SGLANG_MIXED_KV_RECENT_TOKENS:-512}"
export ABSORB_V="${ABSORB_V:-1}"
export DISABLE_RADIX="${DISABLE_RADIX:-0}"   # prefix cache ON by default
export ATTN_BACKEND="${ATTN_BACKEND:-triton}" PREFILL_BACKEND="${PREFILL_BACKEND:-triton}"
launch
