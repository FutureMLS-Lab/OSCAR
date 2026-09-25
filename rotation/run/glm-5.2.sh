#!/usr/bin/env bash
# GLM-5.2-FP8 -- MLA, packed 2-bit latent at 4.00x.
#
# MLA_ROT_PATH is the branch selector in eval_oscar_gpqa.sh: non-empty takes the
# MLA path, empty falls back to the MHA path and forces int2, which DSA rejects.
# MLA_KV_CACHE_DTYPE must be pinned: sglang picks fp8_e4m3 for a DSA model on
# SM100+, which puts ~28% of every 512x512 rotation subnormal (that arm scored 5.56).
source "$(dirname "$0")/_common.sh"
LAT="${LAT:-${OSCAR_ROTATIONS:-/oscar/rotations}/glm52-rotations}"
[ "$(ls "$LAT"/layer_*.pt 2>/dev/null | wc -l)" -gt 0 ] || { echo "FATAL: no latent rotations at $LAT"; exit 1; }
echo "[run] latent rotations: $LAT ($(ls "$LAT"/layer_*.pt | wc -l) layers)"
export MLA_ROT_PATH="$LAT" ROT_DIR="$LAT"
export MODEL="${MODEL:-zai-org/GLM-5.2-FP8}" TP_SIZE="${TP_SIZE:-8}"
export MLA_KV_CACHE_DTYPE=bfloat16
export MLA_GROUP_SIZE="${MLA_GROUP_SIZE:-128}"
export MLA_PACKED=1 MLA_PACKED_SELFCHECK=0
export SGLANG_MIXED_KV_PREFIX_TOKENS="${SGLANG_MIXED_KV_PREFIX_TOKENS:-64}"
export SGLANG_MIXED_KV_RECENT_TOKENS="${SGLANG_MIXED_KV_RECENT_TOKENS:-512}"
export ATTN_BACKEND="${ATTN_BACKEND:-triton}" PREFILL_BACKEND="${PREFILL_BACKEND:-triton}"
launch
