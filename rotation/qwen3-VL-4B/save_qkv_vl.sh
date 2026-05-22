#!/usr/bin/env bash
# Phase 2 E2: Dump post-RoPE Q/K/V tensors for Qwen3-VL-4B-Instruct using
# OCRBench multimodal prompts. Mirrors qwen3-VL-8B/save_qkv_vl.sh.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export MODEL="${MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
export TP_SIZE="${TP_SIZE:-2}"
export PORT="${PORT:-31062}"
export DIST_PORT="${DIST_PORT:-41062}"
export DATASET="${DATASET:-OCRBench}"
export GROUP_SIZE="${GROUP_SIZE:-32}"
export CALIB_BASE_DIR="${CALIB_BASE_DIR:-${SCRIPT_DIR}}"
export DUMP_KVCACHE_TOKENS="${DUMP_KVCACHE_TOKENS:-150000}"
export NUM_PROMPTS="${NUM_PROMPTS:-150}"

# Delegate to the VL-8B script — it's already parameterised.
exec bash "${REPO_ROOT}/rotation/qwen3-VL-8B/save_qkv_vl.sh"
