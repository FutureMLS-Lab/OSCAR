#!/usr/bin/env bash
# GPQA eval wrapper for Qwen/Qwen3-8B (base hybrid).
# Per-model defaults validated for OSCAR Qwen3-8B INT2 KV-cache.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-Qwen/Qwen3-8B}"
export ROT_DIR="${ROT_DIR:-${SCRIPT_DIR}/rotations}"
export RUN_DIR="${RUN_DIR:-$(dirname "${ROT_DIR}")/_eval_gpqa_oscar}"
export TP_SIZE="${TP_SIZE:-4}"
export GROUP_SIZE="${GROUP_SIZE:-128}"
export K_CLIP="${K_CLIP:-0.96}"
export V_CLIP="${V_CLIP:-0.92}"
# Qwen3-8B's published recipe is Lloyd-Max ON and a 512-token BF16 recent
# window, and both are load-bearing: PLAN.md measures each at roughly +3.5
# points on BFCL. The shared launcher defaults to LLOYD_MAX=0 and
# RECENT_TOKENS=256, so without these two lines this wrapper silently served a
# materially worse configuration than the one every published 8B number was
# measured with -- and nothing errors, the score just comes out low.
export LLOYD_MAX="${LLOYD_MAX:-1}"
export SGLANG_MIXED_KV_RECENT_TOKENS="${SGLANG_MIXED_KV_RECENT_TOKENS:-512}"
export NAME="${NAME:-gpqa_oscar_qwen3_8b}"

exec bash "${SCRIPT_DIR}/../eval_oscar_gpqa.sh"
