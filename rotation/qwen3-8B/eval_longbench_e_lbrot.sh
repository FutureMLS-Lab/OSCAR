#!/usr/bin/env bash
# LongBench-E eval wrapper for Qwen/Qwen3-8B (OSCAR INT2 KV cache).
# Defaults to the LongBench-calibrated data-aware OSCAR rotation
# (composition: R · H · Π_BR) at group_size=32 and clip 0.96/0.92 — the
# config that produced longbench_e_mean=50.25, beating OScaR-KV-Quant's 48.74.
# Override ROT_DIR/GROUP_SIZE/K_CLIP/V_CLIP via env to sweep configs.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

ROT_ROOT_DEFAULT="${SCRIPT_DIR}/LongBench/seq150000_prompt44_group128/rotations"

export MODEL="${MODEL:-Qwen/Qwen3-8B}"
export ROT_DIR="${ROT_DIR:-${ROT_ROOT_DEFAULT}}"
export RUN_DIR="${RUN_DIR:-/home/charlie/CoQuant/.RUD/task/runs/longbench_e_qwen3_8b_lb_rot}"
export TP_SIZE="${TP_SIZE:-4}"
export GROUP_SIZE="${GROUP_SIZE:-32}"
export K_CLIP="${K_CLIP:-0.96}"
export V_CLIP="${V_CLIP:-0.92}"
export MAX_INPUT_LEN="${MAX_INPUT_LEN:-32768}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export USE_CHAT="${USE_CHAT:-true}"

exec bash "${SCRIPT_DIR}/../eval_oscar_longbench_e.sh"
