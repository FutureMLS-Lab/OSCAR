#!/usr/bin/env bash
# OCRBench eval wrapper for Qwen/Qwen3-VL-4B-Instruct (OSCAR INT2 KV cache).
# Defaults: OCRBench-calibrated data-aware OSCAR rotation (R · H · Π_BR)
# + group_size=128 + Lloyd-Max INT2 kernel + clip 0.96/0.92.
# Memory-saver config — 4× smaller scales/zeros table vs g=32; score 848/1000.
# Set ROT_DIR=${SCRIPT_DIR}/Hadamard for the data-free Hadamard baseline.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

ROT_ROOT_DEFAULT="${SCRIPT_DIR}/OCRBench/seq150000_prompt160_group128/rotations"

export MODEL="${MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
export ROT_DIR="${ROT_DIR:-${ROT_ROOT_DEFAULT}}"
export RUN_DIR="${RUN_DIR:-/home/charlie/CoQuant/.RUD/task/runs/ocrbench_qwen3vl4b}"
export TP_SIZE="${TP_SIZE:-1}"
export GROUP_SIZE="${GROUP_SIZE:-128}"
export K_CLIP="${K_CLIP:-0.96}"
export V_CLIP="${V_CLIP:-0.92}"
export NUM_WORKERS="${NUM_WORKERS:-8}"

exec bash "${SCRIPT_DIR}/../eval_oscar_ocrbench.sh"
