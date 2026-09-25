#!/usr/bin/env bash
# GPQA eval wrapper for Qwen/Qwen3-30B-A3B, CUDA graph on (the configuration all published
# OSCAR GPQA numbers were measured under). Override CUDA_GRAPH_MAX_BS to change it.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
export ROT_DIR="${ROT_DIR:-${SCRIPT_DIR}/rotations}"
export RUN_DIR="${RUN_DIR:-$(dirname "${ROT_DIR}")/_eval_gpqa_oscar}"
export TP_SIZE="${TP_SIZE:-2}"
export GROUP_SIZE="${GROUP_SIZE:-128}"
export K_CLIP="${K_CLIP:-0.96}"
export V_CLIP="${V_CLIP:-0.92}"
export CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-32}"
# This model needs PER-HEAD rotations (format_version 2). A shared rotation does
# not cost a few points here, it collapses the model: GPQA 43.9 against 58.6.
# Fit them with ../fit_perhead_rotation.py, not compute_rotation.sh.
# These must match the names published in Zhongzhu/OSCAR-RotationZoo under
# Qwen3-30B-A3B/, which are k_rotation_perhead.pt / v_rotation_perhead.pt --
# NOT k_perhead.pt / v_perhead.pt. With the old names a reader who downloaded
# the zoo got a FileNotFoundError, and the obvious "fix" is to fall back to the
# shared k_rotation_qqt_r_h_pbr.pt sitting next to them in the same directory,
# which is the one configuration that collapses this model (GPQA 58.6 -> 43.9,
# HumanEval 89 -> 25.8, MATH500 92 -> 60.8, AIME 66 -> 14).
# Per-head files are ~12.6 MB ([4,128,128]/layer); the shared ones are ~3.2 MB.
export K_ROT_FILENAME="${K_ROT_FILENAME:-k_rotation_perhead.pt}"
export V_ROT_FILENAME="${V_ROT_FILENAME:-v_rotation_perhead.pt}"
# V-rotation absorption folds R_v into o_proj and assumes one rotation per
# layer, so it is invalid for per-head checkpoints.
export ABSORB_V="${ABSORB_V:-0}"
export NAME="${NAME:-gpqa_oscar_qwen3_30b_a3b_perhead}"

exec bash "${SCRIPT_DIR}/../eval_oscar_gpqa.sh"
