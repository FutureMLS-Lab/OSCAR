#!/usr/bin/env bash
# GPQA-Diamond under INT2 OSCAR KV, for any supported model.
#
#   examples/gpqa_int2_oscar.sh qwen3-8b
#   examples/gpqa_int2_oscar.sh qwen3-30b-a3b
#
# It runs rotation/run/<model>.sh, which carries that model's recipe and execs
# rotation/eval_oscar_gpqa.sh. Scoring is simple_evals (vendored at
# third_party/simple_evals).
#
# THE RECIPE IS PER MODEL -- do not copy one model's flags onto another:
#
#   per-head rotation   Qwen3-30B-A3B only. It has 4 KV heads that are near
#                       orthogonal (mean |diag(R0^T R1)| ~= 0.07), so a single
#                       shared basis cannot whiten them: shared scores GPQA 43.9
#                       against per-head's 58.6, HumanEval 25.8 against 89.0.
#                       Both formats sit in the same zoo directory, so the file
#                       must be named explicitly.
#   Lloyd-Max           MiniMax-M2.7, Qwen3-8B, Qwen3.5-35B-A3B. Everything else
#                       uses uniform codebooks.
#   group size          256 for Qwen3.5-4B and Qwen3.5-35B-A3B, 128 elsewhere.
#   rotation filename   MiniMax-M3 ships published Hadamard
#                       (k_rotation_hadamard.pt), not the qqt_r_h_pbr form.
#   packed MLA          GLM-5.2, GLM-5.3, Kimi-K3. KV dtype must be pinned to
#                       bfloat16 -- sglang picks fp8_e4m3 for a DSA model on
#                       SM100+, which drives ~28% of every 512x512 rotation
#                       subnormal.
set -uo pipefail
MODEL_KEY=${1:?usage: gpqa_int2_oscar.sh <model-key>   (see rotation/run/)}
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${OSCAR_SRC:-$(cd -- "$HERE/.." && pwd)}"
RUN="$ROOT/rotation/run/${MODEL_KEY}.sh"
[ -x "$RUN" ] || { echo "no run script for '$MODEL_KEY'"; ls "$ROOT/rotation/run/" | sed 's/\.sh$//' | grep -v _common; exit 1; }
echo "[gpqa] model=$MODEL_KEY  recipe=$RUN"
echo "[gpqa] rotations=${OSCAR_ROTATIONS:-$ROOT/../rotations}"
exec bash "$RUN"
