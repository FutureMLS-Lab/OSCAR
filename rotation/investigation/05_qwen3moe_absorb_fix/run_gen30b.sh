#!/bin/bash
set -euo pipefail
W=/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/CoQuant
INV=$W/rotation/investigation/05_qwen3moe_absorb_fix
CAL=$W/rotation/qwen3-30b-a3b/GPQA/seq30000_prompt117_group128/rotations
ZOO=$W/rotation/OSCAR-RotationZoo/Qwen3-8B/seq20000_prompt83_group128
OUT=$INV/ppl_out; mkdir -p $OUT
source /home/charlie/miniconda3/etc/profile.d/conda.sh; conda activate oscar
export HF_HOME=/shared/huggingface
export CUDA_VISIBLE_DEVICES=$(ls /var/run/nvidia-container-devices 2>/dev/null | paste -sd, || echo 0,1)
python3 $INV/gen_test.py --model-glob "/shared/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/*/" \
  --pairs "shared=$CAL/k_rotation_qqt_r_h_pbr.pt,$CAL/v_rotation_sst_r_h_pbr.pt;perhead=$CAL/k_perhead.pt,$CAL/v_perhead.pt" \
  --out $OUT --tag 30b
# 8B per-head payoff for completeness (fit from its own dump is unavailable; use zoo shared vs bf16 anchor)
echo GEN30B_DONE
