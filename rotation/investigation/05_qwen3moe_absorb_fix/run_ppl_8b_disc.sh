#!/bin/bash
# Discrimination test: can the harness tell a known-good rotation from a known-bad one?
set -euo pipefail
W=/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/oscar
INV=$W/rotation/investigation/05_qwen3moe_absorb_fix
ZOO=$W/rotation/OSCAR-RotationZoo/Qwen3-8B/seq20000_prompt83_group128
OUT=$INV/ppl_out; mkdir -p $OUT
source /home/charlie/miniconda3/etc/profile.d/conda.sh; conda activate oscar
export HF_HOME=/shared/huggingface
export CUDA_VISIBLE_DEVICES=$(ls /var/run/nvidia-container-devices 2>/dev/null | paste -sd, || echo 0)
M=$(ls -d /shared/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/ | head -1)
NW=16
python3 $INV/ppl_fakequant.py --model "$M" --mode bf16  --num-windows $NW --out $OUT --tag 8b_bf16_n16
python3 $INV/ppl_fakequant.py --model "$M" --mode quant --num-windows $NW --out $OUT --tag 8b_zoo_n16 \
  --k-rot $ZOO/k_rotation_qqt_r_h_pbr.pt --v-rot $ZOO/v_rotation_sst_r_h_pbr.pt
python3 $INV/ppl_fakequant.py --model "$M" --mode quant --num-windows $NW --out $OUT --tag 8b_hadamard_n16 \
  --k-rot $INV/rotations/hadamard_36L.pt --v-rot $INV/rotations/hadamard_36L.pt
echo PPL8B_DISC_DONE
