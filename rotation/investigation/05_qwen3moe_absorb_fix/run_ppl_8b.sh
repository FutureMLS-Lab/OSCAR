#!/bin/bash
# Harness validation on Qwen3-8B: bf16 vs Zoo-shared INT2 fake-quant.
# Known target from the serving path: INT2+OSCAR = +2.8-3.7% PPL over bf16.
set -euo pipefail
W=/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/oscar
INV=$W/rotation/investigation/05_qwen3moe_absorb_fix
ZOO=$W/rotation/OSCAR-RotationZoo/Qwen3-8B/seq20000_prompt83_group128
OUT=$INV/ppl_out; mkdir -p $OUT
source /home/charlie/miniconda3/etc/profile.d/conda.sh; conda activate oscar
export HF_HOME=/shared/huggingface
export CUDA_VISIBLE_DEVICES=$(ls /var/run/nvidia-container-devices 2>/dev/null | paste -sd, || echo 0)
M=$(ls -d /shared/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/ | head -1)
python3 $INV/ppl_fakequant.py --model "$M" --mode bf16 --num-windows 6 --out $OUT --tag 8b_bf16
python3 $INV/ppl_fakequant.py --model "$M" --mode quant --num-windows 6 --out $OUT --tag 8b_zoo_shared \
  --k-rot $ZOO/k_rotation_qqt_r_h_pbr.pt --v-rot $ZOO/v_rotation_sst_r_h_pbr.pt
echo PPL8B_DONE
