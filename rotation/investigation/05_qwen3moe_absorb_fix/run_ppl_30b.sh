#!/bin/bash
# 30B-A3B: fit per-head rotations, then PPL across bf16 / pipeline-shared / per-head / per-head+LM.
set -euo pipefail
W=/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/oscar
INV=$W/rotation/investigation/05_qwen3moe_absorb_fix
CAL=$W/rotation/qwen3-30b-a3b/GPQA/seq30000_prompt117_group128
OUT=$INV/ppl_out; mkdir -p $OUT
source /home/charlie/miniconda3/etc/profile.d/conda.sh; conda activate oscar
export HF_HOME=/shared/huggingface
export CUDA_VISIBLE_DEVICES=$(ls /var/run/nvidia-container-devices 2>/dev/null | paste -sd, || echo 0,1)
M=$(ls -d /shared/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/*/ | head -1)

if [ ! -f $CAL/rotations/k_perhead.pt ]; then
  echo "[30b] fitting per-head rotations (48 layers x 4 kv heads)"
  python3 $INV/fit_perhead.py --dump $CAL/qkv_dumps/gpqa --layers 48 \
    --out-k $CAL/rotations/k_perhead.pt --out-v $CAL/rotations/v_perhead.pt
fi

NW=10
python3 $INV/ppl_fakequant.py --model "$M" --mode bf16  --num-windows $NW --out $OUT --tag 30b_bf16
python3 $INV/ppl_fakequant.py --model "$M" --mode quant --num-windows $NW --out $OUT --tag 30b_shared \
  --k-rot $CAL/rotations/k_rotation_qqt_r_h_pbr.pt --v-rot $CAL/rotations/v_rotation_sst_r_h_pbr.pt
python3 $INV/ppl_fakequant.py --model "$M" --mode quant --num-windows $NW --out $OUT --tag 30b_perhead \
  --k-rot $CAL/rotations/k_perhead.pt --v-rot $CAL/rotations/v_perhead.pt
python3 $INV/ppl_fakequant.py --model "$M" --mode quant --num-windows $NW --out $OUT --tag 30b_perhead_lm --lloyd-max \
  --k-rot $CAL/rotations/k_perhead.pt --v-rot $CAL/rotations/v_perhead.pt
echo PPL30B_DONE
