#!/bin/bash
set -euo pipefail
W=/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/CoQuant
INV=$W/rotation/investigation/05_qwen3moe_absorb_fix
FAKE=$W/rotation/qwen3-30b-a3b/GPQA/seq30000_prompt15_group128/rotations
OUT=$INV/ppl_out; mkdir -p $OUT
source /home/charlie/miniconda3/etc/profile.d/conda.sh; conda activate oscar
export HF_HOME=/shared/huggingface
export CUDA_VISIBLE_DEVICES=$(ls /var/run/nvidia-container-devices 2>/dev/null | paste -sd, || echo 0,1)
M=$(ls -d /shared/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/*/ | head -1)
python3 $INV/ppl_fakequant.py --model "$M" --mode quant --num-windows 10 --out $OUT --tag 30b_hadamard \
  --k-rot $INV/rotations/hadamard_48L.pt --v-rot $INV/rotations/hadamard_48L.pt
python3 $INV/ppl_fakequant.py --model "$M" --mode quant --num-windows 10 --out $OUT --tag 30b_fakecalib \
  --k-rot $FAKE/k_rotation_qqt_r_h_pbr.pt --v-rot $FAKE/v_rotation_sst_r_h_pbr.pt
echo PPL30B_BAD_DONE
