#!/bin/bash
set -euo pipefail
INV=/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/CoQuant/rotation/investigation/05_qwen3moe_absorb_fix
source /home/charlie/miniconda3/etc/profile.d/conda.sh; conda activate oscar
export HF_HOME=/shared/huggingface
export CUDA_VISIBLE_DEVICES=$(ls /var/run/nvidia-container-devices 2>/dev/null | paste -sd, || echo 0)
python3 $INV/selftest_fq.py
echo SELFTEST_DONE
