#!/bin/bash
# Calibrate OSCAR rotations for Qwen3-30B-A3B: TP=1 dump (no rank merge) + qqt/sst fit.
set -euo pipefail
W=/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/CoQuant
source /home/charlie/miniconda3/etc/profile.d/conda.sh; conda activate oscar
export PYTHONPATH=$W/sglang-dump-qkv/python:${PYTHONPATH:-}
export TRITON_CACHE_DIR=/tmp/triton_calib; mkdir -p $TRITON_CACHE_DIR
export CUDA_VISIBLE_DEVICES=$(ls /var/run/nvidia-container-devices 2>/dev/null | paste -sd, || echo 0)
mkdir -p $W/rotation/qwen3-30b-a3b
cp $W/rotation/qwen3-8B/save_qkv_8b.sh $W/rotation/qwen3-30b-a3b/save_qkv.sh 2>/dev/null || true
cd $W/rotation/qwen3-30b-a3b
MODEL=Qwen/Qwen3-30B-A3B TP_SIZE=1 GPU=0 MEM_FRACTION_STATIC=0.85 MAX_RUNNING_REQUESTS=16 \
DUMP_KVCACHE_TOKENS=30000 PORT=31610 DIST_PORT=41610 \
bash save_qkv.sh 2>&1 | tail -20
echo CALIB_DUMP_DONE
ls GPQA/ | head -3
