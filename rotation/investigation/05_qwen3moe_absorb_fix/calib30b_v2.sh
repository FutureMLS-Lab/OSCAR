#!/bin/bash
# Proper 30B-A3B calibration dump: modest max-running-requests so alloc_req_slots holds.
set -euo pipefail
W=/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/CoQuant
source /home/charlie/miniconda3/etc/profile.d/conda.sh; conda activate oscar
export PYTHONPATH=$W/sglang-dump-qkv/python:${PYTHONPATH:-}
export TRITON_CACHE_DIR=/tmp/triton_calib2; mkdir -p $TRITON_CACHE_DIR
export CUDA_VISIBLE_DEVICES=$(ls /var/run/nvidia-container-devices 2>/dev/null | paste -sd, || echo 0)
cd $W/rotation/qwen3-30b-a3b
rm -rf GPQA/latest
MODEL=Qwen/Qwen3-30B-A3B TP_SIZE=1 GPU=0 MEM_FRACTION_STATIC=0.80 \
MAX_RUNNING_REQUESTS=8 MAX_QUEUED_REQUESTS=16 \
DUMP_KVCACHE_TOKENS=30000 PORT=31620 DIST_PORT=41620 \
bash save_qkv.sh 2>&1 | tail -12
echo "--- captured tokens per layer_0/k:"
D=$(ls -d GPQA/seq*/qkv_dumps/gpqa 2>/dev/null | head -1)
python3 - <<PY
import glob, torch
fs = sorted(glob.glob("$D/layer_0/k/*.pt"))
tot = 0
for f in fs:
    t = torch.load(f, map_location="cpu", weights_only=False)
    tot += t.shape[0]
print("files:", len(fs), "tokens:", tot)
PY
echo CALIB2_DONE
