#!/bin/bash
# 30B-A3B calibration, TP=2 (weights 30GB/GPU leaves real KV room), then merge rank shards and fit.
set -euo pipefail
W=/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/CoQuant
INV=$W/rotation/investigation/05_qwen3moe_absorb_fix
source /home/charlie/miniconda3/etc/profile.d/conda.sh; conda activate oscar
export PYTHONPATH=$W/sglang-dump-qkv/python:${PYTHONPATH:-}
export TRITON_CACHE_DIR=/tmp/triton_calib3; mkdir -p $TRITON_CACHE_DIR
export CUDA_VISIBLE_DEVICES=$(ls /var/run/nvidia-container-devices 2>/dev/null | paste -sd, || echo 0,1)
cd $W/rotation/qwen3-30b-a3b
rm -rf GPQA/latest
MODEL=Qwen/Qwen3-30B-A3B TP_SIZE=2 GPU=0,1 MEM_FRACTION_STATIC=0.85 \
MAX_RUNNING_REQUESTS=16 MAX_QUEUED_REQUESTS=32 \
DUMP_KVCACHE_TOKENS=30000 PORT=31621 DIST_PORT=41621 \
bash save_qkv.sh 2>&1 | tail -8
D=$(ls -dt GPQA/seq*/qkv_dumps/gpqa 2>/dev/null | head -1)
echo "[calib] dump dir: $D"
python3 - <<PY
import glob, torch, os
D = "$D"
subs = sorted(os.listdir(f"{D}/layer_0"))
print("layer_0 entries:", subs[:6])
for sub in ("k",):
    fs = sorted(glob.glob(f"{D}/layer_0/{sub}/*.pt")) or sorted(glob.glob(f"{D}/layer_0/*/{sub}/*.pt"))
    tot = 0; shp = None
    for f in fs:
        t = torch.load(f, map_location="cpu", weights_only=False)
        tot += t.shape[0]; shp = tuple(t.shape[1:])
    print(f"{sub}: files={len(fs)} tokens={tot} per-token shape={shp}")
PY
echo CALIB3_DONE
