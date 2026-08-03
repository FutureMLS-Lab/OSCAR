#!/bin/bash
# BF16 control for the tiny/mid/long probe (is tiny degeneracy real or a thinking-model artifact?)
set -euo pipefail
INV=/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/oscar/rotation/investigation/05_qwen3moe_absorb_fix
SGL=/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/oscar/sglang-research
PORT=31616
OUT=$INV/out_v13; mkdir -p "$OUT"
rm -rf /dev/shm/nccl* /dev/shm/torch_* 2>/dev/null || true
source /home/charlie/miniconda3/etc/profile.d/conda.sh; conda activate oscar
export PYTHONPATH=${SGL}/python:${PYTHONPATH:-}
export TRITON_CACHE_DIR=/tmp/triton_v13; mkdir -p $TRITON_CACHE_DIR
export CUDA_VISIBLE_DEVICES=$(ls /var/run/nvidia-container-devices 2>/dev/null | paste -sd, || echo 0)
run () {
  local NAME=$1 MODELDIR=$2
  local MODEL=$(ls -d /shared/huggingface/hub/$MODELDIR/snapshots/*/ | head -1)
  echo "=== $NAME (bf16 control)"
  SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 python -m sglang.launch_server \
    --model-path "$MODEL" --served-model-name q3 --tensor-parallel-size 1 \
    --prefill-attention-backend fa3 --decode-attention-backend triton \
    --mem-fraction-static 0.75 --max-running-requests 4 --disable-cuda-graph \
    --disable-radix-cache --host 127.0.0.1 --port $PORT --trust-remote-code > "$OUT/server_$NAME.log" 2>&1 &
  local SP=$!
  for i in $(seq 1 150); do curl -s localhost:$PORT/health >/dev/null 2>&1 && break; kill -0 $SP 2>/dev/null || { echo died; tail -4 "$OUT/server_$NAME.log"; return 0; }; sleep 5; done
  python3 $INV/probe_tiny.py --port $PORT --out $OUT 2>&1 | sed "s/^/[$NAME] /" || true
  mv $OUT/tiny_probe.json $OUT/tiny_$NAME.json 2>/dev/null || true
  kill -KILL $SP 2>/dev/null || true; pkill -KILL -P $SP 2>/dev/null || true; sleep 8
}
run bf16_8b models--Qwen--Qwen3-8B
echo V13_DONE
