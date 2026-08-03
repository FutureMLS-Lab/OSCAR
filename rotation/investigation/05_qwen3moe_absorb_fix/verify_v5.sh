#!/bin/bash
set -euo pipefail
INV=/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/CoQuant/rotation/investigation/05_qwen3moe_absorb_fix
RADIX=/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/CoQuant/rotation/investigation/04_radix_mixedkv_regression
SGL=/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/CoQuant/sglang-research
ZOO=/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/CoQuant/rotation/OSCAR-RotationZoo/Qwen3-8B/seq20000_prompt83_group128
MODEL=$(ls -d /shared/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/ | head -1)
PORT=31605
OUT=$INV/out_v5; mkdir -p "$OUT"
rm -rf /dev/shm/nccl* /dev/shm/torch_* /dev/shm/sglang* 2>/dev/null || true
source /home/charlie/miniconda3/etc/profile.d/conda.sh; conda activate oscar
export PYTHONPATH=${SGL}/python:${PYTHONPATH:-}
export TRITON_CACHE_DIR=/tmp/triton_v4; mkdir -p $TRITON_CACHE_DIR
export CUDA_VISIBLE_DEVICES=$(ls /var/run/nvidia-container-devices 2>/dev/null | paste -sd, || echo 0)
BASE=(SGLANG_ENABLE_MIXED_KV_WINDOWS=1 SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
  SGLANG_MIXED_KV_HP_MAX_SPLITS=8 SGLANG_MIXED_KV_PREFIX_TOKENS=64 SGLANG_MIXED_KV_RECENT_TOKENS=256
  SGLANG_MIXED_KV_HP_DTYPE=bfloat16 SGLANG_MIXED_KV_SCALE_DTYPE=float32
  SGLANG_OSCAR_ABSORB_V_ROTATION=0 SGLANG_OSCAR_K_CLIP_RATIO=0.96 SGLANG_OSCAR_V_CLIP_RATIO=0.92 SGLANG_LLOYD_MAX=0)
run_arm () {
  local NAME=$1 KROT=$2 VROT=$3
  echo "=== ARM $NAME"
  env "${BASE[@]}" SGLANG_OSCAR_K_ROTATION_PATH=$KROT SGLANG_OSCAR_V_ROTATION_PATH=$VROT \
    python -m sglang.launch_server --model-path "$MODEL" --served-model-name q3 \
    --tensor-parallel-size 1 --prefill-attention-backend fa3 --decode-attention-backend triton \
    --kv-cache-dtype int2 --kv-cache-quant-group-size 128 \
    --mem-fraction-static 0.75 --max-running-requests 32 --cuda-graph-max-bs 16 \
    --disable-radix-cache --host 127.0.0.1 --port $PORT --trust-remote-code \
    > "$OUT/server_$NAME.log" 2>&1 &
  local SP=$!
  for i in $(seq 1 120); do curl -s localhost:$PORT/health >/dev/null 2>&1 && break; kill -0 $SP 2>/dev/null || { echo "ARM $NAME died"; tail -5 "$OUT/server_$NAME.log"; return 0; }; sleep 5; done
  python3 "$RADIX/probe_client.py" --port $PORT --arm "$NAME" --out "$OUT" || true
  kill -TERM $SP 2>/dev/null || true; sleep 5; kill -KILL $SP 2>/dev/null || true; pkill -KILL -P $SP 2>/dev/null || true
  sleep 8
}
run_arm randortho $INV/rotations/randortho_36L.pt $INV/rotations/randortho_36L.pt
run_arm randortho_hpbr $INV/rotations/randortho_hpbr_36L.pt $INV/rotations/randortho_hpbr_36L.pt
run_arm had_zoowrap $INV/rotations/hadamard_zoowrap_36L.pt $INV/rotations/hadamard_zoowrap_36L.pt
echo V5_DONE
