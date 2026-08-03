#!/bin/bash
set -euo pipefail
INV=/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/oscar/rotation/investigation/05_qwen3moe_absorb_fix
RADIX=/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/oscar/rotation/investigation/04_radix_mixedkv_regression
SGL=/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/oscar/sglang-research
MODEL=$(ls -d /shared/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/*/ | head -1)
PORT=31603
OUT=$INV/out_v3; mkdir -p "$OUT"
rm -rf /dev/shm/nccl* /dev/shm/torch_* /dev/shm/sglang* 2>/dev/null || true
source /home/charlie/miniconda3/etc/profile.d/conda.sh; conda activate oscar
export PYTHONPATH=${SGL}/python:${PYTHONPATH:-}
export TRITON_CACHE_DIR=/tmp/triton_v3; mkdir -p $TRITON_CACHE_DIR
export CUDA_VISIBLE_DEVICES=$(ls /var/run/nvidia-container-devices 2>/dev/null | paste -sd, || echo 0,1)
BASE=(SGLANG_ENABLE_MIXED_KV_WINDOWS=1 SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
  SGLANG_MIXED_KV_HP_MAX_SPLITS=8 SGLANG_MIXED_KV_HP_DTYPE=bfloat16 SGLANG_MIXED_KV_SCALE_DTYPE=float32
  SGLANG_OSCAR_ABSORB_V_ROTATION=0 SGLANG_OSCAR_K_CLIP_RATIO=0.96 SGLANG_OSCAR_V_CLIP_RATIO=0.92 SGLANG_LLOYD_MAX=0)
HAD=(SGLANG_OSCAR_K_ROTATION_PATH=$INV/rotations/k_rotation_hadamard.pt SGLANG_OSCAR_V_ROTATION_PATH=$INV/rotations/v_rotation_hadamard.pt)
IDN=(SGLANG_OSCAR_K_ROTATION_PATH=$INV/rotations/identity.pt SGLANG_OSCAR_V_ROTATION_PATH=$INV/rotations/identity.pt)

run_arm () {
  local NAME=$1 TP=$2; shift 2
  echo "=== ARM $NAME (tp=$TP)"
  env "$@" python -m sglang.launch_server --model-path "$MODEL" --served-model-name q3 \
    --tensor-parallel-size $TP --prefill-attention-backend fa3 --decode-attention-backend triton \
    --kv-cache-dtype int2 --kv-cache-quant-group-size 128 \
    --mem-fraction-static 0.75 --max-running-requests 32 --cuda-graph-max-bs 16 \
    --disable-radix-cache --host 127.0.0.1 --port $PORT --trust-remote-code \
    > "$OUT/server_$NAME.log" 2>&1 &
  local SP=$!
  for i in $(seq 1 180); do curl -s localhost:$PORT/health >/dev/null 2>&1 && break; kill -0 $SP 2>/dev/null || { echo "ARM $NAME died"; tail -5 "$OUT/server_$NAME.log"; return 0; }; sleep 5; done
  python3 "$RADIX/probe_client.py" --port $PORT --arm "$NAME" --out "$OUT" || true
  kill -TERM $SP 2>/dev/null || true; sleep 5; kill -KILL $SP 2>/dev/null || true; pkill -KILL -P $SP 2>/dev/null || true
  sleep 8
}
run_arm bigprefix 2 "${BASE[@]}" "${HAD[@]}" SGLANG_MIXED_KV_PREFIX_TOKENS=4096 SGLANG_MIXED_KV_RECENT_TOKENS=256 SGLANG_MIXED_KV_HP_PREFIX_POOL_TOKENS=262144
run_arm bigrecent 2 "${BASE[@]}" "${HAD[@]}" SGLANG_MIXED_KV_PREFIX_TOKENS=64 SGLANG_MIXED_KV_RECENT_TOKENS=8192
run_arm identity 2 "${BASE[@]}" "${IDN[@]}" SGLANG_MIXED_KV_PREFIX_TOKENS=64 SGLANG_MIXED_KV_RECENT_TOKENS=256
run_arm tp4 4 "${BASE[@]}" "${HAD[@]}" SGLANG_MIXED_KV_PREFIX_TOKENS=64 SGLANG_MIXED_KV_RECENT_TOKENS=256
echo V3_DONE
