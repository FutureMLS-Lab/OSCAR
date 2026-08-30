#!/bin/bash
# Minimal repro: does a radix-cache prefix hit corrupt mixed-KV INT2 generation?
# Runs the same server twice (radix ON / OFF); for each, sends a long prompt
# sequentially twice (2nd hits the cached prefix) and 5x concurrently
# (N_REPEATS-style), and scores degeneration of each completion.
set -euo pipefail
INV=/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/CoQuant/rotation/investigation/04_radix_mixedkv_regression
ROT=/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/CoQuant/rotation/OSCAR-RotationZoo/Qwen3-8B/seq20000_prompt83_group128
SGL=/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/CoQuant/sglang-research
MODEL=$(ls -d /shared/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/ | head -1)
PORT=31599
OUT=${INV}/out_${RUN_TAG:-run1}
mkdir -p "$OUT"

rm -rf /dev/shm/nccl* /dev/shm/torch_* /dev/shm/sglang* 2>/dev/null || true
source /home/charlie/miniconda3/etc/profile.d/conda.sh
conda activate oscar
export PYTHONPATH=${SGL}/python:${PYTHONPATH:-}
export TRITON_CACHE_DIR=/tmp/triton_radixprobe; mkdir -p $TRITON_CACHE_DIR
export CUDA_VISIBLE_DEVICES=$(ls /var/run/nvidia-container-devices 2>/dev/null | paste -sd, || echo 0)

run_arm () {  # $1 = radix_on|radix_off
  local ARM=$1; local EXTRA=()
  [ "$ARM" = "radix_off" ] && EXTRA+=( --disable-radix-cache )
  echo "=== ARM $ARM ==="
  SGLANG_ENABLE_MIXED_KV_WINDOWS=1 SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  SGLANG_MIXED_KV_HP_MAX_SPLITS=8 SGLANG_MIXED_KV_PREFIX_TOKENS=64 SGLANG_MIXED_KV_RECENT_TOKENS=256 \
  SGLANG_MIXED_KV_HP_DTYPE=bfloat16 SGLANG_MIXED_KV_SCALE_DTYPE=float32 \
  SGLANG_OSCAR_ABSORB_V_ROTATION=1 \
  SGLANG_OSCAR_K_ROTATION_PATH=${ROT}/k_rotation_qqt_r_h_pbr.pt \
  SGLANG_OSCAR_V_ROTATION_PATH=${ROT}/v_rotation_sst_r_h_pbr.pt \
  SGLANG_OSCAR_K_CLIP_RATIO=0.96 SGLANG_OSCAR_V_CLIP_RATIO=0.92 SGLANG_LLOYD_MAX=0 \
  python -m sglang.launch_server --model-path "$MODEL" --served-model-name q3 \
    --tensor-parallel-size 1 --prefill-attention-backend fa3 --decode-attention-backend triton \
    --kv-cache-dtype int2 --kv-cache-quant-group-size 128 \
    --mem-fraction-static 0.75 --max-running-requests 32 --cuda-graph-max-bs 16 \
    "${EXTRA[@]}" --host 127.0.0.1 --port $PORT --trust-remote-code > "$OUT/server_$ARM.log" 2>&1 &
  local SP=$!
  for i in $(seq 1 120); do curl -s localhost:$PORT/health >/dev/null 2>&1 && break; kill -0 $SP 2>/dev/null || { echo "server died"; tail -5 "$OUT/server_$ARM.log"; exit 1; }; sleep 5; done
  python3 "$INV/probe_client.py" --port $PORT --arm "$ARM" --out "$OUT"
  kill -TERM $SP 2>/dev/null || true; sleep 5; kill -KILL $SP 2>/dev/null || true; pkill -KILL -P $SP 2>/dev/null || true
  sleep 5
}

run_arm radix_on
run_arm radix_off
echo "PROBE_DONE"
