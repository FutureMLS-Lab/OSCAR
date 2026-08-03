#!/bin/bash
# Isolate the serving knob that flips 8B INT2 between clean and broken.
set -euo pipefail
INV=/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/oscar/rotation/investigation/05_qwen3moe_absorb_fix
SGL=/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/oscar/sglang-research
ZOO=/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/oscar/rotation/OSCAR-RotationZoo/Qwen3-8B/seq20000_prompt83_group128
MODEL=$(ls -d /shared/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/ | head -1)
PORT=31618
OUT=$INV/out_v15; mkdir -p "$OUT"
source /home/charlie/miniconda3/etc/profile.d/conda.sh; conda activate oscar
export PYTHONPATH=${SGL}/python:${PYTHONPATH:-}
export TRITON_CACHE_DIR=/tmp/triton_v15; mkdir -p $TRITON_CACHE_DIR
export CUDA_VISIBLE_DEVICES=$(ls /var/run/nvidia-container-devices 2>/dev/null | paste -sd, || echo 0)

run_arm () {
  local NAME=$1; shift
  rm -rf /dev/shm/nccl* /dev/shm/torch_* 2>/dev/null || true
  echo "=== ARM $NAME : $*"
  SGLANG_ENABLE_MIXED_KV_WINDOWS=1 SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  SGLANG_MIXED_KV_HP_MAX_SPLITS=8 SGLANG_MIXED_KV_PREFIX_TOKENS=64 SGLANG_MIXED_KV_RECENT_TOKENS=256 \
  SGLANG_MIXED_KV_HP_DTYPE=bfloat16 SGLANG_MIXED_KV_SCALE_DTYPE=float32 \
  SGLANG_OSCAR_ABSORB_V_ROTATION=0 SGLANG_OSCAR_K_CLIP_RATIO=0.96 SGLANG_OSCAR_V_CLIP_RATIO=0.92 SGLANG_LLOYD_MAX=0 \
  SGLANG_OSCAR_K_ROTATION_PATH=$ZOO/k_rotation_qqt_r_h_pbr.pt SGLANG_OSCAR_V_ROTATION_PATH=$ZOO/v_rotation_sst_r_h_pbr.pt \
  python -m sglang.launch_server --model-path "$MODEL" --served-model-name q3 \
    --tensor-parallel-size 1 --prefill-attention-backend fa3 --decode-attention-backend triton \
    --kv-cache-dtype int2 --kv-cache-quant-group-size 128 \
    --host 127.0.0.1 --port $PORT --trust-remote-code "$@" > "$OUT/server_$NAME.log" 2>&1 &
  local SP=$!
  for i in $(seq 1 150); do curl -s localhost:$PORT/health >/dev/null 2>&1 && break; kill -0 $SP 2>/dev/null || { echo "ARM $NAME died"; tail -4 "$OUT/server_$NAME.log"; return 0; }; sleep 5; done
  python3 $INV/probe_sweep.py --port $PORT --out $OUT --tag $NAME 2>&1 | sed "s/^/[$NAME] /" || true
  kill -KILL $SP 2>/dev/null || true; pkill -KILL -P $SP 2>/dev/null || true; sleep 8
}

# A = the v4/v6 config that scored 13/13 clean
run_arm A_graphON_run32 --mem-fraction-static 0.75 --max-running-requests 32 --cuda-graph-max-bs 16 --disable-radix-cache
# B = A but graphs OFF (single knob)
run_arm B_graphOFF_run32 --mem-fraction-static 0.75 --max-running-requests 32 --disable-cuda-graph --disable-radix-cache
# C = A but maxrun 4 (single knob)
run_arm C_graphON_run4 --mem-fraction-static 0.75 --max-running-requests 4 --cuda-graph-max-bs 16 --disable-radix-cache
echo V15_DONE
