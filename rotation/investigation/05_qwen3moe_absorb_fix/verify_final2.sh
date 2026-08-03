#!/bin/bash
set -euo pipefail
INV=/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/CoQuant/rotation/investigation/05_qwen3moe_absorb_fix
RADIX=/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/CoQuant/rotation/investigation/04_radix_mixedkv_regression
SGL=/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/CoQuant/sglang-research
MODEL=$(ls -d /shared/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/*/ | head -1)
PORT=31609
OUT=$INV/out_final2; mkdir -p "$OUT"
rm -rf /dev/shm/nccl* /dev/shm/torch_* 2>/dev/null || true
source /home/charlie/miniconda3/etc/profile.d/conda.sh; conda activate oscar
export PYTHONPATH=${SGL}/python:${PYTHONPATH:-}
export TRITON_CACHE_DIR=/tmp/triton_f2; mkdir -p $TRITON_CACHE_DIR
export CUDA_VISIBLE_DEVICES=$(ls /var/run/nvidia-container-devices 2>/dev/null | paste -sd, || echo 0,1)
run_arm () {
  local NAME=$1 TP=$2
  echo "=== ARM $NAME tp=$TP"
  SGLANG_ENABLE_MIXED_KV_WINDOWS=1 SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1   SGLANG_MIXED_KV_HP_MAX_SPLITS=8 SGLANG_MIXED_KV_PREFIX_TOKENS=64 SGLANG_MIXED_KV_RECENT_TOKENS=256   SGLANG_MIXED_KV_HP_DTYPE=bfloat16 SGLANG_MIXED_KV_SCALE_DTYPE=float32   SGLANG_OSCAR_ABSORB_V_ROTATION=0 SGLANG_OSCAR_K_CLIP_RATIO=0.96 SGLANG_OSCAR_V_CLIP_RATIO=0.92 SGLANG_LLOYD_MAX=0   SGLANG_OSCAR_K_ROTATION_PATH=/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/CoQuant/rotation/qwen3-30b-a3b/GPQA/seq30000_prompt15_group128/rotations/k_rotation_qqt_r_h_pbr.pt   SGLANG_OSCAR_V_ROTATION_PATH=/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/CoQuant/rotation/qwen3-30b-a3b/GPQA/seq30000_prompt15_group128/rotations/v_rotation_sst_r_h_pbr.pt   python -m sglang.launch_server --model-path "$MODEL" --served-model-name q3     --tensor-parallel-size $TP --prefill-attention-backend fa3 --decode-attention-backend triton     --kv-cache-dtype int2 --kv-cache-quant-group-size 128     --mem-fraction-static 0.85 --max-running-requests 8 --disable-cuda-graph     --disable-radix-cache --host 127.0.0.1 --port $PORT --trust-remote-code > "$OUT/server_$NAME.log" 2>&1 &
  local SP=$!
  for i in $(seq 1 180); do curl -s localhost:$PORT/health >/dev/null 2>&1 && break; kill -0 $SP 2>/dev/null || { echo died; tail -5 "$OUT/server_$NAME.log"; return 0; }; sleep 5; done
  python3 "$RADIX/probe_client.py" --port $PORT --arm "$NAME" --out "$OUT" || true
  kill -KILL $SP 2>/dev/null || true; pkill -KILL -P $SP 2>/dev/null || true; sleep 8
}
run_arm tp1 1
run_arm tp2 2
echo FINAL2_DONE
