#!/bin/bash
# Dump decode-step tensors for offline numerics (H rotation on 8B, layer 0).
set -euo pipefail
INV=/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/oscar/rotation/investigation/05_qwen3moe_absorb_fix
SGL=/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/oscar/sglang-research
MODEL=$(ls -d /shared/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/ | head -1)
PORT=31613
OUT=$INV/out_v10; mkdir -p "$OUT"
rm -rf /dev/shm/nccl* /dev/shm/torch_* 2>/dev/null || true
source /home/charlie/miniconda3/etc/profile.d/conda.sh; conda activate oscar
export PYTHONPATH=${SGL}/python:${PYTHONPATH:-}
export TRITON_CACHE_DIR=/tmp/triton_v7; mkdir -p $TRITON_CACHE_DIR
export CUDA_VISIBLE_DEVICES=$(ls /var/run/nvidia-container-devices 2>/dev/null | paste -sd, || echo 0)
run_arm () {
  local NAME=$1 KROT=$2
  local DD=$OUT/dump_$NAME; rm -rf $DD
  \
  SGLANG_ENABLE_MIXED_KV_WINDOWS=1 SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  SGLANG_MIXED_KV_HP_MAX_SPLITS=8 SGLANG_MIXED_KV_PREFIX_TOKENS=64 SGLANG_MIXED_KV_RECENT_TOKENS=256 \
  SGLANG_MIXED_KV_HP_DTYPE=bfloat16 SGLANG_MIXED_KV_SCALE_DTYPE=float32 \
  SGLANG_OSCAR_ABSORB_V_ROTATION=0 SGLANG_OSCAR_K_CLIP_RATIO=0.96 SGLANG_OSCAR_V_CLIP_RATIO=0.92 SGLANG_LLOYD_MAX=0 \
  SGLANG_OSCAR_K_ROTATION_PATH=$KROT SGLANG_OSCAR_V_ROTATION_PATH=/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/oscar/rotation/OSCAR-RotationZoo/Qwen3-8B/seq20000_prompt83_group128/v_rotation_sst_r_h_pbr.pt \
  python -m sglang.launch_server --model-path "$MODEL" --served-model-name q3 \
    --tensor-parallel-size 1 --prefill-attention-backend fa3 --decode-attention-backend triton \
    --kv-cache-dtype int2 --kv-cache-quant-group-size 128 \
    --mem-fraction-static 0.75 --max-running-requests 4 --disable-cuda-graph \
    --disable-radix-cache --host 127.0.0.1 --port $PORT --trust-remote-code > "$OUT/server_$NAME.log" 2>&1 &
  local SP=$!
  for i in $(seq 1 120); do curl -s localhost:$PORT/health >/dev/null 2>&1 && break; kill -0 $SP 2>/dev/null || { echo died; tail -5 "$OUT/server_$NAME.log"; return 0; }; sleep 5; done
  python3 - <<PY
import json, urllib.request
body = json.dumps({"model":"q3","messages":[{"role":"user","content":"Context: " + " ".join(f"fact {i} value {i*3%17}." for i in range(700)) + " Question: what is 2+2? Answer briefly."}],"max_tokens":32,"temperature":0}).encode()
req = urllib.request.Request("http://127.0.0.1:$PORT/v1/chat/completions", data=body, headers={"Content-Type":"application/json"})
print(json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"][:120])
PY
  kill -KILL $SP 2>/dev/null || true; pkill -KILL -P $SP 2>/dev/null || true; sleep 5
}
run_arm q38b_tiny /home/charlie/CoQuant/.RUD/hybridmodel-testing/work/oscar/rotation/OSCAR-RotationZoo/Qwen3-8B/seq20000_prompt83_group128/k_rotation_qqt_r_h_pbr.pt
echo V10_DONE
