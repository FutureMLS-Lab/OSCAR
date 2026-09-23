#!/usr/bin/env bash
# Packed 2-bit MLA latent verification (GLM-5.2 / GLM-5.3 / Kimi-K3).
# Usage: mla.sh <name> <repo> <rot-dir> <tp> [memfrac] [ctx]
source "$(dirname "$0")/_common.sh"
NAME=${1:?name}; REPO=${2:?repo}; ROT=${3:?rot}; TP=${4:?tp}
MF=${5:-0.90}; CTX=${6:-4096}
LOG=$OUT/$NAME.log
LAT=$OSCAR_ROTATIONS/$ROT

echo "=== $NAME  packed MLA  tp=$TP mf=$MF ctx=$CTX  (image $(cat /oscar/IMAGE_TAG 2>/dev/null || echo '?'))"
N=$(ls "$LAT"/layer_*.pt 2>/dev/null | wc -l)
[ "$N" -gt 0 ] || { echo "RESULT=$NAME FAIL(no latent rotations in $LAT)"; exit 1; }
echo "  latent rotations: $LAT ($N layers)"

# Per-model JIT cache. Two GLM jobs sharing one DeepGEMM cache dir put one at
# 2.22 s/it -- a 20-hour warmup -- while the other ran at 9800 it/s.
export HOME=/scratch/home_$NAME TRITON_CACHE_DIR=/scratch/home_$NAME/triton
mkdir -p "$TRITON_CACHE_DIR"

verify_prune_weights "$REPO"
MD=$(verify_download "$REPO")
[ -n "$MD" ] || { echo "RESULT=$NAME SKIP(download)"; exit 0; }
verify_drain_gpus
verify_release_page_cache

PORT=$((35000 + RANDOM % 1500))
# KV dtype must be pinned: sglang picks fp8_e4m3 for a DSA model on SM100+, which
# puts ~28% of every 512x512 rotation subnormal -- that arm once scored 5.56.
SGLANG_OSCAR_MLA_KV_ROTATION_PATH=$LAT \
SGLANG_OSCAR_MLA_KV_GROUP_SIZE=128 \
SGLANG_OSCAR_MLA_KV_PACKED=1 SGLANG_OSCAR_MLA_PACKED_SELFCHECK=0 \
SGLANG_ENABLE_MIXED_KV_WINDOWS=1 \
SGLANG_MIXED_KV_PREFIX_TOKENS=64 SGLANG_MIXED_KV_RECENT_TOKENS=512 \
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((TP-1))) \
python3 -m sglang.launch_server --model-path "$MD" --trust-remote-code \
  --tp "$TP" --port $PORT --host 127.0.0.1 \
  --attention-backend triton --prefill-attention-backend triton \
  --decode-attention-backend triton \
  --kv-cache-dtype bfloat16 \
  --mem-fraction-static "$MF" --context-length "$CTX" \
  --max-running-requests 4 --cuda-graph-max-bs 4 > "$LOG" 2>&1 &
SRV=$!
if ! verify_wait_serve $SRV $PORT 260; then
  grep -aE "Error|assert|OutOfMemory|Traceback" "$LOG" | grep -avE "Ignore import" | tail -5
  kill -9 $SRV 2>/dev/null; wait $SRV 2>/dev/null
  echo "RESULT=$NAME FAIL(no-serve)"; exit 1
fi
verify_probe_and_verdict "$NAME" $PORT "$LOG"
kill -TERM $SRV 2>/dev/null; wait $SRV 2>/dev/null
