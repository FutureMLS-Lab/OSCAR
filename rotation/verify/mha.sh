#!/usr/bin/env bash
# INT2 KV verification for the MHA / hybrid-linear models.
# Usage: mha.sh <name> <repo> <tp> <group> <rot-subdir> [sink] [recent] [memfrac] [extra]
source "$(dirname "$0")/_common.sh"
NAME=${1:?name}; REPO=${2:?repo}; TP=${3:?tp}; GS=${4:?group}; ROT=${5:?rot}
SINK=${6:-64}; RECENT=${7:-256}; MF=${8:-0.55}; EXTRA=${9:-}
MAMBA_STRATEGY=${MAMBA_STRATEGY:-}
NO_AUTOTUNE=${NO_AUTOTUNE:-}
LOG=$OUT/$NAME.log
R=$OSCAR_ROTATIONS/zoo/$ROT

echo "=== $NAME  tp=$TP g$GS mf=$MF sink/recent=$SINK/$RECENT mamba='${MAMBA_STRATEGY:-none}' extra='$EXTRA'"
# Discover rotations rather than assuming one naming: MiniMax-M3 ships published
# Hadamard rotations (k_rotation_hadamard.pt), not the qqt_r_h_pbr form the rest
# use, and a hardcoded name reported it as "no rotation" -- which reads as "not
# supported" when the files were there all along.
KROT=$(ls "$R"/k_rotation_*.pt 2>/dev/null | head -1)
VROT=$(ls "$R"/v_rotation_*.pt 2>/dev/null | head -1)
[ -n "$KROT" ] && [ -n "$VROT" ] || { echo "RESULT=$NAME SKIP(no rotation in $R)"; exit 0; }
echo "  rotations: $(basename "$KROT") / $(basename "$VROT")"

verify_prune_weights "$REPO"
MD=$(verify_download "$REPO")
[ -n "$MD" ] || { echo "RESULT=$NAME SKIP(download)"; exit 0; }
verify_drain_gpus
verify_release_page_cache

PORT=$((34000 + RANDOM % 1500))
# radix cache ON (no --disable-radix-cache) and cuda graph ON (no
# --disable-cuda-graph) -- that is the whole point of this run.
SGLANG_ENABLE_MIXED_KV_WINDOWS=1 \
SGLANG_MIXED_KV_PREFIX_TOKENS=$SINK SGLANG_MIXED_KV_RECENT_TOKENS=$RECENT \
SGLANG_OSCAR_K_ROTATION_PATH=$KROT SGLANG_OSCAR_V_ROTATION_PATH=$VROT \
SGLANG_COQUANT_ROTATION_MODE=coquant \
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((TP-1))) \
python3 -m sglang.launch_server --model-path "$MD" --trust-remote-code \
  --tp "$TP" --port $PORT --host 127.0.0.1 \
  --attention-backend triton --prefill-attention-backend triton \
  --decode-attention-backend triton --mm-attention-backend triton_attn \
  --kv-cache-dtype int2 --kv-cache-quant-group-size "$GS" \
  --mem-fraction-static "$MF" --context-length "${CTX:-8192}" \
  --max-running-requests 2 --cuda-graph-max-bs 2 \
  ${MAMBA_STRATEGY:+--mamba-scheduler-strategy $MAMBA_STRATEGY} \
  ${NO_AUTOTUNE:+--disable-flashinfer-autotune} \
  $EXTRA > "$LOG" 2>&1 &
SRV=$!
if ! verify_wait_serve $SRV $PORT 150; then
  grep -aE "Error|assert|OutOfMemory|Page size|not divisible|no attribute" "$LOG" \
    | grep -avE "Ignore import" | tail -3
  kill -9 $SRV 2>/dev/null; wait $SRV 2>/dev/null
  echo "RESULT=$NAME FAIL(no-serve)"; exit 1
fi
verify_probe_and_verdict "$NAME" $PORT "$LOG"
kill -TERM $SRV 2>/dev/null; wait $SRV 2>/dev/null
