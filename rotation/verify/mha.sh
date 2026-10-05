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
# CALIBRATE_DIR=<dir>: ignore the zoo and point the pair at <dir>; a missing
# pair makes the server calibrate it at startup, an existing one is loaded.
if [ -n "${CALIBRATE_DIR:-}" ]; then
  mkdir -p "$CALIBRATE_DIR"
  KROT=$CALIBRATE_DIR/k_rotation_qqt_r_h_pbr.pt; VROT=$CALIBRATE_DIR/v_rotation_qqt_r_h_pbr.pt
  echo "  rotations: startup calibration into $CALIBRATE_DIR ($([ -f "$KROT" ] && echo present || echo missing))"
else
  [ -n "$KROT" ] && [ -n "$VROT" ] || { echo "RESULT=$NAME SKIP(no rotation in $R)"; exit 0; }
  echo "  rotations: $(basename "$KROT") / $(basename "$VROT")"
fi
# KV_QUANT selects the quant-tier encoders; codebooks live beside the rotation
# they were trained against. kpq: PQ K + INT2 V; krvq: residual PQ K + INT2 V;
# kpq-vpq: PQ K + PQ V.
K_QUANTIZER=int2; V_QUANTIZER=int2; PQ_K_CB=""; PQ_V_CB=""
case "${KV_QUANT:-int2}" in
  int2) ;;
  kpq)     K_QUANTIZER=pq; PQ_K_CB=$R/codebooks/k_pq_n16_c256_d8.pt ;;
  krvq)    K_QUANTIZER=pq; PQ_K_CB=$R/codebooks/k_rvq_n16_c256x16_d8.pt ;;
  kpq-vpq) K_QUANTIZER=pq; V_QUANTIZER=pq; PQ_K_CB=$R/codebooks/k_pq_n16_c256_d8.pt; PQ_V_CB=$R/codebooks/v_pq_n16_c256_d8.pt ;;
  *) echo "RESULT=$NAME FAIL(unknown KV_QUANT=$KV_QUANT)"; exit 1 ;;
esac
for cb in $PQ_K_CB $PQ_V_CB; do
  [ -f "$cb" ] || { echo "RESULT=$NAME SKIP(no codebook $cb)"; exit 0; }
done
[ "$K_QUANTIZER" = int2 ] || echo "  quantizers: K=$K_QUANTIZER V=$V_QUANTIZER ($(basename "$PQ_K_CB")${PQ_V_CB:+ / $(basename "$PQ_V_CB")})"

verify_prune_weights "$REPO"
MD=$(verify_download "$REPO")
[ -n "$MD" ] || { echo "RESULT=$NAME SKIP(download)"; exit 0; }
verify_drain_gpus
verify_release_page_cache

# VERIFY_GPUS: the GPUs this launch may use (default 0..TP-1), so several harness
# runs can share a pod without landing on the same cards.
PORT=$((34000 + RANDOM % 1500))
# radix cache ON (no --disable-radix-cache) and cuda graph ON (no
# --disable-cuda-graph) -- that is the whole point of this run.
# the recall probe sizes its budget from the server context
export VERIFY_CTX="${CTX:-16384}"
SGLANG_ENABLE_MIXED_KV_WINDOWS=1 \
SGLANG_MIXED_KV_PREFIX_TOKENS=$SINK SGLANG_MIXED_KV_RECENT_TOKENS=$RECENT \
SGLANG_OSCAR_K_ROTATION_PATH=$KROT SGLANG_OSCAR_V_ROTATION_PATH=$VROT \
SGLANG_OSCAR_K_QUANTIZER=$K_QUANTIZER SGLANG_OSCAR_V_QUANTIZER=$V_QUANTIZER \
SGLANG_OSCAR_PQ_K_CODEBOOK=$PQ_K_CB SGLANG_OSCAR_PQ_V_CODEBOOK=$PQ_V_CB \
CUDA_VISIBLE_DEVICES=${VERIFY_GPUS:-$(seq -s, 0 $((TP-1)))} \
python3 -m sglang.launch_server --model-path "$MD" --trust-remote-code \
  --tp-size "$TP" --port $PORT --host 127.0.0.1 \
  --attention-backend triton --prefill-attention-backend triton \
  --decode-attention-backend triton --mm-attention-backend triton_attn \
  --kv-cache-dtype int2 --kv-cache-quant-group-size "$GS" \
  --mem-fraction-static "$MF" --context-length "${CTX:-16384}" \
  --max-running-requests 2 --cuda-graph-max-bs-decode 2 \
  ${MAMBA_STRATEGY:+--mamba-radix-cache-strategy $MAMBA_STRATEGY} \
  ${NO_AUTOTUNE:+--disable-flashinfer-autotune} \
  $EXTRA > "$LOG" 2>&1 &
SRV=$!
# 10 s per try. A tp=8 model reads 60-90 shards off a shared volume; under
# contention that alone passed 25 minutes and a healthy MiniMax-M3 was
# reported as FAIL(no-serve) with the server still loading.
if ! verify_wait_serve $SRV $PORT "${WAIT_TRIES:-$(( TP >= 8 ? 360 : 150 ))}"; then
  grep -aE "Error|assert|OutOfMemory|Page size|not divisible|no attribute" "$LOG" \
    | grep -avE "Ignore import" | tail -3
  kill -9 $SRV 2>/dev/null; wait $SRV 2>/dev/null
  echo "RESULT=$NAME FAIL(no-serve)"; exit 1
fi
verify_probe_and_verdict "$NAME" $PORT "$LOG"
kill -TERM $SRV 2>/dev/null; wait $SRV 2>/dev/null
