#!/bin/bash
# One MiniMax-M2.7 GPQA arm for the re-check. Runs inside a charlie-ns pod.
#
# Env contract (set by the manifest):
#   ARM             run name -> /shared/zz-m27-recheck/runs/$ARM
#   TREE            /shared/zz-m27-recheck/tree_post (fixed, 4b06c0593e)
#                   /shared/zz-m27-recheck/tree_pre  (page-0 reservation reverted:
#                                                     the auditor positive control)
#   KV              int2 | bf16
#   DISABLE_RADIX   0|1
#   CUDA_GRAPH_MAX_BS / NUM_WORKERS / MAX_RUNNING / MAX_NEW_TOKENS
#   MIXED_KV_AUDIT  0|1        NUM_EXAMPLES (probe arms only)
set -uo pipefail

: "${ARM:?set ARM}"
TREE="${TREE:-/shared/zz-m27-recheck/tree_post}"
ROOT=/shared/zz-m27-recheck
HUB=/shared/huggingface/hub
KV="${KV:-int2}"

source /home/charlie/miniconda3/etc/profile.d/conda.sh
conda activate oscar
export HF_HOME=/shared/huggingface
# /home/charlie (home-charlie-rwx PVC) is 100% full with 0 bytes available, so
# any cache defaulting under $HOME dies with ENOSPC. Keep HOME pod-local; every
# path used below is absolute. CONDA_BASE keeps eval_oscar_gpqa.sh's own
# `conda activate` pointed at the real prefix.
export HOME=/tmp/h
export CONDA_BASE=/home/charlie/miniconda3 CONDA_ENV_NAME=oscar
export XDG_CACHE_HOME=/tmp/xdg TMPDIR=/tmp
export TVM_FFI_CACHE_DIR=/tmp/tvmffi TORCH_EXTENSIONS_DIR=/tmp/torch_ext
export SGLANG_CACHE_DIR=/tmp/sglcache
mkdir -p /tmp/h /tmp/xdg /tmp/tvmffi /tmp/torch_ext /tmp/sglcache
export SGLANG_RESEARCH_DIR=${TREE}/sglang-research
export PYTHONUNBUFFERED=1

MODEL=$(ls -d ${HUB}/models--MiniMaxAI--MiniMax-M2.7/snapshots/*/ 2>/dev/null | head -1)
[ -n "${MODEL:-}" ] || { echo "FATAL: M2.7 snapshot not found under $HUB"; exit 2; }
export MODEL
export RUN_DIR=${ROOT}/runs/${ARM}
mkdir -p "$RUN_DIR"
export OSCAR_TRITON_PER_RANK_BASE=/tmp/triton_${ARM}
export TRITON_CACHE_DIR=/tmp/triton_${ARM}/main
mkdir -p "$TRITON_CACHE_DIR"

TP_SIZE=${TP_SIZE:-4}
PORT=${PORT:-31940}
MEM_FRAC=${MEM_FRAC:-0.85}
MAX_RUNNING=${MAX_RUNNING:-16}
NUM_WORKERS=${NUM_WORKERS:-16}
CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-32}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-95000}
DISABLE_RADIX=${DISABLE_RADIX:-0}

export CUDA_VISIBLE_DEVICES=$(ls /var/run/nvidia-container-devices 2>/dev/null | paste -sd, || echo 0)
rm -rf /dev/shm/nccl* /dev/shm/torch_* /dev/shm/sglang* 2>/dev/null || true

# Preflight 1: never share a GPU. A squatted card silently halves the KV pool or
# OOMs, and the resulting score reads as a quantization result.
for u in $(ls /var/run/nvidia-container-devices 2>/dev/null); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=$u 2>/dev/null | tr -d ' ')
  echo "[preflight] gpu $u used=${used:-?}MiB"
  [ -n "$used" ] && [ "$used" -gt 2000 ] && { echo "PREFLIGHT_FAIL squatted $u ${used}MiB"; exit 3; }
done

# Preflight 2: which tree is this really? Acceptance item 1 is "page 0 reserved
# in the tree you actually ran", so print it from the file the server imports.
echo "[preflight] tree=${TREE}"
grep -n "hp_prefix_free_pages = torch.arange" -A3 \
  ${TREE}/sglang-research/python/sglang/srt/mem_cache/unified_kv_allocator.py

# Preflight 3: weights complete (indexed shards all present), not "a dir exists".
python3 - "$MODEL" <<'PY' || exit 4
import glob, json, os, sys
p = sys.argv[1]
idx = os.path.join(p, "model.safetensors.index.json")
assert os.path.isfile(idx), "no model.safetensors.index.json"
want = set(json.load(open(idx))["weight_map"].values())
have = {os.path.basename(f) for f in glob.glob(os.path.join(p, "*.safetensors"))}
assert not (want - have), f"missing shards: {sorted(want - have)[:5]}"
for f in ("config.json", "tokenizer.json", "tokenizer_config.json"):
    assert os.path.isfile(os.path.join(p, f)), f"missing {f}"
print(f"[preflight] weights ok: {len(have)} shards, index lists {len(want)}")
PY

echo "[arm] ARM=$ARM KV=$KV tree=$TREE radix_off=$DISABLE_RADIX maxbs=$CUDA_GRAPH_MAX_BS" \
     "workers=$NUM_WORKERS max_running=$MAX_RUNNING budget=$MAX_NEW_TOKENS" \
     "audit=${MIXED_KV_AUDIT:-0} examples=${NUM_EXAMPLES:-all} start=$(date -Is)"

if [ "$KV" = "int2" ]; then
  export ROT_DIR=${ROOT}/rotations
  export TP_SIZE GROUP_SIZE=${GROUP_SIZE:-128}
  export K_CLIP=0.96 V_CLIP=0.92
  export LLOYD_MAX=0    # recipe: uniform quantizer, Lloyd-Max off
  export ABSORB_V=0     # recipe: no V-rotation absorption
  export SGLANG_MIXED_KV_PREFIX_TOKENS=64    # project-wide invariant
  export SGLANG_MIXED_KV_RECENT_TOKENS=256   # project-wide invariant
  export MEM_FRAC MAX_RUNNING NUM_WORKERS CUDA_GRAPH_MAX_BS MAX_NEW_TOKENS
  export DISABLE_RADIX
  export MIXED_KV_AUDIT=${MIXED_KV_AUDIT:-0}
  export MIXED_KV_AUDIT_EVERY=${MIXED_KV_AUDIT_EVERY:-10}
  export PORT DIST_PORT=$((PORT + 10000))
  export GPUS="$CUDA_VISIBLE_DEVICES"
  export EXTRA_SERVER_ARGS="--tool-call-parser minimax-m2"
  export NAME="gpqa_m27_${ARM}"
  export PYTHONPATH=/shared/charlie/py_extra
  [ -n "${NUM_EXAMPLES:-}" ] && export NUM_EXAMPLES
  cd ${TREE}/rotation
  bash eval_oscar_gpqa.sh
  rc=$?
else
  # BF16 paired control: identical serving config, only the KV dtype and the
  # OSCAR rotation differ, so the delta is attributable to the KV cache.
  export PYTHONPATH=${TREE}/rotation/_triton_per_rank:${TREE}/sglang-research/python:/shared/charlie/py_extra
  RADIX_ARGS=()
  [ "$DISABLE_RADIX" = "1" ] && RADIX_ARGS=(--disable-radix-cache)
  python -m sglang.launch_server \
    --model-path "$MODEL" --tensor-parallel-size $TP_SIZE \
    --attention-backend fa3 --prefill-attention-backend fa3 \
    --decode-attention-backend triton \
    --kv-cache-dtype auto \
    --mem-fraction-static ${MEM_FRAC} --max-running-requests ${MAX_RUNNING} \
    --cuda-graph-max-bs ${CUDA_GRAPH_MAX_BS} --enable-cache-report \
    "${RADIX_ARGS[@]}" \
    --host 127.0.0.1 --port $PORT --dist-init-addr 127.0.0.1:$((PORT + 10000)) \
    --tool-call-parser minimax-m2 \
    --trust-remote-code > "$RUN_DIR/server.log" 2>&1 &
  SPID=$!
  cleanup(){ kill -TERM $SPID 2>/dev/null; sleep 3; kill -KILL $SPID 2>/dev/null;
             pkill -KILL -P $SPID 2>/dev/null; }
  trap cleanup EXIT INT TERM
  for _ in $(seq 1 300); do
    curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    kill -0 $SPID 2>/dev/null || { echo "SERVER DIED"; tail -40 "$RUN_DIR/server.log"; exit 1; }
    sleep 5
  done
  curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || {
    echo "SERVER NOT READY"; tail -40 "$RUN_DIR/server.log"; exit 1; }
  echo "[arm] server ready $(date -Is)"
  grep -ao "disable_radix_cache=[A-Za-z]*\|kv_cache_dtype='[a-z0-9_]*'\|cuda_graph_bs=\[[^]]*\]" \
    "$RUN_DIR/server.log" | head -3
  python "${TREE}/rotation/_eval_runner/run_simple_eval.py" \
    --task gpqa --model "$MODEL" --base-url "http://127.0.0.1:$PORT/v1" \
    --max-tokens ${MAX_NEW_TOKENS} --temperature 1.0 --top-p 0.95 --top-k 40 \
    --n-repeats 1 --num-threads ${NUM_WORKERS} \
    ${NUM_EXAMPLES:+--num-examples ${NUM_EXAMPLES}} \
    --output-dir "$RUN_DIR" 2>&1 | tee "$RUN_DIR/runner.log" | tail -20
  rc=$?
fi

echo "[arm] ARM=$ARM rc=$rc end=$(date -Is)"
python3 ${TREE}/rotation/investigation/06_m27_recheck/analyze.py "$RUN_DIR" \
  --arm "$ARM" --json > "${RUN_DIR}/summary.json" 2>&1 || true
cat "${RUN_DIR}/summary.json"
exit $rc
