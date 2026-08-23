#!/bin/bash
# One MiniMax-M2.7 GPQA arm for the pre-merge regression sweep.
#
# Byte-identical to investigation/10_m27_graphbs/run_arm10.sh except for two
# things, both of which exist because 10's defaults are actively wrong for this
# sweep:
#
#   ROOT is overridable   10 hardcodes /shared/zz-m27-recheck, so every arm's
#                         run dir and rotation dir landed in the investigation-10
#                         area. The pre-merge sweep writes to
#                         /shared/zz-premerge/runs/<arm>/.
#
#   FIX ASSERTION         10 defaults TREE to /shared/zz-m27-recheck/tree_post,
#                         which is at 4b06c05 -- BEFORE 20a8b73452 introduced
#                         _safe_block_h. ("post" there means a different, older
#                         page-0 fix.) Running the default silently measures
#                         PRE-fix code and would produce exactly the wrong
#                         conclusion. Preflight 0 refuses to start a server
#                         unless the head-tiling clamp is present in the tree
#                         that will actually be imported.
#
# Env contract (set by the manifest):
#   ARM             run name -> $ROOT/runs/$ARM
#   ROOT            /shared/zz-premerge
#   TREE            code tree; MUST contain _safe_block_h
#   EXPECT_SHA      optional; asserted against `git rev-parse HEAD`
#   KV              int2 | bf16
#   DISABLE_RADIX   0|1
#   CUDA_GRAPH_MAX_BS / NUM_WORKERS / MAX_RUNNING / MAX_NEW_TOKENS / TP_SIZE
#   HP_RECENT / HP_PREFIX / LLOYD_MAX / K_ROT_FILENAME / V_ROT_FILENAME
set -uo pipefail

: "${ARM:?set ARM}"
TREE="${TREE:-/shared/zz-m27-recheck/tree_post}"
ROOT="${ROOT:-/shared/zz-m27-recheck}"
HUB=/shared/huggingface/hub
KV="${KV:-int2}"

source /home/charlie/miniconda3/etc/profile.d/conda.sh
conda activate oscar
export HF_HOME=/shared/huggingface
# /home/charlie (home-charlie-rwx PVC) is 100% full, so any cache defaulting
# under $HOME dies with ENOSPC. Keep HOME pod-local; all paths below absolute.
export HOME=/tmp/h
export CONDA_BASE=/home/charlie/miniconda3 CONDA_ENV_NAME=oscar
export XDG_CACHE_HOME=/tmp/xdg TMPDIR=/tmp
export TVM_FFI_CACHE_DIR=/tmp/tvmffi TORCH_EXTENSIONS_DIR=/tmp/torch_ext
export SGLANG_CACHE_DIR=/tmp/sglcache
mkdir -p /tmp/h /tmp/xdg /tmp/tvmffi /tmp/torch_ext /tmp/sglcache
# /shared/... trees are shared: other jobs `git checkout -f` into them while
# this one imports Python from them. Copy to pod-local storage and pin, so two
# arms can never silently measure different code.
if [ "${ISOLATE_TREE:-1}" = "1" ]; then
  echo "[arm] copying ${TREE} -> /tmp/tree"
  rm -rf /tmp/tree && cp -a "${TREE}" /tmp/tree || exit 8
  TREE=/tmp/tree
fi
git config --global --add safe.directory '*' 2>/dev/null || true
echo "[arm] TREE=${TREE} sha=$(git -C "${TREE}" rev-parse HEAD 2>&1)"
if [ -n "${TREE_PATCH_FILE:-}" ]; then
  git -C "${TREE}" apply -v "${TREE_PATCH_FILE}" || exit 8
  echo "[arm] applied patch ${TREE_PATCH_FILE} ($(wc -l < "${TREE_PATCH_FILE}") lines)"
fi

# Preflight 0: the tree that will actually be imported must carry the
# head-tiling clamp. Without this, TREE=/shared/zz-m27-recheck/tree_post (10's
# default, at 4b06c05) measures PRE-fix code and reports it as a fixed-tree
# result. Assert on the post-isolation, post-patch TREE -- that is the code
# Python imports.
DEC=${TREE}/sglang-research/python/sglang/srt/layers/attention/triton_ops/decode_attention.py
NCLAMP=$(grep -c "BLOCK_H = _safe_block_h" "$DEC" 2>/dev/null || echo 0)
ACTUAL_SHA=$(git -C "${TREE}" rev-parse HEAD 2>/dev/null || echo unknown)
echo "[preflight0] tree=${TREE} sha=${ACTUAL_SHA} clamp_call_sites=${NCLAMP}"
if [ "${NCLAMP:-0}" -lt 2 ]; then
  echo "PREFLIGHT_FAIL head-tiling clamp missing: only ${NCLAMP} '_safe_block_h' call site(s)" \
       "in ${DEC} -- this tree is PRE-fix, refusing to run"
  exit 9
fi
if [ -n "${EXPECT_SHA:-}" ] && [ "${ACTUAL_SHA}" != "${EXPECT_SHA}" ]; then
  echo "PREFLIGHT_FAIL sha mismatch: expected ${EXPECT_SHA}, tree is ${ACTUAL_SHA}"
  exit 9
fi
sed -n '/^def _safe_block_h/,/^    return block_h/p' "$DEC" | tail -4

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
PORT=${PORT:-32100}
MEM_FRAC=${MEM_FRAC:-0.85}
MAX_RUNNING=${MAX_RUNNING:-32}
NUM_WORKERS=${NUM_WORKERS:-15}
CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-32}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32768}
DISABLE_RADIX=${DISABLE_RADIX:-0}
HP_PREFIX=${HP_PREFIX:-64}
HP_RECENT=${HP_RECENT:-256}

export CUDA_VISIBLE_DEVICES=$(ls /var/run/nvidia-container-devices 2>/dev/null | paste -sd, || echo 0)
rm -rf /dev/shm/nccl* /dev/shm/torch_* /dev/shm/sglang* 2>/dev/null || true

# Preflight 1: never share a GPU. A squatted card silently halves the KV pool or
# OOMs, and the resulting score reads as a quantization result.
for u in $(ls /var/run/nvidia-container-devices 2>/dev/null); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=$u 2>/dev/null | tr -d ' ')
  echo "[preflight] gpu $u used=${used:-?}MiB"
  [ -n "$used" ] && [ "$used" -gt 2000 ] && { echo "PREFLIGHT_FAIL squatted $u ${used}MiB"; exit 3; }
done

echo "[preflight] tree=${TREE}"
grep -n "hp_prefix_free_pages = torch.arange" -A3 \
  ${TREE}/sglang-research/python/sglang/srt/mem_cache/unified_kv_allocator.py

# Preflight 2: weights complete (indexed shards all present), not "a dir exists".
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
     "workers=$NUM_WORKERS max_running=$MAX_RUNNING budget=$MAX_NEW_TOKENS tp=$TP_SIZE" \
     "hp_prefix=$HP_PREFIX hp_recent=$HP_RECENT lloyd_max=${LLOYD_MAX:-0}" \
     "k_rot=${K_ROT_FILENAME:-k_rotation_qqt_r_h_pbr.pt} start=$(date -Is)"

if [ "$KV" = "int2" ]; then
  export ROT_DIR=${ROT_DIR_OVERRIDE:-${ROOT}/rotations}
  export TP_SIZE GROUP_SIZE=${GROUP_SIZE:-128}
  export K_CLIP=${K_CLIP:-0.96} V_CLIP=${V_CLIP:-0.92}
  export LLOYD_MAX=${LLOYD_MAX:-0}
  export ABSORB_V=${ABSORB_V:-0}
  # The two knobs 06 pinned. eval_oscar_gpqa.sh already honours these env names
  # (`SGLANG_MIXED_KV_*_TOKENS=${SGLANG_MIXED_KV_*_TOKENS:-...}`), so exporting
  # them here is sufficient -- no edit to the eval script is needed.
  export SGLANG_MIXED_KV_PREFIX_TOKENS=${HP_PREFIX}
  export SGLANG_MIXED_KV_RECENT_TOKENS=${HP_RECENT}
  # The shared HP-prefix pool defaults to req_slots * P * 16 slots, which at
  # P=1024 would reserve ~32 GB of BF16 KV and never fit alongside the FP8
  # weights. eval_oscar_gpqa.sh forwards this wrapper var to
  # SGLANG_MIXED_KV_HP_PREFIX_POOL_TOKENS; 0 keeps the default formula.
  export HP_PREFIX_POOL_TOKENS=${HP_PREFIX_POOL_TOKENS:-0}
  [ -n "${K_ROT_FILENAME:-}" ] && export K_ROT_FILENAME
  [ -n "${V_ROT_FILENAME:-}" ] && export V_ROT_FILENAME
  export MEM_FRAC MAX_RUNNING NUM_WORKERS CUDA_GRAPH_MAX_BS MAX_NEW_TOKENS
  export DISABLE_RADIX
  export MIXED_KV_AUDIT=${MIXED_KV_AUDIT:-0}
  export PORT DIST_PORT=$((PORT + 10000))
  export GPUS="$CUDA_VISIBLE_DEVICES"
  export EXTRA_SERVER_ARGS="--tool-call-parser minimax-m2 ${EXTRA_ARGS:-}"
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
    --tool-call-parser minimax-m2 ${EXTRA_ARGS:-} \
    --trust-remote-code > "$RUN_DIR/server.log" 2>&1 &
  SPID=$!
  cleanup(){ kill -TERM $SPID 2>/dev/null; sleep 3; kill -KILL $SPID 2>/dev/null;
             pkill -KILL -P $SPID 2>/dev/null; }
  trap cleanup EXIT INT TERM
  for _ in $(seq 1 360); do
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
echo "[arm] tree sha actually run: $(git -C "${TREE}" rev-parse HEAD 2>&1)"
python3 ${TREE}/rotation/investigation/06_m27_recheck/analyze.py "$RUN_DIR" \
  --arm "$ARM" --json > "${RUN_DIR}/summary.json" 2>&1 || true
cat "${RUN_DIR}/summary.json"
JUDGE=${TREE}/rotation/investigation/11_premerge_regression/coherence_judge.py
if [ -f "$JUDGE" ] && [ -f "${RUN_DIR}/io_log.jsonl" ]; then
  python3 "$JUDGE" "${RUN_DIR}/io_log.jsonl" --json \
    > "${RUN_DIR}/coherence.json" 2>&1 || true
  python3 "$JUDGE" "${RUN_DIR}/io_log.jsonl" --show 2 \
    > "${RUN_DIR}/coherence.txt" 2>&1 || true
  cat "${RUN_DIR}/coherence.json"
fi
exit $rc
