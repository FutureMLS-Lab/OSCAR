#!/usr/bin/env bash
# MiniMax-M3 pre-merge regression probe (investigation 11), 2-node TP=16.
#
# One INT2 OSCAR arm under the fixed CONTRACT.md protocol:
#   GPQA-diamond NUM_EXAMPLES=24, MAX_NEW_TOKENS=8192, MAX_RUNNING=32,
#   client concurrency 16, radix cache ON, CUDA_GRAPH_MAX_BS=32,
#   temp 1.0 / top_p 0.95 / top_k 40, ABSORB_V=0, no Lloyd-Max,
#   shared published Hadamard rotation, window 64/256, --dtype bfloat16.
#
# Required env: ROLE=head|worker  PORT  DIST_PORT  MASTER_SVC
set -uo pipefail
rm -f /dev/shm/nccl* /dev/shm/torch_* /dev/shm/sglang* 2>/dev/null || true

SRC=/shared/zz-premerge/tree
TREE=/tmp/tree
WANT_SHA=afdcad68b771d312b0eae3d6edde5c08a3881f87
ROT_SRC=/shared/zhongzhu/m3radix/OSCAR/rotation/MiniMax-M3/rotations
OUT=/shared/zz-premerge/runs/zz-pm-m3

# ---- charlie hygiene: the /home/charlie PVC is 100% full, keep caches off it
export HOME=/tmp/h-${ROLE}
export XDG_CACHE_HOME=/tmp/xdg
export TMPDIR=/tmp
export TORCH_EXTENSIONS_DIR=/tmp/torch_ext
export TVM_FFI_CACHE_DIR=/tmp/tvmffi
export SGLANG_CACHE_DIR=/tmp/sglcache
export TRITON_CACHE_DIR=/tmp/triton
export FLASHINFER_WORKSPACE_BASE="$HOME"
export SGLANG_DG_CACHE_DIR=/tmp/dg
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
mkdir -p "$HOME" /tmp/xdg /tmp/torch_ext /tmp/tvmffi /tmp/sglcache /tmp/triton /tmp/dg
chmod -R a+rwX "$HOME" /tmp/tvmffi /tmp/triton /tmp/dg 2>/dev/null || true

export HF_HOME=/shared/huggingface
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA='^mlx5_0'
export NCCL_NET=IB
export NCCL_IB_DISABLE=0

echo "[pm-m3/${ROLE}] host=$(hostname) date=$(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader || true
df -h /tmp | tail -1

# ---- tree assertions BEFORE anything expensive -------------------------------
PINNED="$(cat "$SRC/PINNED_SHA" 2>/dev/null || echo MISSING)"
if [ "$PINNED" != "$WANT_SHA" ]; then
    echo "[pm-m3/${ROLE}] FATAL PINNED_SHA=$PINNED want=$WANT_SHA"; exit 90
fi
DA="$SRC/sglang-research/python/sglang/srt/layers/attention/triton_ops/decode_attention.py"
NSB="$(grep -c '_safe_block_h' "$DA" 2>/dev/null || echo 0)"
if [ "$NSB" -lt 1 ]; then
    echo "[pm-m3/${ROLE}] FATAL _safe_block_h count=$NSB -- this is a PRE-fix tree"; exit 91
fi
echo "[pm-m3/${ROLE}] shared tree ok: PINNED_SHA=$PINNED _safe_block_h=$NSB"

# ---- weights: verify every indexed shard before burning 16 GPUs --------------
python3 - <<'PY' || exit 94
import json, os, struct, sys
snap = "/shared/huggingface/hub/models--MiniMaxAI--MiniMax-M3/snapshots"
rev = open("/shared/huggingface/hub/models--MiniMaxAI--MiniMax-M3/refs/main").read().strip()
snap = os.path.join(snap, rev)
idx = os.path.join(snap, "model.safetensors.index.json")
d = json.load(open(idx))
shards = sorted(set(d["weight_map"].values()))
bad = []
for s in shards:
    p = os.path.realpath(os.path.join(snap, s))
    if not os.path.exists(p):
        bad.append((s, "missing")); continue
    fsz = os.path.getsize(p)
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    end = max(v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__")
    if 8 + n + end != fsz:
        bad.append((s, f"size {fsz} != {8+n+end}"))
print(f"[weights] rev={rev} shards={len(shards)} tensors={len(d['weight_map'])} bad={len(bad)}")
if bad:
    print("[weights] FATAL", bad[:5]); sys.exit(1)
PY

# ---- pod-local copy: NEVER import python from a shared tree ------------------
rm -rf "$TREE"; mkdir -p "$TREE"
cp -a "$SRC/." "$TREE/" || { echo "[pm-m3/${ROLE}] FATAL tree copy failed"; exit 95; }
mkdir -p "$TREE/rotation/MiniMax-M3/rotations"
cp -a "$ROT_SRC/k_rotation_hadamard.pt" "$ROT_SRC/v_rotation_hadamard.pt" \
      "$TREE/rotation/MiniMax-M3/rotations/" || { echo "FATAL rotation copy"; exit 96; }
[ -f "$TREE/third_party/simple_evals/common.py" ] || { echo "FATAL simple_evals empty"; exit 92; }
[ -f "$TREE/rotation/MiniMax-M3/rotations/k_rotation_hadamard.pt" ] || { echo "FATAL no k rot"; exit 93; }
LOCAL_SHA="$(cd "$TREE" && git rev-parse HEAD 2>/dev/null || echo UNKNOWN)"
LOCAL_NSB="$(grep -c '_safe_block_h' "$TREE/sglang-research/python/sglang/srt/layers/attention/triton_ops/decode_attention.py")"
echo "[pm-m3/${ROLE}] LOCAL TREE SHA=$LOCAL_SHA _safe_block_h=$LOCAL_NSB"
[ "$LOCAL_SHA" = "$WANT_SHA" ] || { echo "FATAL local sha drift"; exit 97; }
[ "$LOCAL_NSB" -ge 1 ] || { echo "FATAL local pre-fix"; exit 98; }

# ---- rendezvous --------------------------------------------------------------
if [ "$ROLE" = head ]; then
    MASTER_IP=$(hostname -I | awk '{print $1}')
    NODE_RANK=0
else
    for _ in $(seq 1 360); do
        MASTER_IP=$(getent hosts "$MASTER_SVC" 2>/dev/null | awk '{print $1}' | head -1)
        [ -n "${MASTER_IP:-}" ] && break
        sleep 5
    done
    [ -n "${MASTER_IP:-}" ] || { echo "[pm-m3/${ROLE}] no master via ${MASTER_SVC}"; exit 1; }
    NODE_RANK=1
fi
echo "[pm-m3/${ROLE}] master=${MASTER_IP}:${DIST_PORT} node_rank=${NODE_RANK}"

RUN="$OUT/$ROLE"
mkdir -p "$RUN"
echo "$LOCAL_SHA" > "$RUN/tree_sha"
echo "$(hostname)" > "$RUN/hostname"

cd "$TREE"
export PYTHONPATH="/shared/charlie/py_extra${PYTHONPATH:+:$PYTHONPATH}"
export CONDA_BASE=/home/charlie/miniconda3
export CONDA_ENV_NAME=oscar

export MODEL=MiniMaxAI/MiniMax-M3
export ROT_DIR="$TREE/rotation/MiniMax-M3/rotations"
export K_ROT_FILENAME=k_rotation_hadamard.pt
export V_ROT_FILENAME=v_rotation_hadamard.pt
export RUN_DIR="$RUN"
export NAME=gpqa_pm_m3
export TP_SIZE=16 NNODES=2 NODE_RANK=$NODE_RANK DIST_ADDR=${MASTER_IP}:${DIST_PORT}
export GPUS=0,1,2,3,4,5,6,7
export MEM_FRAC=0.85

# ---- fixed CONTRACT protocol -------------------------------------------------
export NUM_EXAMPLES=24
export MAX_NEW_TOKENS=8192
export MAX_RUNNING=32          # captured set -> [1,2,4,8,12,16,24,32]
export CUDA_GRAPH_MAX_BS=32
export NUM_WORKERS=16          # client concurrency, decoupled from MAX_RUNNING
export N_REPEATS=1 TEMPERATURE=1.0 TOP_P=0.95 TOP_K=40
export DISABLE_RADIX=0         # prefix cache ON: we are verifying it works
export SGLANG_MIXED_KV_PREFIX_TOKENS=64
export SGLANG_MIXED_KV_RECENT_TOKENS=256
export GROUP_SIZE=128 K_CLIP=0.96 V_CLIP=0.92
export LLOYD_MAX=0             # tested and worse on M3; INT2 saturates
export ABSORB_V=0
export HEALTH_WAIT_STEPS=${HEALTH_WAIT_STEPS:-2160}   # M3 weight load ~21.5 min at TP=16
# --dist-init-addr must come last: the harness emits a loopback one earlier and
# argparse is last-wins. --dtype bfloat16 because text_config has no
# torch_dtype and sglang would otherwise pick fp16.
export EXTRA_SERVER_ARGS="--context-length 40960 --dtype bfloat16 --skip-server-warmup --watchdog-timeout 3600 --dist-timeout 7200 --log-requests --log-requests-level 0 --dist-init-addr ${MASTER_IP}:${DIST_PORT}"

env | grep -E '^(SGLANG_|LLOYD|ABSORB|K_CLIP|V_CLIP|GROUP_SIZE|MAX_|NUM_|CUDA_GRAPH|DISABLE_RADIX|TP_SIZE|NNODES|MEM_FRAC|TEMPERATURE|TOP_)' | sort \
    > "$RUN/probe_env.txt"

T0=$(date +%s)
bash rotation/eval_oscar_gpqa.sh
RC=$?
WALL=$(( $(date +%s) - T0 ))
echo "$WALL" > "$RUN/wall_seconds"
echo "[pm-m3/${ROLE}] rc=${RC} wall=${WALL}s"
exit $RC
