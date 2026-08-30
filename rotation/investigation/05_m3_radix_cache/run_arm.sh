#!/usr/bin/env bash
# One MiniMax-M3 GPQA arm, 2-node TP=16, CUDA graph ON, 3 seeds per server
# lifetime. The only intended difference between the int2 arms is DISABLE_RADIX.
#
# Required env: ROLE=head|worker  ARM=<tag>  PORT  DIST_PORT  MASTER_SVC
# Optional:     DISABLE_RADIX KV_MODE MIXED_KV_AUDIT SEEDS NUM_WORKERS CGBS
#               MAX_RUNNING MAX_NEW_TOKENS NUM_EXAMPLES
set -uo pipefail
rm -f /dev/shm/nccl* /dev/shm/torch_* /dev/shm/sglang* 2>/dev/null || true

BASE=/shared/zhongzhu/m3radix
R=$BASE/OSCAR

# /home/charlie is 100% full on this cluster, so keep every cache off it.
export HOME=/tmp/m3r-${ARM}-${ROLE}
mkdir -p "$HOME"
export FLASHINFER_WORKSPACE_BASE="$HOME"
export TVM_FFI_CACHE_DIR="$HOME/tvm-ffi"
export TRITON_CACHE_DIR="$HOME/triton"
export SGLANG_DG_CACHE_DIR="$HOME/dg"
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
mkdir -p "$TVM_FFI_CACHE_DIR" "$TRITON_CACHE_DIR" "$SGLANG_DG_CACHE_DIR"
chmod -R a+rwX "$HOME" 2>/dev/null || true

export HF_HOME=/shared/huggingface
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA='^mlx5_0'
export NCCL_NET=IB
export NCCL_IB_DISABLE=0

RUN=$BASE/out/${ARM}/${ROLE}
mkdir -p "$RUN"

if [ "$ROLE" = head ]; then
    MASTER_IP=$(hostname -I | awk '{print $1}')
    NODE_RANK=0
else
    # Resolve the head through the headless Service: with hostNetwork the pod
    # IP is the node IP, so there is no master-IP file to go stale.
    for _ in $(seq 1 360); do
        MASTER_IP=$(getent hosts "$MASTER_SVC" 2>/dev/null | awk '{print $1}' | head -1)
        [ -n "${MASTER_IP:-}" ] && break
        sleep 5
    done
    [ -n "${MASTER_IP:-}" ] || { echo "[arm ${ARM}/${ROLE}] no master via ${MASTER_SVC}"; exit 1; }
    NODE_RANK=1
fi
echo "[arm ${ARM}/${ROLE}] master=${MASTER_IP}:${DIST_PORT} node_rank=${NODE_RANK} host=$(hostname)"

cd "$R"
echo "[arm ${ARM}/${ROLE}] code under test:"; git log --oneline -1

# The shared `oscar` conda env predates this sglang revision and is missing
# jsonschema (serving_chat imports it at module scope) and tiktoken. /home/charlie
# is 100% full, so they live on /shared instead of being installed into the env;
# only genuinely-absent packages are here, so nothing in the env gets shadowed.
export PYTHONPATH="$BASE/pydeps${PYTHONPATH:+:$PYTHONPATH}"

export CONDA_BASE=/home/charlie/miniconda3
export CONDA_ENV_NAME=oscar
export MODEL=MiniMaxAI/MiniMax-M3
export ROT_DIR=$R/rotation/MiniMax-M3/rotations
export K_ROT_FILENAME=k_rotation_hadamard.pt
export V_ROT_FILENAME=v_rotation_hadamard.pt
export RUN_DIR=$RUN
export NAME=gpqa_m3_${ARM}
export TP_SIZE=16 NNODES=2 NODE_RANK=$NODE_RANK DIST_ADDR=${MASTER_IP}:${DIST_PORT}
export GPUS=0,1,2,3,4,5,6,7
export MEM_FRAC=0.85
export MAX_RUNNING=${MAX_RUNNING:-16}
export CUDA_GRAPH_MAX_BS=${CGBS:-8}
export KV_MODE=${KV_MODE:-int2}
# Project-wide invariant, set explicitly rather than inherited: BF16 sink 64 +
# BF16 recent 256. Grep the server log for "Enable unified mixed KV (int2):
# prefix=64 recent=256" to confirm it reached the pool.
export SGLANG_MIXED_KV_PREFIX_TOKENS=64
export SGLANG_MIXED_KV_RECENT_TOKENS=256
export GROUP_SIZE=128 K_CLIP=0.96 V_CLIP=0.92 LLOYD_MAX=0 ABSORB_V=0
export NUM_WORKERS=${NUM_WORKERS:-8}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32768}
export N_REPEATS=1 TEMPERATURE=1.0 TOP_P=0.95 TOP_K=40
export SEEDS="${SEEDS:-0 1 2}"
export DISABLE_RADIX=${DISABLE_RADIX:-0}
export MIXED_KV_AUDIT=${MIXED_KV_AUDIT:-0}
export MIXED_KV_AUDIT_EVERY=${MIXED_KV_AUDIT_EVERY:-25}
# M3 weight load alone is ~21.5 min at TP=16 (measured Load weight end
# elapsed=1292 s), and the pair may also wait for its partner pod to be
# scheduled, so the stock 20 min /health cap can never be met.
export HEALTH_WAIT_STEPS=${HEALTH_WAIT_STEPS:-2160}
# The harness puts `--dist-init-addr 127.0.0.1:$DIST_PORT` after the multi-node
# block, so the loopback value wins on argparse. EXTRA_SERVER_ARGS is appended
# last, which is the only place a real head address survives. --dist-timeout
# covers a partner pod that is still queueing for GPUs.
export EXTRA_SERVER_ARGS="--context-length 40960 --dtype bfloat16 --skip-server-warmup --watchdog-timeout 3600 --dist-timeout 7200 --log-requests --log-requests-level 0 --dist-init-addr ${MASTER_IP}:${DIST_PORT}"

T0=$(date +%s)
bash rotation/eval_oscar_gpqa.sh
RC=$?
WALL=$(( $(date +%s) - T0 ))
echo "$WALL" > "$RUN/wall_seconds"
echo "[arm ${ARM}/${ROLE}] rc=${RC} wall=${WALL}s"
exit $RC
