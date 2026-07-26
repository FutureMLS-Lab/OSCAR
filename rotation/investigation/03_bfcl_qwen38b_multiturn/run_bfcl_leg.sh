#!/bin/bash
# BFCL multi-turn on Qwen3-8B, one LEG per invocation (runs inside the K8s pod).
# Repro of external report: OSCAR INT2 KV dropping BFCL multi-turn 41->30.5.
# Legs isolate serving-config deltas from the quant method itself:
#   bf16          - baseline, reporter-style serving (radix ON, cuda graphs ON)
#   int2_repro    - reporter's exact INT2 serving (radix ON, cuda-graph-max-bs 16)
#   int2_protocol - same quant, OUR eval protocol (--disable-radix-cache --disable-cuda-graph)
#   int2_lm       - protocol + SGLANG_LLOYD_MAX=1
#   int2_hp2048   - protocol + SGLANG_MIXED_KV_PREFIX_TOKENS=2048 (cover tool schemas)
set -euo pipefail
LEG=${LEG:?set LEG}
RUN_TAG=${RUN_TAG:-run0}
PORT=${PORT:-31500}
ZOO=/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/CoQuant/rotation/OSCAR-RotationZoo
case "${LEG}" in
  q332_*)  # Qwen3-32B legs
    TP=${TP:-4}
    MODEL=/shared/huggingface/hub/models--Qwen--Qwen3-32B/snapshots
    ROT=${ZOO}/Qwen3-32B/seq16000_prompt69_group128
    BFCL_MODEL="Qwen/Qwen3-32B-FC"
    SERVED_NAME="Qwen/Qwen3-32B" ;;
  *)
    TP=${TP:-2}
    MODEL=/shared/huggingface/hub/models--Qwen--Qwen3-8B/snapshots
    ROT=${ZOO}/Qwen3-8B/seq20000_prompt83_group128
    BFCL_MODEL="Qwen/Qwen3-8B-FC"
    SERVED_NAME="Qwen/Qwen3-8B" ;;
esac
MODEL=$(ls -d ${MODEL}/*/ | head -1)

INV=/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/CoQuant/rotation/investigation/03_bfcl_qwen38b_multiturn
SGL=/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/CoQuant/sglang-research
BFCL_SRC=/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/gorilla/berkeley-function-call-leaderboard
OUT=${INV}/legs/${LEG}/${RUN_TAG}
mkdir -p "${OUT}"

# scrub stale IPC from any prior SIGKILL'd server (GLM-5.2 lesson)
rm -rf /dev/shm/nccl* /dev/shm/torch_* /dev/shm/sglang* 2>/dev/null || true
rm -f /tmp/*file_baton* 2>/dev/null || true

source /home/charlie/miniconda3/etc/profile.d/conda.sh
conda activate oscar
export TRITON_CACHE_DIR=/tmp/triton_cache_${LEG}   # pod-local: weka-shared ~/.triton races across pods
mkdir -p "${TRITON_CACHE_DIR}" 
export PYTHONPATH=${SGL}/python:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=$(ls /var/run/nvidia-container-devices 2>/dev/null | paste -sd, || echo 0,1)

# ---- preflight: fail fast (job retries elsewhere) instead of wedging 10 min ----
# 1) CUDA must init in this container (node 057-style cuInit=999 breakage)
python3 - <<'PYEOF' || { echo "PREFLIGHT_FAIL cuInit"; exit 3; }
import ctypes, sys
sys.exit(0 if ctypes.CDLL("libcuda.so.1").cuInit(0) == 0 else 1)
PYEOF
# 2) our two ALLOCATED GPUs must be free (privileged neighbors sometimes squat them);
#    we never take non-allocated GPUs instead — that's someone else's allocation.
for u in $(ls /var/run/nvidia-container-devices 2>/dev/null); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=$u 2>/dev/null | tr -d ' ')
  echo "[preflight] $u used=${used:-?}MiB"
  [ -n "$used" ] && [ "$used" -gt 2000 ] && { echo "PREFLIGHT_FAIL squatted $u ${used}MiB"; exit 3; }
done

MEMFRAC=0.85; MAXRUN=128
case "${LEG}" in
  int2_hp2048) MEMFRAC=0.70; MAXRUN=16 ;;  # HP-prefix pool = prefix_tokens x max_running in BF16
  int2_*|q332_int2_*) MEMFRAC=0.75 ;;             # INT2 pool sizing leaves ~0GB for cuda-graph capture at 0.85 (reporter also used 0.75)
esac
SERVER_ARGS=(
  --model-path "${MODEL}" --served-model-name ${SERVED_NAME}
  --tensor-parallel-size ${TP}
  --prefill-attention-backend fa3 --decode-attention-backend triton
  --mem-fraction-static ${MEMFRAC} --max-running-requests ${MAXRUN}
  --host 127.0.0.1 --port ${PORT} --trust-remote-code
)
INT2_ARGS=( --kv-cache-dtype int2 --kv-cache-quant-group-size 128 )

ENV_COMMON=(
  SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
)
ENV_INT2=(
  SGLANG_ENABLE_MIXED_KV_WINDOWS=1
  SGLANG_MIXED_KV_HP_MAX_SPLITS=8
  SGLANG_MIXED_KV_PREFIX_TOKENS=64
  SGLANG_MIXED_KV_RECENT_TOKENS=256
  SGLANG_MIXED_KV_HP_DTYPE=bfloat16
  SGLANG_MIXED_KV_SCALE_DTYPE=float32
  SGLANG_OSCAR_ABSORB_V_ROTATION=1
  SGLANG_OSCAR_K_ROTATION_PATH=${ROT}/k_rotation_qqt_r_h_pbr.pt
  SGLANG_OSCAR_V_ROTATION_PATH=${ROT}/v_rotation_sst_r_h_pbr.pt
  SGLANG_OSCAR_K_CLIP_RATIO=0.96
  SGLANG_OSCAR_V_CLIP_RATIO=0.92
  SGLANG_LLOYD_MAX=0
)

case "${LEG}" in
  bf16)
    ARGS=( "${SERVER_ARGS[@]}" --cuda-graph-max-bs 16 )
    ENVV=( "${ENV_COMMON[@]}" ) ;;
  int2_repro)   # reporter-verbatim semantics: radix cache ON, graphs ON
    ARGS=( "${SERVER_ARGS[@]}" "${INT2_ARGS[@]}" --cuda-graph-max-bs 16 )
    ENVV=( "${ENV_COMMON[@]}" "${ENV_INT2[@]}" ) ;;
  int2_protocol) # our accuracy protocol
    ARGS=( "${SERVER_ARGS[@]}" "${INT2_ARGS[@]}" --disable-radix-cache --disable-cuda-graph )
    ENVV=( "${ENV_COMMON[@]}" "${ENV_INT2[@]}" ) ;;
  int2_radixoff) # single-delta: radix cache OFF, cuda graphs stay ON (deployable config)
    ARGS=( "${SERVER_ARGS[@]}" "${INT2_ARGS[@]}" --disable-radix-cache --cuda-graph-max-bs 16 )
    ENVV=( "${ENV_COMMON[@]}" "${ENV_INT2[@]}" ) ;;
  int2_lm_g) # deployable config + Lloyd-Max (zero extra bits)
    ARGS=( "${SERVER_ARGS[@]}" "${INT2_ARGS[@]}" --disable-radix-cache --cuda-graph-max-bs 16 )
    ENVV=( "${ENV_COMMON[@]}" "${ENV_INT2[@]}" SGLANG_LLOYD_MAX=1 ) ;;
  int2_lm_g_r512) # + recent 512 (~+3% bits at BFCL median ctx)
    ARGS=( "${SERVER_ARGS[@]}" "${INT2_ARGS[@]}" --disable-radix-cache --cuda-graph-max-bs 16 )
    ENVV=( "${ENV_COMMON[@]}" "${ENV_INT2[@]}" SGLANG_LLOYD_MAX=1 SGLANG_MIXED_KV_RECENT_TOKENS=512 ) ;;
  int2_best_noclip) # winner config, clip disabled (zero-bit; retrieval outliers stay exact)
    ARGS=( "${SERVER_ARGS[@]}" "${INT2_ARGS[@]}" --disable-radix-cache --cuda-graph-max-bs 16 )
    ENVV=( "${ENV_COMMON[@]}" "${ENV_INT2[@]}" SGLANG_LLOYD_MAX=1 SGLANG_MIXED_KV_RECENT_TOKENS=512 SGLANG_OSCAR_K_CLIP_RATIO=0 SGLANG_OSCAR_V_CLIP_RATIO=0 ) ;;
  int2_best_g64) # winner config, group 64 (+0.5 bpe scale overhead; precision probe)
    ARGS=( "${SERVER_ARGS[@]}" --kv-cache-dtype int2 --kv-cache-quant-group-size 64 --disable-radix-cache --cuda-graph-max-bs 16 )
    ENVV=( "${ENV_COMMON[@]}" "${ENV_INT2[@]}" SGLANG_LLOYD_MAX=1 SGLANG_MIXED_KV_RECENT_TOKENS=512 ) ;;
  int2_best_g256) # winner config, group 256 (-0.25 bpe)
    ARGS=( "${SERVER_ARGS[@]}" --kv-cache-dtype int2 --kv-cache-quant-group-size 256 --disable-radix-cache --cuda-graph-max-bs 16 )
    ENVV=( "${ENV_COMMON[@]}" "${ENV_INT2[@]}" SGLANG_LLOYD_MAX=1 SGLANG_MIXED_KV_RECENT_TOKENS=512 ) ;;
  int2_lm)
    ARGS=( "${SERVER_ARGS[@]}" "${INT2_ARGS[@]}" --disable-radix-cache --disable-cuda-graph )
    ENVV=( "${ENV_COMMON[@]}" "${ENV_INT2[@]}" SGLANG_LLOYD_MAX=1 ) ;;
  int2_hp2048)
    ARGS=( "${SERVER_ARGS[@]}" "${INT2_ARGS[@]}" --disable-radix-cache --disable-cuda-graph )
    ENVV=( "${ENV_COMMON[@]}" "${ENV_INT2[@]}" SGLANG_MIXED_KV_PREFIX_TOKENS=2048 SGLANG_MIXED_KV_HP_PREFIX_POOL_TOKENS=65536 ) ;;
  q332_bf16)
    ARGS=( "${SERVER_ARGS[@]}" --cuda-graph-max-bs 16 )
    ENVV=( "${ENV_COMMON[@]}" ) ;;
  q332_int2_best) # winner recipe on 32B: radix OFF + graphs ON + LM + recent512
    ARGS=( "${SERVER_ARGS[@]}" "${INT2_ARGS[@]}" --disable-radix-cache --cuda-graph-max-bs 16 )
    ENVV=( "${ENV_COMMON[@]}" "${ENV_INT2[@]}" SGLANG_LLOYD_MAX=1 SGLANG_MIXED_KV_RECENT_TOKENS=512 ) ;;
  *) echo "unknown LEG=${LEG}"; exit 1 ;;
esac

kill_server(){ [[ -n "${SPID:-}" ]] && { kill -TERM ${SPID} 2>/dev/null||true; sleep 3; kill -KILL ${SPID} 2>/dev/null||true; pkill -KILL -P ${SPID} 2>/dev/null||true; }; }
trap kill_server EXIT INT TERM

echo "[bfcl-leg] LEG=${LEG} tag=${RUN_TAG} model=${MODEL}"
env "${ENVV[@]}" python -m sglang.launch_server "${ARGS[@]}" > "${OUT}/server.log" 2>&1 &
SPID=$!
for i in $(seq 1 120); do
  curl -s http://127.0.0.1:${PORT}/health >/dev/null 2>&1 && break
  kill -0 ${SPID} 2>/dev/null || { echo "server-died-during-boot"; tail -5 "${OUT}/server.log"; exit 1; }
  sleep 5; [[ $i == 120 ]] && { echo server-timeout; exit 1; }
done
echo "[bfcl-leg] server up"

# ---- BFCL client (multi_turn = base, miss_func, miss_param, long_context) ----
conda activate bfcl
export BFCL_PROJECT_ROOT=${OUT}/bfcl_root
rm -rf "${BFCL_PROJECT_ROOT}/result" "${BFCL_PROJECT_ROOT}/score"   # stale entries from a crashed attempt would be kept by BFCL's resume
mkdir -p "${BFCL_PROJECT_ROOT}"
export LOCAL_SERVER_ENDPOINT=127.0.0.1
export LOCAL_SERVER_PORT=${PORT}
cd "${BFCL_SRC}"
bfcl generate --model "${BFCL_MODEL}" --test-category multi_turn \
  --num-threads ${BFCL_THREADS:-16} --skip-server-setup --allow-overwrite \
  > "${OUT}/generate.log" 2>&1
# a dead server mid-generation leaves "Error during inference" entries and a completed-looking run
curl -s http://127.0.0.1:${PORT}/health >/dev/null 2>&1 || { echo "SERVER_DEAD_AFTER_GENERATE"; exit 1; }
nerr=$(grep -hc "Error during inference" "${BFCL_PROJECT_ROOT}"/result/*/multi_turn/*.json 2>/dev/null | awk '{s+=$1} END{print s+0}')
[ "${nerr}" -gt 40 ] && { echo "TOO_MANY_INFERENCE_ERRORS ${nerr}"; exit 1; }
bfcl evaluate --model "${BFCL_MODEL}" --test-category multi_turn \
  > "${OUT}/evaluate.log" 2>&1 || true
cp -r "${BFCL_PROJECT_ROOT}/score" "${OUT}/" 2>/dev/null || true
tail -30 "${OUT}/evaluate.log" | tee "${OUT}/summary.txt"
echo "[bfcl-leg] DONE ${LEG}/${RUN_TAG}"
