#!/usr/bin/env bash
# Qwen3-30B-A3B accuracy sweep: BF16 vs INT2 shared-rotation vs INT2 per-head.
#
# One arm x one benchmark per pod. ARM=bf16|shared|perhead, BENCH=<name>.
# Both INT2 arms run with SGLANG_OSCAR_ABSORB_V_ROTATION=0: absorption cannot
# fold a per-head rotation (it would need R_v.T on a 3D tensor), so keeping it
# off is what makes the two INT2 arms differ only in the rotation itself.
set -uo pipefail
: "${ARM:?ARM=bf16|shared|perhead}"; : "${BENCH:?BENCH required}"
W=/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/oscar
git config --global --add safe.directory "$W" 2>/dev/null || true
# Deliberately no git fetch/merge here: every arm shares one weka worktree, and
# 12 pods merging into it concurrently race on index.lock and can be importing
# sglang while another pod rewrites the files. Update the worktree once, before
# launching, and let the pods read it.
echo "[bench] ARM=$ARM BENCH=$BENCH HEAD=$(cd $W && git log --oneline -1)"

MODEL=$(ls -d /shared/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/*/ | head -1)
ROT=$W/rotation/qwen3-30b-a3b/GPQA/seq30000_prompt117_group128/rotations
OUT=$W/rotation/investigation/07_perhead_bench/out/$ARM
mkdir -p "$OUT"

# simple_evals is a submodule and is empty in fresh worktrees
if [ ! -f "$W/third_party/simple_evals/gpqa_eval.py" ]; then
  mkdir -p $W/third_party
  cp -r /home/charlie/CoQuant/third_party/simple_evals $W/third_party/ 2>/dev/null || true
  find $W/third_party/simple_evals -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
fi
[ -f "$W/third_party/simple_evals/gpqa_eval.py" ] || { echo "[bench] ABORT: simple_evals missing"; exit 1; }

export HF_HOME=/shared/huggingface HF_DATASETS_CACHE=/shared/huggingface/datasets
DEVS=$(ls /var/run/nvidia-container-devices 2>/dev/null | paste -sd,)
[ -n "$DEVS" ] && export CUDA_VISIBLE_DEVICES="$DEVS"
export GPUS="${CUDA_VISIBLE_DEVICES:-0,1}"

COMMON=(MODEL="$MODEL" OUT_BASE="$OUT" TP_SIZE=2 GPUS="$GPUS"
        BENCHES="$BENCH" SEEDS="${SEEDS:-0 1 2}" MAX_NEW_TOKENS=32768
        NUM_WORKERS="${NUM_WORKERS:-32}" MEM_FRAC=0.80
        PORT=$((31300 + RANDOM % 300)) DIST_PORT=$((41300 + RANDOM % 300)))

case "$ARM" in
  bf16)    env "${COMMON[@]}" MODE=bf16 bash $W/rotation/_eval_runner/run_bench_matrix.sh ;;
  shared)  env "${COMMON[@]}" MODE=calibrated ROT_DIR="$ROT" \
             K_ROT_FILENAME=k_rotation_qqt_r_h_pbr.pt V_ROT_FILENAME=v_rotation_sst_r_h_pbr.pt \
             SGLANG_OSCAR_ABSORB_V_ROTATION=0 SGLANG_LLOYD_MAX=0 \
             bash $W/rotation/_eval_runner/run_bench_matrix.sh ;;
  perhead) env "${COMMON[@]}" MODE=calibrated ROT_DIR="$ROT" \
             K_ROT_FILENAME=k_perhead.pt V_ROT_FILENAME=v_perhead.pt \
             SGLANG_OSCAR_ABSORB_V_ROTATION=0 SGLANG_LLOYD_MAX=0 \
             bash $W/rotation/_eval_runner/run_bench_matrix.sh ;;
  *) echo "bad ARM=$ARM"; exit 1 ;;
esac

# the per-head arm must actually have taken the per-head path
if [ "$ARM" = perhead ] && ! grep -q "per-head: 4 kv heads" "$OUT/server.log"; then
  echo "[bench] WARNING: perhead arm did not report the per-head loader path"
fi
echo "[bench] $ARM/$BENCH done"
