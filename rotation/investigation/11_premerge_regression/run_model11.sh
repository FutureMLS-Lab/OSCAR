#!/bin/bash
# One model row of the pre-merge regression sweep (investigation/11).
#
# Generalized from investigation/10_m27_graphbs/run_arm10.sh: the tree
# isolation, HOME/cache redirection away from the 100%-full /home/charlie PVC,
# GPU-squat preflight and weight-completeness verification are kept as-is so
# rows stay comparable with 06/08/10. What is new here:
#
#   * TREE defaults to /shared/zz-premerge/tree, NOT zz-m27-recheck/tree_post.
#     tree_post is at 4b06c059 and contains ZERO occurrences of _safe_block_h,
#     i.e. it is a PRE-fix tree -- "post" there refers to an older, unrelated
#     page-0 fix. Using run_arm10's default would measure code that is not
#     what is being merged. Both facts are asserted below, not assumed.
#   * MODEL_DIR selects which rotation/<model>/eval_gpqa.sh wrapper drives the
#     run, so each model keeps its own validated clip ratios.
#   * The rotation files are verified to be real calibrated rotations before a
#     single GPU-second is spent. A missing or failed rotation does NOT error --
#     it silently serves at Hadamard quality, which reads as a quantization
#     result. Hadamard has every |entry| identical, so |entry| spread/mean is
#     exactly 0; a calibrated rotation is far from 0.
#
# Env contract (set by the manifest):
#   ARM         run name -> $ROOT/runs/$ARM
#   MODEL_REPO  hub dir name, e.g. models--Qwen--Qwen3-8B
#   MODEL_DIR   rotation/<dir> holding eval_gpqa.sh, e.g. qwen3-8B
#   ROT_SUBDIR  rotation dir relative to $TREE/rotation (empty = wrapper default)
#   TP_SIZE / LLOYD_MAX / HP_PREFIX / HP_RECENT / PORT
#   NUM_EXAMPLES / MAX_NEW_TOKENS / MAX_RUNNING / NUM_WORKERS
#   CUDA_GRAPH_MAX_BS / DISABLE_RADIX / MEM_FRAC / ABSORB_V
set -uo pipefail

: "${ARM:?set ARM}"
: "${MODEL_REPO:?set MODEL_REPO}"
: "${MODEL_DIR:?set MODEL_DIR}"
TREE="${TREE:-/shared/zz-premerge/tree}"
ROOT=/shared/zz-premerge
HUB=/shared/huggingface/hub
PINNED="${PINNED:-afdcad68b771d312b0eae3d6edde5c08a3881f87}"

source /home/charlie/miniconda3/etc/profile.d/conda.sh
conda activate oscar
export HF_HOME=/shared/huggingface
# /home/charlie (home-charlie-rwx PVC) is 100% full / 0 bytes free, so any cache
# defaulting under $HOME dies with ENOSPC. Keep HOME pod-local; paths absolute.
export HOME=/tmp/h
export CONDA_BASE=/home/charlie/miniconda3 CONDA_ENV_NAME=oscar
export XDG_CACHE_HOME=/tmp/xdg TMPDIR=/tmp
export TVM_FFI_CACHE_DIR=/tmp/tvmffi TORCH_EXTENSIONS_DIR=/tmp/torch_ext
export SGLANG_CACHE_DIR=/tmp/sglcache
mkdir -p /tmp/h /tmp/xdg /tmp/tvmffi /tmp/torch_ext /tmp/sglcache
git config --global --add safe.directory '*' 2>/dev/null || true

# --- Preflight 0: the tree is the one being merged, read from the source ------
echo "[arm] source TREE=${TREE}"
echo "[arm] source PINNED_SHA file: $(cat "${TREE}/PINNED_SHA" 2>&1)"
echo "[arm] source git HEAD:        $(git -C "${TREE}" rev-parse HEAD 2>&1)"
DECODE_ATTN=${TREE}/sglang-research/python/sglang/srt/layers/attention/triton_ops/decode_attention.py
SBH=$(grep -c "_safe_block_h" "${DECODE_ATTN}" 2>/dev/null || echo 0)
echo "[arm] source _safe_block_h occurrences: ${SBH}"
[ "${SBH}" -ge 1 ] || { echo "PREFLIGHT_FAIL pre-fix tree: no _safe_block_h in ${DECODE_ATTN}"; exit 7; }

# /shared/... trees are shared: other jobs `git checkout -f` into them while
# this one imports Python from them. Copy to pod-local storage and pin, so two
# rows can never silently measure different code.
if [ "${ISOLATE_TREE:-1}" = "1" ]; then
  echo "[arm] copying ${TREE} -> /tmp/tree"
  rm -rf /tmp/tree && cp -a "${TREE}" /tmp/tree || exit 8
  TREE=/tmp/tree
fi
# Re-read every assertion from the tree the server will actually import from.
TREE_SHA=$(git -C "${TREE}" rev-parse HEAD 2>&1)
echo "[arm] LIVE TREE=${TREE} sha=${TREE_SHA}"
echo "[arm] LIVE PINNED_SHA=$(cat "${TREE}/PINNED_SHA" 2>&1)"
[ "${TREE_SHA}" = "${PINNED}" ] || { echo "PREFLIGHT_FAIL live sha ${TREE_SHA} != pinned ${PINNED}"; exit 7; }
LIVE_SBH=$(grep -c "_safe_block_h" "${TREE}/sglang-research/python/sglang/srt/layers/attention/triton_ops/decode_attention.py" 2>/dev/null || echo 0)
echo "[arm] LIVE _safe_block_h occurrences: ${LIVE_SBH}"
[ "${LIVE_SBH}" -ge 1 ] || { echo "PREFLIGHT_FAIL live tree lacks _safe_block_h"; exit 7; }
git -C "${TREE}" status --porcelain | head -20

# third_party/simple_evals is a gitlink: a plain clone leaves it empty and every
# grader imports it, so it fails only AFTER the 10-40 minute weight load.
for f in common.py gpqa_eval.py types.py; do
  [ -f "${TREE}/third_party/simple_evals/${f}" ] || { echo "PREFLIGHT_FAIL simple_evals/${f} missing"; exit 6; }
done
echo "[arm] simple_evals ok ($(ls "${TREE}/third_party/simple_evals" | wc -l) files)"

export SGLANG_RESEARCH_DIR=${TREE}/sglang-research
export PYTHONUNBUFFERED=1

MODEL=$(ls -d ${HUB}/${MODEL_REPO}/snapshots/*/ 2>/dev/null | head -1)
[ -n "${MODEL:-}" ] || { echo "FATAL: ${MODEL_REPO} snapshot not found under $HUB"; exit 2; }
export MODEL
export RUN_DIR=${ROOT}/runs/${ARM}
mkdir -p "$RUN_DIR"
export OSCAR_TRITON_PER_RANK_BASE=/tmp/triton_${ARM}
export TRITON_CACHE_DIR=/tmp/triton_${ARM}/main
mkdir -p "$TRITON_CACHE_DIR"

TP_SIZE=${TP_SIZE:-2}
PORT=${PORT:-32200}
MEM_FRAC=${MEM_FRAC:-0.85}
MAX_RUNNING=${MAX_RUNNING:-32}
NUM_WORKERS=${NUM_WORKERS:-16}
CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-32}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}
NUM_EXAMPLES=${NUM_EXAMPLES:-24}
DISABLE_RADIX=${DISABLE_RADIX:-0}
HP_PREFIX=${HP_PREFIX:-64}
HP_RECENT=${HP_RECENT:-256}
LLOYD_MAX=${LLOYD_MAX:-0}
ABSORB_V=${ABSORB_V:-0}

export CUDA_VISIBLE_DEVICES=$(ls /var/run/nvidia-container-devices 2>/dev/null | paste -sd, || echo 0)
rm -rf /dev/shm/nccl* /dev/shm/torch_* /dev/shm/sglang* 2>/dev/null || true

# --- Preflight 1: never share a GPU. A squatted card silently halves the KV
# pool or OOMs, and the resulting score reads as a quantization result. -------
for u in $(ls /var/run/nvidia-container-devices 2>/dev/null); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=$u 2>/dev/null | tr -d ' ')
  echo "[preflight] gpu $u used=${used:-?}MiB"
  [ -n "$used" ] && [ "$used" -gt 2000 ] && { echo "PREFLIGHT_FAIL squatted $u ${used}MiB"; exit 3; }
done

# --- Preflight 2: weights complete (indexed shards all present) --------------
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

# --- Preflight 3: the rotation is real, not a silent Hadamard fallback -------
# This cannot be skipped. A missing/failed rotation load does not raise; it
# serves at Hadamard quality and the score reads as a quantization result.
if [ -n "${ROT_SUBDIR:-}" ]; then
  export ROT_DIR="${TREE}/rotation/${ROT_SUBDIR}"
else
  export ROT_DIR="${TREE}/rotation/${MODEL_DIR}/rotations"
fi
K_ROT="${ROT_DIR}/${K_ROT_FILENAME:-k_rotation_qqt_r_h_pbr.pt}"
V_ROT="${ROT_DIR}/${V_ROT_FILENAME:-v_rotation_sst_r_h_pbr.pt}"
echo "[preflight] ROT_DIR=${ROT_DIR}"
python3 - "$K_ROT" "$V_ROT" <<'PY' || exit 5
import sys, torch
for path in sys.argv[1:]:
    o = torch.load(path, map_location="cpu", weights_only=False)
    layers = o["layers"] if isinstance(o, dict) and "layers" in o else o
    items = sorted(layers.items(), key=lambda kv: str(kv[0]))
    rots = [v["rotation"] if isinstance(v, dict) else v for _, v in items]
    # A per-(layer, KV head) V2 rotation is [H, d, d]; `A.T` raises on a 3-D
    # tensor in torch 2.x, so the original 2-D-only check REJECTED the real
    # per-head format -- and the tempting workaround (fall back to the shared
    # file) is exactly the silent-Hadamard failure this gate exists to prevent.
    # Validate every head's matrix independently instead, and refuse anything
    # that is neither 2-D nor 3-D rather than letting it through unchecked.
    worst, sms, n_mats = 0.0, [], 0
    for R in rots:
        A = R.float()
        if A.dim() == 2:
            blocks = [A]
        elif A.dim() == 3:
            blocks = [A[i] for i in range(A.shape[0])]
        else:
            raise AssertionError(
                f"unexpected rotation ndim={A.dim()} shape={tuple(A.shape)} "
                f"in {path} -- refusing to serve an unvalidated rotation")
        for B in blocks:
            assert B.shape[-1] == B.shape[-2], f"non-square block {tuple(B.shape)}"
            I = torch.eye(B.shape[-1], dtype=B.dtype)
            worst = max(worst, (B @ B.T - I).abs().max().item())
            n_mats += 1
        a = A.abs()
        sms.append((a.std() / a.mean()).item())
    sm = sum(sms) / len(sms)
    fv = o.get("format_version") if isinstance(o, dict) else None
    print(f"[preflight] rotation {path.split('/')[-1]}: format_version={fv} "
          f"n_layers={len(rots)} shape={tuple(rots[0].shape)} "
          f"matrices_checked={n_mats} "
          f"max|RRt-I|={worst:.3e} |entry|spread/mean={sm:.4f}")
    if rots[0].dim() == 3 and fv not in (2, "2"):
        print(f"[preflight] WARNING: 3-D rotation but format_version={fv!r}; "
              f"per-head files should declare format_version 2")
    assert worst < 1e-3, f"NOT ORTHOGONAL ({worst:.3e})"
    # Hadamard: every |entry| identical -> spread/mean == 0 exactly.
    assert sm > 0.20, f"HADAMARD-LIKE spread/mean {sm:.4f} -- silent fallback"
print("[preflight] rotations are real calibrated rotations")
PY

echo "[arm] ARM=$ARM model=$MODEL_REPO wrapper=${MODEL_DIR}/eval_gpqa.sh tp=$TP_SIZE" \
     "sha=${TREE_SHA} radix_off=$DISABLE_RADIX maxbs=$CUDA_GRAPH_MAX_BS" \
     "client_concurrency=$NUM_WORKERS max_running=$MAX_RUNNING n=$NUM_EXAMPLES" \
     "budget=$MAX_NEW_TOKENS hp_prefix=$HP_PREFIX hp_recent=$HP_RECENT" \
     "lloyd_max=$LLOYD_MAX absorb_v=$ABSORB_V start=$(date -Is)"

export TP_SIZE GROUP_SIZE=${GROUP_SIZE:-128}
export LLOYD_MAX ABSORB_V
export SGLANG_MIXED_KV_PREFIX_TOKENS=${HP_PREFIX}
export SGLANG_MIXED_KV_RECENT_TOKENS=${HP_RECENT}
export HP_PREFIX_POOL_TOKENS=${HP_PREFIX_POOL_TOKENS:-0}
export MEM_FRAC MAX_RUNNING NUM_WORKERS CUDA_GRAPH_MAX_BS MAX_NEW_TOKENS NUM_EXAMPLES
export DISABLE_RADIX
export MIXED_KV_AUDIT=${MIXED_KV_AUDIT:-0}
export PORT DIST_PORT=$((PORT + 10000))
export GPUS="$CUDA_VISIBLE_DEVICES"
export NAME="gpqa_${ARM}"
export PYTHONPATH=/shared/charlie/py_extra   # jsonschema + tiktoken overlay
[ -n "${EXTRA_SERVER_ARGS:-}" ] && export EXTRA_SERVER_ARGS

# Record every knob so the row can be audited without the manifest.
{
  echo "arm=$ARM"; echo "tree=$TREE"; echo "tree_sha=$TREE_SHA"
  echo "safe_block_h_occurrences=$LIVE_SBH"
  echo "model=$MODEL"; echo "model_repo=$MODEL_REPO"
  echo "wrapper=${MODEL_DIR}/eval_gpqa.sh"
  echo "rot_dir=$ROT_DIR"; echo "k_rot=$K_ROT"; echo "v_rot=$V_ROT"
  echo "tp=$TP_SIZE"; echo "group_size=$GROUP_SIZE"
  echo "lloyd_max=$LLOYD_MAX"; echo "absorb_v=$ABSORB_V"
  echo "hp_prefix=$HP_PREFIX"; echo "hp_recent=$HP_RECENT"
  echo "num_examples=$NUM_EXAMPLES"; echo "max_new_tokens=$MAX_NEW_TOKENS"
  echo "max_running=$MAX_RUNNING"; echo "client_concurrency=$NUM_WORKERS"
  echo "cuda_graph_max_bs=$CUDA_GRAPH_MAX_BS"; echo "disable_radix=$DISABLE_RADIX"
  echo "mem_frac=$MEM_FRAC"; echo "gpus=$GPUS"
  echo "node=$(hostname)"
} > "$RUN_DIR/config.env"
cat "$RUN_DIR/config.env"

cd ${TREE}/rotation
bash "${MODEL_DIR}/eval_gpqa.sh"
rc=$?

echo "[arm] ARM=$ARM rc=$rc end=$(date -Is)"
python3 ${TREE}/rotation/investigation/06_m27_recheck/analyze.py "$RUN_DIR" \
  --arm "$ARM" --json > "${RUN_DIR}/summary.json" 2>&1 || true
cat "${RUN_DIR}/summary.json"
echo "[arm] coherence judge:"
python3 ${TREE}/rotation/investigation/11_premerge_regression/coherence_judge.py \
  "$RUN_DIR/io_log.jsonl" --show 2 2>&1 | tail -40 || true
exit $rc
