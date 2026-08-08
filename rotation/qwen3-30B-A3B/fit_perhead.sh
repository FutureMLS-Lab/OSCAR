#!/usr/bin/env bash
# Per-KV-head rotations for Qwen3-30B-A3B (the recipe this model needs).
# Writes format_version 2 checkpoints: rotation is [num_kv_heads, hd, hd].
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FIT="${SCRIPT_DIR}/../fit_perhead_rotation.py"

DATASET="${DATASET:-GPQA}"
if [[ -z "${CALIB_DIR:-}" ]]; then
  # earlier calibrations were written under the lower-case sibling dir;
  # look there too so an existing dump is not silently ignored
  CALIB_DIR="$(ls -1dt "${SCRIPT_DIR}/${DATASET}"/*/ "${SCRIPT_DIR}/../qwen3-30b-a3b/${DATASET}"/*/ 2>/dev/null | head -1 | sed 's:/$::')"
fi
DUMP="${DUMP:-${CALIB_DIR}/qkv_dumps/gpqa}"
OUT="${OUT:-${CALIB_DIR}/rotations}"
NUM_LAYERS="${NUM_LAYERS:-48}"    # Qwen3-30B-A3B
PY="${PY:-${HOME}/miniconda3/envs/oscar/bin/python3}"

[[ -d "${DUMP}" ]] || { echo "no dump at ${DUMP}; run save_qkv_30b.sh first" >&2; exit 1; }
mkdir -p "${OUT}"
echo "[fit] dump=${DUMP} layers=${NUM_LAYERS} out=${OUT}"
"${PY}" "${FIT}" --dump "${DUMP}" --layers "${NUM_LAYERS}" \
  --out-k "${OUT}/k_perhead.pt" --out-v "${OUT}/v_perhead.pt"
"${PY}" - <<PY
import torch
for name in ("k_perhead.pt", "v_perhead.pt"):
    d = torch.load("${OUT}/" + name, map_location="cpu")
    r = next(iter(d["layers"].values()))["rotation"]
    print(f"  {name}: format_version={d.get('format_version')} rotation={tuple(r.shape)}")
    assert d.get("format_version") == 2 and r.dim() == 3, "not a per-head checkpoint"
PY
