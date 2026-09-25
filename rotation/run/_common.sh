# Shared preamble for the per-model run scripts.
#
# These are thin wrappers over eval_oscar_gpqa.sh on purpose. An earlier smoke
# harness re-implemented the launch inline and silently dropped DISABLE_RADIX,
# which is how Qwen3.5 spent three rounds crashing with an illegal memory access
# that was really "Page size must be 1 for MambaRadixCache v1, got 8". The
# serving path lives in one place; these files only carry the per-model recipe.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Rotations: prefer the ones baked into the OSCAR image, fall back to a local
# RotationZoo checkout. They are per model and do NOT transfer -- a missing file
# reads as "no rotation", which at 2 bits is collapse, not a mild regression.
ZOO="${ZOO:-${OSCAR_ROTATIONS:-$HERE/OSCAR-RotationZoo}/zoo}"
[ -d "$ZOO" ] || ZOO="${OSCAR_ROTATIONS:-$HERE/OSCAR-RotationZoo}"

need_rot() {
  local dir="$1"
  if [ -n "${K_ROTATION_FILE:-}" ] || [ -n "${V_ROTATION_FILE:-}" ]; then
    for f in "${K_ROTATION_FILE:?}" "${V_ROTATION_FILE:?}"; do
      [ -f "$dir/$f" ] || { echo "FATAL: missing $dir/$f (named explicitly)"; exit 1; }
    done
    export K_ROT_FILENAME="$K_ROTATION_FILE" V_ROT_FILENAME="$V_ROTATION_FILE"
    echo "[run] rotations: $dir ($K_ROTATION_FILE / $V_ROTATION_FILE)"
    return
  fi
  local k v
  k=$(ls "$dir"/k_rotation_*.pt 2>/dev/null | head -1)
  v=$(ls "$dir"/v_rotation_*.pt 2>/dev/null | head -1)
  [ -n "$k" ] && [ -n "$v" ] || { echo "FATAL: no k_rotation_*.pt / v_rotation_*.pt in $dir"; exit 1; }
  if [ "$(ls "$dir"/k_rotation_*.pt 2>/dev/null | wc -l)" -gt 1 ]; then
    echo "FATAL: $dir holds several k_rotation_*.pt; set K_ROTATION_FILE/V_ROTATION_FILE"
    ls "$dir"/k_rotation_*.pt | sed 's#^#       #'
    exit 1
  fi
  export K_ROTATION_FILE="$(basename "$k")" V_ROTATION_FILE="$(basename "$v")"
  export K_ROT_FILENAME="$K_ROTATION_FILE" V_ROT_FILENAME="$V_ROTATION_FILE"
  echo "[run] rotations: $dir ($K_ROTATION_FILE / $V_ROTATION_FILE)"
}

launch() { exec bash "$HERE/eval_oscar_gpqa.sh"; }
