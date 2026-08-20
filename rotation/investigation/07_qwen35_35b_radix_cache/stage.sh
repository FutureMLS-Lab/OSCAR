#!/usr/bin/env bash
# Stage a clean CoQuant checkout at a given commit for the 35B prefix-cache A/B.
#
# Runs inside the staging pod. The 4B campaign's controls were taken at
# 4b06c0593e, which predates all three MambaRadixCache mixed-KV fixes, so
# reusing them would leave a code-version confound on the one comparison this
# task exists to make. This stages HEAD once and all three arms run from it.
#
# Usage: stage.sh <workspace_dir> <commit> <source_repo_for_objects>
set -euo pipefail

NEW="${1:?workspace dir}"
COMMIT="${2:?commit}"
SRC="${3:-/shared/CoQuant/radixq35-hybrid/CoQuant}"
REMOTE_URL="https://github.com/FutureMLS-Lab/OSCAR.git"

mkdir -p "${NEW}"
if [[ ! -d "${NEW}/CoQuant/.git" ]]; then
    # Local clone: git hardlinks the object store, so this costs a checkout not
    # a copy of history.
    git clone --quiet "${SRC}" "${NEW}/CoQuant"
fi
cd "${NEW}/CoQuant"
git remote remove upstream 2>/dev/null || true
git remote add upstream "${REMOTE_URL}" 2>/dev/null || git remote set-url upstream "${REMOTE_URL}"
git fetch --quiet upstream zhongzhu/hybrid-model
git checkout --quiet --detach "${COMMIT}"
echo "[stage] HEAD=$(git rev-parse HEAD)"

# The submodule is empty in a fresh clone and simple_evals is what scores GPQA.
if [[ ! -f third_party/simple_evals/gpqa_eval.py ]]; then
    echo "[stage] populating third_party/simple_evals from ${SRC}"
    mkdir -p third_party
    cp -a "${SRC}/third_party/simple_evals/." third_party/simple_evals/
    find third_party/simple_evals -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
fi
ls third_party/simple_evals | head -5

# eval_oscar_gpqa.sh has no --seed passthrough on this branch, and every arm
# here is a 3-seed run. Same one-line patch the 4B campaign used.
if ! grep -q 'SEED:+--seed' rotation/eval_oscar_gpqa.sh; then
    python3 - <<'PY'
import re, pathlib
p = pathlib.Path("rotation/eval_oscar_gpqa.sh")
s = p.read_text()
needle = '    --n-repeats "${N_REPEATS}" \\\n'
assert needle in s, "anchor for --seed patch not found"
s = s.replace(needle, needle + '    ${SEED:+--seed ${SEED}} \\\n', 1)
p.write_text(s)
print("[stage] patched rotation/eval_oscar_gpqa.sh with --seed passthrough")
PY
fi
grep -n 'SEED:+--seed' rotation/eval_oscar_gpqa.sh

mkdir -p "${NEW}/runs" "${NEW}/home"
chmod -R a+rwX "${NEW}/home" 2>/dev/null || true
echo "[stage] done: ${NEW}"
