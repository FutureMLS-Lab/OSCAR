#!/usr/bin/env bash
# Copy the working tree into the image build context, so the image is built
# FROM the branch rather than from whatever the context happened to hold.
#
# The image and the branch drifting apart is the failure this guards: v27 was
# built before a month of fixes that only ever reached running pods through a
# tarball overlay, so `docker pull v27` and `git checkout` disagreed.
#
#   bash rotation/verify/sync_image_src.sh <build-context-dir>
set -euo pipefail
CTX=${1:?usage: sync_image_src.sh <build-context-dir>}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DEST="$CTX/oscar-src"

[ -d "$DEST" ] || { echo "no build context at $DEST" >&2; exit 1; }
cd "$ROOT"

git diff --quiet && git diff --cached --quiet || {
  echo "working tree is dirty; commit first so the image has a commit to name" >&2
  exit 1
}
REV=$(git rev-parse --short HEAD)
BRANCH=$(git branch --show-current)

rsync -a --delete \
  --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
  --exclude '.RUD' --exclude 'rotation/*/GPQA' --exclude '*.pt' \
  ./ "$DEST/"

printf '%s %s\n' "$BRANCH" "$REV" > "$DEST/BUILT_FROM"
echo "synced $BRANCH@$REV -> $DEST"

# Spot-check the fixes the Dockerfile asserts, here rather than 40 minutes into
# a build.
#
# The gate-rebase check is a PAIR on purpose: the caller must scale the gate
# into log2 space (RCP_LN2 in kda.py) and the kernels must read it with exp2.
# Having one without the other was the original defect -- decay applied as
# exp(0.693*g) instead of exp(g), too weak and compounding with length. An
# earlier hand-fix put the literal in chunk_intra.py; porting upstream's
# kernels moved it to the caller, so pinning the old location would have
# failed a build over a constant that had simply relocated.
for pat in \
  'sglang-research/python/sglang/srt/layers/attention/fla/kda.py:RCP_LN2' \
  'sglang-research/python/sglang/srt/layers/attention/fla/chunk_intra.py:exp2' \
  'sglang-research/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py:_decode_query_start_loc' \
  'sglang-research/python/sglang/srt/models/kimi_k3.py:local_num_heads' \
  'sglang-research/python/sglang/srt/environ.py:SGLANG_K3_AR_FUSION' \
  'rotation/eval_oscar_gpqa.sh:DECODE_BACKEND'; do
  f=${pat%%:*}; needle=${pat#*:}
  grep -q "$needle" "$DEST/$f" || { echo "MISSING in context: $f -> $needle" >&2; exit 1; }
done
test -f "$DEST/sglang-research/python/sglang/srt/runtime_context.py"
test -x "$DEST/rotation/verify/preflight_port.py"
echo "build context carries this branch's fixes"
