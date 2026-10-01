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

# Spot-check that the context carries this tree's OSCAR hooks, here rather than
# 40 minutes into a build. Each pair is <file>:<needle>.
for pat in \
  'python/sglang/srt/mem_cache/unified_kv_pool.py:class UnifiedInt2HPKVPool' \
  'python/sglang/srt/mem_cache/kv_cache_configurator.py:_build_oscar_unified_kv_pool' \
  'python/sglang/srt/environ.py:SGLANG_ENABLE_MIXED_KV_WINDOWS' \
  'python/sglang/srt/layers/attention/triton_backend.py:mixed_kv_enabled' \
  'python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py:notify_kv_pool_of_forward_batch' \
  'rotation/eval_oscar_gpqa.sh:DECODE_BACKEND'; do
  f=${pat%%:*}; needle=${pat#*:}
  grep -q "$needle" "$DEST/$f" || { echo "MISSING in context: $f -> $needle" >&2; exit 1; }
done
test -f "$DEST/python/sglang/srt/runtime_context.py"
test -x "$DEST/rotation/verify/preflight_port.py"
echo "build context carries this branch's fixes"
