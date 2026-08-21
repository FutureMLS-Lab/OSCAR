#!/usr/bin/env python3
"""Generate the GLM-5.2-FP8 recent-window sweep at the FULL 198 GPQA questions.

Why a 64/256 arm is re-run rather than reused. The archived 64/256 full-198 arm
(``/shared/gpqa_glm52_win64_256_full``, 76.77) cannot be compared to a new arm
on one specific knob: ``SGLANG_LLOYD_MAX``. Nothing in that run's ``server.log``
records it -- the pool logs the write path, the window sizes and every dtype,
but not the codebook -- and the launching Job is long deleted. Lloyd-Max is not
a cosmetic knob on this project (it measurably HURT Gemma-4 LCB and MiniMax-M3
long generations), and GLM-5.2's entire deficit is long generations, so an arm
whose LM setting is unknown cannot anchor a sweep.

So the control is re-run inside the sweep, with every knob pinned and dumped to
``$RUN_DIR/config.env``. That buys two things at the cost of one node:

* the window sweep is internally consistent -- 256 vs 512 vs 1024 differ in
  exactly one environment variable, so the direction of the effect is clean
  regardless of the absolute level;
* it identifies the archived arm's LM setting by elimination. If the re-run
  256 arm lands near 76.77 the archived arm was LM=1 and the sweep is directly
  commensurable with it; if it lands far away, the archived arm was LM=0 and
  the sweep is reported as its own family instead of being spliced onto it.

Operational choices that are not incidental:

* ``preemptionPolicy: Never`` via ``zz-k3-nopreempt``. b200 runs priority-1e9
  CI; matching that priority keeps these arms from being evicted 5 hours in,
  while Never means they evict nobody else.
* No ``nodeSelector``. The ``gpu-queue.xp/admit`` webhook gates node-pinned
  pods indefinitely -- they sit ``SchedulingGated`` until the queue's 3600 s
  timeout and then become permanently unschedulable, so pinning costs the arm
  its FIFO position for nothing.
* ``backoffLimit: 0`` and a long ``ttlSecondsAfterFinished``, plus a ``tee``
  of all pod output onto the PVC. A head-side gloo "connection closed" is
  usually a *downstream* symptom of another rank dying, and the real traceback
  is only in the dead rank's own output -- which is gone once the pod is
  reclaimed. Persisting it is the difference between diagnosing a crash and
  re-running blind.
"""
from __future__ import annotations

import argparse
import base64
from pathlib import Path

# (arm suffix, recent-window tokens)
ARMS = [("w256", 256), ("w512", 512), ("w1024", 1024)]

SINK = 64
LLOYD_MAX = 1
GROUP = 128
# Exact commit the BF16 full-198 reference arm ran. Pinning to a SHA rather
# than to origin/zhongzhu/hybrid-model is deliberate: the branch is shared and
# moved twice mid-session, so a branch-tracking sweep measures whatever landed
# between launches. Verified to contain the latent-window code.
PIN = "7f0f986c0"

SCRIPT = r"""export HEALTH_WAIT_STEPS=480
set -uo pipefail
R=/shared; SRC=$R/oscar; W=/work/oscar; PIN=__PIN__
# The container is already isolated by nvidia.com/gpu; setting
# CUDA_VISIBLE_DEVICES to UUIDs makes torch read them as ordinals
# (AcceleratorError: invalid device ordinal).
export GPUS=$(seq -s, 0 7)

# PRIVATE, PINNED CHECKOUT -- both halves matter, and the first version of this
# sweep got both wrong and had to be killed 5 minutes in.
#
# Private: the shared clone /shared/oscar is ONE working tree, and every arm
# used to `git checkout -f` into it while importing Python from it. Run three
# arms at once and they fight: the observed symptom was `cannot lock ref ...
# is at d5716136 but expected 7f0f986c` on one pod, which then reset the shared
# tree back to the older commit *while another arm's server was importing from
# it*. Two arms reported different HEADs for what was supposed to be a
# one-variable sweep, and a checkout can swap .py files under a live import.
#
# Pinned: the branch is shared with other agents and moved twice during this
# session (7f0f986c -> f895f0ca -> d5716136). Tracking a moving branch means
# arms launched minutes apart measure different code. PIN is the exact commit
# the BF16 full-198 reference arm ran, so the sweep is comparable to it.
mkdir -p /work
git clone -q --local --no-hardlinks "$SRC" "$W" || { echo "FATAL clone failed"; exit 1; }
git -C "$W" checkout -q -f --detach "$PIN" || { echo "FATAL pin $PIN failed"; exit 1; }
# third_party/simple_evals is a submodule (gitlink), so a plain clone leaves it
# EMPTY and the eval would die after the 30-min weight load. Copy the populated
# tree from the source instead of fetching it, so every arm scores against
# byte-identical grading code with no network dependency.
cp -a "$SRC/third_party/simple_evals/." "$W/third_party/simple_evals/"
find "$W/third_party/simple_evals" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
for f in third_party/simple_evals/gpqa_eval.py rotation/eval_oscar_gpqa.sh \
         sglang-research/python/sglang/srt/mem_cache/mla_int2_kv_pool.py; do
  test -s "$W/$f" || { echo "FATAL missing $f in private clone"; exit 1; }
done
. $R/oscar-venv/bin/activate
export HF_HOME=/hf PYTHONPATH=$W/sglang-research/python
echo "[glm] HEAD=$(git -C $W log --oneline -1)  (pinned $PIN, private clone $W)"

# B200 is SM100; FA3 covers SM80-90 only.
export ATTN_BACKEND=triton PREFILL_BACKEND=triton
# Rotations present -> the latent IS quantized (the BF16 arm points this at a
# nonexistent dir on purpose; here it must load 78 layers).
export MLA_ROT_PATH=$R/glm52-rotations ROT_DIR=$R/glm52-rotations
echo "[glm] rotation layers: $(ls $ROT_DIR/layer_*.pt | wc -l)"
# Pin the KV cache dtype. sglang otherwise picks fp8_e4m3 on SM100+ for a DSA
# model, which pushes ~28% of each 512x512 rotation subnormal and scored 5.56.
export MLA_KV_CACHE_DTYPE=bfloat16
export LLOYD_MAX=__LM__ MLA_GROUP_SIZE=__GROUP__
export SGLANG_MIXED_KV_PREFIX_TOKENS=__SINK__
export SGLANG_MIXED_KV_RECENT_TOKENS=__RECENT__
export MODEL=zai-org/GLM-5.2-FP8 TP_SIZE=8 NAME=__NAME__
# MEM_FRAC, not MEM_FRACTION_STATIC: eval_oscar_gpqa.sh reads MEM_FRAC first.
export CUDA_GRAPH_MAX_BS=32 MEM_FRAC=0.85 NUM_WORKERS=16
export RUN_DIR=$R/__RUNDIR__; mkdir -p $RUN_DIR

# Self-documenting arm: dump every knob that decides WHICH METHOD this score
# belongs to. This exists because the archived 64/256 arm's Lloyd-Max setting
# is unrecoverable from its server.log, which cost this sweep a whole node.
{
  echo "HEAD=$(git -C $W log --oneline -1)"
  env | grep -E '^(MLA_|ROT_|LLOYD_MAX|SGLANG_|MODEL|TP_SIZE|CUDA_GRAPH_MAX_BS|MEM_FRAC|NUM_WORKERS|ATTN_BACKEND|PREFILL_BACKEND|DISABLE_RADIX|NAME)' | sort
} > $RUN_DIR/config.env
cat $RUN_DIR/config.env

# tee onto the PVC: backoffLimit=0 means no retry, and once the pod is
# reclaimed its stdout is the only copy of a dying rank's traceback.
bash $W/rotation/eval_oscar_gpqa.sh 2>&1 | tee -a $RUN_DIR/pod.log | tail -25
echo "[glm] score:"; grep -iE "gpqa/score" $RUN_DIR/eval.log 2>/dev/null | head -2
# Read the arm's identity back from what the SERVER reported, not from intent.
echo "[glm] server says:"
grep -ohE "\[Int2HPKVPool\] write path=.*|\[MLAInt2\].*" $RUN_DIR/server.log \
  | sort -u | head -6
grep -oE "disable_radix_cache=[A-Za-z]+|kv_cache_dtype='[a-z0-9_]*'|max_total_num_tokens=[0-9]+" \
  $RUN_DIR/server.log | sort -u
"""

JOB = """---
apiVersion: batch/v1
kind: Job
metadata:
  name: {name}
  namespace: charlie
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 604800
  template:
    spec:
      restartPolicy: Never
      priorityClassName: zz-k3-nopreempt
      tolerations:
      - operator: Exists
      containers:
      - name: g
        image: nvcr.io/nvidia/pytorch:25.01-py3
        imagePullPolicy: IfNotPresent
        command: ["bash", "-lc", "echo {b64} | base64 -d > /tmp/g.sh; bash /tmp/g.sh"]
        env:
        - {{name: PYTHONUNBUFFERED, value: "1"}}
        - {{name: HOME, value: /tmp/h}}
        resources:
          limits: {{nvidia.com/gpu: "8"}}
          requests: {{cpu: "48", memory: 800Gi, nvidia.com/gpu: "8"}}
        volumeMounts:
        - {{name: sh, mountPath: /shared}}
        - {{name: hf, mountPath: /hf}}
        - {{name: shm, mountPath: /dev/shm}}
        - {{name: work, mountPath: /work}}
      volumes:
      - {{name: sh, persistentVolumeClaim: {{claimName: shared-data}}}}
      - {{name: hf, persistentVolumeClaim: {{claimName: hf-cache-pvc}}}}
      - {{name: shm, emptyDir: {{medium: Memory, sizeLimit: 200Gi}}}}
      # Node-local scratch for this arm's private clone: per-pod by
      # construction, so no two arms can share a working tree.
      - {{name: work, emptyDir: {{sizeLimit: 20Gi}}}}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).with_name("sweep09.yaml")))
    ap.add_argument("--tag", default="r3")
    a = ap.parse_args()

    docs = []
    for suffix, recent in ARMS:
        run = f"gpqa_glm52_{suffix}_{a.tag}"
        body = (SCRIPT
                .replace("__PIN__", PIN)
                .replace("__LM__", str(LLOYD_MAX))
                .replace("__GROUP__", str(GROUP))
                .replace("__SINK__", str(SINK))
                .replace("__RECENT__", str(recent))
                .replace("__NAME__", run)
                .replace("__RUNDIR__", run))
        b64 = base64.b64encode(body.encode()).decode()
        docs.append(JOB.format(name=f"zz-glm52-{suffix}-{a.tag}", b64=b64))

    Path(a.out).write_text("".join(docs))
    print(f"wrote {a.out} with {len(docs)} arms: "
          + ", ".join(f"sink{SINK}/recent{r} (LM={LLOYD_MAX})" for _, r in ARMS))


if __name__ == "__main__":
    main()
