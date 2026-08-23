#!/usr/bin/env python3
"""Jobs for the packed-INT2 MLA latent work.

Two shapes:

* ``unit`` -- one GPU, minutes. Runs ``test_packed_latent.py``: pack/unpack
  against the torch fake-quant reference, the window arena's owner-tag
  fallback, and the fused decode kernel against a materialized-row reference.
  Everything downstream is meaningless if this fails, and it costs one GPU
  instead of eight for half an hour of weight loading.
* ``glm`` -- eight GPUs, GLM-5.2-FP8. ``MODE=smoke`` is a short generation with
  the self-check on (shadow BF16 pool, every read asserted against the
  fake-quant value); ``MODE=gpqa`` is the full 198.

Operational choices carried over from ``09_glm52_latent_windows`` because each
one was paid for once already: private pinned clone in an emptyDir (the shared
``/shared/oscar`` tree is ONE working tree and parallel arms ``git checkout -f``
each other out from under a live import), ``simple_evals`` copied rather than
cloned (it is a gitlink, and an empty one fails only *after* the 25-minute
weight load), ``preemptionPolicy: Never`` via ``zz-k3-nopreempt``, no
``nodeSelector`` (the admit webhook gates node-pinned pods forever),
``backoffLimit: 0``, and every byte teed onto the PVC.
"""
from __future__ import annotations

import argparse
import base64
from pathlib import Path

PIN = "90daa7add9"

_PROLOGUE = r"""set -uo pipefail
R=/shared; SRC=$R/oscar; W=/work/oscar; PIN=__PIN__
mkdir -p /work
git clone -q --local --no-hardlinks "$SRC" "$W" || { echo "FATAL clone failed"; exit 1; }
git -C "$W" checkout -q -f --detach "$PIN" || { echo "FATAL pin $PIN failed"; exit 1; }
cp -a "$SRC/third_party/simple_evals/." "$W/third_party/simple_evals/" 2>/dev/null || true
find "$W/third_party/simple_evals" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
. $R/oscar-venv/bin/activate
export HF_HOME=/hf PYTHONPATH=$W/sglang-research/python
echo "[job] HEAD=$(git -C $W log --oneline -1)"
"""

UNIT = _PROLOGUE + r"""
export RUN_DIR=$R/__RUNDIR__; mkdir -p $RUN_DIR
rc=0
for t in test_packed_latent test_packed_pool; do
  echo "==== $t ===="
  python3 $W/rotation/investigation/13_mla_packed_int2/$t.py 2>&1 | tee -a $RUN_DIR/unit.log
  [ "${PIPESTATUS[0]}" = 0 ] || rc=1
done
echo "[job] exit=$rc"
"""

KBENCH = _PROLOGUE + r"""
export RUN_DIR=$R/__RUNDIR__; mkdir -p $RUN_DIR
python3 -u $W/rotation/investigation/13_mla_packed_int2/bench_kernel.py 16 16 20000 \
  2>&1 | tee $RUN_DIR/kbench.log
echo "[job] exit=${PIPESTATUS[0]}"
"""

CAPACITY = _PROLOGUE + r"""
export RUN_DIR=$R/__RUNDIR__; mkdir -p $RUN_DIR
export OUT_DIR=$RUN_DIR HF_HUB_OFFLINE=1
# Short prompt, long generation, many streams. The first sweep used a
# 6000-token prompt with 128-token generations, which is prefill-COMPUTE bound:
# both arms plateaued at ~22 tok/s with zero retracts and the 3.3x resident-KV
# advantage converted into nothing. A KV-capacity win needs the decode side to
# be the bottleneck, which is the shape of the earlier 4B out=512 result.
export PROMPT_TOK="${PROMPT_TOK:-512}" GEN_TOK="${GEN_TOK:-2048}"
export CONC="${CONC:-8,16,32,64,128}" CTX_SWEEP="${CTX_SWEEP:-2000,6000,12000}"
python3 -u $W/rotation/investigation/13_mla_packed_int2/bench_capacity.py \
  2>&1 | tee $RUN_DIR/capacity.log
echo "[job] exit=${PIPESTATUS[0]}"
"""

PROBE = _PROLOGUE + r"""
export RUN_DIR=$R/__RUNDIR__; mkdir -p $RUN_DIR
export OUT_DIR=$RUN_DIR
# HF_HUB_OFFLINE: the weights are already in the PVC cache and a hub round trip
# is the difference between "runs" and "runs if the network feels like it".
export HF_HUB_OFFLINE=1
python3 -u $W/rotation/investigation/13_mla_packed_int2/serve_probe.py \
  2>&1 | tee $RUN_DIR/probe.log
echo "[job] exit=${PIPESTATUS[0]}"
"""

GLM = _PROLOGUE + r"""
export GPUS=$(seq -s, 0 7)
# B200 is SM100; the NSA auto-backends are bypassed because the harness names
# triton explicitly, which is also what makes a Python-level dequant reachable.
export ATTN_BACKEND=triton PREFILL_BACKEND=triton
export MLA_ROT_PATH=$R/glm52-rotations ROT_DIR=$R/glm52-rotations
echo "[glm] rotation layers: $(ls $ROT_DIR/layer_*.pt | wc -l)"
# Pin the KV dtype. sglang picks fp8_e4m3 for a DSA model on SM100+, which puts
# ~28% of every 512x512 rotation subnormal; that arm scored 5.56.
export MLA_KV_CACHE_DTYPE=bfloat16
export LLOYD_MAX=1 MLA_GROUP_SIZE=128
export SGLANG_MIXED_KV_PREFIX_TOKENS=64
export SGLANG_MIXED_KV_RECENT_TOKENS=512
export MLA_PACKED=__PACKED__ MLA_PACKED_SELFCHECK=__SELFCHECK__
export MODEL=zai-org/GLM-5.2-FP8 TP_SIZE=8 NAME=__NAME__
# MEM_FRAC, not MEM_FRACTION_STATIC: eval_oscar_gpqa.sh reads MEM_FRAC first.
export CUDA_GRAPH_MAX_BS=32 MEM_FRAC=0.85 NUM_WORKERS=__WORKERS__
export RUN_DIR=$R/__RUNDIR__; mkdir -p $RUN_DIR
__EXTRA__
{
  echo "HEAD=$(git -C $W log --oneline -1)"
  env | grep -E '^(MLA_|ROT_|LLOYD_MAX|SGLANG_|MODEL|TP_SIZE|CUDA_GRAPH_MAX_BS|MEM_FRAC|NUM_WORKERS|ATTN_BACKEND|PREFILL_BACKEND|NUM_EXAMPLES|NAME)' | sort
} > $RUN_DIR/config.env
cat $RUN_DIR/config.env

bash $W/rotation/eval_oscar_gpqa.sh 2>&1 | tee -a $RUN_DIR/pod.log | tail -30

echo "[glm] ==== score ===="
grep -iE "gpqa/score" $RUN_DIR/eval.log 2>/dev/null | head -2
echo "[glm] ==== pool / dtype / cache (read back from the SERVER, not intent) ===="
grep -oE "max_total_num_tokens=[0-9]+|kv_cache_dtype='[a-z0-9_]*'|disable_radix_cache=[A-Za-z]+|page_size=[0-9]+|max_running_requests=[0-9]+" \
  $RUN_DIR/server.log | sort -u
grep -ohE "\[MLAPacked\].*|\[Int2HPKVPool\] write path=.*|\[MLAInt2\] loaded.*" $RUN_DIR/server.log | sort -u | head -12
echo "[glm] ==== cuda graph ===="
grep -oE "Capture cuda graph bs \[[^]]*\]" $RUN_DIR/server.log | sort -u | head -2
echo "graph_true=$(grep -c 'cuda graph: True' $RUN_DIR/server.log) graph_false=$(grep -c 'cuda graph: False' $RUN_DIR/server.log)"
grep -oE "#running-req: [0-9]+" $RUN_DIR/server.log | sort | uniq -c | sort -rn | head -8
echo "[glm] ==== prefix cache ===="
grep -oE "#cached-token: [0-9]+|cache hit rate: [0-9.]+%" $RUN_DIR/server.log | tail -5
echo "[glm] ==== decode throughput ===="
grep -oE "gen throughput \(token/s\): [0-9.]+" $RUN_DIR/server.log | tail -5
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
          limits: {{nvidia.com/gpu: "{gpus}"}}
          requests: {{cpu: "{cpu}", memory: {mem}, nvidia.com/gpu: "{gpus}"}}
        volumeMounts:
        - {{name: sh, mountPath: /shared}}
        - {{name: hf, mountPath: /hf}}
        - {{name: shm, mountPath: /dev/shm}}
        - {{name: work, mountPath: /work}}
      volumes:
      - {{name: sh, persistentVolumeClaim: {{claimName: shared-data}}}}
      - {{name: hf, persistentVolumeClaim: {{claimName: hf-cache-pvc}}}}
      - {{name: shm, emptyDir: {{medium: Memory, sizeLimit: 200Gi}}}}
      - {{name: work, emptyDir: {{sizeLimit: 20Gi}}}}
"""


def emit(name, body, gpus, cpu, mem):
    return JOB.format(
        name=name,
        b64=base64.b64encode(body.encode()).decode(),
        gpus=gpus,
        cpu=cpu,
        mem=mem,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["unit", "probe", "kbench", "capacity", "smoke", "gpqa", "gpqa-fake", "gpqa-noradix"])
    ap.add_argument("--tag", default="a")
    ap.add_argument("--pin", default=PIN)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.kind == "unit":
        run = f"mlapacked_unit_{a.tag}"
        body = UNIT.replace("__PIN__", a.pin).replace("__RUNDIR__", run)
        doc = emit(f"zz-mlapk-unit-{a.tag}", body, 1, "8", "64Gi")
    elif a.kind == "kbench":
        run = f"mlapacked_kbench_{a.tag}"
        body = KBENCH.replace("__PIN__", a.pin).replace("__RUNDIR__", run)
        doc = emit(f"zz-mlapk-kbench-{a.tag}", body, 1, "16", "200Gi")
    elif a.kind == "capacity":
        run = f"mlapacked_capacity_{a.tag}"
        body = CAPACITY.replace("__PIN__", a.pin).replace("__RUNDIR__", run)
        doc = emit(f"zz-mlapk-cap-{a.tag}", body, 1, "16", "220Gi")
    elif a.kind == "probe":
        run = f"mlapacked_probe_{a.tag}"
        body = PROBE.replace("__PIN__", a.pin).replace("__RUNDIR__", run)
        doc = emit(f"zz-mlapk-probe-{a.tag}", body, 1, "16", "200Gi")
    else:
        packed = "0" if a.kind == "gpqa-fake" else "1"
        if a.kind == "smoke":
            run = f"mlapacked_smoke_{a.tag}"
            # 8 questions x 2048 new tokens: enough to exercise prefill, graph
            # replay at the client concurrency, a radix hit across questions,
            # and -- the part that matters -- generations several times longer
            # than the 512-token recent window, so tokens actually age out of
            # the BF16 ring and are served from the packed tier. Not enough to
            # score; scoring is the gpqa arm's job.
            extra = "export NUM_EXAMPLES=8 MAX_NEW_TOKENS=2048"
            selfcheck, workers = "1", "4"
        else:
            run = f"mlapacked_{a.kind.replace('gpqa-','').replace('gpqa','gpqa')}_{a.tag}"
            # The decisive test for the -12.20 pp regression. GPQA's shared
            # prefix is exactly 64 tokens = the sink width, so on a radix hit
            # the WHOLE cached region is served from the packed tier here while
            # both reference arms keep it BF16. Turning the cache off removes
            # that difference and nothing else: if the gap collapses the sink is
            # the cause, if it survives the sink is exonerated and the bf16
            # end-rotations become the primary suspect.
            extra = ("export DISABLE_RADIX=1"
                     if a.kind == "gpqa-noradix" else "")
            # 16 workers is what the 80.30 / 82.32 reference arms ran at.
            # Concurrency selects which captured graph replays, so it is an
            # experimental variable and has to match, not be tuned.
            selfcheck, workers = "0", "16"
        body = (
            GLM.replace("__PIN__", a.pin)
            .replace("__RUNDIR__", run)
            .replace("__NAME__", run)
            .replace("__PACKED__", packed)
            .replace("__SELFCHECK__", selfcheck)
            .replace("__WORKERS__", workers)
            .replace("__EXTRA__", extra)
        )
        doc = emit(f"zz-mlapk-{a.kind}-{a.tag}", body, 8, "48", "800Gi")

    out = a.out or str(Path(__file__).with_name(f"job_{a.kind}_{a.tag}.yaml"))
    Path(out).write_text(doc)
    print(f"wrote {out}  (pin {a.pin})")


if __name__ == "__main__":
    main()
