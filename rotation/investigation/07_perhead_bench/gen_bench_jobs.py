#!/usr/bin/env python3
"""Emit the benchmark Job manifests for the Qwen3-30B-A3B sweep.

One Job per (arm, benchmark). Kept as a generator rather than 12 checked-in
manifests, which were 755 lines of near-identical YAML.

Usage:
  python gen_bench_jobs.py                    > jobs.yaml   # all 12
  python gen_bench_jobs.py --arms perhead --benches aime25 --seeds "0 1 2 3 4 5 6 7"
"""

import argparse
import base64
import os

ARMS = ["bf16", "shared", "perhead"]
BENCHES = {"gpqa": "gpqa", "humaneval": "he", "aime25": "aime", "math500": "m500"}

TEMPLATE = """apiVersion: batch/v1
kind: Job
metadata: {{name: bq30-{arm}-{short}, namespace: charlie}}
spec:
  backoffLimit: 1
  ttlSecondsAfterFinished: 604800
  template:
    spec:
      restartPolicy: Never
      runtimeClassName: nvidia
      priorityClassName: normal
      containers:
      - name: bench
        image: nvcr.io/nvidia/pytorch:24.12-py3
        env:
        - {{name: HOME, value: /home/charlie}}
        - {{name: PYTHONUNBUFFERED, value: "1"}}
        - {{name: ARM, value: "{arm}"}}
        - {{name: BENCH, value: "{bench}"}}
        - {{name: SEEDS, value: "{seeds}"}}
        - {{name: PYTORCH_CUDA_ALLOC_CONF, value: "expandable_segments:True"}}
        resources:
          limits: {{nvidia.com/gpu: "2", memory: 220Gi, cpu: "32"}}
          requests: {{nvidia.com/gpu: "2", memory: 96Gi, cpu: "16"}}
        command: ["bash","-lc"]
        args: ["echo {b64} | base64 -d > /tmp/bench.sh && bash /tmp/bench.sh"]
        volumeMounts:
        - {{name: shared, mountPath: /shared}}
        - {{name: h, mountPath: /home/charlie}}
        - {{name: shm, mountPath: /dev/shm}}
      volumes:
      - {{name: shared, persistentVolumeClaim: {{claimName: shared-data}}}}
      - {{name: h, persistentVolumeClaim: {{claimName: home-charlie-rwx}}}}
      - {{name: shm, emptyDir: {{medium: Memory, sizeLimit: 64Gi}}}}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=ARMS, choices=ARMS)
    ap.add_argument("--benches", nargs="*", default=list(BENCHES), choices=list(BENCHES))
    ap.add_argument("--seeds", default="0 1 2")
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "bench.sh"), "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()

    docs = [
        TEMPLATE.format(arm=arm, bench=b, short=BENCHES[b], seeds=a.seeds, b64=b64)
        for arm in a.arms
        for b in a.benches
    ]
    print("---\n".join(docs))


if __name__ == "__main__":
    main()
