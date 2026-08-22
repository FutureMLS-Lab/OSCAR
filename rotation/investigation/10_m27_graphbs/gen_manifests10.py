#!/usr/bin/env python3
"""charlie-ns Job manifests for root-causing the M2.7 bs>=16 graph-replay defect.

Investigation 08 established the trigger with 3-seed full GPQA-198 arms:

    conc  7 -> bs  7 (97.7% padded)  graph 100%   score 77.8   reptail  0
    conc  8 -> bs  8 (unpadded)      graph 100%   score 80.8   reptail  0
    conc 12 -> bs 12 (unpadded)      graph 100%   score 78.8   reptail  0
    conc 15 -> bs 16 (96.9% padded)  graph 100%   score 51.0   reptail 26
    conc 16 -> bs 16 (unpadded)      graph 100%   score 53.5   reptail 27
    conc 15, --cuda-graph-max-bs 1   graph  5%    score 75.8   reptail  0

Two things are already excluded by direct measurement rather than argument:

* padding (conc 7 is 97.7% padded and healthy; conc 16 is unpadded and broken),
* the mixed-KV INT2 decode attention kernels themselves -- an offline probe
  replayed ``decode_attention_fwd_int2_unified`` through sglang's own capture
  order and shared scratch at every captured size and got BIT-IDENTICAL output
  against the eager path at bs 1..32,
* split counts -- with ``triton_attention_num_kv_splits=8`` and M2.7's 12q/2kv
  heads per TP rank, ``get_num_kv_splits_triton`` returns exactly 8 for every
  batch size in 1..32 at any decode-length sequence, so there is nothing for a
  captured batch size to freeze differently.

So the remaining question is what ELSE the batch size changes under replay.
Each arm below kills one class, at 48 questions instead of 198 so an arm is
~1 h rather than ~2 h. ``repetitive_tail`` and ``resp_chars_median`` are the
read-outs, not the score: they separate the two regimes by 20+ counts and 2x
respectively, which survives the smaller sample far better than the score does.

  base    conc 16, everything default -- positive control, must reproduce
  c8      conc  8, everything default -- negative control; if this does not
          come back clean the 48-question protocol is not sensitive enough and
          nothing else in the family can be read
  bf16    conc 16 with a BF16 KV cache. THE arm: 08 never ran a BF16 control
          above concurrency 8, so "INT2 degrades at bs>=16" and "this model
          degrades at bs>=16" are not yet distinguishable
  noovl   conc 16, --disable-overlap-schedule. The flush apply phase writes
          quant KV and req_to_token on the schedule stream while the previous
          forward may still be running; the ordering rests on a stashed
          forward-done event that only exists in the overlap path
  norad   conc 16, --disable-radix-cache. Mixed KV x radix cache has already
          cost 22 pp on Qwen3-8B BFCL once
  mbs16   conc 16, --cuda-graph-max-bs 16, so 16 is the LARGEST captured size
          instead of a middle one. All graphs share one memory pool and are
          captured largest-first; this flips whether graph 16 is the first
          capture or the third without changing which graph replays

Usage: python3 gen_manifests10.py [> arms.yaml]
"""
import sys

EXCLUDE = [
    "research-common-h100-117", "research-common-h100-073",
    "research-common-h100-014", "research-common-h100-064",
    "research-common-h100-092", "research-common-h100-099",
    "research-common-h100-105", "research-common-h100-122",
    "research-common-h100-071", "research-common-h100-057",
    "research-common-h100-041", "research-common-h100-080",
    "research-common-h100-048", "research-common-h100-055",
    "research-common-h100-096", "research-common-h100-116",
    "research-common-h100-046", "research-common-h100-090",
]

HEAD = """---
apiVersion: batch/v1
kind: Job
metadata:
  name: {name}
  namespace: charlie
  labels: {{app: zz-m27-graphbs, family: {family}}}
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 604800
  template:
    metadata:
      labels: {{app: zz-m27-graphbs, family: {family}}}
    spec:
      restartPolicy: Never
      runtimeClassName: nvidia
      hostIPC: true
      priorityClassName: normal
      schedulerName: volcano
      tolerations:
      - {{key: node-group, operator: Equal, value: nccl, effect: NoSchedule}}
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: kubernetes.io/hostname
                operator: NotIn
                values:
{excludes}
      containers:
      - name: arm
        image: nvcr.io/nvidia/pytorch:24.12-py3
        imagePullPolicy: IfNotPresent
        securityContext: {{privileged: true}}
        command: ["/bin/bash", "-lc"]
        args: ["base64 -d /shared/zz-m27-recheck/inv10/run_arm10.b64 > /tmp/run_arm10.sh && bash /tmp/run_arm10.sh"]
        env:
{env}
        resources:
          limits: {{nvidia.com/gpu: "4"}}
          requests: {{nvidia.com/gpu: "4"}}
        volumeMounts:
        - {{name: shared, mountPath: /shared}}
        - {{name: home-charlie, mountPath: /home/charlie}}
        - {{name: scratch, mountPath: /tmp}}
      volumes:
      - {{name: shared, persistentVolumeClaim: {{claimName: shared-data}}}}
      - {{name: home-charlie, persistentVolumeClaim: {{claimName: home-charlie-rwx}}}}
      - {{name: scratch, emptyDir: {{}}}}
"""


def job(name, family, env):
    excludes = "\n".join(
        f"                - {h}.cloud.together.ai" for h in EXCLUDE
    )
    envs = "\n".join(
        f'        - {{name: {k}, value: "{v}"}}' for k, v in env.items()
    )
    return HEAD.format(name=name, family=family, excludes=excludes, env=envs)


def base(arm, port):
    return {
        "ARM": arm,
        "PORT": str(port),
        "TREE": "/shared/zz-m27-recheck/tree_post",
        "TP_SIZE": "4",
        "MEM_FRAC": "0.85",
        "MAX_RUNNING": "32",
        "CUDA_GRAPH_MAX_BS": "32",
        "MAX_NEW_TOKENS": "32768",
        "DISABLE_RADIX": "0",
        "HP_PREFIX": "64",
        "HP_RECENT": "256",
        "LLOYD_MAX": "0",
        "NUM_EXAMPLES": "48",
    }


ARMS = [
    # (suffix, family, KV, conc, overrides)
    ("base",  "control", "int2", 16, {}),
    ("c8",    "control", "int2", 8,  {}),
    ("bf16",  "dtype",   "bf16", 16, {}),
    ("noovl", "sched",   "int2", 16, {"EXTRA_ARGS": "--disable-overlap-schedule"}),
    ("norad", "radix",   "int2", 16, {"DISABLE_RADIX": "1"}),
    ("mbs16", "capture", "int2", 16, {"CUDA_GRAPH_MAX_BS": "16"}),
]


def main():
    out, port = [], 32400
    for suffix, family, kv, conc, over in ARMS:
        env = base(f"g10_{suffix}", port)
        env["KV"] = kv
        env["NUM_WORKERS"] = str(conc)
        env.update(over)
        out.append(job(f"zz-m27-g10-{suffix.replace('_', '-')}", family, env))
        port += 4
    sys.stdout.write("".join(out))


if __name__ == "__main__":
    main()
