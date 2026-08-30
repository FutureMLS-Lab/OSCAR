#!/usr/bin/env python3
"""charlie-ns Job manifests for the MiniMax-M2.7 pre-merge regression check.

The head-tiling defect (`_safe_block_h`, commit 20a8b73452) only fired when
``kv_group_num % BLOCK_H != 0``, and the batch heuristic only dropped BLOCK_H to
4 at ``batch >= 16``. M2.7 is 48Q/8KV, so at TP=4 each rank sees 12 q heads /
2 kv heads -> kv_group_num = 6, and 6 % 4 != 0. Investigation 08/10 measured the
resulting cliff end-to-end:

    conc  8 -> bs  8   graph 100%   score 80.8   reptail  0
    conc 12 -> bs 12   graph 100%   score 78.8   reptail  0
    conc 15 -> bs 16   graph 100%   score 51.0   reptail 26
    conc 16 -> bs 16   graph 100%   score 53.5   reptail 27

So the two arms below are the same run twice, differing only in client
concurrency, which is the variable that selects the decode batch size and
therefore whether the defect could fire at all:

    c16   concurrency 16 -> steady-state decode bs ~16, the broken regime
    c8    concurrency  8 -> decode bs  <16, the regime that was always fine

Three things this generator refuses to get wrong, each of which has already
invalidated a run here:

* ``CUDA_GRAPH_MAX_BS = 32``, never 1. ``CudaGraphRunner.can_run`` tests
  ``cuda_graph_bs <= self.max_bs``, so max_bs=1 means batch >= 2 cannot replay
  any graph and silently runs eager. M2.7's published config was pinned to 1,
  i.e. it had zero graph exposure (0 padded replays in 64547 decode steps), so
  a max_bs=1 arm proves nothing about graphs.
* ``MAX_RUNNING = 32`` decoupled from concurrency.
  ``get_batch_sizes_to_capture`` truncates the captured set to
  max-running-requests *and appends it*, so MAX_RUNNING=16 would make 16 a
  captured size and blind the c16 arm exactly where it needs to see.
* ``TREE`` is passed explicitly and asserted. run_arm11.sh defaults TREE to
  /shared/zz-m27-recheck/tree_post, which is at 4b06c05 -- BEFORE the fix.

Usage: python3 gen_manifests11.py [> arms11.yaml]
"""
import sys

# 117 corrupts INT2 output and 073 is driver-broken: both are permanent
# exclusions, not scheduling hints. The rest were busy or 7-GPU nodes during
# investigation 10 and are kept because the list is known to schedule.
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

TREE = "/shared/zz-premerge/tree_m27"
EXPECT_SHA = "afdcad68b771d312b0eae3d6edde5c08a3881f87"
ROOT = "/shared/zz-premerge"
RUNNER_B64 = "/shared/zz-premerge/run_arm11.b64"

HEAD = """---
apiVersion: batch/v1
kind: Job
metadata:
  name: {name}
  namespace: charlie
  labels: {{app: zz-pm-m27, family: {family}}}
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 604800
  template:
    metadata:
      labels: {{app: zz-pm-m27, family: {family}}}
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
        args: ["base64 -d {runner} > /tmp/run_arm11.sh && bash /tmp/run_arm11.sh"]
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
    return HEAD.format(name=name, family=family, excludes=excludes, env=envs,
                       runner=RUNNER_B64)


def base(arm, port):
    return {
        "ARM": arm,
        "PORT": str(port),
        "ROOT": ROOT,
        "TREE": TREE,
        "EXPECT_SHA": EXPECT_SHA,
        "ISOLATE_TREE": "1",
        # Rotations live with investigation 06/10; only the *code* tree needed
        # replacing, and re-copying 12 MB of .pt would only add a way to get a
        # different rotation than every prior M2.7 arm used.
        "ROT_DIR_OVERRIDE": "/shared/zz-m27-recheck/rotations",
        "K_ROT_FILENAME": "k_rotation_qqt_r_h_pbr.pt",
        "V_ROT_FILENAME": "v_rotation_sst_r_h_pbr.pt",
        "KV": "int2",
        "TP_SIZE": "4",
        "MEM_FRAC": "0.85",
        "MAX_RUNNING": "32",          # -> captured [1,2,4,8,12,16,24,32]
        "CUDA_GRAPH_MAX_BS": "32",    # NEVER 1: 1 turns graphs off above bs 1
        "MAX_NEW_TOKENS": "8192",
        "NUM_EXAMPLES": "24",
        "DISABLE_RADIX": "0",         # radix ON: we are verifying it works
        "HP_PREFIX": "64",
        "HP_RECENT": "256",
        "LLOYD_MAX": "0",             # uniform
        "ABSORB_V": "0",
    }


ARMS = [
    # (suffix, family, client concurrency)
    ("c16", "batch16", 16),   # the previously-broken regime
    ("c8",  "batch8",  8),    # sanity contrast, always fine
]


def main():
    out, port = [], 32700
    for suffix, family, conc in ARMS:
        env = base(f"pm_m27_{suffix}", port)
        env["NUM_WORKERS"] = str(conc)
        out.append(job(f"zz-pm-m27-{suffix}", family, env))
        port += 4
    sys.stdout.write("".join(out))


if __name__ == "__main__":
    main()
