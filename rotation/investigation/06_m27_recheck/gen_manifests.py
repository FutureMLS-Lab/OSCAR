#!/usr/bin/env python3
"""Emit the charlie-ns Job manifests for the MiniMax-M2.7 re-check.

Two families:

* probes  -- short, auditor ON. Each is only interpretable if it actually
             exercises padded replays, so concurrency is chosen to be OUTSIDE
             the server's captured CUDA-graph batch sizes, and every probe has
             a pre-fix twin as a positive control.
* scored  -- full GPQA-198, auditor OFF (it syncs the decode hot path).

Usage: python3 gen_manifests.py probes|scored > out.yaml
"""
import sys

# research-common-h100-117 corrupts INT2 sglang outputs (known bad node); 073
# carries nvidia.com/cuda-error=driver-broken; the rest are the nodes the
# concurrent Qwen/Gemma agent excluded, so treat them as suspect too.
EXCLUDE = [
    "research-common-h100-117", "research-common-h100-073",
    "research-common-h100-014", "research-common-h100-064",
    "research-common-h100-092", "research-common-h100-099",
    "research-common-h100-105", "research-common-h100-122",
    "research-common-h100-071", "research-common-h100-057",
    "research-common-h100-041", "research-common-h100-080",
    "research-common-h100-048", "research-common-h100-055",
    "research-common-h100-096", "research-common-h100-116",
]

HEAD = """---
apiVersion: batch/v1
kind: Job
metadata:
  name: {name}
  namespace: charlie
  labels: {{app: zz-m27-recheck, family: {family}}}
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 604800
  template:
    metadata:
      labels: {{app: zz-m27-recheck, family: {family}}}
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
        args: ["bash /shared/zz-m27-recheck/{tree}/rotation/investigation/06_m27_recheck/run_arm.sh"]
        env:
{env}
        resources:
          limits: {{nvidia.com/gpu: "4"}}
          requests: {{cpu: "24", memory: 320Gi, nvidia.com/gpu: "4"}}
        volumeMounts:
        - {{name: shared, mountPath: /shared}}
        - {{name: home, mountPath: /home/charlie}}
      volumes:
      - {{name: shared, persistentVolumeClaim: {{claimName: shared-data}}}}
      - {{name: home, persistentVolumeClaim: {{claimName: home-charlie-rwx}}}}
"""


def job(name, family, tree, env):
    e = "\n".join(f'        - {{name: {k}, value: "{v}"}}' for k, v in env.items())
    ex = "\n".join(f"                - {h}.cloud.together.ai" for h in EXCLUDE)
    return HEAD.format(name=name, family=family, tree=tree, env=e, excludes=ex)


def base(arm, port):
    return {
        "ARM": arm,
        "HOME": "/tmp/h",
        "HF_HOME": "/shared/huggingface",
        "PYTHONUNBUFFERED": "1",
        "PORT": port,
        "TP_SIZE": "4",
        "MEM_FRAC": "0.85",
    }


def probes():
    out = []
    # Pair 1 -- the published server shape. cuda_graph_max_bs=1 captures ONLY
    # bs=1, so any concurrency above 1 makes EVERY decode step a padded replay:
    # maximum exposure, no ambiguity about whether the probe was exercised.
    for tag, tree in (("pre", "tree_pre"), ("post", "tree_post_audit")):
        out.append(job(
            f"zz-m27-p1-{tag}-mbs1", "probe", tree,
            {**base(f"p1_{tag}_mbs1_c4", "31941" if tag == "pre" else "31943"),
             "TREE": f"/shared/zz-m27-recheck/{tree}",
             "KV": "int2", "DISABLE_RADIX": "0",
             "CUDA_GRAPH_MAX_BS": "1", "MAX_RUNNING": "4", "NUM_WORKERS": "4",
             "MAX_NEW_TOKENS": "4096", "NUM_EXAMPLES": "8",
             "MIXED_KV_AUDIT": "1", "MIXED_KV_AUDIT_EVERY": "5",
             "SGLANG_MIXED_KV_AUDIT_PAGE0": "1"},
        ))
    # Pair 2 -- the config every other model uses. 7 is absent from the
    # capture list [1,2,4,8,12,16,24,32], so decode sits at a non-captured size
    # while all 7 requests are alive.
    for tag, tree in (("pre", "tree_pre"), ("post", "tree_post_audit")):
        out.append(job(
            f"zz-m27-p2-{tag}-mbs32", "probe", tree,
            {**base(f"p2_{tag}_mbs32_c7", "31945" if tag == "pre" else "31947"),
             "TREE": f"/shared/zz-m27-recheck/{tree}",
             "KV": "int2", "DISABLE_RADIX": "0",
             "CUDA_GRAPH_MAX_BS": "32", "MAX_RUNNING": "7", "NUM_WORKERS": "7",
             "MAX_NEW_TOKENS": "8192", "NUM_EXAMPLES": "14",
             "MIXED_KV_AUDIT": "1", "MIXED_KV_AUDIT_EVERY": "5",
             "SGLANG_MIXED_KV_AUDIT_PAGE0": "1"},
        ))
    return out


def scored():
    out = []
    port = 31950
    # INT2 cache-ON (the arm the published row claims), INT2 cache-OFF control,
    # and the same-session BF16 paired arm -- 3 seeds each. Seeds differ only by
    # sampling entropy: simple_evals has no seed flag, temp=1.0 supplies it.
    for kv, radix, tag in (("int2", "0", "on"), ("int2", "1", "off"), ("bf16", "0", "bf16")):
        for s in (1, 2, 3):
            # BF16 KV is ~8x larger per token; at 62 layers x 8 KV heads the
            # pool holds ~180K tokens, so 16 concurrent 95K generations would
            # thrash. Halve concurrency for BF16 and report it.
            conc = "8" if kv == "bf16" else "16"
            out.append(job(
                f"zz-m27-{tag}-s{s}", "scored", "tree_post",
                {**base(f"{tag}_s{s}", str(port)),
                 "TREE": "/shared/zz-m27-recheck/tree_post",
                 "KV": kv, "DISABLE_RADIX": radix,
                 "CUDA_GRAPH_MAX_BS": "32",
                 "MAX_RUNNING": conc, "NUM_WORKERS": conc,
                 "MAX_NEW_TOKENS": "95000",
                 "MIXED_KV_AUDIT": "0"},
            ))
            port += 2
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "probes"
    print("".join(probes() if which == "probes" else scored()))
