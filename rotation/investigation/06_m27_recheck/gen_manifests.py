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
    # Pair 1 -- the published server shape, and it is NOT an exposure test.
    # CudaGraphRunner.can_run gates on `cuda_graph_bs <= self.max_bs`, so with
    # cuda_graph_max_bs=1 a batch of 2+ cannot replay any graph at all: it runs
    # EAGER, and padding only exists on replay. So this pair measures the
    # opposite of what one might assume -- zero padded replays at any
    # concurrency -- and doubles as the control that isolates the mechanism:
    # tree_pre here is broken code at concurrency 4 with graphs effectively off,
    # so if it comes back clean, the damage needs padding, not concurrency.
    # (It also explains the published 115 vs 8 tok/s cliff: that was eager.)
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
    # Pair 2 -- maximal exposure with CUDA graphs actually replaying.
    #
    # MAX_RUNNING and NUM_WORKERS must be DECOUPLED. get_batch_sizes_to_capture
    # clamps the capture list to req_to_token_pool.size (= max-running-requests)
    # and *appends* that value, so --max-running-requests 7 produced
    # `Capture cuda graph bs [1, 2, 4, 7]` -- measured, first attempt -- which
    # makes bs=7 a captured size and the probe blind for the whole steady-state
    # phase. Server pool 32 -> capture [1,2,4,8,12,16,24,32]; client pinned at
    # 7 -> every decode step is bs=7, not captured, padded up to 8. 100 %
    # padded replays with graphs live.
    for tag, tree in (("pre", "tree_pre"), ("post", "tree_post_audit")):
        out.append(job(
            f"zz-m27-p2-{tag}-mbs32", "probe", tree,
            {**base(f"p2_{tag}_mbs32_c7", "31945" if tag == "pre" else "31947"),
             "TREE": f"/shared/zz-m27-recheck/{tree}",
             "KV": "int2", "DISABLE_RADIX": "0",
             "CUDA_GRAPH_MAX_BS": "32", "MAX_RUNNING": "32", "NUM_WORKERS": "7",
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
            # Client concurrency is deliberately NOT a captured CUDA-graph
            # batch size. With --max-running-requests 32 the capture list is
            # [1,2,4,8,12,16,24,32]; driving 15 (INT2) or 7 (BF16) concurrent
            # requests pins decode at a non-captured size, so essentially every
            # decode step is a padded replay -- the arm can actually expose bug
            # 1 rather than sitting on a captured size and proving nothing.
            # BF16 gets 7 because its KV pool is only ~193K tokens (measured):
            # 8x larger per token than INT2's ~1.29M.
            conc = "7" if kv == "bf16" else "15"
            out.append(job(
                f"zz-m27-{tag}-s{s}", "scored", "tree_post",
                {**base(f"{tag}_s{s}", str(port)),
                 "TREE": "/shared/zz-m27-recheck/tree_post",
                 "KV": kv, "DISABLE_RADIX": radix,
                 "CUDA_GRAPH_MAX_BS": "32",
                 "MAX_RUNNING": "32", "NUM_WORKERS": conc,
                 # 32768, not the recipe's 95000. The published GPQA 80.3 and
                 # every historical M2.7 GPQA arm on disk used 32K; 95K is the
                 # budget behind the SWE-bench/LCB numbers. Re-checking the
                 # GPQA row at 95K would not be paired with the number under
                 # test, and costs ~3x the wall clock. Truncation counts are
                 # reported so a reader can see whether the budget binds
                 # (historically ~12% of responses hit the 32K cap), and one
                 # extra arm below repeats INT2 cache-ON at 95K to measure it.
                 "MAX_NEW_TOKENS": "32768",
                 "MIXED_KV_AUDIT": "0"},
            ))
            port += 2
    # One 95K arm so the budget effect is measured rather than assumed.
    out.append(job(
        "zz-m27-on-s1-95k", "scored", "tree_post",
        {**base("on_s1_95k", str(port)),
         "TREE": "/shared/zz-m27-recheck/tree_post",
         "KV": "int2", "DISABLE_RADIX": "0", "CUDA_GRAPH_MAX_BS": "32",
         "MAX_RUNNING": "32", "NUM_WORKERS": "15",
         "MAX_NEW_TOKENS": "95000", "MIXED_KV_AUDIT": "0"},
    ))
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "probes"
    print("".join(probes() if which == "probes" else scored()))
