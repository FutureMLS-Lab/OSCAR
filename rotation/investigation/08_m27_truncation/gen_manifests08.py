#!/usr/bin/env python3
"""charlie-ns Job manifests for the MiniMax-M2.7 INT2 truncation investigation.

Three families, each answering one hypothesis about why INT2 truncates far more
than BF16 (measured: INT2 caps 66-81/198 vs BF16 20-23/198 at a 32K budget).

* ``budget``  -- the same INT2 and BF16 recipes at the model's own 95K budget.
                 The 32K arms established that essentially every unanswered
                 response is a ``finish_reason == "length"`` truncation, so the
                 budget is the first thing that has to be varied, and it has to
                 be varied in BOTH arms: a 95K INT2 number against a 32K BF16
                 number is not a pairing. This family is the one that decides
                 how much of the -21.05 pp is a budget artifact.
* ``window``  -- HP-recent BF16 window sweep at the 32K budget. The candidate
                 fix. 06 pinned 256; GLM-5.2's truncation gap and Gemma-4's
                 long-generation gap both moved on this knob.
* ``bracket`` -- rotation-quality and codebook controls at 32K. A data-free
                 Hadamard rotation bounds from below how much the calibrated
                 rotation is contributing: if Hadamard ties the calibrated file,
                 the calibration is not what is costing 21 points. Lloyd-Max is
                 the other inherited default that was never validated here.

Usage: python3 gen_manifests08.py budget|window|bracket > out.yaml
"""
import sys

# research-common-h100-117 corrupts INT2 sglang outputs (known bad node); 073
# carries nvidia.com/cuda-error=driver-broken; 046/055/080/090 advertise only
# 7 GPUs so a 4-GPU pod can land badly interleaved. Keep 06's list so arms from
# both investigations draw from the same node population.
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
  labels: {{app: zz-m27-trunc, family: {family}}}
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 604800
  template:
    metadata:
      labels: {{app: zz-m27-trunc, family: {family}}}
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
        args: ["bash /shared/zz-m27-recheck/inv08/run_arm08.sh"]
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


def job(name, family, env):
    e = "\n".join(f'        - {{name: {k}, value: "{v}"}}' for k, v in env.items())
    ex = "\n".join(f"                - {h}.cloud.together.ai" for h in EXCLUDE)
    return HEAD.format(name=name, family=family, env=e, excludes=ex)


def base(arm, port):
    return {
        "ARM": arm,
        "TREE": "/shared/zz-m27-recheck/tree_post",
        "HOME": "/tmp/h",
        "HF_HOME": "/shared/huggingface",
        "PYTHONUNBUFFERED": "1",
        "PORT": str(port),
        "TP_SIZE": "4",          # M2.7's published TP; identical in every arm
        "MEM_FRAC": "0.85",
        "CUDA_GRAPH_MAX_BS": "32",
        "MAX_RUNNING": "32",
        "DISABLE_RADIX": "0",
        "MIXED_KV_AUDIT": "0",
    }


# Client concurrency is deliberately not a captured CUDA-graph batch size
# (capture list at --max-running-requests 32 is [1,2,4,8,12,16,24,32]), matching
# 06 exactly. BF16 gets 7 rather than 15 because its KV pool is ~8x smaller per
# token; that asymmetry is inherited from 06 so the two investigations pair.
CONC = {"int2": "15", "bf16": "7"}


def budget():
    """95K-budget arms: 3 BF16 seeds + 2 more INT2 seeds (on_s1_95k exists)."""
    out, port = [], 32100
    for s in (1, 2, 3):
        out.append(job(f"zz-m27-t-bf16-95k-s{s}", "budget", {
            **base(f"t_bf16_95k_s{s}", port), "KV": "bf16",
            "NUM_WORKERS": CONC["bf16"], "MAX_NEW_TOKENS": "95000"}))
        port += 2
    for s in (2, 3):
        out.append(job(f"zz-m27-t-int2-95k-s{s}", "budget", {
            **base(f"t_int2_95k_s{s}", port), "KV": "int2",
            "NUM_WORKERS": CONC["int2"], "MAX_NEW_TOKENS": "95000",
            "HP_PREFIX": "64", "HP_RECENT": "256"}))
        port += 2
    return out


def window():
    """HP-recent sweep at 32K. 1024 gets 3 seeds; 2048/512 scout at 1 seed.

    Cost is honest and must be reported with the score: the BF16 window is not
    free, and its amortized bits/element depends on the *actual* generation
    length, not the budget. At the measured INT2 median (~15K tokens) a 1024
    recent window is ~7% of the sequence in BF16.
    """
    out, port = [], 32120
    for recent, seeds in ((512, (1,)), (1024, (1, 2, 3)), (2048, (1,))):
        for s in seeds:
            out.append(job(f"zz-m27-t-w{recent}-s{s}", "window", {
                **base(f"t_w{recent}_s{s}", port), "KV": "int2",
                "NUM_WORKERS": CONC["int2"], "MAX_NEW_TOKENS": "32768",
                "HP_PREFIX": "64", "HP_RECENT": str(recent)}))
            port += 2
    return out


def bracket():
    """Rotation-quality and codebook brackets at 32K, windows pinned at 06's."""
    out, port = [], 32160
    for s in (1, 2):
        out.append(job(f"zz-m27-t-had-s{s}", "bracket", {
            **base(f"t_had_s{s}", port), "KV": "int2",
            "NUM_WORKERS": CONC["int2"], "MAX_NEW_TOKENS": "32768",
            "HP_PREFIX": "64", "HP_RECENT": "256",
            # Data-free Hadamard, generated by make_hadamard.py into the same
            # rotations dir. If this ties the calibrated qqt/sst file, the
            # calibration cannot be what costs 21 points.
            "K_ROT_FILENAME": "k_rotation_hadamard.pt",
            "V_ROT_FILENAME": "v_rotation_hadamard.pt"}))
        port += 2
    out.append(job("zz-m27-t-lm-s1", "bracket", {
        **base("t_lm_s1", port), "KV": "int2",
        "NUM_WORKERS": CONC["int2"], "MAX_NEW_TOKENS": "32768",
        "HP_PREFIX": "64", "HP_RECENT": "256", "LLOYD_MAX": "1"}))
    return out


def sink():
    """HP-prefix (sink) sweep at 32K.

    Direct precedent: on Qwen3-8B a 1-bit K arena failed GPQA at HP_PREFIX=64
    and was fixed by HP_PREFIX=512, because 64 BF16 tokens do not cover the
    *question*. A GPQA prompt is ~810 tokens here, so at P=64 the model
    re-reads a 2-bit copy of its own question on every one of ~15K decode
    steps. P=1024 puts the whole question in BF16.

    The default prefix-pool formula (req_slots * P * 16) would reserve ~32 GB
    at P=1024, so the pool is pinned explicitly instead: 65536 slots is 64
    request-equivalents of headroom over the 15 concurrent requests.
    """
    out, port = [], 32180
    for s in (1, 2, 3):
        out.append(job(f"zz-m27-t-p1024-s{s}", "sink", {
            **base(f"t_p1024_s{s}", port), "KV": "int2",
            "NUM_WORKERS": CONC["int2"], "MAX_NEW_TOKENS": "32768",
            "HP_PREFIX": "1024", "HP_RECENT": "256",
            "HP_PREFIX_POOL_TOKENS": "65536"}))
        port += 2
    return out


def group():
    """Quant group size 64 instead of the inherited 128, at 32K.

    M2.7 uses partial RoPE (rotary_dim=64 of head_dim=128), so the two halves
    of every K head have different statistics before the rotation mixes them.
    group_size=128 gives one min/max scale for the whole head; 64 gives the two
    halves their own. Costs bits -- 2.5 -> 3.0 bpe on the quant tier -- so the
    score has to beat the window arms by enough to justify it.
    """
    out, port = [], 32190
    for s in (1, 2):
        out.append(job(f"zz-m27-t-g64-s{s}", "group", {
            **base(f"t_g64_s{s}", port), "KV": "int2",
            "NUM_WORKERS": CONC["int2"], "MAX_NEW_TOKENS": "32768",
            "HP_PREFIX": "64", "HP_RECENT": "256", "GROUP_SIZE": "64"}))
        port += 2
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "window"
    print("".join({"budget": budget, "window": window, "bracket": bracket,
                   "sink": sink, "group": group}[which]()))
