#!/usr/bin/env python3
"""charlie-ns Job manifests for the MiniMax-M2.7 INT2 truncation investigation.

Six families, each answering one hypothesis about why INT2 truncates far more
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

* ``bigwin``  -- 3-seed replication of this model's own `bigwindow` ablation
                 (P=256 R=1024), the only prior config that moved GPQA.
* ``sink``    -- HP-prefix sweep. At P=64 only 8% of an ~810-token GPQA
                 question is BF16; Qwen3-8B needed P=512 for the same reason.
* ``group``   -- quant group 64 vs the inherited 128, motivated by M2.7's
                 partial RoPE (rotary_dim=64 of head_dim=128).

Usage: python3 gen_manifests08.py budget|window|bracket|bigwin|sink|group
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


def padding():
    """Isolate PADDED replay from graph replay in general.

    The max-bs 1 vs 32 control (family ``graphbs``) showed eager decode recovers
    INT2 by ~18 correct answers out of ~69 paired questions, 3/3 seeds. But it
    changes two things at once: graphs go from replaying-every-step to never
    replaying, AND padding goes from 100% to 0%. Those are different bugs to fix.

    This pair keeps CUDA graphs fully active in both arms and flips only whether
    the decode batch size is one the graph was captured at. With
    --max-running-requests 32 the capture list is [1,2,4,8,12,16,24,32]:

      concurrency 8 -> bs 8 IS captured  -> graph replays, NOTHING padded
      concurrency 7 -> bs 7 is NOT captured -> graph replays, padded to 8

    If 8 looks like the eager arm and 7 looks like the broken arm, the defect is
    specifically in padded replay and the search space is the padding fill-in.
    If both look broken, padding is innocent and the problem is on the graph
    path generally -- a much wider search. Either answer materially narrows the
    work for whoever fixes it, which is why it is worth 6 arms.

    Concurrency 7 vs 8 is the smallest change that flips padding status, so the
    confound from batching differences is as small as it can be made.
    """
    out, port = [], 32300
    for conc in (7, 8):
        for s in (1, 2, 3):
            out.append(job(f"zz-m27-t-c{conc}-s{s}", "padding", {
                **base(f"t_c{conc}_s{s}", port), "KV": "int2",
                "NUM_WORKERS": str(conc), "MAX_NEW_TOKENS": "32768",
                "HP_PREFIX": "64", "HP_RECENT": "256"}))
            port += 2
    return out


def blockrot():
    """End-to-end check of the partial-RoPE hypothesis: block-diagonal K rotation.

    M2.7 is the only model here with partial RoPE (rotary_dim=64 of head_dim=128),
    so one 128x128 rotation mixes the position-dependent dims 0-63 with the
    static dims 64-127. The MLA precedent (GLM-5.2 keeps k_pe in BF16 and
    quantizes only c_kv) says not to put the positional component through the
    2-bit path. `fit_block_rotation.py` fits the same qqt objective and r_h_pbr
    composition INDEPENDENTLY per 64-dim block, so the rotation cannot move
    energy across the boundary. Bit budget, quant group and windows are
    unchanged, so this isolates the block structure.

    The offline diagnostic (rope_subspace_error.py) already predicts this LOSES,
    and says why the premise fails:
      rms_rope 1.566 vs rms_nope 1.579   -- the halves have the same magnitude,
                                            so no shared scale is being swamped
      relerr_rope 0.505 vs relerr_nope 0.488 -- error is spread evenly, not
                                            concentrated in the positional half
      logit error 0.319 full vs 0.361 block  -- block-diagonal is 13% WORSE
    A full 128-dim Hadamard spreads outliers over 128 coordinates; two 64-dim
    blocks spread them over 64 each, so constraining the rotation reduces the
    incoherence processing OSCAR depends on. That also explains why quant group
    64 measured worse (55.31 vs 57.91) -- same mechanism, same direction.

    Run anyway at 3 seeds, because the offline metric is a logit-error proxy and
    the claim under test is an end-to-end score. Only the 64/256 windows are
    run: comparing against the 57.91 baseline is the clean single-variable test,
    and spending six more arms to also pair it against 256/1024 is not
    justifiable for a hypothesis the diagnostic already predicts will lose.
    """
    out, port = [], 32270
    for s in (1, 2, 3):
        out.append(job(f"zz-m27-t-blk-s{s}", "blockrot", {
            **base(f"t_blk_s{s}", port), "KV": "int2",
            "NUM_WORKERS": CONC["int2"], "MAX_NEW_TOKENS": "32768",
            "HP_PREFIX": "64", "HP_RECENT": "256",
            "K_ROT_FILENAME": "k_rotation_qqt_block64.pt"}))
        port += 2
    return out


def graphbs():
    """THE control that separates "2-bit quality" from "graph-replay defect".

    Every scored INT2 arm in this investigation ran --cuda-graph-max-bs 32,
    where essentially every decode step is a padded graph replay. Every
    historical M2.7 number that looked good -- SWE-bench-verified 70.8,
    LiveCodeBench v6 68.4, AIME25 90.0 -- ran --cuda-graph-max-bs 1, where
    CudaGraphRunner.can_run gates on cuda_graph_bs <= max_bs so a batch of 2+
    cannot replay at all and runs EAGER: zero padded replays. So "INT2 is
    lossy on long generations" and "something is still wrong on the
    graph-replay path" are perfectly confounded across everything measured so
    far, and the allocator check cannot separate them -- page 0 being
    unallocated rules out the one known padding bug, not a different one, and
    the auditor's damage counter is known to read zero even on broken code.

    SWE-bench and LCB are long-generation long-context tasks. If INT2 fell into
    non-terminating loops past ~8.6K words as an intrinsic property, those
    numbers could not exist. That is the contradiction this arm resolves.

    Single variable. Both sides are run fresh here rather than reusing the
    earlier 64/256 arms, so the comparison is same-session end to end.

    Two things held deliberately constant, because varying them is how the
    fictitious published +2.0 was produced:
    * NUM_WORKERS is IDENTICAL (15) in both arms. Client concurrency and
      cuda-graph-max-bs are separate knobs; the old wrapper forced concurrency
      to 1 whenever max-bs was 1, which compared INT2 at concurrency 1 against
      BF16 at concurrency 32.
    * MAX_RUNNING stays 32 in both, so req_to_token_pool.size -- and therefore
      the HP arena geometry and max_req_slots -- are identical. It also keeps
      get_batch_sizes_to_capture from appending a different value to the
      captured list in each arm.

    Expect the max-bs 1 arm to be slow: at concurrency 15 every decode step
    runs eager. That is the point of the arm, not a fault in it.
    """
    out, port = [], 32240
    for mbs in (1, 32):
        for s in (1, 2, 3):
            env = {**base(f"t_mbs{mbs}_s{s}", port), "KV": "int2",
                   "NUM_WORKERS": CONC["int2"], "MAX_NEW_TOKENS": "32768",
                   "HP_PREFIX": "64", "HP_RECENT": "256"}
            env["CUDA_GRAPH_MAX_BS"] = str(mbs)
            out.append(job(f"zz-m27-t-mbs{mbs}-s{s}", "graphbs", env))
            port += 2
    return out


def window2():
    """Second window pass, driven by the first pass's result.

    Measured at 32K with the calibrated rotation and P=64:
        R=256   57.91  (n=3: 52.53 / 61.11 / 60.10)
        R=512   57.58  (n=1)
        R=1024  59.93  (n=3: 60.61 / 58.08 / 61.11)
        R=2048  65.66  (n=1)

    R=1024 is +2.0 on three seeds, which is inside this model's seed spread and
    cannot be called a win. R=2048 is +7.8 but on one seed, and one seed is
    exactly how the pre-existing `bigwindow` ablation overstated itself. So
    R=2048 gets its two missing seeds, and R=4096 is added to say whether the
    curve keeps climbing or 2048 was a lucky draw -- a monotone dose-response
    over four window sizes is far harder to explain as noise than any single
    arm, which is the argument this sweep has to be able to make.
    """
    out, port = [], 32220
    for recent, seeds in ((2048, (2, 3)), (4096, (1, 2))):
        for s in seeds:
            out.append(job(f"zz-m27-t-w{recent}-s{s}", "window2", {
                **base(f"t_w{recent}_s{s}", port), "KV": "int2",
                "NUM_WORKERS": CONC["int2"], "MAX_NEW_TOKENS": "32768",
                "HP_PREFIX": "64", "HP_RECENT": str(recent)}))
            port += 2
    return out


def bigwin():
    """Replicate the pre-existing `bigwindow` ablation properly: P=256, R=1024.

    That configuration is the only thing in this model's own ablation history
    that moved GPQA materially -- 66.16 against the 55.56 baseline, same clip,
    same 32K budget, same cuda_graph_max_bs. But it was ONE seed, and this
    model's seed spread is 8.6 pp (52.53 / 61.11 / 60.10), so +10.6 pp single
    seed is barely outside noise and cannot be trusted as reported.

    It also moved both windows at once, so it cannot say which one mattered.
    The `window` and `sink` families here vary them one at a time; this family
    reproduces the combination at 3 seeds so the comparison against the paired
    BF16 arms is like-for-like.
    """
    out, port = [], 32200
    for s in (1, 2, 3):
        out.append(job(f"zz-m27-t-bw-s{s}", "bigwin", {
            **base(f"t_bw_s{s}", port), "KV": "int2",
            "NUM_WORKERS": CONC["int2"], "MAX_NEW_TOKENS": "32768",
            "HP_PREFIX": "256", "HP_RECENT": "1024",
            "HP_PREFIX_POOL_TOKENS": "32768"}))
        port += 2
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
                   "sink": sink, "group": group, "bigwin": bigwin, "window2": window2, "graphbs": graphbs, "blockrot": blockrot, "padding": padding}[which]()))
