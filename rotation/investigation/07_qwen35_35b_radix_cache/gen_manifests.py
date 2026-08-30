#!/usr/bin/env python3
"""Emit the K8s Jobs for the Qwen3.5-35B-A3B prefix-cache A/B.

Four arms, one Job each, 4 H100s each:

  on    INT2 OSCAR, prefix cache ON   (--mamba-scheduler-strategy extra_buffer)
  off   INT2 OSCAR, prefix cache OFF  (necessarily no_buffer; see below)
  bf16  same-session BF16 baseline, prefix cache OFF
  audit one short cache-ON arm with the mixed-KV auditor on, at a client
        concurrency deliberately absent from the CUDA-graph capture list

Everything the three scored arms can share is shared: same commit, same
rotations, same seeds, same client concurrency, same --max-running-requests,
same --cuda-graph-max-bs (so the same capture list), same mem fraction, CUDA
graphs ON, overlap scheduler at its default (ON) in all three.

The one asymmetry that cannot be removed: ``extra_buffer`` and
``--disable-radix-cache`` raise ValueError together, so the cache-OFF control
runs ``no_buffer``. It is reported, not hidden.

HOME and every JIT cache point at pod-local /tmp. The /home/charlie PVC has been
100% full, and a flashinfer cache copied from elsewhere is unusable anyway
because build.ninja hard-codes absolute paths. PYTHONUSERBASE still points into
that PVC read-only -- it is where jsonschema lives, which is missing from the
shared oscar conda env and kills the server on import.
"""
import sys

WS = "/shared/CoQuant/q35b35"
ROT = "/shared/CoQuant/radixq35-hybrid/rotations/Qwen3.5-35B-A3B"
INV = f"{WS}/CoQuant/rotation/investigation/07_qwen35_35b_radix_cache"
MODEL = "Qwen/Qwen3.5-35B-A3B"
PREFIX = "zz-q35b35"

# research-common-h100-117 corrupts INT2 sglang output (project note, 2026-05-22).
BAD_NODES = [
    "research-common-h100-014", "research-common-h100-064",
    "research-common-h100-092", "research-common-h100-099",
    "research-common-h100-117", "research-common-h100-105",
]

# Captured CUDA-graph list at --cuda-graph-max-bs 32 is [1,2,4,8,12,16,24,32]
# (verified in the server log). The audit arm's concurrency must not be in it,
# and --max-running-requests must stay at 32 so get_batch_sizes_to_capture does
# not append a smaller req_to_token_pool.size onto the captured list and blind
# the probe.
AUDIT_CONCURRENCY = 14

ARMS = {
    "on": dict(
        script="run_arm.sh", disable_radix="0", lloyd="1",
        extra="--mamba-scheduler-strategy extra_buffer",
        port=36871, dist=46871, seeds="0 1 2", audit="0",
        workers="32", examples=None, max_new="32768",
    ),
    "off": dict(
        script="run_arm.sh", disable_radix="1", lloyd="1",
        extra="", port=36873, dist=46873, seeds="0 1 2", audit="0",
        workers="32", examples=None, max_new="32768",
    ),
    "bf16": dict(
        script="run_bf16_arm.sh", disable_radix="1", lloyd="0",
        extra="", port=36875, dist=46875, seeds="0 1 2", audit="0",
        workers="32", examples=None, max_new="32768",
    ),
    "audit": dict(
        script="run_arm.sh", disable_radix="0", lloyd="1",
        extra="--mamba-scheduler-strategy extra_buffer --disable-overlap-schedule",
        port=36877, dist=46877, seeds="0", audit="1",
        workers=str(AUDIT_CONCURRENCY), examples=str(AUDIT_CONCURRENCY),
        max_new="2048",
    ),
    # Engagement probe. Runs from a SECOND workspace carrying
    # patch_mixin_diag.py, which is not in the branch and must not touch the
    # scored arms (it logs inside match_prefix, on the hot path). It answers a
    # question the score cannot: did the duck-typed probe actually return True
    # on MambaRadixCache, and does the tier cap actually bite? More examples
    # than workers on purpose -- with concurrency == examples every request
    # prefills before any of them commits, so nothing can match and the cap
    # would never fire regardless of whether it works.
    "diag": dict(
        script="run_arm.sh", disable_radix="0", lloyd="1",
        extra="--mamba-scheduler-strategy extra_buffer",
        port=36879, dist=46879, seeds="0", audit="0",
        workers="8", examples="40", max_new="2048", ws=f"{WS}diag",
    ),
}

TEMPLATE = """---
apiVersion: batch/v1
kind: Job
metadata:
  name: {prefix}-{arm}
  namespace: charlie
  labels: {{app: {prefix}, arm: "{arm}"}}
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 604800
  template:
    metadata:
      labels: {{app: {prefix}, arm: "{arm}"}}
    spec:
      restartPolicy: Never
      schedulerName: default-scheduler
      runtimeClassName: nvidia
      hostNetwork: true
      hostIPC: true
      dnsPolicy: ClusterFirstWithHostNet
      priorityClassName: normal
      nodeSelector: {{node-pool: compute}}
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - {{key: app, operator: In, values: [{prefix}]}}
            topologyKey: kubernetes.io/hostname
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - {{key: kubernetes.io/hostname, operator: NotIn, values: [{bad}]}}
      tolerations:
      - {{key: node-group, operator: Equal, value: nccl, effect: NoSchedule}}
      containers:
      - name: eval
        image: nvcr.io/nvidia/pytorch:24.12-py3
        imagePullPolicy: IfNotPresent
        securityContext: {{privileged: true}}
        workingDir: {ws}
        env:
        # Pod-local HOME: the /home/charlie PVC is 100% full and every JIT
        # cache below hangs off HOME.
        - {{name: HOME, value: /tmp/oscarhome}}
        - {{name: XDG_CACHE_HOME, value: /tmp/oscarhome/.cache}}
        - {{name: TORCH_EXTENSIONS_DIR, value: /tmp/oscarhome/.cache/torch_extensions}}
        - {{name: TVM_FFI_CACHE_DIR, value: /tmp/oscarhome/.cache/tvm-ffi}}
        - {{name: SGLANG_CACHE_DIR, value: /tmp/oscarhome/.cache/sglang}}
        - {{name: TRITON_CACHE_DIR, value: /tmp/oscarhome/.cache/triton}}
        - {{name: FLASHINFER_WORKSPACE_BASE, value: /tmp/oscarhome}}
        # Read-only: this is where jsonschema lives (absent from the shared env).
        - {{name: PYTHONUSERBASE, value: /home/charlie/.local}}
        - {{name: HF_HOME, value: /shared/huggingface}}
        - {{name: HF_DATASETS_CACHE, value: /shared/huggingface/datasets}}
        - {{name: NCCL_NET, value: Socket}}
        - {{name: PYTORCH_CUDA_ALLOC_CONF, value: "expandable_segments:True"}}
        - {{name: PYTHONUNBUFFERED, value: "1"}}
        resources: {{limits: {{nvidia.com/gpu: "4"}}, requests: {{nvidia.com/gpu: "4"}}}}
        volumeMounts:
        - {{name: shared, mountPath: /shared}}
        - {{name: home, mountPath: /home/charlie}}
        command:
        - /bin/bash
        - -lc
        - |
          set -uo pipefail
          mkdir -p "$HOME/.cache" && chmod -R a+rwX "$HOME" 2>/dev/null || true
          # Scope to the GPUs the device plugin actually gave this pod. This
          # container is privileged, so nvidia-smi lists all 8 node GPUs and
          # picking indices 0..N-1 from it would run on someone else's card.
          export CUDA_VISIBLE_DEVICES=$(ls /var/run/nvidia-container-devices | paste -sd,)
          NG=$(ls /var/run/nvidia-container-devices | wc -l)
          echo "[pod] node=$(hostname) allocated_gpus=$NG cvd=$CUDA_VISIBLE_DEVICES"
          nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader | grep -F -f <(ls /var/run/nvidia-container-devices) || true
          ARM={arm} \\
          MODEL={model} \\
          ROT_DIR={rot} \\
          RUN_DIR={ws}/runs/{arm} \\
          W={ws}/CoQuant \\
          TP_SIZE=4 GPUS=$CUDA_VISIBLE_DEVICES \\
          PORT={port} DIST_PORT={dist} \\
          DISABLE_RADIX={disable_radix} LLOYD_MAX={lloyd} \\
          GROUP_SIZE=256 ABSORB_V=0 \\
          MEM_FRAC=0.80 MAX_RUNNING=32 CUDA_GRAPH_MAX_BS=32 \\
          NUM_WORKERS={workers} MAX_NEW_TOKENS={max_new} \\
          SEEDS='{seeds}' \\
{extra_lines}          CONDA_BASE=/home/charlie/miniconda3 \\
          bash {inv}/{script}
      volumes:
      - {{name: shared, persistentVolumeClaim: {{claimName: shared-data}}}}
      - {{name: home, persistentVolumeClaim: {{claimName: home-charlie-rwx}}}}
"""


def main():
    want = sys.argv[1:] or list(ARMS)
    out = []
    for arm in want:
        c = ARMS[arm]
        extra = ""
        if c["extra"]:
            extra += f'          EXTRA_SERVER_ARGS="{c["extra"]}" \\\n'
        if c["audit"] != "0":
            extra += f'          MIXED_KV_AUDIT={c["audit"]} \\\n'
            extra += "          MIXED_KV_AUDIT_EVERY=5 \\\n"
        if c["examples"]:
            extra += f'          NUM_EXAMPLES={c["examples"]} \\\n'
        ws = c.get("ws", WS)
        out.append(TEMPLATE.format(
            prefix=PREFIX, arm=arm, ws=ws, rot=ROT, model=MODEL,
            inv=f"{ws}/CoQuant/rotation/investigation/07_qwen35_35b_radix_cache",
            bad=", ".join(f"{n}.cloud.together.ai" for n in BAD_NODES),
            port=c["port"], dist=c["dist"], disable_radix=c["disable_radix"],
            lloyd=c["lloyd"], seeds=c["seeds"], workers=c["workers"],
            max_new=c["max_new"], script=c["script"], extra_lines=extra,
        ))
    sys.stdout.write("".join(out))


if __name__ == "__main__":
    main()
