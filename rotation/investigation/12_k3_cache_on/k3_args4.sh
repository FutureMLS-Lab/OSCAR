#!/bin/bash
# Shared launch-argument construction for the Kimi-K3 KV/cache arms.
#
# Same contract as k3_args.sh (both ranks source THIS file so the two nodes
# cannot drift), extended with the PREFIX-CACHE mode, which is now part of the
# arm name rather than a separate knob:
#
#     int2on  INT2 OSCAR KV, prefix cache ON   (extra_buffer)
#     int2    INT2 OSCAR KV, prefix cache OFF  (no_buffer + --disable-radix-cache)
#     bf16on  BF16 KV,       prefix cache ON
#     bf16    BF16 KV,       prefix cache OFF
#
# Putting the cache mode in the arm name matters because head4.sh keys the
# results file off the arm (results.$ARM.jsonl).  On run gpqa9 a single logical
# run produced one cache-ON and one cache-OFF results file and the two were
# nearly mistaken for one another; a name that encodes the whole configuration
# makes blending two server configs into one score impossible.
#
# Expects in the environment: MEMFRAC MAXREQ PORT MASTER NODERANK MODEL ROT
# Sets: ENVV[] ARGS[] and ARM_CACHE_MODE (on|off) for the caller to assert on.

resolve_model() {
  # The original moonshotai/Kimi-K3 snapshot was deleted from the HF cache PVC
  # on 2026-08-21 (~21:45 UTC).  Prefer it if it ever comes back.
  #
  # Otherwise use /shared/kimi-k3-base: a symlink view of Together's mirror with
  # its one MTP shard removed.  The mirror CANNOT be used raw -- the base runtime
  # RAISES on MTP tensors (kimi_k3.py adapt_checkpoint_name) rather than skipping
  # them, which killed an earlier control four shards into the load.  Dropping
  # that one shard leaves 96, exactly the original repo's shard count, and the
  # same-weights INT2 arm came up at exactly 483592 KV tokens, which is how we
  # know the geometry is identical to the run that produced the n=168 reference.
  local m
  if [ -f /shared/kimi-k3-base/config.json ] &&
     [ "$(ls /shared/kimi-k3-base 2>/dev/null | grep -c 'safetensors$')" -ge 90 ]; then
    m=$(ls -d /hf/hub/models--moonshotai--Kimi-K3/snapshots/*/ 2>/dev/null | head -1)
    if [ -n "$m" ] && [ -f "$m/config.json" ]; then echo "$m"; return 0; fi
    echo /shared/kimi-k3-base; return 0
  fi
  m=$(ls -d /hf/hub/models--moonshotai--Kimi-K3/snapshots/*/ 2>/dev/null | head -1)
  if [ -n "$m" ] && [ -f "$m/config.json" ]; then echo "$m"; return 0; fi
  return 1
}

build_launch() {   # $1 = arm: int2on | int2 | bf16on | bf16
  local arm=$1
  ENVV=(PYTHONUNBUFFERED=1)
  local kvarg
  case "$arm" in
    int2*)
      kvarg=int2
      ENVV+=(SGLANG_ENABLE_MIXED_KV_WINDOWS=1
             SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
             SGLANG_MIXED_KV_PREFIX_TOKENS=64
             SGLANG_MIXED_KV_RECENT_TOKENS=256
             SGLANG_MIXED_KV_HP_DTYPE=bfloat16
             SGLANG_MIXED_KV_SCALE_DTYPE=float32
             SGLANG_OSCAR_ABSORB_V_ROTATION=0
             SGLANG_LLOYD_MAX=0
             SGLANG_OSCAR_K_CLIP_RATIO=0.96
             SGLANG_OSCAR_V_CLIP_RATIO=0.92
             SGLANG_OSCAR_K_ROTATION_PATH=$ROT/k_rotation_qqt_r_h_pbr.pt
             SGLANG_OSCAR_V_ROTATION_PATH=$ROT/v_rotation_sst_r_h_pbr.pt) ;;
    *)
      kvarg=bfloat16 ;;
  esac

  # Prefix cache.  extra_buffer is REQUIRED for cache-ON here: INT2 mixed-KV
  # needs page_size == N_Q == 8 and MambaRadixCache v1 asserts page_size == 1
  # without it.  extra_buffer also raises ValueError together with
  # --disable-radix-cache, so the two modes are genuinely exclusive and cache-OFF
  # must be no_buffer.  This asymmetry cannot be removed on a linear-attention
  # model and is reported rather than hidden.
  local cacheargs=()
  case "$arm" in
    *on)
      ARM_CACHE_MODE=on
      cacheargs=(--mamba-scheduler-strategy extra_buffer) ;;
    *)
      ARM_CACHE_MODE=off
      cacheargs=(--mamba-scheduler-strategy no_buffer --disable-radix-cache) ;;
  esac

  # Every integer 1..MAXREQ is captured, so no decode batch is ever a padded
  # graph replay -- the confound that produced the M2.7 defect.
  local cg=(); for i in $(seq 1 "$MAXREQ"); do cg+=("$i"); done
  ARGS=(--model-path "$MODEL"
        --tensor-parallel-size 8 --pipeline-parallel-size 2 --nnodes 2
        --node-rank "$NODERANK" --dist-init-addr "${MASTER}:41255"
        --kv-cache-dtype "$kvarg" --disable-custom-all-reduce
        --attention-backend triton
        --prefill-attention-backend triton
        --decode-attention-backend triton
        --mem-fraction-static "$MEMFRAC" --moe-runner-backend triton_kernel
        --max-running-requests "$MAXREQ" --cuda-graph-max-bs "$MAXREQ"
        --cuda-graph-bs "${cg[@]}"
        --page-size 8
        "${cacheargs[@]}"
        # --kv-cache-quant-group-size deliberately UNSET: K head dim 192 = 3x64,
        # and every group size dividing both 192 and 128 yields a group count
        # with a factor of 3 that no INT2 write path supports.
        --trust-remote-code --skip-server-warmup --watchdog-timeout 3600
        --host 0.0.0.0 --port "$PORT")
}
