#!/usr/bin/env bash
# Kimi-K3 -- MLA latent (c_kv 512 + k_pe 64), packed 2-bit, only the NoPE half
# is quantised; k_pe stays BF16 throughout.
#
# THE LATENT PATH IS THE POINT. An expanded-MHA fallback stores 96 heads x 128 x
# 2 = 24,576 values per token per layer against the latent's 512 + 64 = 576, so
# it is not MLA any more and the KV cache it is supposed to compress is ~40x
# larger. Measured end to end it is also the worst decode result of any model
# here (1.09x against BF16 at 64K, 70 ms/tok). Do not "fix" this file by
# switching to expanded MHA.
#
# The latent path is now the only path -- the expanded-MHA layout and its
# SGLANG_OSCAR_K3_MLA_LATENT switch have been removed, so there is nothing left
# to select and nothing left to get wrong. It mattered while both existed: the
# flag drove the FORWARD and MLA_PACKED drove the POOL, and setting one without
# the other died with
#     RuntimeError: shape '[-1, 6, 576]' is invalid for input of size 4608
# because q arrived un-absorbed at head_dim 128 where the layer wanted
# kv_lora_rank 512 + qk_rope 64 = 576.
#
# 1.4 TB of MXFP4 weights need 16 GPUs across two nodes:
#   --tp 16 --nnodes 2 --node-rank N --dist-init-addr <leader>:29500
# The two ranks reach the rendezvous minutes apart, so --dist-timeout must
# exceed torch's 600 s default or the leader gives up with "8/16 clients joined".
#
# --moe-runner-backend triton_kernel keeps those weights PACKED. Without it
# mxfp4.py's process_weights_after_loading calls upcast_from_mxfp(bfloat16) on
# w13/w2 and 1.4 TB will not fit: rank 0 OOMs at 175.7 GiB/GPU with 90% of the
# shards loaded. The loader threads are raised for load time only.
#
# KEEP mmap ON. The 1.4 TB checkpoint is mmapped by default and its page cache
# is charged to the container, so loading creeps toward the memory ceiling and
# the loader was SIGKILLed around 91/96 shards, the per-shard rate degrading
# from 32 s to 70 s. looked like the fix and is the
# opposite: without mmap the shards land in ANONYMOUS memory the kernel cannot
# reclaim, and the rank was OOMKilled outright after six minutes. Page cache is
# reclaimable; give the cgroup headroom instead (pod limit 1100Gi -- past ~1100
# too few nodes can host a rank and the two-rank gang stops being schedulable).
source "$(dirname "$0")/_common.sh"
LAT="${LAT:-${OSCAR_ROTATIONS:-/oscar/rotations}/k3_latent_rot}"
[ "$(ls "$LAT"/layer_*.pt 2>/dev/null | wc -l)" -gt 0 ] || { echo "FATAL: no latent rotations at $LAT"; exit 1; }
echo "[run] latent rotations: $LAT ($(ls "$LAT"/layer_*.pt | wc -l) layers)"
export MLA_ROT_PATH="$LAT" ROT_DIR="$LAT"
# TP 8 x PP 2, not TP 16. Both spread K3 over the same two nodes, but TP16 puts
# every one of the 93 layers' all-reduces on the inter-node network, and at the
# small batches this eval runs that latency dominates: the packed latent path
# measured ~700 ms/token at TP16 against 70 ms/token for the same model at
# TP8 x PP2. Pipelining keeps the all-reduces inside a node and crosses the
# network only at the stage boundary. This is a parallelism choice, not a
# quantization one -- the MLA latent path below is unchanged.
# Sampling: the harness default (temperature 1.0 / top_p 0.95 / top_k 40) is what
# these numbers were measured with. K3's generation_config.json carries NO sampling
# parameters, so that default is applied silently -- but top_k is NOT the reason
# K3 underperforms here. REFUTED by a matched BF16 pair at a 128k budget, TP8 x PP2,
# n=48, everything else held fixed:
#     top_k=40   score 41.67, 38% of responses terminate, 71% parseable
#     top_k=-1   score 35.42, 50% terminate, 73% parseable  <- WORSE
# The early ticks of the top_k=-1 arm read 19/19 terminating; that was survivorship
# (short answers finish first) and it did not survive to n=48. Do not re-litigate
# top_k on partial results.
export MODEL="${MODEL:-moonshotai/Kimi-K3}" TP_SIZE="${TP_SIZE:-8}" PP_SIZE="${PP_SIZE:-2}"
export NNODES="${NNODES:-2}" DIST_TIMEOUT="${DIST_TIMEOUT:-3600}"
export MLA_KV_CACHE_DTYPE=bfloat16
export MLA_GROUP_SIZE="${MLA_GROUP_SIZE:-128}"
export MLA_PACKED=1 MLA_PACKED_SELFCHECK=0
export SGLANG_MIXED_KV_PREFIX_TOKENS="${SGLANG_MIXED_KV_PREFIX_TOKENS:-64}"
export SGLANG_MIXED_KV_RECENT_TOKENS="${SGLANG_MIXED_KV_RECENT_TOKENS:-256}"
export ATTN_BACKEND="${ATTN_BACKEND:-triton}" PREFILL_BACKEND="${PREFILL_BACKEND:-triton}"
# K3 is a linear-attention hybrid, so it selects MambaRadixCache, which asserts
# page_size == 1 unless the extra buffer is enabled.
export MAMBA_SCHEDULER_STRATEGY="${MAMBA_SCHEDULER_STRATEGY:-extra_buffer}"
# --disable-piecewise-cuda-graph, NOT --disable-cuda-graph. Ordinary CUDA graph
# capture stays ON; only the torch.compile-based PIECEWISE variant is off, and it
# has to be: piecewise graphs trace the model with dynamo, and the
# triton_kernel MoE routing kernel takes a custom object argument that dynamo
# rejects with
#     torch._dynamo.exc.Unsupported: Unexpected argument type for a Triton
#     kernel: UserDefinedObjectVariable(Tensor)
# That backend is not optional here -- without it mxfp4.py upcasts w13/w2 to
# bfloat16 and 1.4 TB does not fit on 16 GPUs -- so the compile layer is what
# gives way, not the graphs and not the packing.
export EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-} --pipeline-parallel-size ${PP_SIZE} --moe-runner-backend triton_kernel --disable-piecewise-cuda-graph --model-loader-extra-config {\"num_threads\":6}"
launch
