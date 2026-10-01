# MiniMax-M3 — INT2 KV recipe

428B VL-MoE. A single rotation shared per layer is **saturated** on this model:
an all-head calibrated rotation scores 85.35 % on GPQA against 84.85 % for plain
Hadamard, a one-question difference at n=198. Fit it anyway — the calibrated
rotation is measurably better on reconstruction error and costs nothing at serve
time — but do not expect calibration to be the lever here.

| KV | GPQA (198) | GPQA PPL | vs BF16 |
|---|:---:|:---:|:---:|
| BF16 | 88.38 % | 1.3140 | — |
| INT2, Hadamard | 84.85 % | – | – |
| INT2, rank-0 calib (**deficient dump**) | 84.85 % | – | – |
| **INT2, all-head calib** | **85.35 %** | 1.3479 | +2.6 % |
| INT2, all-head + Lloyd-Max | 84.85 % | 1.3672 | +4.0 % |

**Do not enable Lloyd-Max.** It is worse on both perplexity and GPQA.

## The dump must cover every TP rank

This is the one trap specific to M3. The dump hook writes per rank, and a rank-0
dump captures **4 of 64 query heads and 1 of 4 KV heads** — an unrepresentative
head sample that fits a rotation no better than Hadamard, with nothing in the
logs to say so. Dump all 16 ranks and merge before fitting.

Reconstruction error on the merged data, INT2-asymmetric relL2:

| rotation | V | K |
|---|:---:|:---:|
| all-head | **0.4794** | ~0.49 |
| rank-0 only | 0.4937 | ~0.49 |
| Hadamard | 0.5000 | ~0.49 |

K stays flat because `qqt` targets the Q covariance, not raw-K error. Verify V
improves before trusting a fit; if it sits at 0.50 the dump is deficient.

Calibration token count is **not** a lever — the covariance converges by roughly
5K tokens per layer.

## Steps

```bash
# 1. dump post-RoPE Q/K/V, every TP rank. TP=16 across 2 nodes.
#    DUMP_M3_QKV_DIR makes minimax_m3.py write layer_<id>/rank_<r>/{q,k,v}/.
bash save_qkv_m3.sh

# 2. merge the 16 per-rank shards into full q[T,64,128] / k,v[T,4,128]
python3 merge_qkv_allhead.py \
  --dump  <dump-dir>/qkv_dumps_perrank \
  --out   <dump-dir>/qkv_dumps_merged \
  --tp 16 --num-q-heads 64 --num-kv-heads 4

# 3. fit
bash compute_rotation.sh          # qqt_sst / r_h_pbr, ~12K tok/layer

# 4. verify before spending an eval on it
python3 verify_rotation.py \
  --dump <dump-dir>/qkv_dumps_merged \
  --k-rotation <out>/k_rotation_qqt_r_h_pbr.pt \
  --v-rotation <out>/v_rotation_sst_r_h_pbr.pt
```

## Serving

```bash
SGLANG_ENABLE_MIXED_KV_WINDOWS=1 SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
SGLANG_MIXED_KV_PREFIX_TOKENS=64 SGLANG_MIXED_KV_RECENT_TOKENS=256 \
SGLANG_MIXED_KV_HP_DTYPE=bfloat16 SGLANG_MIXED_KV_SCALE_DTYPE=float32 \
SGLANG_OSCAR_ABSORB_V_ROTATION=0 SGLANG_LLOYD_MAX=0 \
SGLANG_OSCAR_K_CLIP_RATIO=0.96 SGLANG_OSCAR_V_CLIP_RATIO=0.92 \
SGLANG_OSCAR_K_ROTATION_PATH=<out>/k_rotation_qqt_r_h_pbr.pt \
SGLANG_OSCAR_V_ROTATION_PATH=<out>/v_rotation_sst_r_h_pbr.pt \
python -m sglang.launch_server --model-path MiniMaxAI/MiniMax-M3 \
  --tensor-parallel-size 16 --nnodes 2 --node-rank <0|1> \
  --dist-init-addr <head>:20000 \
  --kv-cache-dtype int2 --page-size 8 --disable-radix-cache \
  --prefill-attention-backend fa3 --decode-attention-backend triton \
  --mem-fraction-static 0.85 --context-length 40960
```

INT2 KV capacity is 3,408,248 tokens against 532,539 for BF16 — about **6.4×**.

## Operational notes

- **Pin the revision.** M3's shared-cache `refs/main` was once bumped to an
  undownloaded revision, which broke every `HF_HUB_OFFLINE=1` load. Pass
  `--revision` explicitly in manifests.
- The `nccl` node pool is often saturated. `node-pool=compute,node-group=default`
  is the same H100+IB hardware and TP=16 across 2 nodes works there — swap the
  nodeSelector rather than waiting.
