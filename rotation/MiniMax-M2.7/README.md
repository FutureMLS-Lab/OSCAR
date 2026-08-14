# MiniMax-M2.7 — INT2 KV recipe

Dense MHA. Shared per-layer rotation (`qqt_r_h_pbr` for K, `sst_r_h_pbr` for V)
with the **uniform** quantizer — Lloyd-Max off.

| Benchmark | BF16 | INT2 | Δ |
|---|:---:|:---:|:---:|
| GPQA | 78.28 | 80.3 | +2.0 |
| HumanEval | 88.17 | 92.1 | +3.9 |
| AIME 2025 (95K budget) | 76.67 | 90.0 | +13.3 |
| MATH500 | 93.79 | 94.6 | +0.8 |
| SWE-bench-Verified | – | 70.8 | – |
| LiveCodeBench v6 (95K) | – | 68.4 | – |

Read the deltas with care: the BF16 column came from the model authors under a
different harness and seed count, so these are **not** paired measurements. The
honest claim is that INT2 lands in the same band as BF16 on all four, not that
it beats it. LCB 68.4 is a hard-window effect, not quantization — GLM-4.7 in
BF16 scores 65.5 on the same window.

## Two settings that are not optional

- **Mixed-KV windows must be on.** With `SGLANG_ENABLE_MIXED_KV_WINDOWS=0` this
  model degenerates into repetition loops.
- **Budget 95K generation tokens.** M2.7 is a long-thinking model; a smaller
  budget measures truncation, not accuracy. AIME and LCB above both use 95K.

## Steps

```bash
# 1. dump post-RoPE Q/K/V on GPQA (TP=4)
bash save_qkv_m27.sh

# 2. fit
bash compute_rotation.sh    # qqt_sst / r_h_pbr
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
python -m sglang.launch_server --model-path MiniMaxAI/MiniMax-M2.7 \
  --tensor-parallel-size 4 \
  --kv-cache-dtype int2 --page-size 8 --disable-radix-cache \
  --tool-call-parser minimax-m2
```

## Evaluating it

If you serve with `--cuda-graph-max-bs 1`, drive the eval client with **one**
thread. Two threads collapse throughput from ~115 tok/s to ~8 tok/s, because
every concurrent batch misses the single captured graph shape and falls back to
eager. This looks like a hung eval rather than a configuration mistake.
