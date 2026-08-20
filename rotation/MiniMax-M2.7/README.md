# MiniMax-M2.7 — INT2 KV recipe

Dense MHA, 62 layers, 48 Q / 8 KV heads, head_dim 128. Shared per-layer rotation
(`qqt_r_h_pbr` for K, `sst_r_h_pbr` for V) with the **uniform** quantizer —
Lloyd-Max off. Note the checkpoint itself is FP8 (`quant_method: fp8`,
`weight_block_size [128,128]`), so INT2 KV here sits on top of already-quantized
weights, and this is the only model in the sweep with **partial RoPE**
(`rotary_dim=64` of `head_dim=128`).

## Paired GPQA-198, measured

Same code, same session, 3 seeds per arm, TP=4, CUDA graphs on
(`--cuda-graph-max-bs 32`), 32K budget. From
`investigation/08_m27_truncation`:

| arm | seeds | mean |
|---|---|:---:|
| BF16 | 78.28 / 77.78 / 80.81 | **78.96** |
| INT2, windows 256/1024 (current default) | 63.13 / 67.68 / 66.16 | **65.66** |
| INT2, windows 64/256 (old inherited floor) | 52.53 / 61.11 / 60.10 | 57.91 |

**INT2 − BF16 = −13.3 pp** at the tuned windows, −21.05 pp at the old ones. This
model is not near-lossless under INT2 KV, and the earlier table below claiming
otherwise was a harness artifact.

The failure is length, not wrong reasoning in the ordinary sense. At the old
windows INT2's median generation was 2× BF16's (11.4K vs 5.5K words), it hit the
budget on 73/198 responses against BF16's 22, and 12–26 responses per seed
degenerated into non-terminating repetition loops — BF16 produced **zero** loops
in three seeds at any length. Loop onset is at a median of 8.6K–12.4K words and
never at the start, i.e. the error accumulates over the model's own reasoning
chain. Raising the budget to 95K recovers the answered count (134 → 180, equal
to BF16) but not the accuracy: on the 149 questions where both arms terminate
normally, INT2 is 71.8% against BF16's 87.2%.

## Superseded published table

| Benchmark | BF16 | INT2 | Δ |
|---|:---:|:---:|:---:|
| GPQA | 78.28 | 80.3 | +2.0 |
| HumanEval | 88.17 | 92.1 | +3.9 |
| AIME 2025 (95K budget) | 76.67 | 90.0 | +13.3 |
| MATH500 | 93.79 | 94.6 | +0.8 |
| SWE-bench-Verified | – | 70.8 | – |
| LiveCodeBench v6 (95K) | – | 68.4 | – |

**Do not quote the Δ column.** The BF16 numbers came from the model authors under
a different harness and seed count, so these were never paired. The GPQA +2.0
specifically compared INT2 at concurrency 1 against a BF16 arm at concurrency 32
from a different harness, and at `--cuda-graph-max-bs 1` a batch of 2+ runs
eager, so the config was not the one being described. The paired table above
replaces it. LCB 68.4 is a hard-window effect, not quantization — GLM-4.7 in
BF16 scores 65.5 on the same window.

## What is *not* the problem

Measured, so that nobody re-spends the GPU hours:

- **Rotation quality.** The calibrated rotation was fit on 10,000 tokens/layer
  with the full head axis present (8/8 KV, 48/48 Q — not a TP shard), spectrum
  participation ratio 10–29 of 128 and condition number 77–270, orthogonality
  2.6e-08, all in the same band as the shipped Qwen3-4B-Thinking Zoo rotation.
  Swapping in a **data-free Hadamard** scores 56.32 (2 seeds) against the
  calibrated 57.91 — a tie. Refitting cannot buy 13 pp; see
  `investigation/08_m27_truncation/rotation_audit.py`.
- **Lloyd-Max**: 50.0, i.e. −7.9. Refuted here as on Gemma-4 and M3.
- **Quant group 64** (motivated by the partial-RoPE split): 55.31, −2.6.
- **The sink alone**: P=1024 R=256 is 58.76, +0.9 — within noise.
- **The prefix cache**: cache-ON minus cache-OFF is +2.35, inside the seed
  spread.
- **The CUDA-graph padding bug**: fixed. Page 0 is never allocated in the tree
  these arms ran (`hp_prefix_page0_ever_allocated` false, min allocated page 1),
  while the pre-fix twin shows page 0 at 85–100% padded replays.

## Two settings that are not optional

- **Mixed-KV windows must be on.** With `SGLANG_ENABLE_MIXED_KV_WINDOWS=0` this
  model degenerates into repetition loops.
- **Budget 95K generation tokens.** M2.7 is a long-thinking model; a smaller
  budget measures truncation, not accuracy. AIME and LCB above both use 95K.
  Two cautions when using it. First, vary it in *both* arms — a 95K INT2 number
  against a 32K BF16 number is not a comparison, and that asymmetry is where the
  implausible published AIME +13.3 came from. Second, a bigger budget raises the
  answered count without raising accuracy proportionally: INT2 at 95K answers
  180/198 (BF16's count) but still scores ~60, because the responses the budget
  rescues are disproportionately the ones INT2 gets wrong. Report the
  full-denominator score, never accuracy conditioned on having answered.

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
SGLANG_MIXED_KV_PREFIX_TOKENS=256 SGLANG_MIXED_KV_RECENT_TOKENS=1024 \
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
