# MiniMax-M2.7 — INT2 KV recipe

Dense MHA, 62 layers, 48 Q / 8 KV heads, head_dim 128. Shared per-layer rotation
(`qqt_r_h_pbr` for K, `sst_r_h_pbr` for V) with the **uniform** quantizer —
Lloyd-Max off. Note the checkpoint itself is FP8 (`quant_method: fp8`,
`weight_block_size [128,128]`), so INT2 KV here sits on top of already-quantized
weights, and this is the only model in the sweep with **partial RoPE**
(`rotary_dim=64` of `head_dim=128`).

## Read this first: max-bs 32 is broken for INT2 on this model

Everything in the next section was measured at `--cuda-graph-max-bs 32`, and a
later control showed that configuration is itself defective. Same session, same
rotation, same 64/256 windows, **identical** client concurrency 15, varying only
`--cuda-graph-max-bs`, compared question-by-question over the same ~69 GPQA
items, 3 seeds:

| arm | decode path | correct (matched subset) | capped | length |
|---|---|---|---|---|
| max-bs 32 | 100% padded graph replay | 38 / 38 / 40 | 26 / 20 / 22 | 1.0× |
| max-bs 1 | 100% eager, 0 padded | **56 / 57 / 56** | **6 / 5 / 6** | 0.63–0.73× |

The 2×2 is 19–20 gains against 1–3 losses in every seed (McNemar p < 0.001).
Eager decode recovers INT2 to roughly BF16 parity on the matched subset.

Caveat on that subset: the eager arms run ~3× slower (that is the cost of never
replaying a graph), so this was read while they were mid-flight, over the ~69
questions both arms had finished. Those are the *shorter* questions, so the
absolute levels are inflated for both arms — 38/69 and 56/69 are not full-198
scores. The comparison itself is sound because it is the same question set in
both arms, and the effect is 3/3 seeds with a 20-vs-2 asymmetry. Full-198
numbers should replace these when the eager arms finish. So the
big INT2 deficit below is **predominantly a CUDA-graph-replay defect, not 2-bit
quality**, and it reconciles the good historical numbers — SWE-bench 70.8, LCB
v6 68.4, AIME25 90.0 and GPQA 80.3 all ran max-bs 1 and single-stream, i.e.
never on the broken path.

It is **not** the known page-0 padding bug: that is fixed here (page 0 reserved,
`hp_prefix_page0_ever_allocated` false, minimum allocated page 1, while the
pre-fix twin still shows page 0 at 85–100% padded replays). Something else on
the replay path is wrong and is not yet root-caused.

**Serve INT2 with `--cuda-graph-max-bs 1` until this is fixed, and do not read
any max-bs 32 INT2 number on this model as a quantization result.**

## Paired GPQA-198 at max-bs 32 — measured under the defect above

Same code, same session, 3 seeds per arm, TP=4, CUDA graphs on
(`--cuda-graph-max-bs 32`), 32K budget. From
`investigation/08_m27_truncation`:

| arm | seeds | mean |
|---|---|:---:|
| BF16 | 78.28 / 77.78 / 80.81 | **78.96** |
| INT2, windows 256/1024 | 63.13 / 67.68 / 66.16 | 65.66 |
| INT2, windows 64/2048 | 65.66 / 65.66 / 65.66 | 65.66 |
| INT2, windows 64/1024 | 60.61 / 58.08 / 61.11 | 59.93 |
| INT2, windows 64/256 (the recipe) | 52.53 / 61.11 / 60.10 | 57.91 |

Read these as *the size of the replay defect at each window setting*, not as
quantization results. The window sweep is a real, replicated ordering — and two
large-window configurations land on the same 65.66 from opposite directions with
seed spreads 0.00 (64/2048) and 4.55 (256/1024) — but since a bigger BF16 window
simply exposes less KV to whatever the replay path corrupts, the most likely
reading is that it *masks* the defect. That is why 64/256 remains the default:
adopting 256/1024 would spend 2.79 → 3.65 bits/element papering over a bug.
R=4096 cannot be measured at all here (HP arena would need ~11.6 GB/rank atop
the FP8 weights and OOMs at mem-fraction 0.85).

How the defect presents, all at max-bs 32: INT2's median generation is 2× BF16's
(11.4K vs 5.5K words), it hits the budget on 73/198 responses against BF16's 22,
and 12–26 responses per seed degenerate into non-terminating repetition loops
while BF16 produces **zero** loops in three seeds at any length. Loop onset is at
a median of 8.6K–12.4K words and never at the start. A representative loop is
`Answer: **A**.` repeated to the cap — the model finds the answer and loses the
ability to stop.

Under eager decode the same symptoms largely disappear (capped 26/20/22 → 6/5/6,
length → 0.7×), which is why the length/loop behaviour should be attributed to
the replay path rather than to 2-bit KV. The residual eager-mode loop rate is
consistent with the historical LCB v6 run at 95K, which reported "truncation
10/171 = 6% ... ~10 pathologically non-terminating problems" on a max-bs 1
server — so a small intrinsic non-termination rate does exist, roughly 6%,
against the ~17% seen on the broken path.

Budget is not the cause either. Run at the model's own 95K budget in **both**
arms the gap gets *wider*, because BF16 converts the extra budget into answers
(capping 22/198 → **0/198**, score 78.96 → 85.52 over 3 seeds) while INT2 turns
it into more tokens (capping 73 → 39, score 57.91 → 60.10).

## Superseded published table

| Benchmark | BF16 | INT2 | Δ |
|---|:---:|:---:|:---:|
| GPQA | 78.28 | 80.3 | +2.0 |
| HumanEval | 88.17 | 92.1 | +3.9 |
| AIME 2025 (95K budget) | 76.67 | 90.0 | +13.3 |
| MATH500 | 93.79 | 94.6 | +0.8 |
| SWE-bench-Verified | – | 70.8 | – |
| LiveCodeBench v6 (95K) | – | 68.4 | – |

**Do not quote the Δ column** — the BF16 numbers came from the model authors
under a different harness and seed count, so these were never paired, and the
GPQA +2.0 specifically compared INT2 at concurrency 1 against a BF16 arm at
concurrency 32. But the **INT2 column is not the thing that was wrong**: every
one of these ran `--cuda-graph-max-bs 1` and single-stream, which the control
above shows is the *working* decode path. They are plausible INT2 numbers
obtained off the defective path, which is exactly why they look nothing like the
max-bs 32 measurements. Re-pair them against same-harness BF16 arms at max-bs 1
rather than discarding them. LCB 68.4 is a hard-window effect, not quantization
— GLM-4.7 in BF16 scores 65.5 on the same window.

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
- **The known page-0 CUDA-graph padding bug**: fixed, and therefore *not* the
  replay defect described at the top. Page 0 is never allocated in the tree these
  arms ran (`hp_prefix_page0_ever_allocated` false, min allocated page 1) while
  the pre-fix twin shows page 0 at 85–100% padded replays. Note the mixed-KV
  auditor's damage counter reads zero even on known-broken code, so it cannot be
  used to clear the replay path — only the allocator signal is trustworthy, and
  it only clears this one bug.
- **The partial-RoPE boundary** (M2.7 is the only model here with
  `rotary_dim=64` of `head_dim=128`, so one rotation and one 2-bit scale span
  both the positional and static halves). Measured on the real dump: the halves
  have the *same* magnitude (rms 1.63 vs 1.47) and error is spread evenly
  (relerr 0.46 vs 0.52), so no shared scale is being swamped. A block-diagonal
  64+64 rotation that cannot mix the halves is **worse** offline (logit rel-err
  0.360 vs 0.330), because a full 128-dim Hadamard spreads outliers over 128
  coordinates while two 64-dim blocks spread over 64 — the same mechanism that
  made quant group 64 worse. Keeping dims 0–63 in BF16 does cut logit error to
  47% of baseline (against 82% for keeping the static half instead, so the
  positional subspace does carry more logit signal), but it costs ~9 bits/element
  on K against 2 — a 4.5× increase — so it is a diagnostic, not a fix. See
  `investigation/08_m27_truncation/rope_subspace_error.py`.

## Two settings that are not optional

- **Mixed-KV windows must be on.** With `SGLANG_ENABLE_MIXED_KV_WINDOWS=0` this
  model degenerates into repetition loops.
- **Budget 95K generation tokens.** M2.7 is a long-thinking model; a smaller
  budget measures truncation, not accuracy. AIME and LCB above both use 95K.
  But it must be varied in *both* arms, and doing so does **not** narrow the
  INT2 gap — it widens it. Measured, windows 64/256:

  | budget | BF16 | INT2 | Δ |
  |---|:---:|:---:|:---:|
  | 32K | 78.96 (3 seeds) | 57.91 (3 seeds) | −21.05 |
  | 95K | ~85.5 (85.35 / 86.36 / 84.85) | 60.10 (1 seed) | ~−25.4 |

  BF16 converts the extra budget into answers — it caps **0/198** at 95K
  (against 22/198 at 32K) and answers 196–198/198. INT2 converts it into more
  tokens: capping falls only 73 → 39, and the responses the budget rescues are
  disproportionately ones it gets wrong, so answered rises 140 → 180 while the
  score moves 57.91 → 60.10. A 95K INT2 number against a 32K BF16 number is the
  asymmetry that produced the implausible published AIME +13.3. Always report
  the full-denominator score, never accuracy conditioned on having answered.

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
