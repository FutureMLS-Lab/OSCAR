# GLM-5.2-FP8 — INT2 KV recipe (MLA shared latent)

MLA stores one compressed latent `c_kv` (512-d) shared by all heads, plus a
positional `k_pe`. Only `c_kv` is quantized; `k_pe` stays BF16. Per-head
rotations do not apply here — there is nothing per-head to rotate.

Rotation is `Rcov · P · Hblock` (covariance eigenvectors, bit-reversal
permutation, per-group Hadamard) with **Lloyd-Max** on, group 128.

The BF16 sink and recent windows apply here like everywhere else: sink 64,
recent 256, on by default. That was not true until `NSAInt2HPKVPool` /
`MLAInt2HPKVPool` learned to read `SGLANG_MIXED_KV_PREFIX_TOKENS` and
`SGLANG_MIXED_KV_RECENT_TOKENS` — before that they quantized every latent
token, including the attention sink and the newest ones, and this file told
you setting those vars changed nothing. **Every number in the table below
predates the windows** — read them as the no-window arm, not as the recipe;
the next section has the windowed GPQA numbers.

| Benchmark | BF16 | INT2 (no window) | Δ |
|---|:---:|:---:|:---:|
| GPQA @32K | 81.3 ± 0.0 | 68.7 ± 1.8 | −12.6 |
| HumanEval (≤16K) | 91.5 ± 1.2 | 88.2 ± 1.5 | −3.3 |
| AIME 2025 | 76.7 ± 3.3 | 75.6 ± 6.9 | −1.1 (n.s.) |
| MATH500 | 94.8 ± 0.8 | 93.8 ± 0.3 | −1.0 |

Only the GPQA row has been re-measured with the windows. The other three are
short-generation benchmarks where truncation is not the binding constraint, so
expect less from the windows there — but that is an expectation, not a
measurement.

**The −12.6 at 32K is truncation, not wrong answers.** On questions both arms
answer, they pick the same option 97.1 % of the time; INT2 simply thinks 1.69×
longer, so the truncation rate goes 14 % → 28 %. BF16 itself gains 6.6 points
moving from a 32K to a 64K budget, which confirms the metric is budget-bound.
Give this model a generous `max_tokens` before reading a regression into it.
The INT2 @64K cell is not measured.

## What the windows are worth here

Full GPQA-Diamond 198, B200 TP=8, CUDA graph on, `--kv-cache-dtype bfloat16`,
Lloyd-Max on, 32K budget — the two arms differ only in the windows:

| KV | answered | score | truncated | mean chars |
|---|:---:|:---:|:---:|:---:|
| INT2, no window | 140/198 | 65.15 | 29.3 % | 63 286 |
| INT2, **sink 64 / recent 256** | 159/198 | **76.77** | **19.7 %** | 52 832 |

**+11.6 points, and the truncation rate is what moved**: 29.3 % → 19.7 %, with
the mean generation 17 % shorter. Paired against the no-window arm, 23
questions are newly answered against 5 lost, and of the 135 both arms answer
they agree on 126 (93 %) — single-seed at temperature 1.0, so some of that
churn is sampling, not the windows.

Same picture on the 40-question paired subset (`--num-examples 40`, seed 0, so
the same 40 questions in every arm), which additionally has a BF16 control:

| KV | answered | score | truncated | mean chars |
|---|:---:|:---:|:---:|:---:|
| INT2, no window | 30/40 | 75.0 | 25.0 % | 54 267 |
| INT2, **sink 64 / recent 256** | 35/40 | **87.5** | **12.5 %** | 44 426 |
| INT2, sink 64 / recent 512 | 33/40 | 75.0 | 17.5 % | 46 223 |
| BF16 | 36/40 | 90.0 | 10.0 % | 44 566 |

Score on this subset *is* the answered rate — every question either arm
answered, it answered correctly — so the whole −15.0 no-window gap was
truncation, and 64/256 removes 5 of the 6 truncations that separated INT2 from
BF16. Paired against the no-window arm: 30/30 identical answers on the
questions both answered, 5 newly answered, **0 lost**, mean generation length
down to the BF16 value. The windows do not make the model answer differently;
they stop it from over-thinking past the budget.

Recent 512 is *not* better here — it recovers some truncation (17.5 %) but not
the score. Unlike Gemma-4, this model does not want a bigger recent window;
take the 64/256 floor. The 512 cell is single-seed on 40 questions
(±6.8 pp at 1 SD on the score alone), so read it as "no evidence 512 helps",
not as a measured regression.

One asymmetry between the two tables: the window arms ran with the radix cache
on and the no-window/BF16 arms with `--disable-radix-cache`, because the
harness default flipped between them. Measured impact: the cache hit was
exactly 64 tokens — one page, the shared instruction preamble — on 11 of 26
prefill batches and 0 on the rest. Those tokens are inside the BF16 sink, so
they are BF16 either way, and no generated token is ever cached. Prefill
caching cannot change generation length, which is what truncation measures.

## Do not expect a better rotation to close it

Measured on a real `c_kv` dump (functional error, lower is better):

| variant | error | vs no rotation |
|---|:---:|:---:|
| no rotation | 0.0845 | — |
| Hadamard | 0.0839 | ≈ 0 |
| **`Rcov·P·Hblock`** | **0.0523** | −0.72 dB |
| Hadamard + HP128 | 0.0447 | −2.74 dB |

Only the last row reaches BF16 parity, and it costs ~6.5 bits per element, which
defeats the point of 2-bit. The latent is a *trained* compressed representation —
it has no outlier structure for a rotation to flatten. `Rcov·P·Hblock` is the
best of the cheap options; take it and move on.

## Steps

```bash
# 1. dump c_kv on GPQA (TP=16, 2 nodes)
bash save_ckv_glm52.sh

# 2. fit the per-layer latent rotation
bash compute_rotation.sh     # -> rotations/layer_*.pt
```

Fitted rotations are published as
`Zhongzhu/OSCAR-RotationZoo/GLM-5.2-FP8/c_kv_rcov_phblock_g128/` — download
rather than re-fit unless you are changing the recipe.

## Serving

```bash
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
SGLANG_LLOYD_MAX=1 \
SGLANG_OSCAR_MLA_KV_ROTATION_PATH=<rotations-dir> \
SGLANG_OSCAR_MLA_KV_GROUP_SIZE=128 \
python -m sglang.launch_server --model-path zai-org/GLM-5.2-FP8 \
  --tensor-parallel-size 16 --nnodes 2 --node-rank <0|1> \
  --dist-init-addr <head>:20000 \
  --kv-cache-dtype bfloat16 --disable-radix-cache \
  --mem-fraction-static 0.85
```

Three flags in that command are load-bearing and were each wrong in an earlier
version of this file:

* `SGLANG_OSCAR_MLA_KV_ROTATION_PATH` is the name the pool reads. It is also
  what *creates* the pool — misspell it (this file said
  `SGLANG_OSCAR_MLA_ROTATION_DIR`) and the server starts, serves, and scores
  with a plain BF16 latent cache and no rotation at all.
* `--kv-cache-dtype bfloat16`, not `int2`. `int2` aborts at argument validation
  on a DSA model; the latent is fake-quantized into a float cache, so that
  cache stays float. Leaving it at `auto` is worse than either: sglang picks
  fp8_e4m3 on SM100+ and bfloat16 on Hopper and below, which silently changes
  the method with the GPU generation. Pinning it is what makes a B200 run
  comparable to the H100 numbers in the table above.
* No `--page-size`. A DSA model forces 64; the 8 this file used to pass was
  ignored.

The BF16 windows need no flag — 64/256 is the default the pool applies when
`SGLANG_MIXED_KV_PREFIX_TOKENS` / `SGLANG_MIXED_KV_RECENT_TOKENS` are unset,
because the two vars' own defaults are the per-head pool's older 32/128 and
inheriting those would put this path below the floor. Raise the recent window
for a model that needs more (Gemma-4 and Qwen3-8B use 512). Setting both to 0
turns the windows off; the server logs a warning when you do, since that is an
A/B arm and not a serving configuration. Confirm the line

```
[Int2HPKVPool] BF16 latent windows: sink=64 recent=256 ...
```

in the server log — its absence means the pool fell back to quantizing every
latent token. `SGLANG_MIXED_KV_AUDIT=1` additionally checks each live
request's tiers against the windows every 25 decode steps (it syncs the decode
path — never leave it on for a timed run).

Set `SGLANG_OSCAR_MLA_KV_REAL_KERNEL=1` to use the packed-INT2 latent kernel
instead of the fake-quant path: bit-identical packing, decode cosine 1.000000,
1024 → 160 B/token (**6.4×**). It is correct but not yet fast — measure before
enabling it for throughput.
