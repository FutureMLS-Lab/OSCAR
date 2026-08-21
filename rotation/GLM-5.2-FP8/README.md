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
32K budget, single seed. The BF16 arm is a genuine unquantized pass-through
through the same pool class and harness: its rotation path points at a
nonexistent directory, so `_latent_windows` is False *and* `self.rotations` is
empty, and both latent setters fall through to `parent_setter` untouched
(`windows=off` in its log is therefore expected, not a misconfiguration).

| KV | answered | score | truncated | acc given answered | mean chars |
|---|:---:|:---:|:---:|:---:|:---:|
| BF16 | 169/198 | **82.32** | 14.65 % | 96.45 % | 48 339 |
| INT2, **sink 64 / recent 256** | 159/198 | **76.77** | **19.70 %** | 95.60 % | 52 832 |
| INT2, no window | 140/198 | 65.15 | 29.29 % | 92.14 % | 63 286 |

Paired per question at n=198 (matched on the rendered prompt, not by position):

| pairing | both answered | same letter | net | paired SE | McNemar |
|---|:---:|:---:|:---:|:---:|:---:|
| BF16 → 64/256 | 155 | 150 (**96.8 %**) | −11 q (**−5.56 pp**) | ±2.20 pp | p = 0.019 |
| BF16 → no window | 139 | 131 (94.2 %) | −34 q (−17.17 pp) | ±3.03 pp | p = 1.1e-9 |
| no window → 64/256 | 135 | 125 (92.6 %) | **+23 q (+11.62 pp)** | ±3.07 pp | p = 1.9e-4 |

**The windows recover 68 % of the BF16 gap** (−17.17 → −5.56 pp), and what
moved is truncation: 29.29 % → 19.70 %, mean generation 17 % shorter. Of the 15
questions 64/256 still loses to BF16, **13 (86.7 %) are its own truncations and
only 2 are answered wrong**; on the 155 both arms answer, accuracy is 96.8 % vs
96.1 %. The residual gap is budget, not corrupted answers.

Paired SE is computed from the discordant pairs (`sqrt(b+c)/n`), not from each
arm's own binomial SE (±2.7–3.4 pp here) — the questions both arms get right
carry no information about the difference, so treating the arms as independent
would overstate the uncertainty on the quantity actually being claimed.

**Truncation here is measured, not inferred.** These runs' `io_log.jsonl` has no
`finish_reason` field, so any tool that counts `finish_reason == "length"`
reports *zero* truncation for all three arms — an artifact of the log schema
that would invert the conclusion if taken at face value. Re-encoding every
response with the GLM-5.2 tokenizer instead: every response lacking an
`Answer:` line lands at **32 760–32 776 tokens against the 32 768 budget, and
none below 32 000**, in all three arms. The shortest such response is 74 965
characters, none is under 50 characters, none has a compressible (looping)
tail, and they all end mid-word. So "unanswered" and "hit the cap" are the same
set, with no degenerate or garbled cases hiding in it.

`max_total_num_tokens` is **645 056 in every arm** — INT2 here is fake quant
into a BF16 pool, so unlike MiniMax-M2.7 there is no KV-wall asymmetry making
the arms serve different amounts of context.

Same picture on the 40-question paired subset (`--num-examples 40`, seed 0, so
the same 40 questions in every arm *of this subset*):

> **Do not pool this table with the 198-question one.** `GPQAEval.__init__`
> draws `rng.sample(examples, num_examples)` from the *same* `random.Random(0)`
> before drawing each question's option permutation, so passing
> `--num-examples 40` shifts the rng state and every question gets a **different
> permutation** than it has in the full-198 run — a different correct letter for
> the same question. These 40 are not a subset of those 198 in any usable sense:
> matching the two by rendered prompt finds **1 common item out of 40**. Numbers
> may only be compared within one table, which is also why "512 does not help"
> needed re-testing at n=198 rather than being read off the row below.

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

Recent 512 shows no gain on this subset — it recovers some truncation (17.5 %)
but not the score. Read that as **"no evidence 512 helps"**, not as a measured
regression: the cell is single-seed on 40 questions (±6.8 pp at 1 SD on the
score alone), and per the warning above it cannot be compared against the
198-question rows at all. Gemma-4, whose failure mode is the same
over-thinking-into-the-cap, *did* want 64/512 (+4.6 AIME / +4.2 LCB over
64/256), so "this model does not want a bigger window" is not yet established.

**Being re-tested at n=198** (`rotation/investigation/09_glm52_latent_windows/`):
recent 256 / 512 / 1024 at sink 64, one variable, LM=1, group 128, radix on, all
three pinned to the same commit as the BF16 reference arm, with a 64/256 control
arm so the family is internally valid. Until those land, 64/256 is the default
because it is the only window setting measured at full scope, not because larger
windows were ruled out.

### Which comparisons are clean, read back from each server log

Label every arm by what its server reported, not by what the launch script
intended. Read back per arm:

| arm | `disable_radix_cache` | rotation / windows as logged |
|---|:---:|---|
| BF16 full-198 | **False** (on) | `rotation_path .../glm52-rotations-none not found — no rotation`; `windows=off` |
| INT2 64/256 full-198 | **False** (on) | `loaded 78 rotations`, orthogonality residual 9.11e-04; `windows=sink=64/recent=256` |
| INT2 no-window full-198 | **True** (off) | `loaded 78 rotations`; **no `windows=` field at all** |
| BF16 40-subset | **True** (off) | no rotation |

Two consequences, and the first supersedes what this file used to say here:

* **The headline BF16 → 64/256 comparison is radix-matched** — both ran with the
  cache on. The earlier note that "the BF16 arm ran with `--disable-radix-cache`"
  was true of the *40-question* BF16 control, not of the full-198 BF16 arm.
* **The no-window arm is not a controlled A/B of the windows.** It differs in
  radix *and* its write-path log has no `windows=` field, which means it ran
  code from before the windows landed — it is a record of historical behaviour,
  not "the same code with the windows switched off". The clean evidence for the
  windows is the BF16 → 64/256 row plus the 86.7 %-of-losses-are-truncation
  decomposition, neither of which depends on the no-window arm.

Radix impact was separately measured as negligible: the cache hit was exactly
64 tokens — one page, the shared instruction preamble — on 11 of 26 prefill
batches and 0 on the rest. Those tokens sit inside the BF16 sink, so they are
BF16 either way, and no generated token is ever cached. Prefill caching cannot
change generation length, which is what truncation measures.

### `SGLANG_LLOYD_MAX` is not recoverable from the archived arms

Nothing logs the codebook. The pool prints the write path, the window sizes and
every dtype, but not Lloyd-Max, and the launching Jobs are deleted — so whether
the archived 76.77 and 65.15 arms ran LM=1 or LM=0 cannot be established from
their `server.log`. This is not pedantry: Lloyd-Max has *hurt* long generations
elsewhere in this project (Gemma-4 LCB, MiniMax-M3), and GLM-5.2's entire
deficit is long generations, so it is exactly the knob you would want pinned.
Two things were done about it:

* the write-path log line now also prints `group=` and `lloyd_max=`, so future
  arms are self-identifying;
* the window sweep in `rotation/investigation/09_glm52_latent_windows/` dumps
  every knob to `$RUN_DIR/config.env`, pins an explicit commit rather than
  tracking the branch, and carries its own 64/256 control arm so its
  window comparison is internally valid regardless.

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
