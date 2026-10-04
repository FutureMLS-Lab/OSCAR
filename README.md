<p align="center">
  <img src="materials/oscar_logo_kv_transparent.png" alt="OSCAR INT2 KV-Cache" width="180"/>
</p>

# OSCAR

### Offline Spectral Covariance-Aware Rotation for 2-bit KV Cache Quantization

<p align="center">
  <a href="https://arxiv.org/pdf/2605.17757"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=white" alt="Paper"/></a>
  <a href="https://oscar-quantize.github.io/"><img src="https://img.shields.io/badge/Website-oscar--quantize-1f77b4?logo=googlechrome&logoColor=white" alt="Website"/></a>
  <a href="https://huggingface.co/Zhongzhu/OSCAR-RotationZoo"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-RotationZoo-FFD21E" alt="HuggingFace RotationZoo"/></a>
  <br/>
  <a href="https://huggingface.co/Zhongzhu/OSCAR-LLAMACPP-Qwen3-32B-INT2-KV"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20GGUF%20Qwen3--32B-INT2%20KV-FFD21E" alt="GGUF Qwen3-32B INT2 KV"/></a>
  <a href="https://huggingface.co/Zhongzhu/OSCAR-LLAMACPP-Gemma-4-12B-it-INT2-KV"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20GGUF%20Gemma--4--12B--it-INT2%20KV-FFD21E" alt="GGUF Gemma 4 12B it INT2 KV"/></a>
  <a href="https://huggingface.co/Zhongzhu/OSCAR-LLAMACPP-Qwen3-4B-Thinking-2507-INT2-KV"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20GGUF%20Qwen3--4B--Thinking--2507-INT2%20KV-FFD21E" alt="GGUF Qwen3-4B Thinking 2507 INT2 KV"/></a>
</p>

OSCAR captures Q/K/V activations on a small calibration set, estimates **attention-aware K/V covariance structures** offline, and derives per-layer rotations + clipping thresholds that align KV quantization with the directions attention actually consumes. By storing the bulk of the KV cache in INT2 while retaining only a small BF16 sink and recent window, OSCAR reduces KV-cache memory by approximately **8×** compared with BF16. Under the same memory budget, our **attention kernel** enables up to **7× higher throughput** at large batch sizes, and also accelerates batch-size-1 decoding by up to **3×** by reducing memory-bandwidth overhead.


<p align="center">
  <img src="materials/OSCAR_pipeline.png" alt="OSCAR pipeline" width="720"/>
</p>

OSCAR is built directly into the open-source SGLang framework (main branch), llama.cpp (zhongzhu/llamacpp branch). We also provide a rotation zoo so users can download calibrated rotations directly instead of recomputing them.

## 🔥 Latest News
- **[2026-10-03]** Dense INT2 decode on B200 is now within **1.2–1.5× of FlashInfer BF16** at 64K / batch 1 (it was 2–3.9×): a separate INT2 split cap, a fused HP+INT2 stage-1, batched decode rotations, a parallel stage-2 and a parallel metadata build, all default-on and checked bit-for-bit against the two-launch path. The rotation can also be **fit at server start** (no offline calibration), and 1-bit / 1.5-bit K via product quantization ships as an opt-in tier.
- **[2026-10-03]** OSCAR now lives on top of **upstream SGLang main** (base `67eab57057`, the official nightly of 2026-10-02): the repository root is upstream, OSCAR is the diff. Twelve model families pass the garbling sweep with radix cache and CUDA graphs on, and the [upstream-aligned results](#results-on-the-upstream-aligned-tree) cover GPQA-Diamond at a 64K budget and 64K-context decode speed for thirteen models, with every INT2 arm running the model's own attention (DSA for GLM-5.x, MSA for MiniMax-M3, MLA + KDA across two nodes for Kimi-K3).
- **[2026-06-26]** OSCAR is PRing into **vLLM** too, bringing INT2 KV cache support to another high-throughput serving stack.
- **[2026-06-07]** OSCAR INT2 KV cache now runs **256K Gemma 4 12B under <code style="color : Red">!!16GB!!</code>** and **Qwen3** on the [`zhongzhu/llamacpp` llama.cpp fork](https://github.com/FutureMLS-Lab/OSCAR/tree/zhongzhu/llamacpp) — **~8× smaller KV at near-f16 quality**, with [pre-built `*-rot-kv.gguf` on Hugging Face](https://huggingface.co/Zhongzhu/OSCAR-LLAMACPP-Gemma-4-12B-it-INT2-KV). RUN GEMMA 4 / QWEN3 with LONG CONTEXT on your LOCAL MAC!
  <details>
  <summary> <b>MacBook M5 Max Gemma 4 12B OSCAR INT2 KV Local Run Video</b></summary>
  <img width="960" height="502" alt="Screen Recording 2026-06-08 at 00 47 11 - 2x" src="https://github.com/user-attachments/assets/72f7c51d-fb43-42b7-ac2b-1b5baaf256c5" />
  </details>
- **[2026-06-05]** OSCAR now runs its INT2 KV cache through a fused mixed-precision Flash-Attention kernel on Apple Metal in the [`zhongzhu/llamacpp` llama.cpp fork](https://github.com/FutureMLS-Lab/OSCAR/tree/zhongzhu/llamacpp), making long-context decode up to ~15× faster (near-BF16) at ~7× less KV memory. Try to RUN QWEN-3-32B with LONG CONTEXT in your LOCAL MAC!
  <details>
  <summary><b>MacBook M5 Max Qwen3-32B OSCAR INT2 KV Local Run Screenshot</b></summary>
  <img width="1003" height="654" alt="Screenshot 2026-06-05 at 09 57 31" src="https://github.com/user-attachments/assets/8ea1ccf9-c2f0-4f3e-8232-7f0b2dbdd144" />
  </details>
- **[2026-06-04]** OSCAR now supports **Gemma 4 12B** with **SGLang INT2 KV cache** on the [`zhongzhu/gemma4-12b`](https://github.com/FutureMLS-Lab/OSCAR/tree/zhongzhu/gemma4-12b) branch.
- **[2026-05-31]** OSCAR is now runnable on the zhongzhu/llamacpp branch of **llama.cpp**. Feedback and suggestions are very welcome!
- **[2026-05-23]** OSCAR release the [qwen3.5 4B, 35B-A3B, minimax-m2.7 229B preview results](#main-results). You can use OSCAR for qwen3.5, minimax2.7 beta now! refer to branch zhongzhu/hybrid-model and set SGLANG_LLOYD_MAX=1.
- **[2026-05-18]** Full release: [paper](https://arxiv.org/pdf/2605.17757), code, [website](https://oscar-quantize.github.io/), and [RotationZoo](https://huggingface.co/Zhongzhu/OSCAR-RotationZoo) are all live — runs out of the box on **SGLang**.

## 📖 Table of Contents
- [Main results](#main-results)
- [Layout](#layout)
- [Setup](#setup)
- [Quick start (Qwen3-8B example)](#quick-start-qwen3-8b-example)
- [Model support](#model-support)
- [All configured models](#all-configured-models)
- [How the rotation is fit (spectral covariance)](#how-the-rotation-is-fit-spectral-covariance)
- [Serving with the rotation](#serving-with-the-rotation)
- [Calibration knobs](#calibration-knobs)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License & acknowledgements](#license--acknowledgements)

## Main results

### Results on the upstream-aligned tree

Base `67eab57057` (upstream SGLang main), B200, radix cache and CUDA graphs on; GPQA-Diamond @64K budget, n=198, single seed unless a cell says otherwise (then the mean over the seeds run on this tree); decode at 64K context, batch 1. Details and ms/tok in the [per-model](#per-model-gpqa-bf16-vs-oscar-int2-this-tree) and [speed](#64k-decode-on-b200-this-tree) tables.

| Model | GPQA BF16 | GPQA INT2 | INT2 decode vs triton BF16 | INT2 decode vs FlashInfer family |
|:---:|:---:|:---:|:---:|:---:|
| Qwen3-4B-Thinking-2507 | 63.6 | 64.6 | 6.57× faster | 1.27× slower |
| Qwen3-8B | 58.6 (2 seeds) | 52.8 (4 seeds) | 5.71× faster | 1.22× slower |
| Qwen3-32B | 64.1 | 59.6 | 5.45× faster | 1.20× slower |
| Qwen3-30B-A3B | 61.1 | 55.3 (2 seeds) | 8.39× faster | 1.40× slower |
| Qwen3.5-4B | 79.3 | 75.3 | 3.23× faster | 1.26× slower |
| Qwen3.5-35B-A3B | 81.8 | 83.8 | 3.97× faster | 1.32× slower |
| Gemma-4-12B-it | 63.1 | 64.1 | 2.03× faster | 1.51× slower (trtllm_mha) |
| MiniMax-M2.7 | 86.9 | 87.9 | 6.66× faster | 1.27× slower |
| MiniMax-M3 (MSA sparse) | 90.9 | 88.1 (2 seeds) | 1.34× slower | 1.73× slower |
| GLM-4.7-FP8 | 80.8 | 78.8 | 6.54× faster | 1.18× slower |
| GLM-5.2-FP8 (DSA sparse) | 87.4 | 83.8 | 1.36× slower | same DSA path |
| GLM-5.3 (DSA sparse) | 87.4 | 84.3 | 1.36× slower | same DSA path |
| Kimi-K3 (TP 8 × PP 2) | 90.9 | 90.4 | 1.33× slower | 1.40× slower (trtllm_mla) |

The dense-GQA rows read the same way: INT2 decodes 1.9–3.0× faster than the
triton BF16 arm and 2.0–3.9× slower than the FlashInfer-family kernels. Where
the BF16 arm is already a sparse or MLA-specific kernel (GLM's DSA, MiniMax-M3's
MSA, Kimi-K3's MLA) or the head geometry has no fast path (Gemma-4's 512-wide
heads), INT2 is 1.1–1.4× slower than BF16 on the same backend; the gain there
is the 4× smaller cache, not speed.

<details>
<summary><b>Qwen3.5-4B preview</b></summary>

Qwen3.5 — BF16 vs OSCAR INT2 KV (2-bit, sink 64 / recent 256), mean ± std over 3 seeds (35B-A3B AIME: 8 seeds, N=30 is high-variance). OSCAR quantizer per model best: 4B uniform, 35B-A3B Lloyd-Max.

| Benchmark | BF16 | OSCAR | Δ vs BF16 |
|---|:---:|:---:|:---:|
| GPQA-Diamond | 76.9 ± 1.3 | **75.8 ± 1.6** | −1.2 |
| HumanEval | 81.7 ± 1.8 | **83.9 ± 1.0** | +2.2 |
| AIME 2025 | 47.8 ± 3.1 | **46.7 ± 0.0** | −1.1 |
| MATH500 | 89.5 ± 0.6 | **88.0 ± 0.6** | −1.5 |

</details>

<details>
<summary><b>Qwen3.5-35B-A3B preview</b></summary>

Qwen3.5 — BF16 vs OSCAR INT2 KV (2-bit, sink 64 / recent 256), mean ± std over 3 seeds (35B-A3B AIME: 8 seeds, N=30 is high-variance). OSCAR quantizer per model best: 4B uniform, 35B-A3B Lloyd-Max.

| Benchmark | BF16 | OSCAR | Δ vs BF16 |
|---|:---:|:---:|:---:|
| GPQA-Diamond | 83.3 ± 1.8 | **84.0 ± 1.3** | +0.7 |
| HumanEval | 83.9 ± 0.6 | **86.6 ± 1.8** | +2.6 |
| AIME 2025 † | 66.7 ± 5.3 | **62.1 ± 4.7** | −4.6 |
| MATH500 | 92.8 ± 0.2 | **91.7 ± 0.4** | −1.1 |

<sub>† AIME N=30 is high-variance; measured over 8 seeds. The −4.6 gap is not statistically significant (Welch t=1.72). At 3 seeds it read −6.7, inflated by a favorable BF16 draw.</sub>

</details>

<details>
<summary><b>MiniMax-M2.7 preview</b></summary>

MiniMax-M2.7 — BF16 vs OSCAR INT2 KV (LM_RATIO=1.16), single run per benchmark.

| Benchmark | BF16 | OSCAR (LM_RATIO=1.16) | Δ |
|---|---|---|---|
| GPQA-Diamond | 0.7828 | **0.7929** | +1.0 pp |
| HumanEval | 0.8817 | **0.8854** | +0.4 pp |
| AIME 2025 | 0.7667 | **0.7667** | 0.0 pp |
| MATH500 | 0.9379 | **0.9279** | −1.0 pp |

</details>

<details>
<summary><b>MLA models — packed 2-bit latent (GLM-5.2, GLM-5.3, Kimi-K3)</b> </summary>

**What is quantized.** MLA stores one shared latent `c_kv` plus a positional
`k_pe`. OSCAR quantizes **the latent** and leaves **`k_pe` in BF16** — the
opposite way round from how it is easy to read. Per token per layer, at
`kv_lora_rank=512`, `qk_rope_head_dim=64`, group 128:

| buffer | bytes | contents |
|---|---:|---|
| `c_codes` | 128 | the 512-dim latent at **2 bits**, 4 per byte |
| `c_params` | 32 | (scale, zero) fp32 per quantization group |
| `rope_buf` | 128 | `k_pe`, **BF16, never quantized** |
| **total** | **288** | vs BF16's `(512+64)·2 = 1152` → **4.00×** |

`k_pe` staying BF16 is load-bearing: it is the positional half of the MLA key
and 2-bit'ing it destroys the rope term. It is also **44% of the cell**, which
caps this axis — a latent compressed to zero bits would still leave 256 B, i.e.
**4.5× is the ceiling**, and the shipped 4.00× already sits just under it.

**GPQA-Diamond, full 198 questions, `max_tokens=32768`, single seed.** Same
question set and permutations across arms (`Random(0)`), so the rows are paired.

| Model | BF16 | packed 2-bit (4.00×) | Δ |
|---|---:|---:|---:|
| GLM-5.2 | 82.32 | 74.75 | −7.57 pp |
| GLM-5.3 | 82.83 | **77.78** | **−5.05 pp** |

At a **64K** generation budget both models are measured on the same 48-question
subset (`n=48`, `max_tokens=65536`). Kept as its own table because neither the
question count nor the budget matches the one above, and the two are not
comparable — but the rows here are comparable to each other:

| Model | budget | n | BF16 | packed 2-bit (4.00×) | Δ |
|---|---|---:|---:|---:|---:|
| GLM-5.2 | 64K | 48 | 87.50 | 83.33 | −4.17 pp |
| **Kimi-K3** | 64K | 48 | **93.75** | **83.33** | **−10.42 pp** |

The two land on the same 83.33 from very different starting points: K3's BF16 is
6.25 pp higher, and 2 bits takes all of that back and more. Whatever is costing
K3 its 10 points is specific to K3, not to the packed path — GLM-5.2 runs the
identical quantiser for less than half the loss.

GLM-5.3 required **zero model-code changes** — `GlmMoeDsaForCausalLM` is
identical to GLM-5.2's — but rotations do **not** transfer between models and
must be refitted from each model's own `c_kv` dump.

**Ratio / accuracy frontier (GLM-5.2, all end-to-end at n=198).** Every knob
inside 2 bits has been measured; none closes the 4.00× gap, and bit width buys
more accuracy per unit of ratio surrendered than group size does:

| config | ratio | GPQA | vs BF16 |
|---|---:|---:|---:|
| 2-bit g128 (shipped) | 4.00× | 74.75 | −7.57 |
| 2-bit g32 | 3.00× | 77.78 | −4.54 |
| 4-bit g128 | 2.77× | 80.81 | −1.51 |
| BF16 | — | 82.32 | — |

**Where K3's loss actually sits.** Adding stock upstream to the same 48
questions separates the quantiser from this fork:

| arm | KV | GPQA |
|---|---|---:|
| stock upstream sglang | BF16 | 95.83 |
| this fork, no quantiser | BF16 | **93.75** |
| this fork, packed 2-bit | 4.00× | **83.33** |

So the serving path costs **2.1 pp** — one question out of 48, inside the noise
of a single 48-question run — and 2-bit costs **10.4 pp**. Reaching 90 is a
bit-width question, not a bug hunt.

K3 pays more for 2 bits than the GLM models do (−4.2 pp for GLM-5.2 at the same
n=48/64K, −5.1 pp for GLM-5.3 at n=198/32K). It is a KDA/MLA hybrid where only
24 layers carry a latent rotation, so the quantisation error concentrates in
far fewer layers instead of being spread across all of them.

Both figures are single runs at n=48; the standard error on each is about
±5 pp, so the 10.4 pp separation is real but its decimals are not.

That gap prompted an attempt to replace K3's model file with upstream's
wholesale. The attempt is recorded here because it failed instructively: the
ported file served but produced word salad, and the next 38 commits went to
repairing its fallout. Two of those repairs were genuine defects in shared
code and were kept — a use-after-free on the KDA decode metadata, and a
`gate_up_interleaved` that was threaded through FusedMoE and never read.
Neither was K3's problem. K3's problem was the replacement. The model line is
back on the file that scored 70.83.

Worth knowing before trying again: K3's chat format is Python
(`encoding_k3.py`), not a jinja template, and its expert weights are
compressed-tensors `mxfp4-pack-quantized` with only the routed experts
quantized. Its `--reasoning-parser kimi_k3` does not exist in this fork, so
thinking is not split out of `content` and any length comparison against
upstream is not like-for-like.

</details>

<details>
<summary><b>Multi-Modal & LongBench</b> </summary>
Use Rotation and Run Script in zhongzhu/VL branch. Baseline numbers taken from arxiv.org/abs/2605.19660 (Su et al., 2026).

OCRBench comparison
| Method                          | Qwen3-VL-8B | Qwen3-VL-4B |
|---------------------------------|------------:|------------:|
| 16-bit Baseline                 | 858         | 852         |
| QuaRot (INT2)                   | 722         | 773         |
| RotateKV (INT2)                 | 754         | 638         |
| KIVI (INT2)                     | 851         | 813         |
| OTT (INT2)                      | 850         | 831         |
| TurboQuant+ (2.5-bit)           | 847         | 828         |
| **OSCAR (Lloyd-Max)**           | **854**     | **848**     |

Omni-Modal LLMs: MMAU-Pro

| Method (Qwen3-Omni-30B-A3B) | Open-ended | Good Rate | AIF |
|:---------------------------|:----------:|:---------:|:---:|
| 16-bit Baseline | 66.2 | 27.8 | 87.4 |
| KIVI (INT2) | 65.8 | 27.0 | 78.2 |
| OTT (INT2) | 65.8 | 26.9 | 83.9 |
| TurboQuant+ (2.5-bit) | 66.6 | 27.0 | 79.3 |
| **OSCAR** | **67.4** | **33.8** | **89.7** |

LongBench-E comparison

| Method                          | Qwen3-8B    |
|---------------------------------|------------:|
| 16-bit Baseline                 | 49.56       |
| QuaRot (INT2)                   | 40.13       |
| RotateKV (INT2)                 | 42.95       |
| KIVI (INT2)                     | 47.95       |
| OTT (INT2)                      | 48.21       |
| TurboQuant+ (2.5-bit)           | 47.56       |
| **OSCAR**                       | **50.25**   |
</details>

**Setup.** Each cell is the **MEAN across 5 reasoning / coding benchmarks** — **GPQA**, **HumanEval**, **LiveCodeBench v6**, **AIME 25**, **MATH-500**. To control single-seed variance, **every benchmark is evaluated 5 times per (model, method) cell** (3 times for GLM-4.7-FP8) and the per-seed scores are averaged before being averaged across benchmarks. TurboQuant rows are single-run (\*) because its vLLM path is too slow for repeated 32K-context evaluations under our compute budget. All runs use **32K-token max generation length**. **BPE** = effective bits per KV element at 128K context length. Higher is better; the BF16 row is the upper bound.

| Method | BPE | Qwen3-4B&nbsp;Thinking | Qwen3-8B | Qwen3-32B | GLM-4.7-FP8&nbsp;(358B) |
|:---|:---:|:---:|:---:|:---:|:---:|
| BF16 (upper bound) | 16.00 | 75.64 | 70.84 | 74.19 | 77.89 |
| Saw-INT4 | 4.25 | 73.11 | 69.97 | 74.43 | 77.95 |
| TurboQuant K3V3 \* | 3.25 | 31.74 | 56.88 | 71.99 | 78.15 |
| QuaRot-INT2 | 2.25 | 1.40 | 10.14 | 7.90 | 75.14 |
| Naive INT2 | 2.25 | 0.00 | 0.00 | 0.00 | 60.49 |
| **OSCAR (ours)** | **2.28** | **71.86** | **69.42** | **74.17** | **78.16** |
| _Gap of OSCAR vs BF16_ | | _−3.78_ | _−1.42_ | _−0.02_ | _+0.27_ |

<details>
<summary><b>Details for each task </b> </summary>
<img width="1404" height="1052" alt="image" src="materials/detail_table.png" />
</details>

<details>
<summary><b>Baseline notes</b> — TurboQuant / QuaRot / Saw-INT4 / Naive INT2 configurations</summary>

For a fair comparison at a comparable bit-budget, **TurboQuant** results use
vLLM's implementation
([docs](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/quantization/turboquant/))
modified so that **all layers are quantized** (no mixed precision); the
original TurboQuant keeps the first, last, and selected middle layers in
full precision. We run it in its **K3V3** configuration (3-bit K, 3-bit V)
to land near the OSCAR bit-budget.

**QuaRot-INT2** is the standard 2-bit KV-quant recipe (data-free Hadamard
rotation per layer). **Saw-INT4** is an INT4 reference for context.
**Naive INT2** is per-token symmetric INT2 with no rotation.

\* TurboQuant entries are single-run results because its vLLM path is too
slow for repeated 32K-context evaluations under our compute budget.

</details>

<details>
<summary><b>Comparison with other INT2 KV-cache methods on AIME25</b></summary>

Most prior INT2 KV-cache methods do not provide framework-level support for
efficient long-context generation, so 32K-generation evaluations are extremely
slow and their papers do not report the full benchmark suite above. For this
reason, we compare against the reported AIME25 setting where public numbers are
available.

| Method | BPE | Qwen3-8B | Qwen3-32B |
|:---|:---:|:---:|:---:|
| Original BF16 | 16.00 | 66.00 +/- 7.33 | 72.59 +/- 7.41 |
| KIVI-KV2 | 2.25 | 52.33 +/- 9.00 | 57.41 +/- 9.26 |
| KIVI-KV2* | 2.26 | 57.67 +/- 9.00 | 59.05 +/- 12.38 |
| Kitty | 2.39 | 59.67 +/- 10.33 | 69.26 +/- 9.26 |
| **OSCAR (ours)** | **2.38** | **66.67 +/- 3.33** | **74.00 +/- 5.48** |

OSCAR is the only INT2 method in this comparison that reaches BF16-level AIME25
accuracy at 32K generation while staying near a 2-bit KV-cache budget.

</details>
OSCAR is the only INT2 method that stays within a few pp of BF16 across
every model. QuaRot-INT2 and naive INT2 collapse on reasoning + coding
tasks. Saw-INT4 is a strong INT4 reference, but OSCAR matches or beats it
**at roughly half the storage** (≈2 bits per KV element).

## Layout

```
rotation/
  eval_oscar_gpqa.sh        generic GPQA eval driver
  eval_oscar_lcb.sh         generic LiveCodeBench v6 (128K) eval driver
  compute_kv_rotation.py    eigendecomposition + R·H·P_br composition
  _dump_compat/             sgl_kernel compat shim for dump
  <model>/
    save_qkv_<model>.sh     phase 1 — dump
    compute_rotation.sh     phase 2 — rotation
    eval_gpqa.sh            phase 3 — GPQA eval
    eval_lcb.sh             phase 3 — LCB v6 (128K) eval (where applicable)
    GPQA/
      seq<T>_prompt<N>_group<G>/
        qkv_dumps/          dump output
        rotations/          rotation .pt files
        _eval_gpqa_oscar/   eval results from this rotation
        _eval_lcb_v6_128k/  ...

python/sglang/              sglang (upstream main + the OSCAR low-bit KV cache) — INT2 KV eval
sglang-dump-qkv/            vendored older sglang fork — QKV dump (loaded via shim)
third_party/simple_evals/   git submodule — eval harness (needs git clone --recursive)
```

## Setup

The repository root is upstream SGLang `main` (currently commit `67eab57057`)
with the OSCAR low-bit KV cache added under `python/sglang/`. It is served from
the official `lmsysorg/sglang` nightly image built from that same commit
(`nightly-dev-20261002-67eab570`), with this tree laid over the image's editable
`sglang` install, so there
is no OSCAR-specific environment to assemble: `docker/Dockerfile.oscar` is the
whole recipe. The QKV-dump fork `sglang-dump-qkv/` ships in the repo for
calibration.

### Requirements

- NVIDIA GPUs with the memory the model needs: 1 × 80 GB for the 4B/8B models,
  4 for Qwen3-32B / MiniMax-M2.7 / Gemma-4-12B, 8 for GLM-5.2 / GLM-5.3 /
  MiniMax-M3, 16 across two nodes for Kimi-K3. The verification sweep below
  runs on B200.
- Docker with the NVIDIA container runtime. The base image is the official
  `lmsysorg/sglang` nightly the tree is merged to (CUDA 13.0, torch 2.13,
  transformers 5.17, flashinfer 0.7, sglang-kernel 0.4.8); nothing is
  installed on the host.
- HuggingFace access for the model weights, and the pre-fit rotations from the
  [RotationZoo](https://huggingface.co/Zhongzhu/OSCAR-RotationZoo).

### Clone

```bash
git clone --recursive https://github.com/FutureMLS-Lab/OSCAR.git
cd OSCAR
```

`--recursive` matters: `third_party/simple_evals` is the GPQA scorer, and the
eval driver stops with an explicit error when it is missing instead of scoring
a harness failure as a model result.

### Build the image

The build context holds this tree as `./oscar-src` and the rotations as
`./rotations`; `git archive` leaves submodules out, so the scorer is archived
separately.

```bash
mkdir -p ctx/oscar-src
git archive --format=tar HEAD | tar -x -C ctx/oscar-src
git -C third_party/simple_evals archive --format=tar HEAD \
  | tar -x -C ctx/oscar-src/third_party/simple_evals
hf download Zhongzhu/OSCAR-RotationZoo --local-dir ctx/rotations/zoo
docker build -f docker/Dockerfile.oscar \
  --build-arg IMAGE_TAG="$(git rev-parse --short HEAD)" \
  -t oscar-env:local ctx
```

The build asserts that `import sglang` inside the image resolves to
`/sgl-workspace/sglang/python`, i.e. to this tree, and that the OSCAR modules
parse. GPU kernels are Triton and compile at first use, so the build needs no
GPU.

### Run

```bash
docker run --gpus all --rm -it --shm-size 32g \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -e HF_TOKEN \
  oscar-env:local bash

oscar-selfcheck                                        # tag, commit, torch + device
bash /oscar/src/rotation/run/qwen3-8b.sh               # GPQA under INT2 (default)
KV_MODE=bf16 bash /oscar/src/rotation/run/qwen3-8b.sh  # the paired BF16 control
bash /oscar/src/rotation/verify/all.sh qwen3-8b        # PASS/FAIL smoke, radix + graph on
```

To serve a model directly rather than through a recipe, see
[Serving with the rotation](#serving-with-the-rotation).

### Without Docker

Any environment that runs upstream SGLang at this commit runs OSCAR:
`pip install -e python` replaces the `sglang` package with this tree. Match the
sgl-kernel and flashinfer builds the official image pins for this version;
OSCAR adds no compiled extension of its own.

## Quick start (Qwen3-8B example)

End-to-end on a single H100, ~20 minutes total.

```bash
cd OSCAR

# Phase 1 — dump Q/K/V (TP=1, default DUMP_KVCACHE_TOKENS=30000)
bash rotation/qwen3-8B/save_qkv_8b.sh
# → writes rotation/qwen3-8B/GPQA/seq30000_prompt<N>_group128/qkv_dumps/

# Phase 2 — fit the calibrated rotation
bash rotation/qwen3-8B/compute_rotation.sh
# → writes rotation/qwen3-8B/GPQA/seq30000_prompt<N>_group128/rotations/{k,v}_rotation_qqt_r_h_pbr.pt

# Phase 3 — GPQA eval against the rotation we just produced
ROT_DIR=rotation/qwen3-8B/GPQA/seq30000_prompt<N>_group128/rotations \
  bash rotation/qwen3-8B/eval_gpqa.sh
# → writes results to rotation/qwen3-8B/GPQA/seq30000_prompt<N>_group128/_eval_gpqa_oscar/
```

Pick the actual `seq...prompt..._group...` tag printed by phase 1, or:

```bash
ROT_DIR=$(ls -1d rotation/qwen3-8B/GPQA/seq*_prompt*_group*/rotations | tail -1) \
  bash rotation/qwen3-8B/eval_gpqa.sh
```

## The image

**One image serves every model**; there is no per-model tag to pick. Model
weights are deliberately not included: they are ~2 TB and re-download at tens
of GB/min, whereas the environment is the part that is slow and fragile to
rebuild.

| path in the image | contents |
|---|---|
| `/sgl-workspace/sglang` (also `/oscar/src`) | this tree; the image's editable `sglang` install resolves here |
| `/oscar/rotations/zoo/<model>/` | per-head K/V rotations from the RotationZoo |
| `/oscar/rotations/glm52-rotations/`, `glm53-rotations/`, `k3_latent_rot/` | per-layer MLA latent rotation sets for GLM-5.2, GLM-5.3 and Kimi-K3 |
| `/oscar/IMAGE_TAG`, `/oscar/BUILT_FROM` | the tag, and the branch + commit the tree came from, so a number from a cluster can be matched to the tree that produced it |
| `/usr/local/bin/oscar-selfcheck` | prints tag and commit, torch and the CUDA device, and `OSCAR_SELFCHECK_OK` once `import sglang` resolves to this tree |

Run `oscar-selfcheck` on a GPU node before trusting a long job to an image: it
is the one check that exercises the driver, which a build host without a GPU
cannot.

### One command per model

Each `rotation/run/<model>.sh` is a thin wrapper over `eval_oscar_gpqa.sh`
carrying only that model's recipe — the checkpoint id, the rotation set, the
sink/recent window, the parallelism. Nothing re-implements the launch path, and
the model id is baked in, so a GPQA run is one command with no arguments:

```bash
bash /oscar/src/rotation/run/qwen3-8b.sh                 # INT2 (default)
KV_MODE=bf16 bash /oscar/src/rotation/run/qwen3-8b.sh    # the paired control
bash /oscar/src/rotation/verify/all.sh                   # the PASS/FAIL sweep
```

The recipes differ in ways that matter (per-head rotation for Qwen3-30B-A3B,
Lloyd-Max codebook, group size 256 for Qwen3.5, packed 2-bit latent for the MLA
models, the Mamba radix-cache strategy for the hybrids); `rotation/examples/`
tabulates them.

## Model support

Every model below runs from this tree with `--kv-cache-dtype int2` and the
per-model recipe in `rotation/run/<model>.sh`; there are no feature branches.
Per-head K/V models use the mixed INT2 pool with BF16 sink/recent windows;
MLA models (GLM-5.2, GLM-5.3, Kimi-K3) store the packed 2-bit latent
(288 B/token/layer, 4.00× against BF16) and are served through upstream's own
sparse attention (DSA for GLM, `trtllm_mla` for Kimi-K3); MiniMax-M3 runs its
native MSA sparse attention with the INT2 rows staged into BF16 per layer.

### Garbling sweep (one image, radix cache and CUDA graphs on)

`rotation/verify/all.sh` serves every supported model from a single image with
radix cache and CUDA graphs **on**, sends the probe through the chat template,
and judges the output on four shape checks rather than a letter ratio (a
letter-ratio judge passed four error strings). `PASS` means *did not collapse*,
not *answered correctly*: a fluent, off-task answer passes. Verdicts append to
a file on the volume after each model, so a pre-empted pod costs time and not
results.

The probe goes through the chat template on purpose. Gemma-4-12B-it on a raw
`/generate` prompt echoes the prompt tail (" Answer in. Answer in.") in HF
transformers exactly as in sglang, because its tokenizer under transformers 5.12
prepends no `<bos>` to raw text; the template carries the token and the model
answers. A probe that bypasses the model's real input path measures the probe.

A third probe checks that the prompt is still readable after a long
generation, judged on what the model says *after* its reasoning: a four-option
question whose options are ~35 words each, placed after ~130 tokens of filler
so they sit in the BF16 recent window at prefill and are demoted to int2 slots
by the decode flush; at least 700 words of reasoning are demanded, then the
letter and the chosen option copied verbatim on the last line. A copy that does
not match is `FAIL(no-recall)`. The shape matters: on the MiniMax-M3 pool
before its indexer-key fix (56.6 on GPQA against a 90.9 BF16 control) four
code-word variants of this probe all passed, because a thinking model restates
a short code word in the first lines of its reasoning and later recalls its
own text; the verbatim-quote variant missed on that pool and recalls on the
fixed one.

| model | smoke on this tree (base `67eab57057`) |
|---|---|
| Qwen3-4B-Thinking-2507, Qwen3-8B, Qwen3-32B, Qwen3-30B-A3B | PASS |
| Qwen3.5-4B, Qwen3.5-35B-A3B (hybrid GatedDeltaNet) | PASS |
| Gemma-4-12B-it (hybrid SWA, dual head_dim) | PASS |
| MiniMax-M2.7 | PASS |
| GLM-5.2-FP8, GLM-5.3 (DSA + packed 2-bit latent) | PASS (92 / 90 graph shapes captured, prefix hits logged) |
| MiniMax-M3 (MSA + INT2 staging) | PASS (91 graph shapes, prefix hits logged) |
| Kimi-K3 (TP 8 × PP 2, packed 2-bit latent) | PASS (probed inside its two-node GPQA job: 173 graph shapes, 1,856 prefix tokens hit, packed latent 288 B/token/layer) |

**Kimi-K3 is verified separately, not by this sweep.** It needs 16 GPUs across
two nodes (tp 8 × pp 2, 1.4 TB of MXFP4 weights) and the sweep is one pod with
eight, so its row could only ever report `FAIL(no-serve)` — which reads as a
broken model and means a harness that cannot host it. K3 runs from
`rotation/run/kimi-k3.sh` as a two-node job.

## All configured models

Calibration-pipeline folders included on this branch (`rotation/<model>/`):

| Folder | HF model | TP (dump) | TP (eval) | Notes |
|---|---|---|---|---|
| `rotation/qwen3-4B-thinking-2507/` | `Qwen/Qwen3-4B-Thinking-2507` | 1 | 1 | thinking model |
| `rotation/qwen3-8B/` | `Qwen/Qwen3-8B` | 1 | 1 | |
| `rotation/qwen3-32B/` | `Qwen/Qwen3-32B` | 2-4 | 4 | |
| `rotation/GLM-4.7/` | `zai-org/GLM-4.7-FP8` | 8 | 8 | FP8 weights, 92 layers |
| `rotation/gemma-4-12B-it/` | `google/gemma-4-12B-it` | 1 | 1 | `gemma4_unified` hybrid-SWA, dual head_dim (sliding 8×256 / full 1×512), all INT2; optional vision via `--enable-multimodal` |

### Per-model GPQA: BF16 vs OSCAR INT2 (this tree)

The table is populated only from runs on **this tree at its current upstream
base** (`67eab57057`, nightly image `nightly-dev-20261002-67eab570`); numbers
from earlier bases are not carried over. Both arms of a row share one launch
path (`rotation/run/<model>.sh`) and differ only in `KV_MODE`; GPQA-Diamond is
single-seed unless the cell lists its seeds, all 198 questions, at a 64K generation budget with radix cache
and CUDA graphs on. All thirteen rows below are complete on this base.

**The INT2 arm runs the model's own attention.** GLM-5.2/5.3 run upstream's
DSA sparse attention (flashmla_sparse prefill, trtllm sparse decode) with the
packed 2-bit latent staged per layer; MiniMax-M3 runs upstream's MSA sparse
top-k (indexer on the real token table, INT2 rows dequantized per layer for the
same sparse kernels); Kimi-K3 runs MLA + KDA across two nodes; the rest are
dense GQA. The BF16 control uses the same backend in every row.

| Model | INT2 attention path | n / budget | GPQA (BF16) | GPQA (OSCAR INT2) | Δ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| `Qwen/Qwen3-4B-Thinking-2507` | dense GQA | 198 / 64K | 63.6 | 64.6 | +1.0 |
| `Qwen/Qwen3-8B` | dense GQA | 198 / 64K | 58.6 (2 seeds: 60.6, 56.6) | 52.8 (4 seeds: 50.0, 52.5, 56.1, 52.5) | −5.8 |
| `Qwen/Qwen3-32B` | dense GQA | 198 / 64K | 64.1 | 59.6 | −4.5 |
| `Qwen/Qwen3-30B-A3B` | dense GQA, per-head rotation | 198 / 64K | 61.1 | 55.3 (2 seeds: 55.6, 55.1) | −5.8 (INT2 answers run longer: median 49K vs 32K chars, 12 of 198 hit the budget without a final answer vs 0; same shape as on the previous base) |
| `Qwen/Qwen3.5-4B` | hybrid GDN + GQA | 198 / 64K | 79.3 | 75.3 | −4.0 |
| `Qwen/Qwen3.5-35B-A3B` | hybrid GDN + GQA | 198 / 64K | 81.8 | 83.8 | +2.0 |
| `google/gemma-4-12B-it` | hybrid SWA, two geometries | 198 / 64K | 63.1 | 64.1 | +1.0 |
| `MiniMaxAI/MiniMax-M2.7` | dense GQA | 198 / 64K | 86.9 | 87.9 | +1.0 |
| `MiniMaxAI/MiniMax-M3` | MSA sparse top-k (upstream backend) | 198 / 64K | 90.9 | 88.1 (2 seeds: 88.9, 87.4) | −2.8 (after the indexer-key fix below; 56.6 before it) |
| `zai-org/GLM-5.2-FP8` | DSA sparse (upstream backend), packed latent 4.00× | 198 / 64K | 87.4 | 83.8 | −3.5 (INT2 answers run longer: median 63K vs 32K chars, 24 vs 9 without a final answer) |
| `zai-org/GLM-5.3` | DSA sparse (upstream backend), packed latent 4.00× | 198 / 64K | 87.4 | 84.3 | −3.0 (INT2 answers run longer: median 51K vs 31K chars, 21 vs 8 without a final answer) |
| `zai-org/GLM-4.7-FP8` | dense GQA | 198 / 64K | 80.8 | 78.8 | −2.0 |
| `moonshotai/Kimi-K3` | MLA latent + KDA, packed latent 4.00×, TP 8 × PP 2 | 198 / 64K | 90.9 | 90.4 |  −0.5 (re-measured with the BF16 windows on; the earlier 93.4 came from a run whose packed pool had them silently off) |

**MiniMax-M3 scored 56.6 on the first INT2 run of this base.** The BF16 control
(90.9, same MSA backend) ruled out noise: 47 of 198 INT2 answers had no final
letter and 12 ended by asking for the question after pages of correct reasoning.
The cause was in the INT2 pool, not the kernels: the decode flush demotes
recent-window rows into int2 slots and remaps `req_to_token`, but the lightning
indexer's key rows written at the window slots did not move with them, so after
a few hundred generated tokens the indexer scored every flushed token --
including the question beyond the 64-token BF16 prefix -- against zeros. The
pool now moves those rows in the same step (`on_flush_applied`); the re-run on
the fixed tree scores 88.9 (8/12 discordant against BF16, p = 0.5) with 6
unanswered questions against BF16's 5. The smoke suite gained the verbatim-quote
retention probe described above because every fluency probe had passed on the broken pool.

### 64K decode on B200 (this tree)

Batch size 1, a 65,536-token prompt followed by 512 generated tokens, median
of three repeats; measured on the same base as the GPQA sweep, all thirteen
rows complete. The nine dense rows were then re-measured on the current image
(v88) with the decode-speed knobs below on, one pod per model with all three
arms in that pod; GLM-5.x, MiniMax-M3 and Kimi-K3 run sparse or MLA paths the
knobs do not touch and keep their earlier same-pod numbers. The BF16
column is the same model on the triton backend (what OSCAR's INT2 kernels
replace); the FlashInfer column is the fastest BF16 baseline that serves the
model (for GLM that is upstream's DSA path, for Gemma-4 `trtllm_mha`, the only
FlashInfer-family backend its model file accepts). A GPU keep-alive that the
cluster's idle-pod reaper requires during weight loads is paused while the
server answers `/v1/models`, so no foreign kernel runs during a measurement.
Bench pods get 12 CPU cores per GPU: a 16-core pod throttled a TP 8 model by
~19% in both arms (ratio unchanged), so TP 8 rows come from 96-core pods.

| Model | INT2 ms/tok | BF16 triton ms/tok | INT2 vs BF16 triton | BF16 FlashInfer-family ms/tok | INT2 vs FlashInfer |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen3-4B-Thinking-2507 | 6.10 | 40.09 | 6.57× faster | 4.81 | 1.27× slower |
| Qwen3-8B | 7.23 | 41.23 | 5.71× faster | 5.91 | 1.22× slower |
| Qwen3-32B | 11.23 | 61.17 | 5.45× faster | 9.38 | 1.20× slower |
| Qwen3-30B-A3B | 5.08 | 42.60 | 8.39× faster | 3.62 | 1.40× slower |
| Qwen3.5-4B | 4.09 | 13.21 | 3.23× faster | 3.26 | 1.26× slower |
| Qwen3.5-35B-A3B | 4.08 | 16.18 | 3.97× faster | 3.08 | 1.32× slower |
| Gemma-4-12B-it | 9.08 | 18.43 | 2.03× faster | 6.01 (trtllm_mha) | 1.51× slower |
| MiniMax-M2.7 | 8.85 | 58.98 | 6.66× faster | 6.95 | 1.27× slower |
| GLM-4.7-FP8 | 11.41 | 74.58 | 6.54× faster | 9.71 | 1.18× slower |
| GLM-5.2-FP8 | 11.99 | 8.81 (DSA, same backend both arms) | 1.36× slower | same DSA path (8.81) | 1.36× slower |
| GLM-5.3 | 11.99 | 8.82 (DSA, same backend both arms) | 1.36× slower | same DSA path (8.81) | 1.36× slower |
| MiniMax-M3 | 14.98 | 11.19 (MSA sparse, same backend both arms) | 1.34× slower | 8.67 | 1.73× slower |
| Kimi-K3 (TP 8 × PP 2) | 28.43 | 21.44 (triton MLA) | 1.33× slower | 20.27 (trtllm_mla) | 1.40× slower |

> Every model's serving recipe (rotation set, windows, codebook, parallelism) is `rotation/run/<model>.sh` on this tree; pre-fit rotations for all of them are on the [RotationZoo](https://huggingface.co/Zhongzhu/OSCAR-RotationZoo).

#### Decode-speed knobs (dense INT2 path)

These settings decide how the mixed HP+INT2 decode kernel is launched. All
are on by default; each can be switched off on its own for an A/B.

| Env | Default | What it changes |
|:---|:---:|:---|
| `SGLANG_INT2_MAX_SPLITS` | `64` | Split ceiling for the INT2 tier. The stage-1 grid is `batch × head-tiles × splits` programs, so the shared cap of 8 leaves most of a B200 idle at batch 1 (the kernel moves ~150 GB/s there: it is short of resident programs, not bandwidth). Measured at 64K / bs=1 on Qwen3-8B (B200): 206 µs per layer at 8 splits, 74 at 32, 59 at 64 with the 64-token tile below; FlashInfer BF16 on the same shape is ~56 µs. The per-request count is still adaptive; only the ceiling moves. The HP window keeps `SGLANG_MIXED_KV_HP_MAX_SPLITS`. |
| `SGLANG_OSCAR_FUSED_STAGE1` | `1` | One grid for the HP window and the INT2 tier (`program_id(2)` below the HP split count selects the tier) instead of two launches. Each tier body is the standalone kernel's; `rotation/tests/test_int2_fused_stage1_gpu.py` asserts the partial states are bit-identical to the two-launch path and that the kernel replays inside a CUDA graph. |
| `SGLANG_OSCAR_FAST_ROT` | `1` | Decode `q @ R_k` and `o @ R_vᵀ` through one Triton launch each (`sglang/QuantKernel/oscar_rotate_rows.py`) instead of a cuBLAS GEMM plus its copy kernels; per-KV-head rotations are read in place, with no `repeat_interleave`. bf16 in, fp32 accumulate, bf16 out, held to one bf16 ulp by `rotation/tests/test_rotate_rows_gpu.py`. |
| `SGLANG_INT2_FAST_STAGE2` | `1` | Reduce the split partials with one program per (request, head, 16-wide slice of head_dim) instead of one serial loop per (request, head). The serial kernel's time grows with the split count (6.5 µs at 40 splits, ~21 µs at 72); the parallel one stays under 3 µs. Only when no LSE is requested; held to one bf16 ulp of the serial result. |
| `SGLANG_OSCAR_FAST_METADATA` | `1` | Build the per-step HP/INT2 index lists with one program per 512-token block (three launches) instead of one serial program per request. Runs outside the CUDA graph on the decode critical path: 279 → 94 µs per step at 64K / bs=1. Bit-identical layout, sliding windows included; falls back to the serial build when the one-program prefix tile would not fit. |
| `SGL_INT2_BLOCK_N` / `SGL_INT2_BLOCK_H` / `SGL_INT2_NUM_WARPS` / `SGL_INT2_NUM_STAGES` | shape-dependent | Stage-1 tile. The bs<4 row is `64/8/2/3` (B200-measured; the previous `128/8/4/3` plateaus at 72–75 µs past 32 splits); batch ≥4 rows keep the H100-era settings. |

The dense rows of the table above are measured with all of these on. The
same-pod ablation on Qwen3-8B (64K, bs=1) reads 16.25 → 10.15 (split cap 32)
→ 8.86 (+ fused stage-1, batched rotation) → 8.36 (cap 64 + tile) → 7.50 ms/tok
(+ parallel stage-2), against FlashInfer BF16 at 5.94 in that pod; the parallel
metadata build then took the table row to 7.23 against 5.91.

## How the rotation is fit (spectral covariance)

For each transformer layer, given calibration `(Q, K, V)` activations, OSCAR estimates two attention-aware **covariance** matrices and uses their eigenspectra to derive rotations:

- **K covariance** (`qqt`) — average attention-query covariance seen by K:
  `Σ_K = (1/H_kv) · Σ_h Q_h^T Q_h / n_tokens` (GQA-aware: query heads grouped under the matching KV head)
- **V covariance** (`sst`) — score-weighted V-side covariance:
  `Σ_V = (1/H_kv) · Σ_h V_h^T diag(w_h) V_h / n_tokens` where `w_h[t] = K_h[t] · (Q^T Q) · K_h[t]^T` is the per-token attention-score weight derived from K and the Q covariance
- `torch.linalg.eigh(Σ)` → orthogonal eigenvectors `R` plus the eigenvalues (used for ordering, not for scaling)
- Composition `r_h_pbr`: `R_loaded = R · H_d · P_br`
  - `H_d` — head-dim Hadamard
  - `P_br` — bit-reversal permutation, sorted by eigenvalue magnitude; this interleaves high-variance directions evenly across quant groups so no single group concentrates outliers

Saved as fp32 per-layer `(head_dim, head_dim)` orthogonal matrices in
`<calib_dir>/rotations/{k,v}_rotation_qqt_r_h_pbr.pt`.

## Serving with the rotation

The eval driver `eval_oscar_gpqa.sh` and `eval_oscar_lcb.sh` set everything for you. The underlying sglang server flags are:

```bash
SGLANG_ENABLE_MIXED_KV_WINDOWS=1 \
SGLANG_OSCAR_K_ROTATION_PATH=.../k_rotation_qqt_r_h_pbr.pt \
SGLANG_OSCAR_V_ROTATION_PATH=.../v_rotation_sst_r_h_pbr.pt \
SGLANG_OSCAR_K_CLIP_RATIO=0.96 \
SGLANG_OSCAR_V_CLIP_RATIO=0.92 \
SGLANG_OSCAR_ABSORB_V_ROTATION=1 \
SGLANG_MIXED_KV_PREFIX_TOKENS=64 \
SGLANG_MIXED_KV_RECENT_TOKENS=256 \
SGLANG_MIXED_KV_HP_MAX_SPLITS=8 \
SGLANG_MIXED_KV_HP_DTYPE=bfloat16 \
SGLANG_MIXED_KV_SCALE_DTYPE=float32 \
python -m sglang.launch_server \
  --model-path <model> \
  --tensor-parallel-size <tp> \
  --kv-cache-dtype int2 \
  --kv-cache-quant-group-size 128 \
  --prefill-attention-backend fa3 \
  --decode-attention-backend triton \
  --trust-remote-code
```

Sink (`PREFIX_TOKENS`) and recent window (`RECENT_TOKENS`) stay in BF16; the rest of the cache is INT2-quantized into 128-element groups along head-dim.

### Skip calibration — use a pre-fit rotation from RotationZoo

To serve without running phases 1–2, download a calibrated rotation from the [RotationZoo](https://huggingface.co/Zhongzhu/OSCAR-RotationZoo) and point the env vars at it:

```bash
huggingface-cli download Zhongzhu/OSCAR-RotationZoo --include "Qwen3-8B/*" --local-dir rotzoo
ROT=$(ls -1d rotzoo/Qwen3-8B/seq*_prompt*_group128 | tail -1)
export SGLANG_OSCAR_K_ROTATION_PATH=$ROT/k_rotation_qqt_r_h_pbr.pt
export SGLANG_OSCAR_V_ROTATION_PATH=$ROT/v_rotation_sst_r_h_pbr.pt
```

### Fit the rotation at server startup (no offline calibration)

When the unified mixed pool is active and the rotation pair is unset, or the
configured files are missing, the server fits the pair itself before it
accepts traffic: the pool starts on identity rotations, a prefill-only pass
over calibration prompts (GPQA-Diamond by default, downloaded once) collects
the `qqt` / `sst` moments through the attention layers, the scheduler reduces
them over TP and decomposes them exactly as the offline pipeline does, and the
rotations are installed in place (captured decode CUDA graphs stay valid).
The pair is written atomically under a lock and later launches load it.

```bash
# nothing to download: the pair lands in $HF_HOME/oscar-rotations/<model>-<digest>/
SGLANG_ENABLE_MIXED_KV_WINDOWS=1 \
python -m sglang.launch_server --model-path Qwen/Qwen3-8B --kv-cache-dtype int2 \
  --kv-cache-quant-group-size 128 --attention-backend triton
```

| Env | Default | Effect |
|---|---|---|
| `SGLANG_OSCAR_CALIBRATION_PROMPTS_PATH` | `` | JSONL of `{"messages": [...]}` prompts (or an original GPQA CSV); empty = GPQA-Diamond |
| `SGLANG_OSCAR_CALIBRATION_TOKENS` | 30000 | Prompt tokens observed per layer |
| `SGLANG_OSCAR_CALIBRATION_BATCH_SIZE` | 32 | Prompts per calibration request |
| `SGLANG_OSCAR_CALIBRATION_TIMEOUT` | 1800 | Seconds before the launch fails |
| `SGLANG_OSCAR_CALIBRATION_LOCK_DIR` | `/tmp/sglang-oscar-locks` | Lock files that serialize pair publication |

Single node, DP=PP=1, no HiCache, no torch.compile, no explicit prefill CUDA
graph backend; the first launch also keeps the V rotation at runtime (no
`SGLANG_OSCAR_ABSORB_V_ROTATION`). The smoke harness exercises it with
`CALIBRATE_DIR=<dir> bash rotation/verify/all.sh qwen3-8b`.

## The OSCAR-2 transform family (per-head, centering, non-orthogonal keys, output-aware values)

The rotation checkpoints can carry three optional companions of `rotation`;
each defaults to the plain orthogonal behaviour when absent, so V1 files and
calibrating launches are unchanged:

| Field (file) | Meaning | Where it acts |
|:---|:---|:---|
| `q_rotation` (K) | query-side matrix `R_k⁻ᵀ` of a **non-orthogonal** key transform, so `(q·q_rotation)·(k·rotation) = q·k` exactly | prefill q, decode q |
| `o_rotation` (V) | `R_v⁻ᵀ` of a non-orthogonal value transform; the attention output is un-rotated as `o·o_rotationᵀ` | prefill and decode output |
| `k_mean` (K) | per-head key mean subtracted before the key rotation (**centering**) | every stored key (HP window and INT2 tier) and the extend-time keys, so all logits of a request shift by one constant and the softmax is unchanged; nothing on the read side |

Shapes follow `rotation`: `[hd, hd]` (shared) or `[kv_heads, hd, hd]` (per
head); `k_mean` is `[hd]` or `[kv_heads, hd]`. Per-head fields are sharded per
TP rank like the rotations. The fused rotate-clip-quant write kernel does not
center, so it stays off under `k_mean`.

`rotation/tools/fit_oscar2_variants.py` fits every row of the component
ablation from one startup calibration run: launch once with
`SGLANG_OSCAR_CALIBRATION_SAVE_MOMENTS=1` and the calibrator leaves
`oscar_moments_rank<r>.pt` next to the published pair (per layer and KV head:
the query second moment `M_q`, the key sum and second moment, the
energy-weighted value covariance), then

```bash
python rotation/tools/fit_oscar2_variants.py --moments-dir $CAL --out $OUT \
    --variants perhead,center,nova,flat,stretch,outaware --model Qwen/Qwen3-8B
```

| Variant | Key transform | Values |
|:---|:---|:---|
| `perhead` | orthogonal per KV head, `E_q H P_br` | orthogonal, `E_v H P_br` |
| `center` | `perhead` + `k_mean` | same |
| `nova` | centered, NOVA compact basis `M_q^{1/2} E`, query side `M_q^{-1/2} E` | same |
| `flat` | centered, `M_q^{1/2} E H P_br` (flattened spectrum) | same |
| `stretch` | centered, fixed-rate stretch `X*^{1/2} E* H P_br` | same |
| `outaware` | `--k-base` (default `flat`) | post-`W_O` metric `G^{1/2} E_v H P_br`, `G = Σ W_{O,j}ᵀ W_{O,j}` pooled over the query heads reading the KV head; ships `o_rotation` |

Each variant lands in its own directory as `k_rotation_oscar2_<v>.pt` /
`v_rotation_oscar2_<v>.pt`, which the run recipes discover
(`ROT_DIR=<dir> rotation/run/qwen3-8b.sh`). `--shared` averages the per-head
moments into one basis per layer. The fitter checks on every file that
`q'·k' = q·k` and that `o_rotation` inverts the value transform;
`rotation/tests/test_oscar2_transforms.py` runs quantized decode under a fully
non-orthogonal centered pair against dense attention.
`rotation/_eval_runner/ppl_longctx.py` scores teacher-forced NLL over 32K
WikiText-2 windows past the BF16 recent window, the low-noise metric used to
rank the variants before GPQA.

Measured on Qwen3-8B (every row fitted from one startup-calibration pass;
PPL = nine 32,768-token windows of WikiText-2 test, tokens at positions
≥ 4096 scored through the real serving path; GPQA-Diamond at a 64K budget,
one seed, same pod; recipe windows 128/2048, Lloyd-Max, clip .96/.92):

| Row | Keys | Values | PPL@32K | vs BF16 | GPQA-198 |
|:---|:---|:---|---:|---:|---:|
| BF16 | — | — | 7.980 | — | 58.6 (2 seeds) |
| V1 shared zoo rotation | `U_Q H P_br` | `E_v H P_br` | 8.164 | +2.30% | 48.0 (this pod; 51.0 over 5 seeds) |
| startup-calibrated, shared | `E_q H P_br` | same | 8.134 | +1.93% | 51.5 |
| per-head | per-head orthogonal | same | 8.211 | +2.89% | 51.5 |
| **per-head + centering** | + `k_mean` | same | **8.083** | **+1.28%** | **54.0 / 53.0 (2 seeds)** |
| shared + centering | + `k_mean` | same | 8.123 | +1.79% | — |
| NOVA compact basis | `M_q^{1/2} E` | same | 172 | collapse | 39.9 |
| flat compact basis | `M_q^{1/2} E H P_br` | same | 978 | collapse | every answer ran to the cap |
| fixed-rate stretch | `X*^{1/2} E* H P_br` | same | 102 | collapse | 51.5 (short GPQA prompts hide the long-context collapse) |
| output-aware values on stretch keys | stretch | post-`W_O` | 108 | collapse | 52.5 |
| output-aware values on centered keys | per-head + centering | post-`W_O` | 8.083 | +1.29% | 52.0 |

Centering is the one closed-form change that pays under fixed-rate scalar
INT2: it halves the long-context PPL overhead and gives the best GPQA arm.
Per-head alone does not beat the shared basis on this model (eight KV heads
with similar statistics; the per-head case for OSCAR-2 is the heterogeneous
MoE heads). Every non-orthogonal key metric collapses at long context under
scalar quantization, including the fixed-rate stretch that passes the smoke
probe -- a non-orthogonal basis needs a vector quantizer on the read path,
as NOVA-KV found. On orthogonal centered keys the pooled post-`W_O` value
metric is neutral (PPL 8.083 vs 8.083; GPQA 52.0 vs 54.0, McNemar p = 0.63).
Paired on the same 198 questions, centering beats the shared startup basis by
+2.0pp (p = 0.63) and the V1 zoo rotation by +5.6pp (p = 0.08); single-seed
GPQA cannot separate the orthogonal rows, the 32K PPL can.

The collapse of the non-orthogonal rows is the method, not the engine. An
offline replay on real Qwen3-8B post-RoPE queries and keys (one 4096-token
WikiText-2 window, 36 layers x 8 KV heads, the write kernel's arithmetic in
fp64, serving path bypassed; keys older than the 2048-token window quantized)
reproduces the ordering as attention KL against exact attention: per-head
0.136, centered 0.072, stretch 0.123, NOVA 0.56, flat 2.07 nats. The same
NOVA/flat bases with a 2-bit product quantizer (g = 4, 256 centroids) stay at
0.027/0.028, which is why NOVA-KV's basis works with its vector quantizer and
not with a per-row scalar one (its own ablation reports basis + scalar
quantization = 0.0). Storing the transformed rows in bf16 is not a factor
(BF16-window-only arms: KL <= 0.0005). A Gaussian second-order simulation on
the calibration moments ranks the fixed-rate stretch best, as the theory says;
on real keys its logit RMSE is indeed lowest but its errors are heavier-tailed
(spurious INT2 maxima for 20.7% of queries vs 14.3% for centering), and the
softmax pays for the tail. The tail also explains why stretch passes the
smoke probe and GPQA but collapses at 32K: on a 16384-token window (four
times the INT2 keys) the KL grows 1.4x for per-head and 1.8x for centering
but 6.8x for stretch (0.83 nats), 5x for NOVA and 7.5x for flat. Scripts and
logs: `/home/admin/imgctx/diag/oscar2_nonorth/`.

## 1-bit and 1.5-bit K with product quantization

The quant tier's encoder is selectable per tensor. `pq` stores each row as
`n_sub` uint8 codes against a per-layer codebook trained on the rotated
activations (no per-row scale; 16 codes for head_dim 128 = 1.0 bit/value,
physically 16 bytes per row). A codebook file with a second stage adds a
residual code (RVQ): its 16-centroid codes are packed two per byte, so an RVQ
row is 24 bytes = 1.5 bit/value, still without the INT2 scale/zero.
Prefill encode, the decode-time flush, prefix dequant and a graph-safe split-KV
decode (centroids reconstructed inline, or scored through a per-query lookup
table at small batch) all run as Triton kernels; the BF16 sink/recent windows
are unchanged.

```bash
SGLANG_OSCAR_K_QUANTIZER=pq SGLANG_OSCAR_PQ_K_CODEBOOK=$ROT/codebooks/k_pq_n16_c256_d8.pt   # K 1.0 bit, V INT2
SGLANG_OSCAR_K_QUANTIZER=pq SGLANG_OSCAR_PQ_K_CODEBOOK=$ROT/codebooks/k_rvq_n16_c256x16_d8.pt   # K 1.5 bit (RVQ, residual codes nibble-packed)
SGLANG_OSCAR_V_QUANTIZER=pq SGLANG_OSCAR_PQ_V_CODEBOOK=$ROT/codebooks/v_pq_n16_c256_d8.pt   # with K pq: V 1.0 bit too
```

| Env | Default | Effect |
|---|---|---|
| `SGLANG_OSCAR_K_QUANTIZER` / `SGLANG_OSCAR_V_QUANTIZER` | `int2` | `int2` or `pq`; V may be `pq` only when K is |
| `SGLANG_OSCAR_PQ_K_CODEBOOK` / `SGLANG_OSCAR_PQ_V_CODEBOOK` | `` | Codebook file for the tier (`codebooks_per_layer`, or `codebooks_stage1/2` for RVQ) |
| `SGLANG_OSCAR_PQ_USE_ADC` | -1 | Decode scoring: -1 = lookup table when batch < 4, 0 = reconstruct K, 1 = always lookup table |

Codebooks are bound to the rotation they were trained against. Train them from
the same dumps the rotation came from:

```bash
python rotation/tools/train_pq_codebooks.py --dumps $CALIB/qkv_dumps/gpqa \
  --rotation $CALIB/rotations/k_rotation_qqt_r_h_pbr.pt --tensor k --out codebooks/k_pq_n16_c256_d8.pt
python rotation/tools/train_pq_codebooks.py --dumps $CALIB/qkv_dumps/gpqa \
  --rotation $CALIB/rotations/k_rotation_qqt_r_h_pbr.pt --tensor k --stage2-centroids 16 \
  --out codebooks/k_rvq_n16_c256x16_d8.pt
```

Measured on this tree (Qwen3-8B, GPQA-Diamond at 64K, the model's recipe
windows of 128 BF16 prefix / 2048 BF16 recent, one seed, n=198):

| K / V | bits (K) | GPQA |
|:---|:---:|:---:|
| BF16 | 16 | 60.6 / 57.1 (two runs) |
| INT2 / INT2 | 2.0 | 52.5 (v87), 50.0 / 53.5 (earlier runs) |
| PQ K / INT2 V | 1.0 | 36.4 |
| RVQ K / INT2 V | 1.5 | 36.9 |
| PQ K / PQ V | 1.0 | 36.4 |

The three PQ rows serve cleanly (graphs, prefix cache, no exceptions; the
prefix read is checked row by row against the codebook in
`rotation/tests/test_pq_kv_gpu.py`), but the codes themselves carry little:
at 1–1.5 bit the accuracy lives in the BF16 window. An earlier measurement
with a 512-token BF16 prefix, which kept the whole GPQA question in BF16,
scored 57.2 for PQ K; with the 128-token prefix the question is read back
through the codes and the score drops to the mid thirties, and the residual
stage of RVQ buys nothing on top. The smoke's long-context recall probe fails
for all three for the same reason (INT2 passes it). Treat PQ as a memory
lever for contexts long enough to amortize the BF16 window, not as an
accuracy-neutral tier; PQ decode is also slower than INT2 at long context.
PQ is not available on the two-group (Gemma 4), MiniMax-sparse or packed-MLA
pools. The smoke rows `qwen3-8b-kpq`, `qwen3-8b-krvq` and `qwen3-8b-kpq-vpq`
cover the three configurations and report `FAIL(no-recall)` on the probe.

## Calibration knobs

Override per `bash rotation/<model>/save_qkv_<model>.sh ENV=val`:

| Env | Default | Effect |
|---|---|---|
| `DUMP_KVCACHE_TOKENS` | 30000 | Total token budget for calibration |
| `GROUP_SIZE` | 128 | KV quant group size, encoded in output dir name |
| `DATASET` | GPQA | Calibration dataset name |
| `MODEL` | per-model HF id | HuggingFace model id |
| `TP_SIZE` | per-model | Tensor parallel size for dump |
| `GPU` | per-model | CUDA_VISIBLE_DEVICES |
| `HF_HOME` | `$HOME/.cache/huggingface` | HF cache (override to a shared cache if you have one) |

## Troubleshooting

- **Garbled / mixed-language output in long-context agent sessions.** Update to the latest `main`. Older checkouts (before the mixed-KV slot-accounting fix) could free KV slots still referenced by other in-flight requests under concurrency, corrupting reads.
- **`--kv-cache-dtype int2` only supports full-attention models.** MLA models (GLM-5.1 / DeepSeek-style) and hybrid linear-attention models (Qwen3.5 GatedDeltaNet) are not supported on `main` yet — see [Model support](#model-support).
- **Hybrid models (Qwen3.5, Kimi-K3).** Their prefix cache needs `--mamba-radix-cache-strategy extra_buffer` (the recipes set it); without it the Mamba radix cache asserts `page_size == 1`, which surfaces asynchronously as an illegal memory access.
- **Running locally on a Mac.** Use the `zhongzhu/llamacpp` branch or the pre-built `*-rot-kv.gguf` files on Hugging Face.

## Citation

```bibtex
@misc{zhou2026oscarofflinespectralcovarianceaware,
      title={OSCAR: Offline Spectral Covariance-Aware Rotation for 2-bit KV Cache Quantization},
      author={Zhongzhu Zhou and Donglin Zhuang and Jisen Li and Ziyan Chen and Shuaiwen Leon Song and Ben Athiwaratkun and Xiaoxia Wu},
      year={2026},
      eprint={2605.17757},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.17757},
}
```

## License & acknowledgements

- Released under the MIT License.
- Built on top of [sglang](https://github.com/sgl-project/sglang).
