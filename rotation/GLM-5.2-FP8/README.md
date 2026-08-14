# GLM-5.2-FP8 — INT2 KV recipe (MLA shared latent)

MLA stores one compressed latent `c_kv` (512-d) shared by all heads, plus a
positional `k_pe`. Only `c_kv` is quantized; `k_pe` stays BF16. Per-head
rotations do not apply here — there is nothing per-head to rotate.

Rotation is `Rcov · P · Hblock` (covariance eigenvectors, bit-reversal
permutation, per-group Hadamard) with **Lloyd-Max** on, group 128, sink 64 /
recent 256 BF16.

| Benchmark | BF16 | INT2 | Δ |
|---|:---:|:---:|:---:|
| GPQA @32K | 81.3 ± 0.0 | 68.7 ± 1.8 | −12.6 |
| HumanEval (≤16K) | 91.5 ± 1.2 | 88.2 ± 1.5 | −3.3 |
| AIME 2025 | 76.7 ± 3.3 | 75.6 ± 6.9 | −1.1 (n.s.) |
| MATH500 | 94.8 ± 0.8 | 93.8 ± 0.3 | −1.0 |

**The −12.6 at 32K is truncation, not wrong answers.** On questions both arms
answer, they pick the same option 97.1 % of the time; INT2 simply thinks 1.69×
longer, so the truncation rate goes 14 % → 28 %. BF16 itself gains 6.6 points
moving from a 32K to a 64K budget, which confirms the metric is budget-bound.
Give this model a generous `max_tokens` before reading a regression into it.
The INT2 @64K cell is not measured.

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
SGLANG_ENABLE_MIXED_KV_WINDOWS=1 SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
SGLANG_MIXED_KV_PREFIX_TOKENS=64 SGLANG_MIXED_KV_RECENT_TOKENS=256 \
SGLANG_LLOYD_MAX=1 \
SGLANG_OSCAR_MLA_ROTATION_DIR=<rotations-dir> \
python -m sglang.launch_server --model-path zai-org/GLM-5.2-FP8 \
  --tensor-parallel-size 16 --nnodes 2 --node-rank <0|1> \
  --dist-init-addr <head>:20000 \
  --kv-cache-dtype int2 --page-size 8 --disable-radix-cache \
  --mem-fraction-static 0.85
```

Set `SGLANG_OSCAR_MLA_KV_REAL_KERNEL=1` to use the packed-INT2 latent kernel
instead of the fake-quant path: bit-identical packing, decode cosine 1.000000,
1024 → 160 B/token (**6.4×**). It is correct but not yet fast — measure before
enabling it for throughput.
