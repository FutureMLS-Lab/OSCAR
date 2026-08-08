# Can Vera Rubin Accelerate Attention by 30×? An Accuracy Simulation Reveals the Truth

## TL;DR

1. **What does Vera Rubin provide?** Per-GPU HBM4 bandwidth rises to 22 TB/s. Its new LUT-B Tensor Core can use a 3-bit index to select one of eight E4M3 values and decompress it inside the matrix multiplication, without first restoring BF16 data.
2. **How can attention use it?** Long-context decoding rereads historical K/V for every generated token. We store `Kᵀ` and `V` as 3-bit LUT-B operands and use Q and post-softmax P as MXFP8 A operands, allowing Rubin to consume compressed KV while computing both `Q × Kᵀ` and `P × V`.
3. **Why not simply change KV to 3 bits?** Naive uniform INT3 is destroyed by outliers and a small number of sensitive tokens: Qwen3-4B's 8K PPL jumps from 9.468 in BF16 to 1504.444. Preserving accuracy requires offline per-layer Lloyd-Max codebooks; OSCAR rotation/clipping and a BF16 sink/recent window can further protect sensitive values, while Q/P must use Rubin's supported E4M3 + UE8M0 block32 precision.
4. **What does attention look like afterward, and how much faster could it be?** The hardware-aligned K3V3 + Q/P simulation reaches 8K PPL of 10.434–10.465, still close to BF16, while effective KV storage falls to about 3.63 bits/value. Counting bytes alone gives 29–32× H100 BF16, but that ignores the B-only LUT constraint: MXFP8 has an M tile of 128, while Qwen GQA decode can pack only four Q rows per KV head. Including this constraint gives a single-token attention-only ceiling of about **20.4×** H100; at 60–80% execution efficiency, the range is about **12.3–16.4×**. The 30× figure is a byte-only upper bound, not a credible prediction for the current kernel.

## 1. Introduction: What Exactly Changed with Vera Rubin?

### 1.1 From a “faster GPU” to an agentic inference platform

NVIDIA's official narrative for Rubin is agentic AI: longer contexts, more reasoning steps, greater decode interactivity, and rack-scale orchestration. Published specifications for a single Rubin GPU include 224 SMs, 896 Tensor Cores, 288 GB of HBM4, and 22 TB/s of HBM bandwidth; Vera Rubin NVL72 connects 72 GPUs into a single NVLink 6 scale-up domain.

| Public spec | 1× Rubin GPU | Vera Rubin NVL72 |
|---|---:|---:|
| GPU count | 1 | 72 |
| HBM4 capacity | 288 GB | 20.7 TB |
| HBM bandwidth | 22 TB/s | 1,580 TB/s |
| NVFP4 inference | 50 PFLOP/s | 3,600 PFLOP/s |
| FP8/FP6 published peak | 17.5 PFLOP/s | 1,260 PFLOP/s |
| NVLink bandwidth | 3.6 TB/s | 260 TB/s switch bandwidth |

These figures come from [NVIDIA's Rubin GPU architecture overview][nvidia-rubin-gpu] and the [Vera Rubin NVL72 specifications][nvidia-rubin-nvl72]. They are peak specifications, not the sustained performance an application can necessarily achieve.

### 1.2 Why do we care so much about the KV cache?

The community often summarizes inference with one line: **prefill is compute-bound, decode is bandwidth-bound**. The Medium article on [disaggregated inference][medium-disagg] uses this framework to explain why HBM, KV movement, and prefill/decode separation are becoming increasingly important.

In long-context decoding, every generated token requires the historical K/V to be read again. The smaller the model, the longer the context, and the larger the batch, the more likely KV traffic is to become the primary bottleneck rather than a supporting concern. Rubin's 22 TB/s of HBM4 bandwidth therefore matters, but even more interestingly, it also introduces a new ISA that reduces bytes per value.

### 1.3 CUDA 13.4 turns Rubin from slideware into an inspectable software target

[SemiAnalysis / InferenceX][semianalysis-rubin] reported on the first public SM107 software stack: CUDA 13.4, PyTorch, vLLM, and OpenAI Triton have begun adding Rubin support, and the report also disclosed the 3-bit programmable LUT Tensor Core.

We installed CUDA 13.4.46 Developer Preview and used:

- PTX ISA 9.4
- `sm_107a`
- `ptxas`
- `cuobjdump`
- `nvdisasm`

This gives us, at minimum, a way to perform compile-time legality checks. It is important to note that vLLM's [SM107 tracking issue][vllm-sm107] remains open, while Triton has only [initial Rubin support][triton-sm107]; software bring-up is not yet complete.

## 2. The Most Relevant New Feature: 3-bit Programmable LUT-B

### 2.1 What does LUT-B do?

A conventional FP8 B matrix requires 8 bits per value. Rubin LUT-B instead:

1. Stores only a 3-bit index at each position;
2. Uses the index to select a reconstruction value from an 8-entry E4M3 LUT;
3. Performs the lookup inside `tcgen05.mma` on the Tensor Core;
4. Does not require the kernel to materialize a decompressed B matrix first.

The granule described by SemiAnalysis contains `K=64 × N=8 = 512` values:

```text
512 × 3-bit index = 192 bytes
8 × E4M3 LUT      =   8 bytes
--------------------------------
total             = 200 bytes
effective         = 3.125 bit/value
```

The LUT does not require uniform spacing, so it can hold nonuniform centroids such as those produced by Lloyd-Max. This is also key to why 3 bits may still preserve quality.

### 2.2 What do the CUDA 13.4 compile tests tell us?

We compiled PTX in a clean K8s container and inspected the resulting SASS:

| Probe | Compatibility | Compiler / SASS evidence | Meaning |
|---|---:|---|---|
| `sm_107a` target | SUPPORTED | `ptxas` PASS | Rubin compile CI can be established |
| UE5M3 conversion | SUPPORTED | `F2FP.SATFINITE.UE5M3` | The UE5M3 datatype itself exists |
| Dense E4M3 A × LUT-B E4M3 B | SUPPORTED | `UTCQMMA.LUTB` | The basic QK/PV mapping is legal |
| MXFP8 block32 A × LUT-B B | SUPPORTED | `UTCQMMA.LUTB` + scale TMEM | Scaled Q/P can be fused |
| LUT-B collector B reuse | SUPPORTED | `B_KEEP` → `B_REUSE` | One K/V tile can be reused across GQA heads |
| Sparse A × ordinary B | SUPPORTED | `UTCQMMA` control | Sparse-A MMA itself exists |
| Sparse A × LUT-B B | UNSUPPORTED | `ptxas` rejects `mma.sp...lut::b` | Sparse P cannot be used together with LUT-B V |
| MXFP8 + UE5M3 scale + LUT-B | UNSUPPORTED | PTX 9.4 Table 69 | The FP8 scale must use UE8M0 instead |
| Transposed LUT-B B | UNSUPPORTED | PTX 9.4 restriction | K must be stored physically as Kᵀ |

PTX 9.4 specifies UE8M0 as the block scale for `mxf8f6f4`, while the UE5M3 scale belongs to `mxf4nvf4`. The hardware-valid Q/P contract for K3V3 attention is therefore **E4M3 + UE8M0 block32**.

These probes prove compiler acceptance and SASS emission, not runtime correctness on Rubin. CUDA 13.4 Developer Preview also explicitly prohibits using preview performance data to characterize the hardware.

## 3. What We Did: Mapping LUT-B onto Attention

Rubin LUT-B has one crucial rule: **it decompresses only matrix B inside the MMA**. We therefore do not turn all of attention into 3-bit arithmetic. We compress the cache that is reread at every decode step, while Q and P remain MXFP8 matrix-A operands.

### 3.1 One Diagram: GQA KV and the MLA Latent

![GQA KV and MLA latent mapping to Rubin LUT-B](assets/lutb-gqa-mla-comparison.svg)

Both architectures follow the same four-step path:

1. **Write the cache once.** Prepare codebooks offline. When a token enters the low-precision region, retain only its 3-bit indices and eight E4M3 lookup values.
2. **First LUT-B MMA.** MXFP8 Q is A and the compressed cache is B; the operation produces attention scores.
3. **Keep softmax in FP32.** Max, exp, sum, and normalization are not reduced to INT3. P remains on chip and is converted to MXFP8.
4. **Second LUT-B MMA.** MXFP8 P is A and the compressed cache is B; the operation produces the attention output.

Only the cache target differs:

- **GQA / MHA (Qwen and Gemma)** caches separate `K^T` and `V`, so each layer has two codebooks and may use independent `R_k` / `R_v`. Sink 64, recent 256, and incomplete blocks remain BF16.
- **MLA (Kimi K3)** never expands K/V. It caches one shared 512-d latent `c_KV`, so each layer needs one codebook and one latent rotation. Kimi's 69 KDA states and 64-d RoPE suffix remain native; only the latent in its 24 Gated-MLA layers enters K3V3.

### 3.2 What Exactly Happens Inside Attention?

| Architecture | First multiply: scores | Middle | Second multiply: output |
|---|---|---|---|
| GQA / MHA | MXFP8 `Q` × LUT-B `K^T` | FP32 softmax produces `P` | MXFP8 `P` × LUT-B `V` |
| MLA | MXFP8 absorbed `Q` × LUT-B `c_KV^T`, plus the native RoPE term | FP32 softmax produces `P` | MXFP8 `P` × LUT-B `c_KV`, followed by the value up-projection |

The key point is that the INT3 cache does not have to be decompressed back into HBM. `UTCQMMA.LUTB` performs lookup and multiplication together inside the Tensor Core. Q/P use E4M3 elements with UE8M0 block32 scales. Qwen3-4B's GQA can additionally use `B_KEEP -> B_REUSE`, allowing one K/V tile to serve four Q heads.

### 3.3 Lloyd-Max: Put the Eight Levels Where They Are Needed

Three bits can represent only eight values. Basic uniform quantization spaces those eight levels evenly. If most values are near zero and only a few outliers are large, many levels are wasted in nearly empty regions while the dense region near zero is represented too coarsely.

Lloyd-Max is straightforward:

1. Place eight initial centers on calibration data.
2. Assign every value to its nearest center.
3. Move each center to the mean of the values assigned to it.
4. Repeat steps 2 and 3 until the centers barely move, then round to E4M3 once.

All of this happens offline. Serving performs no iterative fitting; it only finds the nearest center and stores its 3-bit index. GQA fits separate eight-value K and V codebooks per layer. MLA fits one latent codebook per layer.

### 3.4 OSCAR: Spread the Spikes Before Giving Them to 3 Bits

Lloyd-Max decides where to put the eight levels, but 3-bit quantization remains difficult when a few dimensions are much larger than the rest. The simplest interpretation of OSCAR is: **rotate the coordinate system so spikes concentrated in a few dimensions are spread across more dimensions, then quantize.**

The procedure has only three steps:

1. Learn an orthogonal rotation for each layer on calibration data.
2. Rotate before writing the cache, clip only the rarest extreme outliers, and then apply Lloyd-Max K3V3.
3. Apply the corresponding transform on the other side of QK/PV, or apply an inverse rotation, so the original attention relationship is preserved.

GQA may use separate `R_k` and `R_v`. Because MLA shares one `c_KV` between K and V, it can use only one latent rotation; the matching transforms are absorbed into the absorbed-query and value projections. A compact way to remember the difference is: **Lloyd-Max changes the eight marks on the ruler; OSCAR first lays the object flat.**

The H100 experiment simulates this numerical QDQ, the BF16 safety window, and FP32 softmax, but its physical cache remains BF16. A dummy-weight integration smoke containing one KDA layer and one Gated-MLA layer completed both a health check and a real generation request, validating the connection between latent K3V3 and the Q/P kernels; it does not replace distributed-load validation of the full 2.8T checkpoint. A real Rubin implementation still needs 3-bit paged allocation, descriptors/swizzles, TMEM copies, barriers, and a fused SM107 kernel.

## 4. What Did We Measure?

### 4.1 Accuracy setup

| Item | Setting |
|---|---|
| Primary ablation model | Qwen3-4B-Thinking-2507 |
| PPL | WikiText-2 test, 16 × 8192-token blocks |
| KL | BF16→quantized top-50 bucketed forward KL |
| Cross-model accuracy | GPQA Diamond full set, 3 seeds |
| High-precision window | sink 64 + recent 256 |
| K/V | offline per-layer Lloyd-Max E4M3 LUT |
| Q/P | E4M3 element + UE8M0 block32 scale |

`KL₅₀` is sampled at 16 fixed positions in each of 16 WikiText blocks, for 256 samples in total. Each BF16 top-50 token receives its own bucket, while the rest of the vocabulary is merged into a tail bucket; it is therefore a data-processing lower bound on full-vocabulary KL, measured in nats/token.

### 4.2 Accuracy analysis

This section reports only PPL and KL₅₀. PPL measures sequence likelihood, while KL₅₀ directly measures the quantized logit distribution's shift from the BF16 teacher.

#### 4.2.1 Sink/recent ablation

We fix Qwen3-4B-Thinking-2507 and BF16 Q/P, and vary only the high-precision window. Group A uses OSCAR + offline LM K3V3; Group B uses ordinary uniform K3V3 without OSCAR or Lloyd-Max. Each group sweeps `64/256`, `0/256`, `64/0`, and `0/0` on the same 8K WikiText-2 blocks.

**A. OSCAR + offline LM K3V3**

| BF16 Sink | BF16 Recent | PPL↓ | KL₅₀↓ |
|---:|---:|---:|---:|
| 64 | 256 | 10.453 | 0.09482 |
| 0 | 256 | 15.019 | 0.36044 |
| 64 | 0 | 10.659 | 0.14341 |
| 0 | 0 | 16.260 | 0.44066 |

**B. Ordinary K3V3 (uniform LUT, no OSCAR, no Lloyd-Max)**

| BF16 Sink | BF16 Recent | PPL↓ | KL₅₀↓ |
|---:|---:|---:|---:|
| 64 | 256 | 1504.444 | 5.14717 |
| 0 | 256 | 1875.596 | 5.40732 |
| 64 | 0 | 1890.707 | 5.42792 |
| 0 | 0 | 2489.291 | 5.59230 |

With OSCAR + offline LM, retaining sink 64 while removing recent has a comparatively small effect; removing sink causes a much larger PPL/KL regression. Ordinary uniform K3V3 collapses under every window, showing that a high-precision window cannot compensate for the absence of rotation and a nonuniform codebook.

#### 4.2.2 Overall measurement

| Method | BF16 Sink | BF16 Recent | PPL↓ | KL₅₀↓ |
|---|---:|---:|---:|---:|
| BF16 | all | all | **9.468** | **0.0000** |
| OSCAR INT2, no LM | 64 | 256 | 9.893 | 0.10791 |
| Offline LM K3V3, no OSCAR | 64 | 256 | **10.406** | 0.09618 |
| Offline LM K3V3, no OSCAR + FP8/UE8M0 Q/P | 64 | 256 | 10.434 | 0.10447 |
| OSCAR + offline LM K3V3 + FP8/UE8M0 P only | 64 | 256 | 10.446 | 0.09609 |
| OSCAR + offline LM K3V3 | 64 | 256 | 10.453 | **0.09482** |
| OSCAR + offline LM K3V3 + FP8/UE8M0 Q/P | 64 | 256 | 10.465 | 0.10090 |
| OSCAR + offline LM K3V3 + FP8/UE8M0 Q only | 64 | 256 | 10.466 | 0.09935 |
| Ordinary uniform K3V3 | 64 | 256 | 1504.444 | 5.14717 |

At 8K, no-OSCAR offline LM has the lowest PPL (10.406), while OSCAR + offline LM has the lowest KL₅₀ (0.09482); they optimize different objectives. Adding Q/P QDQ slightly increases both PPL and KL. OSCAR INT2 has lower PPL, but higher KL₅₀ than either K3V3 configuration without Q/P QDQ.

#### 4.2.3 Cross-model measurement

PPL/KL uses the 8K protocol, while accuracy uses three-seed GPQA Diamond. Qwen3.5 is a hybrid Gated DeltaNet model, so K3V3 covers only its 10 full-attention layers. Gemma4's 40 sliding-attention layers use 256-d heads, while its 8 full-attention layers use 512-d heads; rotations and per-layer codebooks are fitted separately for the two geometries. Gemma4 uses the base checkpoint for PPL/KL and the instruction-tuned checkpoint for GPQA. For Kimi K3, only the 512-d latent in its 24 Gated-MLA layers is quantized; all 69 KDA states and the 64-d RoPE suffix remain native, and native mode is used only as the KL₅₀ teacher. Kimi PPL follows the native Kimi protocol: NLL is scored only on the final 2,048 tokens of each 8,192-token block, while the context remains the full 8K. Absolute PPL should not be compared across different model families.

| Model | Method | PPL↓ | KL₅₀↓ | GPQA↑ |
|---|---|---:|---:|---:|
| Qwen3.5-35B-A3B | OSCAR + offline LM K3V3 + FP8/UE8M0 Q/P | 6.127 | **0.02295** | 82.49% |
| Qwen3.5-35B-A3B | Offline LM K3V3 + FP8/UE8M0 Q/P (no OSCAR) | **6.109** | 0.02870 | **82.83%** |
| Gemma4-12B | OSCAR + offline LM K3V3 + FP8/UE8M0 Q/P | 6.120 | **0.04806** | 62.29% |
| Gemma4-12B | Offline LM K3V3 + FP8/UE8M0 Q/P (no OSCAR) | 6.388 | 0.07572 | 62.29% |
| Kimi K3 | OSCAR + offline LM K3V3 + FP8/UE8M0 Q/P | **1.564** | **0.00679** | — |
| Kimi K3 | Offline LM K3V3 + FP8/UE8M0 Q/P (no OSCAR) | 1.601 | 0.00800 | — |

The public Kimi K3 checkpoint is a 2.8T-parameter, 1.56 TB MXFP4 model. Both result rows apply K3V3 directly to the MLA latent rather than expanding the 512-d latent into standard K/V and quantizing afterward. The native checkpoint supplies only the KL teacher distribution; it is not reported as a third result. Latent OSCAR lowers PPL from 1.601 to 1.564 (about 2.3%) and KL₅₀ from 0.00800 to 0.00679 (about 15.2%). Three-seed GPQA will be filled only after it completes.

### 4.3 Throughput: What Is Theoretically Achievable on Rubin?

First, consider the hardware of a single GPU. H100 and B200 can serve as bandwidth baselines, but they do not have SM107's `decompress::lut::b`; only Rubin can execute the K3V3 LUT-B path in this article natively.

| GPU | Memory | Capacity | HBM bandwidth | Dense FP8 ceiling | Native LUT-B |
|---|---|---:|---:|---:|---:|
| H100 SXM | HBM3 | 80 GB | 3.35 TB/s | 1.979 PFLOP/s | No |
| B200 SXM | HBM3e | 180 GB | up to 8 TB/s | 4.5 PFLOP/s | No |
| Rubin | HBM4 | 288 GB | 22 TB/s | 8.75 PFLOP/s assumption | **Yes** |

The H100 data come from the [NVIDIA H100 specifications][nvidia-h100]; for B200, we use the 180 GB / up to 8 TB/s figures from the [NVIDIA HGX component specifications][nvidia-hgx] and the publicly reported 4.5 PFLOP/s dense FP8 figure. Rubin's 8.75 PFLOP/s is a modeling assumption obtained by halving the published 17.5 PFLOP/s sparse peak. That peak applies only when the MMA tile is full; it cannot be used directly as the compute ceiling for single-token decode.

Qwen3-4B-Thinking-2507 has 36 layers, 32 Q heads, 8 KV heads, and a head dimension of 128. The logical attention FLOPs per decoded token are `4 × Q_heads × head_dim × context × layers`.

#### 4.3.1 If all three GPUs read BF16 KV

With BF16 KV, arithmetic intensity is only 4 FLOP/B, so all three GPUs are clearly HBM-bound.

| GPU | 8K BF16 roofline | 8K attention-only tok/s | 32K BF16 roofline | 32K attention-only tok/s |
|---|---:|---:|---:|---:|
| H100 SXM | 13.4 TFLOP/s | 2.77K | 13.4 TFLOP/s | 0.69K |
| B200 SXM | 32.0 TFLOP/s | 6.62K | 32.0 TFLOP/s | 1.66K |
| Rubin | 88.0 TFLOP/s | 18.21K | 88.0 TFLOP/s | 4.55K |

Here, TFLOP/s remains constant as context length changes because BF16 arithmetic intensity is fixed; token/s declines as the context grows longer.

#### 4.3.2 If only K3V3 compressed bytes are counted

After adding the BF16 sink/recent window to K3V3, effective storage is 3.628 bit/value at 8K and 3.251 bit/value at 32K; the corresponding arithmetic intensities are 17.64 / 19.69 FLOP/B.

| GPU | 8K compressed-byte roofline | 8K tok/s | 32K compressed-byte roofline | 32K tok/s | Can run natively |
|---|---:|---:|---:|---:|---|
| H100 SXM | 59.1 TFLOP/s | 12.23K | 66.0 TFLOP/s | 3.41K | No: requires software unpack |
| B200 SXM | 141.1 TFLOP/s | 29.21K | 157.5 TFLOP/s | 8.15K | No: no LUT-B |
| Rubin | **388.1 TFLOP/s** | **80.32K** | **433.1 TFLOP/s** | **22.41K** | **Yes: `UTCQMMA.LUTB`** |

The H100/B200 rows are only normalization ceilings for the case where HBM moves no more than these compressed bytes; they are not executable native K3V3 performance figures. Both GPUs must first unpack/dequantize in software, which in practice adds instructions, temporary traffic, and latency. Rubin differs because the lookup occurs inside the Tensor Core MMA.

#### 4.3.3 LUT Can Only Be B: The Single-Token Decode Cost

The byte roofline above misses an important restriction. If decode is written as `K × Q^T` / `V^T × P^T`, the long KV matrix occupies A's M direction and short Q/P occupies B's N direction; for a very skinny GEMV, this orientation fills an asymmetric MMA tile more easily. LUT decompression supports B only, however, so compressed KV must use the `Q × K^T` / `P × V` orientation.

Donglin is correct for the kernel cited by the user. [FlashInfer's SM100 GQA decode][flashinfer-mma] explicitly sets `mma_tile_m=128`; GEMM1 uses K as A and Q as B, while GEMM2 uses V as A and P as B. This fills A's M direction with sequence/head-dimension work and places GQA heads × prediction tokens in B's N direction. Other implementations must still be distinguished: older FlashInfer prefill/Tensor Core paths and FlashAttention-3 also contain Q/P-as-A, KV-as-B schedules. Operand roles are therefore not a mathematical rule of GQA, but this latest CuTe decode kernel does choose KV-as-A.

PTX specifies `M=128` and an N dimension starting at 8 for `kind::mxf8f6f4`. Qwen3-4B has a GQA ratio of four, so one shared K/V tile can pack at most four Q heads into four rows:

| Orientation | Effective tile occupancy | Tile-adjusted compute cap | 8K roofline | 32K roofline |
|---|---:|---:|---:|---:|
| Compressed bytes only; ignore shape | — | 8.75 PFLOP/s | 388.1 TFLOP/s | 433.1 TFLOP/s |
| **LUT-B: Q/P in A, KV in B** | M: 4 / 128 = **3.125%** | **273.4 TFLOP/s** | **273.4 TFLOP/s** | **273.4 TFLOP/s** |
| FlashInfer dense-FP8 decode: KV in A, Q/P in B | N: 4 / 16 = 25% | 2.1875 PFLOP/s | 388.1 TFLOP/s | 433.1 TFLOP/s |
| Hypothetical Rubin compressed-A (minimum N=8) | N: 4 / 8 = 50% | 4.375 PFLOP/s | 388.1 TFLOP/s | 433.1 TFLOP/s |

To test whether swapping operands is always intrinsically faster, we also ran ordinary padded dense-E4M3 GEMMs with equal logical work on B300. Median Q/P-as-A versus KV-as-A latencies were: QK 8K `39.97 / 40.42 µs`, PV 8K `43.71 / 46.70 µs`, QK 32K `52.42 / 53.41 µs`, and PV 32K `57.95 / 57.18 µs`. Every difference was within about 7%. This cuBLASLt microbenchmark lacks FlashInfer's persistent decode schedule, TMEM softmax pipeline, and LUT-B. It therefore shows only that operand names alone do not create a large penalty; it neither invalidates FlashInfer's M/N tile choice nor replaces a Rubin measurement.

For Qwen GQA4 single-token decode, the B-only constraint therefore reduces the byte-only ceiling by about **1.42× @ 8K / 1.58× @ 32K**. MHA/GQA ratio one suffers more. GQA ratio eight, multiple query tokens sharing one KV cache, or speculative decode with `T≥2` puts more rows into A and can return the kernel to the bandwidth-bound regime.

Another testable compromise is dense E4M3 A. `kind::f8f6f4` supports `M=64`, raising Qwen GQA4's tile-adjusted cap to about 546.9 TFLOP/s—enough to cover this HBM roofline. The cost is losing native UE8M0 block32 scaling, adding software scaling, and requiring a new accuracy validation. The quality numbers in this article use MXFP8/UE8M0 and therefore cannot claim the higher ceiling. If SM107 ultimately exposes an unpublished LUT-B small-M special path, this cap will change; the current public PTX and compile probes do not justify assuming that it exists.

#### 4.3.4 Operand-Corrected Practical Range

| Context | Effective KV bits/value | Byte-only HBM roofline | B-only tile-corrected ceiling | Expected @ 60–80% | Corrected tok/s | vs Rubin BF16 |
|---:|---:|---:|---:|---:|---:|---:|
| 8K | 3.628 | 388.1 TFLOP/s | **273.4 TFLOP/s** | **164.1–218.8 TFLOP/s** | 56.59K | 3.11× |
| 32K | 3.251 | 433.1 TFLOP/s | **273.4 TFLOP/s** | **164.1–218.8 TFLOP/s** | 14.15K | 3.11× |

After operand-shape correction, the single-token attention-only ceiling is about **20.4×** H100 BF16 and **8.5×** B200 BF16. Applying 60–80% execution efficiency gives about **12.3–16.4×** and **5.1–6.8×**, respectively. The 30× title is reachable only as a byte-only roofline, or if GQA packing, multi-token decode, or a small-M path fills the Tensor Core again.

If GQA B-tile reuse is lost entirely, the 8K/32K bandwidth ceilings fall to about 97.0 / 108.3 TFLOP/s, making the kernel HBM-bound again. Every token/s figure remains an attention-only ceiling excluding model weights, the MLP, sampling, scheduling, communication, and power throttling; none is a Qwen3-4B end-to-end output rate or a Rubin hardware measurement.

## 5. What Does This Story Tell Us So Far?

1. **Dense KV-on-LUT-B has no legality blocker, but it has a decode-shape blocker.** PTX does not require B to be a static weight; `Kᵀ` and `V` can be dynamic compressed B, but B-only plus MXFP8 M=128 limits single-token tile occupancy.
2. **A compile proof does not mean the kernel is finished.** We generated `UTCQMMA.LUTB` SASS, but do not yet have a production PagedAttention implementation.
3. **The precision contract must be dictated by the ISA.** For LUT-B's MXFP8 A operand, the hardware contract is E4M3 + UE8M0 block32.
4. **Offline LM is a more realistic serving recipe.** It removes runtime centroid fitting and leaves only index assignment.
5. **Performance is not determined by bytes alone.** Packed layout, HBM efficiency, and collector reuse still matter, but GQA packing, multi-token decode, and a small-M path determine whether the Tensor Core can consume the bandwidth savings.

## Reproducibility

- Offline codebooks: `/shared/oscar-int3-rubin-simulate/codebooks/qwen3-4b-thinking-2507/offline-wikitext-v2/`
- PPL / KL: `/shared/oscar-int3-rubin-simulate/offline-v2/`
- CUDA 13.4 compile artifacts: `/shared/oscar-int3-rubin-simulate/analysis/rubin-isa-cuda13.4-clean/`
- Roofline calculator: `rotation/rubin_k3v3_roofline.py --compare-hardware`
- Compile probes: `rotation/rubin_compile_probes/run_compile_probes.py`

## References

- [NVIDIA: Inside NVIDIA Rubin GPU Architecture][nvidia-rubin-gpu]
- [NVIDIA: Inside the Vera Rubin Platform][nvidia-rubin-platform]
- [NVIDIA: Vera Rubin NVL72 specs][nvidia-rubin-nvl72]
- [NVIDIA: H100 product specifications][nvidia-h100]
- [NVIDIA: HGX H100/B200/B300 components][nvidia-hgx]
- [SemiAnalysis / InferenceX: Vera Rubin NVL72 vs GB200 NVL72][semianalysis-rubin]
- [Medium: Prefill is Compute, Decode is Bandwidth][medium-disagg]
- [CUDA 13.4 Developer Preview release notes][cuda134-release]
- [PTX ISA 9.4][ptx94]
- [FlashAttention-3][flashattention3]
- [FlashInfer Tensor Core attention source][flashinfer-mma]
- [OpenAI Triton initial SM107 support][triton-sm107]
- [vLLM SM107 tracking issue][vllm-sm107]

[nvidia-rubin-gpu]: https://developer.nvidia.com/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/
[nvidia-rubin-platform]: https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/
[nvidia-rubin-nvl72]: https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/
[nvidia-h100]: https://www.nvidia.com/en-us/data-center/h100/
[nvidia-hgx]: https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html
[semianalysis-rubin]: https://inferencex.semianalysis.com/blog/vera-rubin-nvl72-vs-gb200-nvl72-inference
[medium-disagg]: https://medium.com/@asheesh.goja/prefill-is-compute-decode-is-bandwidth-the-architectural-case-for-disaggregated-llm-inference-92c8b36d1a88
[cuda134-release]: https://docs.nvidia.com/cuda/developer-preview/13.4/cuda-toolkit-release-notes/index.html
[ptx94]: https://docs.nvidia.com/cuda/developer-preview/13.4/parallel-thread-execution/index.html
[flashattention3]: https://tridao.me/publications/flash3/flash3.pdf
[flashinfer-mma]: https://github.com/flashinfer-ai/flashinfer/blob/b1d95851675b8799d623df4d5a7d6eac3254b3ff/flashinfer/cute_dsl/attention/gqa_decode_paged.py#L222-L259
[triton-sm107]: https://github.com/triton-lang/triton/commit/ab592f012f37f2c2ffc9ceda6c5b49b3c37c9036
[vllm-sm107]: https://github.com/vllm-project/vllm/issues/49735
