# Qwen3 / Gemma 4 INT2 KV cache with OSCAR — llama.cpp fork

A fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) that adds a **~2-bit (INT2) KV cache** with the **OSCAR calibrated rotation**, so the KV footprint drops ~8× while keeping near-f16 quality — targeting edge / MacBook (Apple Silicon / Metal) deployment.

Supported today: **Qwen3** (head dim 128) and **Gemma 4** (head dim 512, incl. sliding-window layers). The Metal GPU path is **validated and fast** — fused mixed-precision flash-attention kernels run the INT2+f16 KV on-GPU (INT2 prefill ≈ f16 parity).

> This README covers only how to **deploy and run** the INT2/OSCAR build. Base llama.cpp docs (full build options, backends, general usage) are upstream [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), preserved as [`README.upstream.md`](README.upstream.md).

---

## Models (pre-built, on Hugging Face)

Ready-to-run `*-rot-kv.gguf` (the OSCAR rotation is already baked in) plus the raw rotation matrices:

| model | head dim | Hugging Face repo |
|---|---|---|
| Qwen3-4B-Thinking-2507 | 128 | [`Zhongzhu/OSCAR-LLAMACPP-Qwen3-4B-Thinking-2507-INT2-KV`](https://huggingface.co/Zhongzhu/OSCAR-LLAMACPP-Qwen3-4B-Thinking-2507-INT2-KV) |
| Gemma-4-12B-it | 512 | [`Zhongzhu/OSCAR-LLAMACPP-Gemma-4-12B-it-INT2-KV`](https://huggingface.co/Zhongzhu/OSCAR-LLAMACPP-Gemma-4-12B-it-INT2-KV) |
| Qwen3-32B | 128 | [`Zhongzhu/OSCAR-LLAMACPP-Qwen3-32B-INT2-KV`](https://huggingface.co/Zhongzhu/OSCAR-LLAMACPP-Qwen3-32B-INT2-KV) |

Download one with:

```bash
hf download Zhongzhu/OSCAR-LLAMACPP-Gemma-4-12B-it-INT2-KV --local-dir ./gemma-4-12b-int2
```

The Qwen3 repos ship the Q4_K_M `*-rot-kv.gguf` (+ `k_/v_rotation_*.pt`) directly. The Gemma repo has several weight variants in subfolders — **`q4km-rot-kv/`** (recommended), `bf16-rot-kv/`, and plain `base-*` — plus a shared `rotation/`; see that repo's README for which is which.

---

## Results — GPQA-Diamond @ 32K (Qwen3-4B-Thinking-2507, Q4_K_M weights)

Full chain-of-thought (`n_predict=16000`), HP buffer `sink=64 / recent=256`, clip 0.96.

| KV cache config | GPQA |
|---|---|
| broken INT2 (32-wide rotation, no clip) | 0/2 |
| INT2 + full-head Hadamard + clip (data-free) | 2/6 = 33% |
| **INT2 + OSCAR calibrated rotation — K only (V=f16, isolation test)** | **6/6** (= f16) |
| **INT2 + OSCAR calibrated rotation — full K+V** | **14/20 = 70%** |
| f16 baseline (same n=20 sample) | 10/20 = 50% |
| sglang OSCAR INT2 (reference, full 198-q) | ~62% |

---

## Runtime knobs (env vars)

INT2/OSCAR is enabled by the `q2_0` cache type plus these env vars. With default cache types and no env vars, the build behaves exactly like upstream llama.cpp.

| var | meaning | recommended |
|---|---|---|
| `LLAMA_KV_FUSED_FA`   | use the fused mixed-precision (INT2+f16) flash-attention kernels (required for the fast Metal GPU path) | `1` |
| `LLAMA_KV_NO_HADAMARD`| skip the in-quant Hadamard (the calibrated rotation already includes it) | `1` |
| `LLAMA_KV_CLIP_RATIO` | per-row outlier clip percentile before quant | `0.96` |
| `LLAMA_KV_HP_SINK`    | keep the first N tokens high-precision | `64` |
| `LLAMA_KV_HP_RECENT`  | keep the last N tokens high-precision | `256` |

`64 / 256` matches the official OSCAR setup; the rest of the context is INT2. Larger values trade more KV memory for output closer to f16.

---

## Build

```bash
# macOS (Apple Silicon): Metal is ON by default — the recommended deployment target
cmake -B build
cmake --build build -j --target llama-server llama-cli

# Linux / server, CPU-only (how the accuracy results above were produced)
cmake -B build -DLLAMA_CURL=OFF -DGGML_METAL=OFF
cmake --build build -j --target llama-server llama-cli
```

## Run

You need a **rotated GGUF** (`*-rot-kv.gguf`) — your base GGUF with the OSCAR `attn_k_rot` / `attn_v_rot` tensors baked in (see [Bake your rotation into the GGUF](#bake-your-rotation-into-the-gguf)). Then run with `-fa on`, `--cache-type-k q2_0 --cache-type-v q2_0`, and the env vars above.

### Qwen3 (head dim 128)

Server:

```bash
LLAMA_KV_FUSED_FA=1 LLAMA_KV_NO_HADAMARD=1 LLAMA_KV_CLIP_RATIO=0.96 \
LLAMA_KV_HP_SINK=64 LLAMA_KV_HP_RECENT=256 \
./build/bin/llama-server -m qwen3-4b-rot-kv.gguf \
  -fa on -ngl 99 -c 32768 \
  --cache-type-k q2_0 --cache-type-v q2_0 \
  --host 127.0.0.1 --port 8080
```

One-shot CLI:

```bash
LLAMA_KV_FUSED_FA=1 LLAMA_KV_NO_HADAMARD=1 LLAMA_KV_CLIP_RATIO=0.96 \
LLAMA_KV_HP_SINK=64 LLAMA_KV_HP_RECENT=256 \
./build/bin/llama-cli -m qwen3-4b-rot-kv.gguf \
  -fa on -ngl 99 -c 32768 -n 16000 \
  --cache-type-k q2_0 --cache-type-v q2_0 \
  -p "your prompt"
```

### Gemma 4 (head dim 512)

Same flags, plus the Gemma 4 chat template (its channel / "thinking" format needs it):

```bash
LLAMA_KV_FUSED_FA=1 LLAMA_KV_NO_HADAMARD=1 LLAMA_KV_CLIP_RATIO=0.96 \
LLAMA_KV_HP_SINK=64 LLAMA_KV_HP_RECENT=256 \
./build/bin/llama-server -m q4km-rot-kv/gemma-4-12b-it-rot-kv.gguf \
  -fa on -ngl 99 -c 16384 \
  --cache-type-k q2_0 --cache-type-v q2_0 \
  --chat-template-file models/templates/google-gemma-4-31B-it.jinja \
  --host 127.0.0.1 --port 8080
```

Notes:

- `-fa on` is **required** (the INT2 KV path runs through flash-attention); `-ngl 99` offloads everything to the Metal GPU; `LLAMA_KV_FUSED_FA=1` selects the fused INT2+f16 kernels.
- `--cache-type-k q2_0 --cache-type-v q2_0` = full K+V INT2. Use `--cache-type-v f16` to keep V high-precision (a bit more quality, more memory).
- A **non-rotated** GGUF with these flags falls back to data-free INT2 (degraded) — always run the `*-rot-kv.gguf`.

---

## Bake your rotation into the GGUF

You provide two things:

1. a **base GGUF** — any quant (bf16, Q4_K_M, …); only the KV cache is INT2, the weights are copied through unchanged.
2. the **OSCAR rotation** for that model — a directory with `k_rotation_qqt_r_h_pbr.pt` and `v_rotation_sst_r_h_pbr.pt` (per-layer orthogonal matrices, data-calibrated from the model's own activations; 128×128 for Qwen3, 512×512 for Gemma 4).

Bake them into a `*-rot-kv.gguf`:

```bash
python3 oscar-rotation/export_rot_kv_gguf.py \
  --base    /path/to/base.gguf \
  --rot-dir /path/to/rotation-dir \
  --out     /path/to/model-rot-kv.gguf
```

This appends the per-layer `blk.{i}.attn_k_rot.weight` / `attn_v_rot.weight` tensors (stored as `Mᵀ` so `ggml_mul_mat(rot, K) == K @ M`) and copies the base weights through unchanged. The resulting `*-rot-kv.gguf` is what you pass to `-m` in the Run section. (Needs `torch` + `numpy`; the repo's `gguf-py` is imported automatically.)

---

## Metal / GPU (Apple Silicon)

The Metal GPU path is **validated and fast** — just run with `-ngl 99 -fa on` and `LLAMA_KV_FUSED_FA=1` (as in the Run examples). The fork ships fused mixed-precision flash-attention kernels that keep the two-tier KV (INT2 history + f16 sink/recent) on-GPU in a single pass:

- **Decode** — a per-query online-softmax kernel over both tiers.
- **Prefill** — a tiled simdgroup-matmul kernel (dual-source q2_0 + f16); INT2 prefill runs at roughly **f16 parity** while the KV cache stays ~8× smaller.

Supported head dims: **128** (Qwen3), **256**, **512** (Gemma 4, incl. its sliding-window layers).

Fallbacks: `LLAMA_KV_PF_NOMM=1` uses the per-query kernel for prefill instead of the matmul one; `-ngl 0` keeps the whole KV path on the CPU backend.

---

## Upstream

Fork of **`ggml-org/llama.cpp`**. INT2/OSCAR is additive and gated behind the `q2_0` cache type + the env vars above; with default cache types this behaves exactly like upstream llama.cpp. Full base docs: [`README.upstream.md`](README.upstream.md).
