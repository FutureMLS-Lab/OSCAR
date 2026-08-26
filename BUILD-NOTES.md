# OSCAR-llamacpp — Build & Run Notes (RTX 5090 / Blackwell)

This is a working copy of the OSCAR INT2-KV llama.cpp fork
(`FutureMLS-Lab/OSCAR`, branch `zhongzhu/llamacpp`), preserved at
`github.com/giveen/OSCAR-llamacpp`.

OSCAR compresses only the **KV cache** to INT2 (`q2_0`) using a calibrated,
per-layer orthogonal rotation baked into the GGUF. Model **weights stay in full
precision** (e.g. Q4_K_M). This lets you run very long contexts (tested to the
model's full 256K) for almost no extra VRAM.

---

## 1. Clone

```bash
git clone --depth 1 --branch zhongzhu/llamacpp \
  https://github.com/FutureMLS-Lab/OSCAR.git OSCAR-llamacpp
cd OSCAR-llamacpp
```

## 2. Build for NVIDIA RTX 5090 (sm_120, CUDA 13.3)

The system's CUDA compiler-id probe is broken (it tries `compute_52`, which
CUDA 13.3's `ptxas` no longer supports → `sm_52 is not defined`). The working
flag set below **lies about the compiler ID** (`-DCMAKE_CUDA_COMPILER_ID=NVIDIA`
+ explicit version) and forces `native` arch. It is derived from the proven
`/mnt/storage/llama-server/rebuild_llama.sh`.

The critical flag for OSCAR GPU-side KV is **`-DGGML_CUDA_FA_ALL_QUANTS=ON`**:
it compiles Flash-Attention kernels for the `q2_0` KV quant type, which is what
lets the INT2 KV cache run on the GPU instead of forcing `-nkvo` (CPU KV).

```bash
cd /mnt/storage/Projects/OSCAR-llamacpp
rm -rf build
export PATH="/usr/local/cuda/bin:$PATH"

cmake -B build -G Ninja \
  -DCMAKE_C_COMPILER=gcc-15 \
  -DCMAKE_CXX_COMPILER=g++-15 \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_FLAGS="-ccbin /usr/bin/g++-15 -isystem /usr/local/cuda/include" \
  -DCMAKE_CUDA_COMPILER_ID=NVIDIA \
  -DCMAKE_CUDA_COMPILER_VERSION=13.3 \
  -DCMAKE_CUDA_STANDARD_COMPUTED_DEFAULT=17 \
  -DCMAKE_CUDA_EXTENSIONS_COMPUTED_DEFAULT=ON \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DGGML_LTO=ON \
  -DGGML_CPU_KLEIDIAI=OFF \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -DCMAKE_LINK_DEPENDS_USE_LINKER=OFF \
  -DLLAMA_CURL=ON .

cmake --build build --config Release -j"$(nproc)"
```

> NOTE: do **not** mix generators — the first attempt used Unix Makefiles; this
> script uses Ninja. Purge `build/` if you switch.

Resulting binaries: `build/bin/llama-server`, `build/bin/llama`.

## 3. Get a model

Pre-baked OSCAR GGUF (weights Q4_K_M + rotation baked in), ~7.4 GB:

- `https://huggingface.co/Zhongzhu/OSCAR-LLAMACPP-Gemma-4-12B-it-INT2-KV`
  - `q4km-rot-kv/gemma-4-12b-it-rot-kv.gguf`

Raw rotation matrices (for self-baking other models), head_dim 256, 48 layers:

- `https://huggingface.co/Zhongzhu/OSCAR-RotationZoo` → `Gemma4-12B/`
  - `k_rotation_qqt_r_h_pbr.pt`, `v_rotation_sst_r_h_pbr.pt`

Self-bake a `-rot-kv` GGUF from a base GGUF + rotation dir:

```bash
python3 oscar-rotation/export_rot_kv_gguf.py \
  --base   /path/to/base-q4km.gguf \
  --rot-dir /path/to/Gemma4-12B \
  --out    /path/to/base-q4km-rot-kv.gguf
```

## 4. Run

### The original crash (and why the new build fixes it)

With the *first* build (single-arch sm_120, no `GGML_CUDA_FA_ALL_QUANTS`),
running `-fa on` with the KV cache on the GPU aborted at context build:

```
ggml-backend.cpp:898: pre-allocated tensor (cache_k_l0 (view)) in a buffer
(CUDA0) that cannot run the operation (SET_ROWS)
```

`-fa off` does **not** help — the server refuses because
`V cache quantization requires flash_attn`.

**Fix (this build):** `-DGGML_CUDA_FA_ALL_QUANTS=ON` compiles the
Flash-Attention kernels for the `q2_0` KV quant, so the INT2 KV cache can live
on the GPU. Drop `-nkvo` and run with GPU KV. Only fall back to `-nkvo` (CPU KV)
if the SET_ROWS crash reappears.

### Recommended launch (GPU KV)

```bash
cd /mnt/storage/Projects/OSCAR-llamacpp
G=/path/to/gemma-4-12b-it-rot-kv.gguf

pkill -f llama-server 2>/dev/null; sleep 2
CUDA_VISIBLE_DEVICES=0 \
GGML_CUDA_DISABLE_GRAPHS=1 \
LLAMA_KV_FUSED_FA=1 \
LLAMA_KV_NO_HADAMARD=1 \
LLAMA_KV_CLIP_RATIO=0.96 \
LLAMA_KV_HP_SINK=512 \
LLAMA_KV_HP_RECENT=2048 \
./build/bin/llama-server \
  -m "$G" \
  -fa on \
  -ngl 99 \
  -c 262144 \
  --cache-type-k q2_0 --cache-type-v q2_0 \
  --host 127.0.0.1 --port 8080
```

### Fallback launch (CPU KV, if GPU KV crashes)

```bash
# same as above but append:
  -nkvo
```

### Notes on the flags

| Flag | Why |
|------|-----|
| `-fa on` | **Required.** `q2_0` KV quantization needs flash attention. |
| `--cache-type-k/v q2_0` | The OSCAR INT2 KV cache. |
| `-nkvo` | **Fallback only.** KV on CPU sidesteps the Blackwell `SET_ROWS` CUDA-buffer crash. Avoid with the FA_ALL_QUANTS build. |
| `LLAMA_KV_FUSED_FA=1` | Fused mixed-precision Flash-Attention kernel (OSCAR path). |
| `LLAMA_KV_NO_HADAMARD=1` | OSCAR stores plain INT2 KV (no in-quant Hadamard). |
| `LLAMA_KV_CLIP_RATIO=0.96` | Per-layer clip ratio baked during calibration. |
| `LLAMA_KV_HP_SINK=512` / `LLAMA_KV_HP_RECENT=2048` | Hybrid-prefix KV: 512 sink + 2048 recent tokens in higher precision. |
| `-c 262144` | Full native Gemma-4-12B context (model trains at 256K). |

`GGML_CUDA_DISABLE_GRAPHS=1` is kept to avoid runtime graph-replay issues on
Blackwell (it does not affect the SET_ROWS reserve path).

## 5. Verified behavior (RTX 5090, 32 GB)

- OSCAR KV engaged: `KV cache dtype: K=q2_0 V=q2_0 (n_embd_k_gqa=512, kv_size=262144)`.
- Full 256K context loads with **4 slots**, no truncation warning.
- VRAM at 256K context: **~10.4 GB** (weights Q4_K_M + 2-bit KV).
- Decode: **~45 tok/s** (CPU-KV path; GPU still does the matmuls).
  - *With the FA_ALL_QUANTS GPU-KV build, expect higher decode once `-nkvo` is dropped — verify after launch.*
- Health: `GET /health` → `{"status":"ok"}`.
- Smoke generation: prompt "The capital of France is" → "**Paris**." (coherent).

## 6. Known limitation / future work

With the old build, `-nkvo` put the KV cache on CPU, capping long-context decode
throughput. The new `GGML_CUDA_FA_ALL_QUANTS=ON` build targets GPU KV directly.
If the SET_ROWS crash persists on GPU, the deeper fix is the scheduler's KV-buffer
backend selection in `src/llama-kv-cache.cpp` (CUDA buffer-type placement) so
`SET_ROWS` into the q2_0 cache runs on the CUDA buffer.
