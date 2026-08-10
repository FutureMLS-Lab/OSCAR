# Rubin INT3 Attention Scripts

This folder contains the runnable code for the Rubin LUT-B INT3 attention
simulation. It intentionally excludes benchmark outputs, model checkpoints,
calibration tensors, generated PDFs, and result tables.

Run every command from the OSCAR repository root:

```bash
git clone https://github.com/FutureMLS-Lab/OSCAR.git
cd OSCAR
git checkout zhongzhu/oscar-int3-rubin-simulate
```

## Hardware and ISA analysis

Rebuild the H100, B200, and Rubin roofline projection:

```bash
python3 rubin/rubin_k3v3_roofline.py --compare-hardware
```

The default contexts are 8K, 32K, and 1M tokens.

NVIDIA publishes 17.5 PFLOP/s as Rubin's dense FP8/FP6 specification, but
does not publish the full-tile throughput of `decompress::lut::b`. The default
uses 17.5 PFLOP/s only as a sensitivity endpoint. Compare it with a hypothetical
half-rate LUT-B mode explicitly:

```bash
python3 rubin/rubin_k3v3_roofline.py \
  --lutb-full-tile-pflop-s 17.5
python3 rubin/rubin_k3v3_roofline.py \
  --lutb-full-tile-pflop-s 8.75
```

Neither command is a hardware prediction. A concrete speedup requires Rubin
measurements of LUT-B tile throughput, achieved HBM bandwidth, and collector
reuse in a fused attention kernel.

The calculator models ordinary MHA/GQA caches, where separate K and V tensors
are shared by `q_heads / kv_heads` query heads. Do not transfer its GQA4
`4 / 128` row occupancy to MLA: an MLA latent is shared by all local query
heads. Tensor/context parallelism determines how many of those heads can be
packed by one GPU.

Compile the SM107 LUT-B probes with a CUDA 13.4 toolchain:

```bash
python3 rubin/run_compile_probes.py \
  --cuda /usr/local/cuda-13.4 \
  --output /tmp/rubin-compile-probes
```

Run the dense-FP8 operand-orientation microbenchmark on an NVIDIA GPU:

```bash
python3 rubin/benchmark_decode_operand_orientation.py \
  > /tmp/decode-operand-orientation.json
```

The block-scaled benchmark is enabled automatically when `sgl_kernel` and the
corresponding SGLang FP8 helpers are installed.

## Kimi K3 MLA-latent K3V3 simulation

The quality simulation quantizes only the 512-dimensional `c_KV` latent in
Kimi K3's Gated-MLA layers. KDA states and the RoPE suffix stay at native
precision. H100/B200 reconstruct the latent into BF16, so this is a numerical
simulation rather than a physical Rubin 3-bit cache implementation.

### 1. Patch a stock Kimi SGLang runtime

Copy SGLang's Python package to a writable directory, then apply the overlay:

```bash
cp -a /sgl-workspace/sglang/python /tmp/kimi-rubin-python
python3 rubin/patch_stock_runtime.py /tmp/kimi-rubin-python
export PYTHONPATH=/tmp/kimi-rubin-python
```

The overlay is pinned to the runtime structure used by
`lmsysorg/sglang:kimi-k3` and fails if an expected source anchor is missing.

### 2. Fit offline codebooks

After a native server has dumped per-layer latent calibration tensors:

```bash
python3 rubin/fit_latent_codebooks.py \
  --dump-path /path/to/latent-dump \
  --output-no-oscar /path/to/latent_lloyd_max.pt \
  --output-oscar /path/to/latent_oscar_lloyd_max.pt \
  --calibration-name wikitext-2-raw-v1/train
```

The two output files correspond to `offline_lm_qp` and
`oscar_offline_lm_qp`.

### 3. Run PPL, KL, or GPQA

`run_quality_client.sh` talks to already-running native or student SGLang
endpoints. Required inputs are supplied through environment variables:

```bash
export SOURCE="$PWD"
export BASE=/path/to/output
export BLOCKS=/path/to/wikitext2_test_8192.pt
export MODEL=moonshotai/Kimi-K3

# Native endpoint: create the KL teacher.
BASE_URL=http://native-server:30000 \
  TASK=kl_teacher MODE=native \
  bash rubin/run_quality_client.sh

# Student endpoint: evaluate one quantization mode.
BASE_URL=http://student-server:30000 \
  TASK=ppl MODE=offline_lm_qp SKIP_CONTROL_MODE_CHECK=1 \
  bash rubin/run_quality_client.sh
BASE_URL=http://student-server:30000 \
  TASK=kl_student MODE=offline_lm_qp SKIP_CONTROL_MODE_CHECK=1 \
  bash rubin/run_quality_client.sh
```

For GPQA, run seeds `0..2` and shards `0..3` for each mode:

```bash
BASE_URL=http://student-server:30000 \
  TASK=gpqa MODE=offline_lm_qp SEED=0 SHARD_INDEX=0 NUM_SHARDS=4 \
  SKIP_CONTROL_MODE_CHECK=1 \
  bash rubin/run_quality_client.sh
```

Repeat the student commands with `MODE=oscar_offline_lm_qp` and its matching
server. Merge block-sharded PPL/KL files with `merge_quality_shards.py`, then
build one machine-readable summary:

```bash
python3 rubin/summarize_quality.py \
  --root "$BASE" \
  --output "$BASE/summary.json"
```

## Script index

| Script | Purpose |
|---|---|
| `rubin_k3v3_roofline.py` | Operand-aware HBM/compute roofline |
| `run_compile_probes.py` | PTX/CUBIN/SASS LUT-B compile checks |
| `benchmark_decode_operand_orientation.py` | Q/P-as-A versus KV-as-A latency |
| `patch_stock_runtime.py` | Kimi latent K3V3 and Q/P QDQ runtime overlay |
| `fit_latent_codebooks.py` | Offline Lloyd-Max and OSCAR fitting |
| `wikitext2_ppl.py` | Resumable WikiText-2 PPL client |
| `wikitext2_kl_tail.py` | Top-50 bucketed forward-KL client |
| `run_simple_eval.py` | Seeded and sharded GPQA client |
| `run_quality_client.sh` | Kimi evaluation orchestrator |
| `merge_quality_shards.py` | PPL/KL shard merger |
| `summarize_quality.py` | Final two-mode JSON summarizer |
