# Kimi K3 MLA-Latent K3V3

This directory contains the smallest self-contained quality-simulation path for
quantizing Kimi K3's MLA latent cache to a Rubin-compatible 3-bit LUT.

## Quantization target

- Quantized: the 512-d `c_KV` latent in Kimi K3's 24 Gated-MLA layers.
- Native precision: all 69 KDA states and the 64-d RoPE suffix.
- Q/P: E4M3 QDQ with a UE8M0 block-32 scale.
- Softmax max/sum: FP32.

The H100/B200 implementation is a numerical oracle: it reconstructs the
quantized latent into BF16 storage. It does not claim to implement Rubin's
physical 3-bit paged cache or `UTCQMMA.LUTB`.

## Core files

- `patch_stock_runtime.py`: overlays the latent hook and Q/P QDQ onto the pinned
  `lmsysorg/sglang:kimi-k3` runtime.
- `fit_latent_codebooks.py`: fits per-layer offline Lloyd-Max codebooks, with
  an optional learned latent OSCAR rotation.
- `wikitext2_kl_tail.py`: computes 8K-context top-50 bucketed forward KL.
- `run_quality_client.sh`: runs calibration, PPL, KL, or GPQA against a server.
- `merge_quality_shards.py`: merges block-sharded PPL/KL outputs.
- `summarize_quality.py`: emits the final two-row Kimi quality summary.

## Fast integration check

`rotation/multimodel/k8s_kimi_k3_two_layer_smoke.yaml` launches one KDA layer
and one Gated-MLA layer with dummy weights. It verifies server startup, latent
K3V3, Q/P kernels, and a real generation request without reading the 1.56 TB
checkpoint.

Passing this smoke test does not prove that the complete model can load: the
full run additionally validates checkpoint I/O, 32-rank sharding, memory, and
distributed barriers.

## Full protocol

1. Run a native-precision server with latent calibration dumping enabled.
2. Fit:

   ```bash
   python3 rotation/kimi_k3_latent/fit_latent_codebooks.py \
     --dump-path /path/to/latent-dump \
     --output-no-oscar /path/to/latent_lloyd_max.pt \
     --output-oscar /path/to/latent_oscar_lloyd_max.pt \
     --calibration-name wikitext-2-raw-v1/train
   ```

3. Evaluate exactly two result rows:
   - `oscar_offline_lm_qp`
   - `offline_lm_qp`
4. Use native Kimi only as the KL teacher.

The Kubernetes manifests under `rotation/multimodel/` show the distributed
launch and block-sharded evaluation used for the reported results.
