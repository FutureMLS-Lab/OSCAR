# Qwen3-30B-A3B — INT2 KV recipe

This model needs **per-head rotations**. With one rotation shared across the
four KV heads, long generations degenerate into digit soup or repetition loops
(GitHub issue #16). The cause is not calibration quality: the four KV heads
want mutually near-orthogonal rotations (`mean |diag(R0ᵀR1)| ≈ 0.07`), so no
single matrix can serve them.

WikiText-2 perplexity through the fake-quant harness, vs BF16:

| rotation | PPL delta |
|---|---|
| shared per layer, calibrated (117 prompts) | +16.6 % |
| shared per layer, plain Hadamard | +26.6 % |
| shared per layer, 12-token junk dump | +15.3 % |
| **per KV head, calibrated** | **+0.35 %** |

The three shared rows land within a few points of each other while per-head is
an order of magnitude better, which is why re-running calibration does not help.
Qwen3-8B has 8 KV heads that tolerate a shared rotation, so this never showed up
before.

## Steps

```bash
# 1. dump post-RoPE Q/K/V on GPQA (~15 min, 1 GPU)
bash save_qkv_30b.sh

# 2a. per-head fit -- the recipe you want (CPU, ~5 min)
bash fit_perhead.sh

# 2b. shared per-layer fit, for comparison only
bash compute_rotation.sh

# 3. serve
bash serve.sh
```

Check step 1 before continuing: the script prints `ok=<n> err=<n>` and aborts
unless every prompt came back. A dump that silently captured a handful of tokens
still produces loadable rotation files that quietly serve garbage.

Confirm step 3 took the per-head path — the pool logs it at startup:

```
[oscar] loaded K rotation ... [per-head: 4 kv heads]
```

## Files

| file | what |
|---|---|
| `save_qkv_30b.sh` | Q/K/V dump for calibration |
| `fit_perhead.sh` | per-KV-head rotations (`format_version: 2`) |
| `compute_rotation.sh` | shared per-layer rotations (V1) |
| `serve.sh` | sglang launch with the full env |
