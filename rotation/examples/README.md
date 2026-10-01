# examples

## GPQA under INT2 OSCAR KV

```bash
examples/gpqa_int2_oscar.sh qwen3-8b
```

Runs GPQA-Diamond (198 questions, scored by the vendored `simple_evals`) against
the model served with INT2 OSCAR KV, using that model's own recipe from
`rotation/run/<model>.sh`.

**The recipe is not shared between models.** The table below is the whole reason
these are separate files:

| knob | which models |
|---|---|
| per-head rotation (V2) | **Qwen3-30B-A3B only** |
| Lloyd-Max codebook | MiniMax-M2.7, Qwen3-8B, Qwen3.5-35B-A3B |
| group size 256 | Qwen3.5-4B, Qwen3.5-35B-A3B (128 elsewhere) |
| published-Hadamard rotation file | MiniMax-M3 |
| packed 2-bit MLA latent | GLM-5.2, GLM-5.3, Kimi-K3 |
| `--mamba-scheduler-strategy extra_buffer` | Qwen3.5-4B, Qwen3.5-35B-A3B (hybrid) |
| two nodes | Kimi-K3 (1.5 TB bf16 does not fit 8 x 183 GB) |

Qwen3-30B-A3B is the sharpest case: its 4 KV heads are near orthogonal, so a
shared per-layer rotation collapses it — GPQA **43.9** against per-head's
**58.6**. Both rotation formats live in the same zoo directory, so the per-head
file has to be named explicitly rather than globbed.
