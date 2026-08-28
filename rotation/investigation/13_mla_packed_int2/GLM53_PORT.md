# GLM-5.3-Flash (`glm5_next`) port map

Authoritative source: **transformers 5.16.1**, `transformers/models/glm5_next/`.
Not 5.16.0 — that release does not contain the architecture, despite the
checkpoint's `transformers_version` field saying `5.16.0`. The checkpoint ships
no remote code, so this is the only reference.

Checkpoint: `/hf/hub/models--zai-org--GLM-5.3-Flash`, 306 GB, 62 shards,
76,108 tensors, weights verified openable.

## Status

| piece | state |
|---|---|
| `Glm5NextConfig` in sglang `_CONFIG_REGISTRY` | **DONE**, verified on the real checkpoint: `get_config` parses, layer-type mapping 0 mismatches |
| mHC hyper-connection | **DONE**, bit-identical to the reference (max\|d\| 0.000e+00 on post/comb/collapsed/write-back) |
| KDA linear layers (34/45) | not started — see the `head_v_dim` trap below |
| DSA layers (11/45 + MTP) | not started |
| model assembly, `model.language_model.` prefix | not started |
| vision tower (347 tensors), MTP layer 45 | not started |
| packed pool at `qk_rope_head_dim=0` | not started (zero-width `rope_buf` untested) |

## Geometry

45 layers, four-layer cycle: 3x `linear_attention` (KDA) + 1x
`deepseek_sparse_attention`. Layer 45 is the MTP layer (`eh_proj`, `enorm`,
`hnorm`): DSA with an indexer, **no `hc_*`**. That is why `index_kpool` appears
on 12 layers while `layer_types` lists 11.

hidden 4096, 64 heads. MLA latent `kv_lora_rank=512`, **`qk_rope_head_dim=0`**
(`mla_use_nope=True`), `qk_nope_head_dim=256`, `v_head_dim=256`,
`q_lora_rank=1536`. DSA `index_head_dim=128`, `index_topk=2048`.
MoE 288 routed + 1 shared, top-8, `first_k_dense_replace=3`.
mHC: `hc_mult=4`, `hc_eps=1e-6`, `hc_sinkhorn_iters=20`.

KDA per layer (all shapes from the checkpoint, layer 0):

    q/k/v_proj    [8192, 4096]     = 64 heads x 128
    q/k/v_conv1d  [8192, 1, 4]     each
    b_proj        [64, 4096]       per-head beta
    f_a_proj      [128, 4096]  f_b_proj [8192, 128]  dt_bias [8192]  A_log [64]
    g_a_proj      [128, 4096]  g_b_proj [8192, 128]
    o_norm        [128]            gated RMS norm over head_dim
    o_proj        [4096, 8192]

## Weight name mapping (`transformers/conversion_mapping.py`, key `glm5_next`)

| checkpoint | module | operation |
|---|---|---|
| `self_attn.{f_a_proj,f_b_proj,dt_bias,A_log}` | `self_attn.forget_gate.*` | rename |
| `hc_{attn,ffn}_{fn,base,scale}` | `{attn,ffn}_hc.{fn,base,scale}` | rename |
| `self_attn.{q,k,v}_conv1d.weight` | `self_attn.conv1d.weight` | **`Concatenate(dim=0)`**, order q→k→v |
| `mlp.experts.*.{gate,up}_proj.weight` | `mlp.experts.gate_up_proj` | `MergeModulelist(0)` + `Concatenate(1)` |
| `mlp.experts.*.down_proj.weight` | `mlp.experts.down_proj` | `MergeModulelist(0)` |

The three separate conv1d tensors merging into one is the easiest of these to
get wrong silently: concatenating in the wrong order produces a correctly
shaped weight and a model that runs.

## Traps found so far

**`kda_layers` indexing differs between models.**
`KimiLinearConfig.is_kda_layer` tests `(layer_idx + 1) in kda_layers` — Kimi's
list is 1-indexed. GLM-5.3's is 0-indexed and agrees with `layer_types`
position for position. Reusing Kimi's predicate mislabels **23 of 45 layers**
and does not raise. Derive from `layer_types`, which is positional.

**`head_v_dim` cannot come from `config.v_head_dim`.**
sglang's `KimiDeltaAttention` sets `self.head_v_dim = config.v_head_dim`. For
Kimi that is the right field. For GLM-5.3 `v_head_dim=256` is the **MLA** value
while the KDA head is **128** (`v_proj` is [8192, 4096] = 64 x 128). Reusing
`KimiDeltaAttention` unmodified would build the KDA layers at the wrong width.
Use `linear_head_dim`.

**mHC cannot be derived from tensor shapes.** `base` [24] and `fn`
[24, 4*hidden] split 4 + 4 + 16 at `hc_mult=4`, which is the right split and
suggests plain linear stream mixing. It is not: `comb` is softmaxed and then
projected onto the doubly-stochastic manifold by 20 Sinkhorn-Knopp iterations
(first step columns only, then iters-1 rows-then-columns), `post` is
`2*sigmoid` (range [0,2]), `pre`/`comb` carry epsilons, and the whole mapping
runs in float32 under an *unweighted* RMS norm. Measured: a plain-sigmoid
`post` is off by rel 5.0e-01, and a comb without Sinkhorn has column sums
spanning [0.401, 2.396] where doubly stochastic is ~1.0. Both variants load
every tensor and emit fluent nonsense.

**`linear_attn_config` is still a nested dict in this checkpoint.** The
reference config flattens it into `linear_head_dim` / `linear_num_heads` /
`linear_conv_kernel_dim` / `linear_lower_bound`. Without that conversion every
KDA layer silently takes the default geometry. sglang's `Glm5NextTextConfig`
now does it.

## Why the pool cares about `qk_rope_head_dim=0`

`k_pe` is never quantized, so it is a fixed `2*rope` byte floor per token per
layer — 128 B of a 288 B cell at rope 64, i.e. 44% of the row. At rope 0 it
vanishes: 160 B against BF16's 1024 = **6.40x** (vs 4.00x on GLM-5.2/K3).
`packed_latent_bytes_per_token` is already parameterized by rope width, so the
arithmetic needs no change; the untested part is allocating a **zero-width**
`rope_buf`.

But **only 11 of 45 layers have a KV cache at all** — the other 34 are KDA with
nothing to compress. Never quote the cell ratio as a model-level saving.

## Order of work

Config → KDA → DSA → assembly → **BF16 control first** → packed pool. The
BF16-first rule is the Kimi-K3 lesson: `use_expanded_mha_cache=True` was hiding
a broken forward path while I spent two cycles tuning the quantizer.

## Why not just take upstream sglang? (checked, 2026-08-28)

Fair question, and it should have been checked before writing anything. The
answer is "half yes", and the half that is yes was a real miss.

**sglang 0.5.18 does NOT support this model.** `glm5_next` / `Glm5Next` is zero
hits across the wheel; its glm models stop at glm4_moe / glm_ocr / glm_image_vl.
So rebasing would not produce a running GLM-5.3-Flash: the config, the KDA
wiring, the k-pool selector and the assembly all still have to be written.

**But upstream HAS an mHC kernel that I hand-wrote in pure torch.**
`sglang/kernels/ops/layernorm/mhc.py` (1618 lines, TileLang-jitted
`hc_split_sinkhorn_kernel`) computes exactly the same thing -- `pre =
sigmoid(mixes*scale[0] + base) + eps`, `post = 2*sigmoid(...)`, Sinkhorn. It is
there for **DeepSeek-V4**, which also uses mHC (`sglang/srt/configs/deepseek_v4.py`
carries `hc_mult` and `hc_sinkhorn_iters`).

**It is not a cherry-pick.** It imports `sglang.kernels.jit.utils`,
`sglang.srt.layers.attention.dsa.utils` and `sglang.srt.layers.utils.common`;
this fork has no `sglang/kernels/` tree at all and has `nsa/` where upstream has
`dsa/`. `tilelang` is not installed either. Taking it means grafting a chunk of
upstream's structure, not copying a file.

**Also corrects an earlier claim of mine:** I said `deepseek_v4` was "zero hits
in the fork" and left the impression the model was unsupported everywhere.
Upstream ships `deepseek_v4.py` and `deepseek_v4_dspark.py`. It is unsupported
*here*, not unsupported.

**Cost of switching base**: this fork carries the whole OSCAR / packed-INT2
stack -- packed pool, GF kernel, rotations, the mixed-KV window arena -- i.e.
everything behind 4.00x and the 86.67% score. Rebasing means porting all of it.

**Decision**: keep this base; write the model here; keep the torch mHC for now.
It is bit-exact against the HF reference, so it also serves as the oracle for
whichever kernel replaces it.

**But the torch mHC has to be profiled once the model runs.** Op count per
decode step, 45 layers x 2 mHC sites: **16,650 small-tensor launches, 13,680 of
them the Sinkhorn loop** (19 iterations x 8 ops). The tensors are
`[tokens, 4, 4]` -- negligible FLOPs, essentially pure launch overhead.
Mitigating: sglang captures decode into CUDA graphs, so launch cost is paid at
capture rather than replay, which makes a naive per-launch extrapolation an
overestimate. This is an op COUNT, not a timing. It says "profile mHC", not
"the torch version is already too slow".
