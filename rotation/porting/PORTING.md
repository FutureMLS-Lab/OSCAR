# Porting OSCAR onto upstream sglang main

Status page for the move from the vendored `sglang-research/` tree (upstream
`f652135d52`, 2026-04-11, plus fork changes) to a repository whose root *is*
upstream sglang. The branch is `sglang-rebase`; the reference delta is
`git diff f652135d52 oscar-vendored` in the porting clone (41.9k patch lines),
summarised per file in `port_checklist.md`.

## Why not a textual rebase

Dry run of `git rebase --onto main f652135d52` on the vendored tree: 111
content conflicts, 18 modify/delete, 2 rename. Upstream has ~7.5k commits since
the base, renamed NSA to DSA, split the DSA indexer, moved kernels into
`sglang.kernels` (and `sgl-kernel/` out of the root), replaced
`cuda_graph_runner.py` with `model_executor/runner/*`, and moved KV-pool
construction into `mem_cache/kv_cache_configurator.py`. The OSCAR hooks are
re-applied by hand against the current code, feature by feature, each with a
smoke test before the next.

## Environment

Upstream pins torch 2.13 / CUDA 13.0 / flashinfer 0.7.0.post1 /
sglang-kernel 0.4.8 / transformers 5.17. The OSCAR image is built FROM the
official `lmsysorg/sglang:v0.5.21-cu130` (main is 166 commits past v0.5.21)
with this repository laid over it. Every number in the verification table has
to be re-measured on this stack; the numbers in PR #26 are the old tree's.

## Phases

| phase | content | smoke | status |
|---|---|---|---|
| 0 | repository layout, porting reference, image base | image boots upstream models | done (`docker/Dockerfile.oscar` on `lmsysorg/sglang:v0.5.21-cu130`) |
| 1 | KV-quant core: `--kv-cache-dtype int2`, `UnifiedInt2HPKVPool` + allocator, mixed HP windows, radix-cache tiers, scheduler flush, triton INT2 prefill/decode, QuantKernel, pool pricing, env knobs | Qwen3-8B INT2 at 8K ctx, radix + graph on | **PASS** (image v55: prefix-cache hit, 62 captured graph shapes, 2048-token probe clean) |
| 2 | MHA/GQA models: qwen3, qwen3_5 (hybrid GDN), qwen3_moe, gemma4_unified (two-group), minimax_m2.7 | each model's smoke | qwen3-4b-think / qwen3-32b / qwen3-30b-a3b / minimax-m2.7 / qwen3.5-4b / qwen3.5-35b **PASS** (v61; the hybrids needed the prefill checkpoint clamped to the mixed-KV insert ceiling so the first request donates FULL+MAMBA). gemma4-12b: the first garbled verdicts were the PROBE, not the port -- the raw `/generate` prompt has no `<bos>` under transformers 5.12 (`GemmaTokenizer.add_bos_token=False`), and the checkpoint echoes the prompt tail in HF transformers exactly as in sglang (BF16 upstream image included); the probe now goes through the chat template, re-verification pending |
| 3 | MLA packed latent (`MLAPackedInt2KVPool`, `NSAPackedInt2KVPool`, packed decode kernels) on upstream's DSA backend for GLM-5.2/5.3, trtllm_mla for Kimi-K3; MSA for MiniMax-M3 on upstream's native backend | GLM-5.2 recall probe past 2K tokens, then smoke | **GLM-5.2 and GLM-5.3 PASS** on upstream's DSA backend (flashmla_sparse prefill / trtllm decode, packed pool 4.00x, graph + prefix cache on); GLM-5.2 passphrase recall at 1.2K/5.3K/8.9K/12.5K/16.0K-token prompts **OK on both the DSA BF16 and the DSA packed-INT2 arm** (the old fork's >2K garbling is gone). MiniMax-M3 INT2 ported into upstream's MSA backend (BF16 staging, commit a5fe0c4568), smoke pending; Kimi-K3 pending |
| 4 | harness: `rotation/` paths (`sglang-research/python` -> `python`), run scripts, verify sweep, bench | 12-model sweep | paths, `dsa` backend name, conda-free eval/bench drivers done; the image must carry the `simple_evals` submodule (git archive drops it, the v61 GPQA job scored nothing and said so) |
| 5 | GPQA @64K + 64K decode speed for all 12, table and README | | Qwen3-8B 64K bench on v61 (B200, bs 1): INT2 13.65 ms/tok vs triton-BF16 34.99 (0.39x) vs flashinfer-BF16 5.55 (2.46x slower); GPQA rerun queued on v62 |

## Notes

* The old tree's `nsa` prefill garbles any prompt past 2048 tokens in BF16
  (passphrase recall OK at 1.2K tokens, garbage from 5.3K; identical for fused
  and unfused top-k and for trtllm vs flashmla_sparse decode). That is the
  defect behind the 33.8% sparse-mode GPQA, and the first thing phase 3 checks
  on the new tree is that upstream's DSA passes the same probe.
* MiniMax-M3's sparse prefill kernel asserts `gqa_group_size * block_size_q
  <= 128`; at TP=8 the group is 8, so the query tile must be capped at 16.

## Phase 1 port notes (2026-10-01)

* Base commit is `bd78095030`, the merge-base of `v0.5.21` and `main`, so the
  tree matches the official `lmsysorg/sglang:v0.5.21-cu130` image's pins
  (torch 2.13.0+cu130, transformers 5.12.1, flashinfer 0.6.18, sglang-kernel
  0.4.7). Later upgrades are `git merge upstream/main`.
* `--kv-cache-dtype int2` is served only by the OSCAR mixed-KV pool. The old
  tree also carried an int2 branch inside `MHATokenToKVPool` (no BF16
  windows); it was not ported, and the configurator raises when int2 is
  requested without the unified gate instead of falling through.
* Upstream's `HybridLinearKVPool` already accepts an externally built
  `full_kv_pool`; the OSCAR inner pools go in through it. Its attribute
  forwarding (`__getattr__`) is enabled only for OSCAR inner pools so a plain
  BF16 hybrid keeps upstream's exact surface.
* Kernels that upstream moved to `sglang.kernels.ops.attention.decode_attention`
  are imported from there; the INT2 decode kernels live in
  `layers/attention/triton_ops/decode_attention.py` next to them.
* `KVWriteLoc`: the OSCAR pools unwrap it themselves, and the triton / FA3
  backends pass bare slot tensors to them.
* CUDA-graph padding: the decode runner points padded `out_cache_loc` entries
  at the mixed pool's reserved HP-prefix page 0 through the buffer registry's
  `out_cache_loc_pad_value` instead of patching the replay path.
* Not ported: K3-only files and env knobs (upstream has Kimi-K3 natively), the
  graph-vs-eager verify diagnostic, the old `minimax_sparse_backend.py`
  (MiniMax-M3 INT2 goes into upstream's native MSA backend in phase 3).
* Dropped as duplicates of what upstream now ships natively: the copied
  `configs/gemma4_unified.py` + `multimodal/processors/gemma4_unified_*`
  (upstream registers `gemma4_unified` through `_Gemma4UnifiedConfigAlias` and
  `processors/gemma4_unified.py`) and `layers/attention/minimax_sparse_kernels/`
  (upstream's MSA kernels live in `layers/attention/minimax_sparse_ops/`).
  `layers/attention/minimax_sparse_staging.py` stays for phase 3, where the
  INT2 staging is attached to upstream's `minimax_sparse_backend.py`.
* Phase 2 still owes the gemma4 model-side INT2 guards (the old
  `models/gemma4_unified.py` skipped its fused KV write when the pool dtype is
  int2) on upstream's `gemma4_causal.py` / `gemma4_unified.py`, and a
  prefix-cache answer for hybrid SSM models: done -- the mixed-KV tree
  semantics are also on `UnifiedRadixCache` (FULL + MAMBA), with the mamba
  checkpoint declined past the insert ceiling (the first request on a hybrid
  model donates FULL only; the cache bootstraps on the second).

## Phase 2/3 plan (from the code surveys, 2026-10-01)

**Hybrid SSM + mixed KV + prefix cache.** Upstream serves Qwen3.5 (GDN) and
Kimi-K3 (KDA) through `UnifiedRadixCache` (FULL + MAMBA components); the
mixed-KV tier semantics live on `RadixCache`. The port puts them on the unified
cache too: the tier cap at the cache level of `match_prefix` (the Rust tree-core
adapter forwards only key/extra_key/cache_salt, so the core cannot see a new
param), the HP-recent trim and the slack cutoff through the per-component
`prepare_for_caching_req` / `floor_cache_len` pair, the slack drop in
`on_release` before its `inserted` early return, the `insert_req` early return,
and `max()` on the two `cache_protected_len` writes. `MambaComponent` requires
`enable_mamba_extra_buffer` once page_size > 1 (`--mamba-radix-cache-strategy
auto` already resolves to `extra_buffer` then). Until that lands the factory
raises for the combination.

**Packed MLA latent on the DSA backend.** `dsa_backend.py` fetches the layer's
KV exactly once per phase (`get_key_buffer` in forward_extend, forward_decode
and `_forward_trtllm`) and the packed pool refuses that call, so the staging
hook has one insertion point per phase. The top-k table is PAGED (absolute
slots, -1 padded) for a bf16 pool because the RAGGED transform is gated on an
fp8 cache; at decode `metadata.page_table_1` may be None (fused top-k drops the
wide table), so the slots come from the per-layer top-k table itself. The
staging buffers (`max_num_tokens x index_topk` rows) are preallocated in
`init_cuda_graph_state`. Rows materialised from the packed pool are in the
ROTATED frame, so `forward_mla.py` rotates `q_nope_out` with
`rotate_latent` before attention and un-rotates `attn_output` before the
`w_vc` bmm, on both the absorbed path and the concat (triton) path, refusing
the fused-bmm / fused-rope producers that bypass the plain `q_nope_out`.
GLM-5.2/5.3 on Blackwell with a bf16 cache resolve to prefill
`flashmla_sparse`, decode `trtllm`; the dense one-shot MHA prefill never reads
the pool.
