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
| 0 | repository layout, porting reference, image base | image boots upstream models | layout committed; base image pulling |
| 1 | KV-quant core: `--kv-cache-dtype int2`, `UnifiedInt2HPKVPool` + allocator, mixed HP windows, radix-cache tiers, scheduler flush, triton INT2 prefill/decode, QuantKernel, pool pricing, env knobs | Qwen3-8B INT2 at 8K ctx, radix + graph on | not started |
| 2 | MHA/GQA models: qwen3, qwen3_5 (hybrid GDN), qwen3_moe, gemma4_unified (two-group), minimax_m2.7 | each model's smoke | not started |
| 3 | MLA packed latent (`MLAPackedInt2KVPool`, `NSAPackedInt2KVPool`, packed decode kernels) on upstream's DSA backend for GLM-5.2/5.3, trtllm_mla for Kimi-K3; MSA for MiniMax-M3 on upstream's native backend | GLM-5.2 recall probe past 2K tokens, then smoke | not started |
| 4 | harness: `rotation/` paths (`sglang-research/python` -> `python`), run scripts, verify sweep, bench | 12-model sweep | not started |
| 5 | GPQA @64K + 64K decode speed for all 12, table and README | | not started |

## Notes

* The old tree's `nsa` prefill garbles any prompt past 2048 tokens in BF16
  (passphrase recall OK at 1.2K tokens, garbage from 5.3K; identical for fused
  and unfused top-k and for trtllm vs flashmla_sparse decode). That is the
  defect behind the 33.8% sparse-mode GPQA, and the first thing phase 3 checks
  on the new tree is that upstream's DSA passes the same probe.
* MiniMax-M3's sparse prefill kernel asserts `gqa_group_size * block_size_q
  <= 128`; at TP=8 the group is 8, so the query tile must be capped at 16.
