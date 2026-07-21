# Plan — OSCAR on MiniMax-M3

## Goal
Get a first, credible **"does OSCAR INT2 KV work on MiniMax-M3"** signal. Port MiniMax-M3 into the
CoQuant sglang fork (dense, text-only), serve it BF16 on the cluster, then run the **same model dense**
with **INT2 OSCAR KV** and compare GPQA-Diamond against the BF16 baseline. Success = INT2-OSCAR within a
few points of dense BF16.

## Context / Decisions (from interview + recon 2026-06-15)
- **Model**: `MiniMaxAI/MiniMax-M3`, `model_type=minimax_m3_vl`, arch `MiniMaxM3SparseForConditionalGeneration`.
  ~428B total / ~23B active MoE (128 experts, 4/tok, 1 shared), 60 layers, hidden 6144, 1M ctx, VL multimodal.
- **Attention = GQA** (64 q-heads, 4 kv-heads, head_dim 128), **partial RoPE** (rotary_dim 64, factor 0.5),
  **per-head qk_norm**, **gemma_norm**, rope_theta 5e6. Plus **MiniMax Sparse Attention (MSA)** = top-16-block
  pre-filter from layer 3. M3 ≈ **M2 + MSA + VL** on a GQA substrate.
- **Why staged works**: OSCAR's INT2 KV path attaches to *dense* GQA attention. The official SGLang M3 **BF16**
  recipe runs **dense** (no MSA flags); web confirms "M3 uses GQA as substrate… kernels reused with little/no
  modification." So dense-GQA M3 (ignore the MSA indexer params) is a valid substrate, exactly like M2.
- **OSCAR wiring is model-agnostic** (confirmed): `UnifiedInt2HPKVPool` + rotation loading + env vars are external
  to the model def. Adding M3 = (1) write `minimax_m3.py`, (2) generate rotations, (3) launch with OSCAR env.
- **Fork base** = M2.7-era (~sglang 0.5.12); **no** M3 def (`minimax_m2.py` exists, no `minimax_m3.py`). Upstream
  M3 support = PR #27944 (≥0.6.x). We must bring M3 **into this fork** (OSCAR's KV pool can't move to new sglang).
- **Compute**: cluster is **H100-80GB only** (no H200). BF16 weights ~856GB → **TP16 across 2× 8-GPU nodes**.
  Many healthy 8-GPU `research-common-h100-*` nodes free. Exclude known-bad `h100-117`.
- **Reusable infra**: launch recipe + `rotation/eval_oscar_gpqa.sh` + `rotation/_eval_runner/run_simple_eval.py`
  + M2.7 K8s manifests. OSCAR env vars: `SGLANG_ENABLE_MIXED_KV_WINDOWS=1`, `SGLANG_MIXED_KV_PREFIX_TOKENS=64`,
  `SGLANG_MIXED_KV_RECENT_TOKENS=256`, `SGLANG_OSCAR_{K,V}_ROTATION_PATH`, `SGLANG_OSCAR_K_CLIP_RATIO=0.96`,
  `SGLANG_OSCAR_V_CLIP_RATIO=0.92`, `SGLANG_OSCAR_ABSORB_V_ROTATION=1`; server `--kv-cache-dtype int2
  --kv-cache-quant-group-size 128 --prefill-attention-backend fa3 --decode-attention-backend triton`.
  Eval needs `--disable-radix-cache` (mixed-KV + radix drops GPQA).

## Constraints / non-goals
- **BF16 weights** (no FP8/FP4), per decision. INT2 applies to **KV only**.
- **MSA runs dense** for this task. **Full INT2-OSCAR↔MSA top-k kernel interop is a NON-GOAL here** (deferred;
  fork already has an `nsa_backend` to build on later).
- **Text-only**: do not load/eval the vision (image/video) path. Skip `vision.*` / VL tensors in the loader.
- **One benchmark first** = GPQA-Diamond (198Q), reusing existing harness. No multi-bench sweep, no throughput/
  perf benchmarking, no commits/PRs unless asked.
- K8s-first; keep the local box free. `/shared/huggingface` cache; `HF_TOKEN` from env; wandb for tracking.

## ★ HEADLINE RESULTS (final, 2026-06-16 → 2026-07-10) — COMPLETE
GPQA-Diamond acc = 198Q, 1-shot, T=1.0, 32K think, single seed (SE≈2.5%). GPQA PPL = teacher-forced NLL of the 40
BF16-generated responses (136,916 identical resp tokens per config) through the serving stack with
`--chunked-prefill-size 512` so quantized KV is actually **read** (Step I).
| Config | GPQA acc | GPQA PPL | Δppl vs BF16 |
|---|---|---|---|
| **M3 BF16 dense** | **88.38%** (175/198) | **1.3140** | — |
| INT2 OSCAR, data-free Hadamard | 84.85% (168/198) | – | – |
| INT2 OSCAR, rank-0 calib (deficient dump, Step F) | 84.85% (168/198) | – | – |
| **INT2 OSCAR, all-head calib (Step G)** | **85.35%** (169/198) | **1.3479** | **+2.6%** |
| INT2 OSCAR, all-head + Lloyd-Max (Step H) | 84.85% (168/198) | 1.3672 | +4.0% |
- **OSCAR INT2 KV works on MiniMax-M3**: ~85% = 96% of BF16 with ~6.4× KV capacity (3,408,248 vs 532,539 tok).
  M3 INT2 84.85–85.35% > prior MiniMax-M2.7 INT2 80.3%.
- **Is there an accuracy drop at all? NOT statistically resolved.** Single-seed Δacc = −3.0 pts vs SE(diff)≈3.4
  (per-run SE≈2.4, T=1.0 sampling) → within ~1σ of run-to-run noise, and all INT2 variants land 168–169/198
  regardless of rotation/quantizer (acc is an insensitive, saturated metric here). The exact evidence of degradation
  is PPL: **+2.6% ppl / +0.026 nats per token** — real but small; whether it translates to any GPQA acc loss would
  need multi-seed (n_repeats≥4) or per-question McNemar to resolve. Plausibly Δacc ≈ 0.
- **All-head calibration is the principled rotation** (V relL2 0.4794 vs Hadamard 0.5000) and the nominal best
  (85.35%, PPL-measured config), but the acc gain (+1Q) is within noise → **data-free Hadamard remains a valid
  practical default** (no calibration pipeline needed).
- **Lloyd-Max is measurably HARMFUL on M3** (+1.4% ppl over uniform INT2; −1Q) → do NOT enable `SGLANG_LLOYD_MAX`.
- Remaining untested gap-closer: **per-head rotations** (one shared 128×128/layer can't fit divergent heads).

## Acceptance criteria — ALL MET
- [x] `minimax_m3.py` loads M3 BF16 on TP16 (2×8 H100), coherent sanity generation.
- [x] Dense BF16 GPQA-Diamond baseline: **88.38%**.
- [x] INT2 OSCAR KV serves stably, within ~3 pts of BF16 (85.35% all-head / 84.85% Hadamard) → OSCAR works on M3.

## Steps — all complete
- [x] **A–C**: port (`minimax_m3.py`, 711/711 weights) · BF16 TP16 serve · BF16 GPQA **88.38%**.
- [x] **D–E**: Hadamard rotations · INT2 GPQA **84.85%**.
- [x] **F**: rank-0 calibrated rotation **84.85%** — dump was deficient (1 TP shard = 4/64 q, 1/4 KV heads).
- [x] **G**: all-head calibrated rotation (merge 16 ranks) **85.35%** — the principled calib; +1Q = within noise.
- [x] **H**: Lloyd-Max (after porting LM into the decode-flush kernel) **84.85%**, PPL-worse → do NOT enable.
- [x] **I**: GPQA PPL harness + 3-way table (BF16 1.3140 / all-head 1.3479 / +LM 1.3672).

## Caveats (still true)
- **Dense ≠ MSA**: all numbers are dense-attention M3; may differ from MiniMax's published MSA numbers. The A/B is
  INT2-vs-BF16-**dense** (valid for the OSCAR question).
- Exclude known-bad `h100-117`; footprint 53.5 GB/GPU weights on TP16 → tune `--mem-fraction-static ~0.85`.

## M3 text-model spec (reverse-engineered from transformers `modeling_minimax_m3_vl.py` + weight index)
Authoritative for the port. Source refs cached in `/home/charlie/CoQuant/.RUD/m3oscar/m3_src/` (config, weight
index) and `m3_src/tf_ref/` (transformers modular/modeling).
- **Norms**: Gemma RMSNorm everywhere = `x_fp32 * rsqrt(mean(x²)+eps) * (1 + w)`, cast at end. Fork's
  `GemmaRMSNorm` (layernorm.py:466, residual-capable) is numerically identical → use for input/post/final + per-head q/k.
- **Attention** (60 layers): GQA 64 q / 4 kv heads, **head_dim=128** (≠ hidden/heads=96 — must use explicit head_dim),
  per-head Gemma q_norm/k_norm over head_dim, THEN partial RoPE (rotary_dim=64, θ=5e6). No output gate. Skip MSA
  `index_{q,k}_{proj,norm}` (layers 3-59) in dense mode.
- **MLP**: layers 0-2 DENSE (`mlp.{gate,up,down}_proj`, inter=dense_intermediate_size=12288); layers 3-59 MoE.
- **Activation = swigluoai** (dense MLP, shared expert, AND routed experts): `gate,up=chunk(2);
  gate.clamp(max=7); up.clamp(±7); out=(up+1)*gate*σ(1.702*gate)`. CHUNKED (M3 ships separate w1/w3), unlike
  gpt-oss interleaved → needs a chunked fused-kernel variant.
- **MoE** (57 layers): 128 experts (`experts.N.w1/w2/w3` = gate/down/up), top-4, sigmoid router + correction bias
  (`e_score_correction_bias`), renormalize, **routed_scaling_factor=2.0**; PLUS 1 shared expert
  (`shared_experts.{gate,up,down}_proj`, inter=3072). Combine: `out = 2.0*routed(x) + shared(x)`.
- **Weights**: all under `language_model.model.*` + `language_model.lm_head` (tie=false). SKIP `vision_tower.*`,
  `multi_modal_projector.*`, `patch_merge_mlp.*`, `*.index_*`. No MTP tensors in released weights. q/k/v separate →
  fuse to qkv_proj; dense+shared gate/up → fuse to gate_up_proj; experts via FusedMoE expert mapping.
- **Wrapper**: EntryClass `MiniMaxM3SparseForConditionalGeneration`; build `language_model` from
  `config.text_config`; sglang's `get_hf_text_config` feeds text dims to KV pool automatically. Text-only (no images).

## Port + eval gotchas (the non-obvious ones; full details in git history / memory)
- **MoE: compute the shared expert BEFORE `self.experts(...)`** — FusedMoE mutates `hidden_states` in-place;
  experts-first fed the shared expert corrupted input → total garbage output. (THE bring-up bug.)
- **`--dtype bfloat16` explicit** — M3 `text_config` has no `torch_dtype` → `dtype=auto` silently downcasts to fp16.
- **Eval client read timeout** — 32K-token thinking traces (~15 min/req) exceed the default ~600s; fixed with
  `httpx.Timeout(7200)` in `run_simple_eval.py` SglangChatSampler.
- **Pure TP only** (no EP/deepep): only the FusedMoE-triton path has the chunked swigluoai branch.
- Weights: 59/59 safetensors, 796GB at `/shared/huggingface`; INT2 KV capacity 3,408,248 tok vs BF16 532,539 (~6.4×).
- **Cluster ops**: nccl pool (~10 nodes) is often saturated; the `node-pool=compute,node-group=default` pool is the
  SAME H100+IB hardware and TP16/2-node NCCL works there — just swap the nodeSelector when nccl is full.
- **Shared HF cache `refs/main` hazard**: an online run bumped M3's `refs/main` (Jun 25) to an undownloaded revision
  → every `HF_HUB_OFFLINE=1` load broke. Fixed by reverting refs/main to `b8682713` AND pinning
  `--revision b8682713...` in manifests.

## Step F: rank-0 calibrated rotation (2026-06-16) — 84.85%, dump was DEFICIENT
QKV dump hook in `minimax_m3.py` (`DUMP_M3_QKV_DIR`, post-RoPE q/k/v) → `compute_kv_rotation.py --method qqt_sst
--composition r_h_pbr` → INT2 GPQA = **84.85% (168/198), identical to Hadamard**. Root cause understood in Step G:
the dump captured only ONE TP shard (4/64 q-heads, 1/4 KV-heads), so the per-layer shared rotation was fit to an
unrepresentative head sample. (Also established: calib token-count is NOT the lever — covariance converges by ~5K
tokens.) Artifacts: `investigation/03_calib_dump/` (dump manifest, rotations), `04_int2_oscar_calib/` (eval).

## Step G: ALL-HEAD calibrated rotation (2026-06-17) — 85.35%, the principled calib; gain within noise
Redo with every TP rank dumped (`layer_<id>/rank_<r>/{q,k,v}/`), merged to full q[T,64,128] / k,v[T,4,128]
(`05_calib_allhead/merge_qkv_allhead.py`; KV dedup verified intra-group identical / inter-group distinct), same
qqt_sst / r_h_pbr compute (~12K tok/layer, orth err ~1e-15).
- **Pre-eval verification** (`05_calib_allhead/verify_rotation.py`, INT2-asym relL2 on the merged all-head data):
  V clearly improves — all-head **0.4794** vs rank-0 0.4937 vs Hadamard 0.5000; K flat ~0.49 (qqt targets Q-cov,
  not raw-K error).
- **INT2 GPQA = 85.35% (169/198)** — +1Q over Hadamard/rank-0; within single-seed noise. Conclusion: the
  single-shared-per-layer OSCAR rotation is SATURATED on M3 GPQA. Eval ran on the default pool (nccl full);
  NCCL TP16 across default-pool nodes verified working.

## Step H: Lloyd-Max INT2 (2026-06-19) — measurably WORSE; do NOT enable
- **KEY GOTCHA found**: `SGLANG_LLOYD_MAX` only reached the prefill/SET kernel; the decode-FLUSH kernel
  (`gpu_flush_int2.py`) had NO LM path — and ~94% of GPQA tokens are generated (quantized via flush) → the flag was
  effectively a no-op for generation. (Same class of bug as the INT1 flush-LM gap.)
- **Ported LM into the flush kernel** (working tree, UNCOMMITTED): `LLOYD_MAX` branch in `_fused_flush_quant_body`
  mirroring the single-scale set-kernel LM (standardize → bucketize ±0.981 → uniform-equivalent scale/zero,
  LM_RATIO=1.16), threaded through grid kernel + `gpu_flush_int2_apply`/`gpu_flush_int2` + `common.py`
  (`kv_pool._lloyd_max`), NUM_GROUPS==1 guard. **Verified bit-exact vs set-LM** (0/2048 bytes, head_dim=128;
  `06_int2_oscar_lloydmax/verify_flush_lm.py`) + existing flush regression tests pass.
- **Result: GPQA 84.85% (−1Q vs all-head), and PPL +1.4% over uniform INT2 (Step I) → LM confirmed harmful on M3.**
  The LM here is the Gaussian closed-form levels (fixed ±0.981) + the LM_RATIO≈1.16 uniform-dequant approximation —
  mismatched for M3's rotated K/V. If revisited: fit data-calibrated per-layer levels from the all-head dump instead.
  LM stays OFF by default (correct).

## Step I: GPQA PPL (2026-07-10) — the sensitive metric; INT2 true cost = +2.6% ppl
Accuracy is saturated (168–169/198) → built a teacher-forced PPL harness (`07_gpqa_ppl/ppl_gpqa.py`): score the 40
BF16-generated GPQA responses' NLL via sglang `/generate` `return_logprob` + `logprob_start_len=len(prompt_ids)`,
server with `--chunked-prefill-size 512` so response tokens attend to earlier context as QUANTIZED cached prefix
(verified: chunked extend writes middle tokens INT2 via `_set_quant_kv_buffer_extend`, later chunks read them via
`dequantize_prefix_kv`; a single unchunked prefill reads bf16 and hides the quant effect entirely).
- **Results** (136,916 identical response tokens per config → deltas are exact):
  BF16 nll 0.2730 / ppl **1.3140** · INT2 all-head 0.2986 / **1.3479 (+2.6%)** · +LM 0.3128 / **1.3672 (+4.0%)**.
- ~35 min/config (vs ~75 min + noise for accuracy) → cheap A/B harness for future M3 KV experiments.
- Harness gotcha: transformers v5 `apply_chat_template(tokenize=True)` returns a **BatchEncoding** — `list()` over it
  yields its string keys → 400 on every request. Render with `tokenize=False` then `tok(txt, add_special_tokens=False)`.

## Progress Log
- 2026-06-15: Recon + port. `minimax_m3.py` written (~470 lines; GemmaRMSNorm, per-head qk-norm, chunked swigluoai
  fused-MoE branch, dense 0-2 / MoE 3-59, text-only wrapper, `mm_disabled_models` entry); offline weight-mapping
  validated 711/711 on meta device; Hadamard rotations generated; TP16 manifests written; 796GB weights downloaded
  via cluster job. Eval note: thinking model (`<mm:think>`), no reasoning parser, max-tokens 32768, T=1.0/top_p 0.95.
- 2026-06-16: BF16 bring-up (MoE shared-before-experts garbage bug found+fixed; dtype=auto→fp16 bug fixed; eval
  httpx timeout fixed) → **BF16 GPQA 88.38%**; INT2 OSCAR Hadamard **84.85%**; rank-0 calib **84.85%** (Step F).
- 2026-06-17: Step G all-head calib: merge 16-rank dump → verify (V relL2 0.4794 < 0.5000) → **GPQA 85.35%**.
  Ran on default pool (nccl saturated) — TP16 NCCL on default pool verified.
- 2026-06-19: Step H Lloyd-Max: found+fixed flush-kernel LM gap (bit-exact-verified port, uncommitted) →
  **GPQA 84.85%**, no help.
- 2026-07-09/10: Step I GPQA PPL: harness built (chunked-prefill methodology verified in code); fixed shared-cache
  refs/main breakage + transformers-v5 BatchEncoding gotcha → **BF16 1.3140 / all-head 1.3479 (+2.6%) /
  +LM 1.3672 (+4.0%)**. LM confirmed harmful; INT2 true cost quantified.

## FINAL STATUS (2026-07-10) — project complete
- **Deploy recipe**: INT2 OSCAR mixed-KV, all-head calibrated rotation (or Hadamard — within noise), LM OFF,
  clip K0.96/V0.92, HP prefix 64 / recent 256, fa3 prefill + triton decode, pure TP16, `--disable-radix-cache`.
- **Quality**: GPQA acc 85.35% vs BF16 88.38% (Δ within single-seed noise — see headline); PPL +2.6% (exact).
- **Uncommitted work in the worktree**: flush-kernel Lloyd-Max support (verified; default OFF) + the
  `minimax_m3.py` port + chunked-swigluoai MoE branch + dump hook. Commit to `zhongzhu/m3oscar` when asked.
- **Follow-ups (not started)**: per-head rotations (the remaining principled lever for the residual gap);
  data-calibrated LM levels from the all-head dump (only if LM is ever revisited); multi-seed GPQA (n_repeats≥4)
  if the accuracy question needs a definitive answer; MSA(top-k)×INT2 interop remains a non-goal/deferred.
