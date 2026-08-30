# gemma-4-12B-it (model_type: gemma4_unified) — OSCAR INT2 KV

Heterogeneous hybrid-SWA model: 40 sliding layers (8 KV heads x head_dim 256,
window 1024) + 8 full-attention layers (1 KV head x head_dim 512, k_eq_v,
partial proportional RoPE). Served via this fork's
`Gemma4UnifiedForConditionalGeneration` shim (reuses gemma4_causal.py).

ENV: needs transformers >= 5.5 (the 5.3.0 default cannot import gemma4).
Set `PY` to such an env (sglang deps + flashinfer/sgl_kernel for serving).

## Pipeline
1. `PY=<py> bash save_qkv_gemma4_12b.sh`   # dump Q/K/V (eval-fork DUMP_KVCACHE hook)
2. `PY=<py> bash compute_rotation.sh`       # per-layer rotations (256/512)
3. `PY=<py> MODE=int2 bash eval_gpqa.sh`     # INT2 OSCAR GPQA (MODE=bf16 for baseline)

## Status (2026-06-04) — DONE ✅
- BF16 GPQA-Diamond = **62.63%** (full 198).
- **INT2 OSCAR GPQA-Diamond = 62.63%** (full 198) — **Δ 0.0pp vs BF16**.
  The full-attention layers (head_dim 512) ARE quantized to INT2.
- INT2 serving uses a **two-geometry-group `UnifiedInt2HPKVPool`** (gemma is the first
  heterogeneous / hybrid-SWA OSCAR model — uniform-geometry OSCAR models are unaffected).
- Key fix: sliding layers must be windowed in the mixed-KV **decode** path (not just
  prefill); without it INT2 scored 50.0% (sliding layers read full context in decode).
- Quant precision (Lloyd-Max) and V-rotation were ruled out as the cause via diagnostics.
