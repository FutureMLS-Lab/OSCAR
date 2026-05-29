"""Benchmark: Triton GQA decode vs FlashInfer at varying num_kv_splits.

Sweeps num_kv_splits from 1 to 64 for batch=1 (the under-served case),
compares against FlashInfer BF16 reference, and prints the optimal split count.

Run:
    python -m sglang.QuantKernel.bench_splits
"""
import os, sys, math
import torch
import triton

DEVICE = "cuda"

# ---------------------------------------------------------------------------
# Triton GQA decode (BF16)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sglang.srt.layers.attention.triton_ops.decode_attention import (
    decode_attention_fwd_grouped,
)

def _make_tensors(batch, q_heads, kv_heads, seq_len, head_dim, max_splits, dtype=torch.bfloat16):
    kv_group = q_heads // kv_heads
    q         = torch.randn(batch, q_heads, head_dim, dtype=dtype, device=DEVICE)
    k_buf     = torch.randn(batch * seq_len, kv_heads, head_dim, dtype=dtype, device=DEVICE)
    v_buf     = torch.randn(batch * seq_len, kv_heads, head_dim, dtype=dtype, device=DEVICE)
    o         = torch.empty(batch, q_heads, head_dim, dtype=dtype, device=DEVICE)
    kv_indptr = torch.arange(0, (batch + 1) * seq_len, seq_len, dtype=torch.int32, device=DEVICE)
    kv_indices = torch.arange(batch * seq_len, dtype=torch.int64, device=DEVICE)
    attn_logits = torch.empty(batch, q_heads, max_splits, head_dim, dtype=torch.float32, device=DEVICE)
    attn_lse    = torch.empty(batch, q_heads, max_splits, dtype=torch.float32, device=DEVICE)
    num_kv_splits = torch.empty(batch, dtype=torch.int32, device=DEVICE)
    sm_scale = head_dim ** -0.5
    return q, k_buf, v_buf, o, kv_indptr, kv_indices, attn_logits, attn_lse, num_kv_splits, sm_scale


def bench_triton(batch, q_heads, kv_heads, seq_len, head_dim, n_splits, warmup=25, rep=200):
    q, k_buf, v_buf, o, kv_indptr, kv_indices, attn_logits, attn_lse, num_kv_splits, sm_scale = \
        _make_tensors(batch, q_heads, kv_heads, seq_len, head_dim, n_splits)
    num_kv_splits.fill_(n_splits)

    def fn():
        decode_attention_fwd_grouped(
            q, k_buf, v_buf, o,
            kv_indptr, kv_indices,
            attn_logits, attn_lse,
            num_kv_splits, n_splits,
            sm_scale, v_scale=1.0,
        )

    fn(); torch.cuda.synchronize()
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    # BF16 K+V bytes touched
    kv_bytes = batch * seq_len * kv_heads * head_dim * 2 * 2  # K+V, bf16
    return ms, kv_bytes / (ms * 1e-3) / 1e9


# ---------------------------------------------------------------------------
# FlashInfer BF16 reference
# ---------------------------------------------------------------------------
def bench_flashinfer(batch, q_heads, kv_heads, seq_len, head_dim, warmup=25, rep=200):
    try:
        import flashinfer
    except ImportError:
        return None, None

    dtype = torch.float16  # FlashInfer fp16
    qo_indptr  = torch.arange(batch + 1, dtype=torch.int32, device=DEVICE)
    kv_indptr  = torch.arange(0, (batch + 1) * seq_len, seq_len, dtype=torch.int32, device=DEVICE)
    q  = torch.randn(batch, q_heads, head_dim, dtype=dtype, device=DEVICE)
    k  = torch.randn(batch * seq_len, kv_heads, head_dim, dtype=dtype, device=DEVICE)
    v  = torch.randn(batch * seq_len, kv_heads, head_dim, dtype=dtype, device=DEVICE)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD")
    wrapper.plan(qo_indptr, kv_indptr, q_heads, kv_heads, head_dim,
                 causal=False, q_data_type=dtype, kv_data_type=dtype)

    def fn():
        wrapper.run(q, k, v)

    fn(); torch.cuda.synchronize()
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    kv_bytes = batch * seq_len * kv_heads * head_dim * 2 * 2
    return ms, kv_bytes / (ms * 1e-3) / 1e9


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_sweep(batch, q_heads, kv_heads, seq_len, head_dim, splits_list):
    fi_ms, fi_gbps = bench_flashinfer(batch, q_heads, kv_heads, seq_len, head_dim)
    fi_str = f"{fi_gbps:6.0f} GB/s ({fi_ms*1e3:.1f} µs)" if fi_gbps else "N/A"

    print(f"\n[B={batch} S={seq_len} Hq={q_heads} Hkv={kv_heads} D={head_dim}]")
    print(f"  FlashInfer: {fi_str}")
    print(f"  {'splits':>6}  {'ms(µs)':>8}  {'GB/s':>8}  {'vs FI':>7}")

    best_gbps, best_splits = 0, -1
    for ns in splits_list:
        if ns > seq_len:
            continue
        ms, gbps = bench_triton(batch, q_heads, kv_heads, seq_len, head_dim, ns)
        vs = f"{gbps/fi_gbps*100:.1f}%" if fi_gbps else "  --"
        print(f"  {ns:>6}  {ms*1e3:>8.1f}  {gbps:>8.0f}  {vs:>7}")
        if gbps > best_gbps:
            best_gbps, best_splits = gbps, ns

    print(f"  → optimal splits={best_splits} ({best_gbps:.0f} GB/s)")
    return best_splits, best_gbps, fi_gbps


if __name__ == "__main__":
    # Qwen3-4B GQA config: q=32, kv=8, group=4, D=128
    Q, KV, D = 32, 8, 128
    splits = [1, 2, 4, 8, 16, 32, 64]

    print("=" * 65)
    print("BF16 GQA decode: Triton split sweep vs FlashInfer")
    print("=" * 65)

    for batch, seq in [(1, 4096), (1, 8192), (1, 16384), (4, 4096), (8, 4096), (32, 4096)]:
        run_sweep(batch, Q, KV, seq, D, splits)

    # Show what the NEW heuristic gives (max_kv_splits=64)
    print("\n=== New heuristic (max_kv_splits=64) ===")
    device_sms = torch.cuda.get_device_properties(0).multi_processor_count
    kv_group = Q // KV
    block_h = min(16, kv_group)
    MAX_SPLITS = 64
    for batch, seq in [(1, 4096), (1, 8192), (1, 16384), (4, 4096), (8, 4096), (32, 4096)]:
        token_grid = batch * math.ceil(Q / block_h)
        sm_fill = min(math.ceil(2 * device_sms / token_grid), MAX_SPLITS)
        ideal_chunk = int(128 * math.sqrt(batch))
        seq_s   = min(math.ceil(seq / ideal_chunk), MAX_SPLITS)
        splits_2 = max(sm_fill, seq_s)
        chunk_2 = math.ceil(seq / splits_2)
        per_seq = math.ceil(seq / chunk_2)
        print(f"  B={batch} S={seq}: splits={per_seq}  (sm_fill={sm_fill}, seq_based={seq_s}, ideal_chunk={ideal_chunk})")
