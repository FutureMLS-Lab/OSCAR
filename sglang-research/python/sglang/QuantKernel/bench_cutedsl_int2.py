"""Benchmark INT2 KV cache quantize + decode kernels: Triton vs CuteDSL.

Defaults match the validated Qwen3-4B Thinking eval config
(``head_dim=128``, ``group_size==head_dim`` per-row scale/zero).

Exit code 0 on speedup >= ``--assert-speedup``; non-zero otherwise.
"""

import argparse
import sys

import torch
import triton

from sglang.srt.layers.attention.triton_ops.decode_attention import (
    decode_attention_fwd_grouped_quant_int2,
    decode_attention_fwd_normal_quant_int2,
)
from sglang.srt.mem_cache.kv_quant_kernels import _launch_quantize_int2
from sglang.QuantKernel.cutedsl_int2_kv import (
    _launch_quantize_one,
    cuda_decode_attention_fwd_int2,
    cutedsl_decode_attention_fwd_int2,
    _load_cuda_decode_extension,
    get_cuda_invocation_counters,
    reset_cuda_invocation_counters,
)


# Optional FlashInfer fp16 reference (skip if not importable).
_HAS_FLASHINFER = False
try:
    import flashinfer  # noqa: F401
    _HAS_FLASHINFER = True
except Exception:
    pass


# H100 HBM3 peak DRAM bandwidth used for utilization percentages.
H100_HBM3_PEAK_GBPS = 3350.0


def _bench(fn, *, warmup=25, rep=200):
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep)


def bench_quantize(num_tokens: int, num_heads: int, head_dim: int):
    device = "cuda"
    x = torch.randn(num_tokens, num_heads, head_dim,
                    dtype=torch.bfloat16, device=device)
    loc = torch.arange(num_tokens, dtype=torch.int32, device=device)
    cache = torch.empty(num_tokens, num_heads, head_dim // 4,
                        dtype=torch.uint8, device=device)
    sz = torch.empty(num_tokens, num_heads, 2,
                     dtype=torch.float32, device=device)
    x2 = torch.randn_like(x)
    cache2 = torch.empty_like(cache)
    sz2 = torch.empty_like(sz)

    def run_tri():
        _launch_quantize_int2(x, loc, cache, sz, None)
        _launch_quantize_int2(x2, loc, cache2, sz2, None)

    def run_dsl():
        _launch_quantize_one(x, loc, cache, sz, None)
        _launch_quantize_one(x2, loc, cache2, sz2, None)

    run_tri(); run_dsl(); torch.cuda.synchronize()
    tri_ms = _bench(run_tri)
    dsl_ms = _bench(run_dsl)
    bytes_read = x.element_size() * x.numel()
    bytes_written = cache.numel() + sz.element_size() * sz.numel()
    total_bytes = bytes_read + bytes_written
    tri_gbps = total_bytes / (tri_ms * 1e-3) / 1e9
    dsl_gbps = total_bytes / (dsl_ms * 1e-3) / 1e9
    return tri_ms, dsl_ms, tri_gbps, dsl_gbps


def _bench_flashinfer_decode(
    batch: int, q_heads: int, kv_heads: int, head_dim: int,
    seq_len: int,
):
    """Run a fp16 single-query decode through FlashInfer for the same shape.

    Returns (ms, gbps) where gbps uses the fp16 K+V bytes touched."""
    if not _HAS_FLASHINFER:
        return None, None
    try:
        import flashinfer
        device = "cuda"
        dtype = torch.float16

        # Paged KV cache, page_size=1, NHD layout. Indices arrange the
        # ``seq_len`` pages contiguously for each request.
        page_size = 1
        num_pages = batch * seq_len
        kv_cache = torch.randn(
            num_pages, 2, kv_heads, page_size, head_dim,
            dtype=dtype, device=device,
        )
        kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
        kv_indptr = torch.arange(
            0, num_pages + seq_len, seq_len, dtype=torch.int32, device=device,
        )[: batch + 1]
        kv_last_page_len = torch.full(
            (batch,), page_size, dtype=torch.int32, device=device,
        )
        q = torch.randn(batch, q_heads, head_dim, dtype=dtype, device=device)

        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")
        wrapper.plan(
            kv_indptr, kv_indices, kv_last_page_len,
            q_heads, kv_heads, head_dim, page_size,
            q_data_type=dtype, kv_data_type=dtype,
        )

        def run():
            wrapper.run(q, kv_cache)

        run(); torch.cuda.synchronize()
        ms = _bench(run)
        fp16_bytes = (
            batch * seq_len * kv_heads * head_dim * 2 * 2  # fp16 K + V
        )
        gbps = fp16_bytes / (ms * 1e-3) / 1e9
        return ms, gbps
    except Exception:
        # FlashInfer's BatchDecodeWithPagedKVCacheWrapper currently has no
        # dispatch for q_heads=32 / kv_heads=8 (group=4) at head_dim=128 in
        # this build — error is reported as ``Unsupported group_size: 32``.
        # Treat as "no reference available" instead of failing the bench.
        return None, None


def bench_decode(
    batch: int, q_heads: int, kv_heads: int, head_dim: int,
    seq_len: int, max_splits: int,
    backend: str = "cutedsl",
):
    device = "cuda"
    dtype = torch.bfloat16
    cache_size = seq_len

    k = torch.randn(seq_len, kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(seq_len, kv_heads, head_dim, dtype=dtype, device=device)
    k_packed = torch.empty(seq_len, kv_heads, head_dim // 4,
                           dtype=torch.uint8, device=device)
    v_packed = torch.empty_like(k_packed)
    k_sz = torch.empty(seq_len, kv_heads, 2, dtype=torch.float32, device=device)
    v_sz = torch.empty_like(k_sz)
    loc = torch.arange(seq_len, dtype=torch.int32, device=device)
    _launch_quantize_int2(k, loc, k_packed, k_sz, None)
    _launch_quantize_int2(v, loc, v_packed, v_sz, None)

    q = torch.randn(batch, q_heads, head_dim, dtype=dtype, device=device)
    kv_indptr = torch.arange(0, (batch + 1) * seq_len, seq_len,
                              dtype=torch.int32, device=device)
    kv_indices = loc.repeat(batch).contiguous()
    num_kv_splits = torch.full((batch,), max_splits, dtype=torch.int32, device=device)
    sm_scale = 1.0 / (head_dim ** 0.5)

    o = torch.empty(batch, q_heads, head_dim, dtype=torch.float32, device=device)
    logits = torch.empty(batch, q_heads, max_splits, head_dim,
                         dtype=torch.float32, device=device)
    lse = torch.empty(batch, q_heads, max_splits, dtype=torch.float32, device=device)

    if kv_heads == q_heads:
        triton_fn = decode_attention_fwd_normal_quant_int2
    else:
        triton_fn = decode_attention_fwd_grouped_quant_int2

    def run_tri():
        triton_fn(
            q, k_packed, v_packed, k_sz, v_sz, o,
            kv_indptr, kv_indices, logits, lse,
            num_kv_splits, max_splits, sm_scale=sm_scale,
        )

    if backend == "cuda":
        opt_fn = cuda_decode_attention_fwd_int2
    elif backend == "cuda-wgmma":
        from sglang.QuantKernel.cutedsl_int2_kv import (
            cuda_decode_attention_fwd_int2_wgmma,
        )
        opt_fn = cuda_decode_attention_fwd_int2_wgmma
    else:
        opt_fn = cutedsl_decode_attention_fwd_int2

    def run_dsl():
        opt_fn(
            q, k_packed, v_packed, k_sz, v_sz, logits, lse,
            kv_indptr, kv_indices, num_kv_splits, max_splits, sm_scale,
        )

    run_tri(); run_dsl(); torch.cuda.synchronize()
    tri_ms = _bench(run_tri)
    dsl_ms = _bench(run_dsl)
    # Effective INT2 bytes touched (packed K + V + scale/zero for active heads).
    int2_bytes = (
        batch * seq_len * kv_heads * (head_dim // 4) * 2  # packed K+V
        + batch * seq_len * kv_heads * 2 * 4 * 2          # scale/zero (fp32)
    )
    tri_gbps = int2_bytes / (tri_ms * 1e-3) / 1e9
    dsl_gbps = int2_bytes / (dsl_ms * 1e-3) / 1e9

    fi_ms, fi_gbps = _bench_flashinfer_decode(batch, q_heads, kv_heads,
                                               head_dim, seq_len)
    return tri_ms, dsl_ms, fi_ms, tri_gbps, dsl_gbps, fi_gbps


def _decode_shape_list(gqa: bool):
    """Return [(batch, q_heads, kv_heads, seq, max_splits), ...].

    GQA shapes mirror Qwen3-4B Thinking production (q=32, kv=8, group=4).
    MHA shapes use q=kv=8 (CuteDSL's historical default).
    Production GPQA serves at batch=32 (MAX_RUNNING) — the high-batch
    shape exercises the GQA-coalesced grid where the kernel has many
    CTAs/SM and amortizes barrier overhead.
    """
    if gqa:
        return [
            (1, 32, 8, 4096, 4),
            (1, 32, 8, 8192, 8),
            (1, 32, 8, 16384, 8),
            (1, 32, 8, 20000, 8),
            (32, 32, 8, 4096, 4),  # production max-batch shape
        ]
    return [
        (1, 8, 8, 4096, 4),
        (1, 8, 8, 8192, 8),
        (1, 8, 8, 16384, 8),
        (1, 8, 8, 20000, 8),
    ]


def _test_correctness(backend: str, head_dim: int = 128,
                      cos_threshold: float = 0.999):
    """Run the chosen optimized backend vs Triton on representative shapes
    and verify cosine similarity. Also reports the CUDA invocation counters
    (delta over this test run) so the harness can confirm the requested
    kernel actually executed.

    Returns (min_cos, exit_code). exit_code is 0 on PASS, 1 on FAIL.
    """
    device = "cuda"
    dtype = torch.bfloat16
    # Small representative shapes: MHA q=8/kv=8 and GQA q=32/kv=8 (Qwen3-4B
    # production). Short sequences keep the test fast.
    shapes = [
        # (batch, q_heads, kv_heads, seq, max_splits)
        (1, 8, 8, 4096, 4),     # MHA
        (1, 32, 8, 4096, 4),    # GQA
        (1, 32, 8, 8192, 4),    # longer GQA
    ]

    if backend == "cuda":
        opt_fn = cuda_decode_attention_fwd_int2
        kernel_name = "wmma"
    elif backend == "cuda-wgmma":
        from sglang.QuantKernel.cutedsl_int2_kv import (
            cuda_decode_attention_fwd_int2_wgmma,
        )
        opt_fn = cuda_decode_attention_fwd_int2_wgmma
        kernel_name = "wgmma"
    else:
        opt_fn = cutedsl_decode_attention_fwd_int2
        kernel_name = "cutedsl"

    print(f"\n=== INT2 decode correctness ({backend} backend) ===")
    print(f"{'shape':<32s} {'cos_sim':>10s} {'kernel':>10s} {'PASS?':>7s}")
    use_counter = backend in ("cuda", "cuda-wgmma")
    if use_counter:
        reset_cuda_invocation_counters()
    pre_w, pre_g = (0, 0)
    if use_counter:
        pre_w, pre_g = get_cuda_invocation_counters()

    min_cos = float("inf")
    for (batch, q_heads, kv_heads, seq, splits) in shapes:
        torch.manual_seed(0xC0FFEE)
        k = torch.randn(seq, kv_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(seq, kv_heads, head_dim, dtype=dtype, device=device)
        k_packed = torch.empty(seq, kv_heads, head_dim // 4,
                               dtype=torch.uint8, device=device)
        v_packed = torch.empty_like(k_packed)
        k_sz = torch.empty(seq, kv_heads, 2, dtype=torch.float32, device=device)
        v_sz = torch.empty_like(k_sz)
        loc = torch.arange(seq, dtype=torch.int32, device=device)
        _launch_quantize_int2(k, loc, k_packed, k_sz, None)
        _launch_quantize_int2(v, loc, v_packed, v_sz, None)

        q = torch.randn(batch, q_heads, head_dim, dtype=dtype, device=device)
        kv_indptr = torch.arange(0, (batch + 1) * seq, seq,
                                 dtype=torch.int32, device=device)
        kv_indices = loc.repeat(batch).contiguous()
        num_kv_splits = torch.full((batch,), splits,
                                   dtype=torch.int32, device=device)
        sm_scale = 1.0 / (head_dim ** 0.5)

        # Triton reference stage-1: returns per-split logits & lse, reduced
        # via a second pass below for a comparable final output.
        ref_logits = torch.empty(batch, q_heads, splits, head_dim,
                                 dtype=torch.float32, device=device)
        ref_lse = torch.empty(batch, q_heads, splits,
                              dtype=torch.float32, device=device)
        opt_logits = torch.empty_like(ref_logits)
        opt_lse = torch.empty_like(ref_lse)

        triton_fn = (decode_attention_fwd_normal_quant_int2
                     if kv_heads == q_heads
                     else decode_attention_fwd_grouped_quant_int2)
        ref_o = torch.empty(batch, q_heads, head_dim,
                            dtype=torch.float32, device=device)
        triton_fn(
            q, k_packed, v_packed, k_sz, v_sz, ref_o,
            kv_indptr, kv_indices, ref_logits, ref_lse,
            num_kv_splits, splits, sm_scale=sm_scale,
        )

        opt_fn(
            q, k_packed, v_packed, k_sz, v_sz,
            opt_logits, opt_lse,
            kv_indptr, kv_indices, num_kv_splits,
            splits, sm_scale,
        )
        torch.cuda.synchronize()

        # Compare per-split logits where both kernels actually wrote.
        # The Triton path only writes splits in use (mask via finite LSE).
        valid = torch.isfinite(ref_lse)
        if not valid.any():
            cos = float("nan")
        else:
            a = ref_logits[valid].flatten().float()
            b = opt_logits[valid].flatten().float()
            cos = torch.nn.functional.cosine_similarity(
                a.unsqueeze(0), b.unsqueeze(0)
            ).item()

        min_cos = min(min_cos, cos)
        shape_label = f"B={batch} Hq={q_heads} Hk={kv_heads} S={seq}"
        passed = cos >= cos_threshold
        print(f"  {shape_label:<32s} {cos:>10.6f} {kernel_name:>10s} "
              f"{'PASS' if passed else 'FAIL':>7s}")

    if use_counter:
        post_w, post_g = get_cuda_invocation_counters()
        d_w, d_g = post_w - pre_w, post_g - pre_g
        print(f"\nCUDA invocation counters delta over this test:")
        print(f"  wmma  launches: {d_w}")
        print(f"  wgmma launches: {d_g}")
        expected = "wmma" if backend == "cuda" else "wgmma"
        actual = d_w if expected == "wmma" else d_g
        other  = d_g if expected == "wmma" else d_w
        if actual <= 0:
            print(f"FAIL: backend={backend} but {expected} counter "
                  f"did not advance (delta={actual}).")
            return min_cos, 1
        if other > 0:
            print(f"WARN: backend={backend} but the other CUDA kernel "
                  f"also fired ({other} launches).")

    if min_cos < cos_threshold:
        print(f"\nFAIL: min cosine_sim {min_cos:.6f} < {cos_threshold:.6f}")
        return min_cos, 1
    print(f"\nPASS: min cosine_sim {min_cos:.6f} >= {cos_threshold:.6f}")
    return min_cos, 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--assert-speedup", type=float, default=None,
                   help="Fail with non-zero exit if min CuteDSL speedup < this.")
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--gqa", action="store_true",
                   help="Use Qwen3-4B GQA shapes (q=32, kv=8, group=4).")
    p.add_argument("--quantize-only", action="store_true",
                   help="Skip decode benchmark.")
    p.add_argument("--backend", choices=("cutedsl", "cuda", "cuda-wgmma"),
                   default="cutedsl",
                   help="Optimized decode backend to bench against Triton.")
    p.add_argument("--test-correctness", action="store_true",
                   help="Run cosine-similarity correctness test vs Triton "
                        "reference for the chosen backend (no benchmark).")
    p.add_argument("--dump-cosine-sim", action="store_true",
                   help="Alias for --test-correctness; prints cos_sim line "
                        "the harness reads.")
    p.add_argument("--cos-threshold", type=float, default=0.999,
                   help="Cosine-sim PASS threshold for --test-correctness.")
    args = p.parse_args()
    if args.backend in ("cuda", "cuda-wgmma"):
        _load_cuda_decode_extension()

    if args.test_correctness or args.dump_cosine_sim:
        min_cos, rc = _test_correctness(
            args.backend, args.head_dim, args.cos_threshold,
        )
        # Emit a machine-readable line so an external harness can grep it.
        print(f"\ncosine_sim_min: {min_cos:.6f}")
        sys.exit(rc)

    head_dim = args.head_dim

    print(f"\n=== INT2 quantize benchmark (head_dim={head_dim}) ===")
    print(f"{'shape':<30s} {'Triton ms':>10s} {'CuteDSL ms':>11s} "
          f"{'Tri GB/s':>10s} {'DSL GB/s':>10s} {'speedup':>9s}")
    q_min_speedup = float("inf")
    for (num_tokens, num_heads) in [
        (4096, 8), (8192, 8), (16384, 8),
    ]:
        tri, dsl, tg, dg = bench_quantize(num_tokens, num_heads, head_dim)
        sp = tri / dsl if dsl > 0 else float("inf")
        q_min_speedup = min(q_min_speedup, sp)
        print(f"  T={num_tokens:>5d} H={num_heads:>2d}{'':<14s} "
              f"{tri:>10.4f} {dsl:>11.4f} {tg:>10.1f} {dg:>10.1f} {sp:>8.2f}x")

    if args.quantize_only:
        print(f"\nmin quantize speedup: {q_min_speedup:.4f}x")
        return

    shapes = _decode_shape_list(args.gqa)
    tag = "GQA (q=32,kv=8)" if args.gqa else "MHA (q=8,kv=8)"
    backend_name = {"cutedsl": "CuteDSL", "cuda": "CUDA C++ wmma",
                    "cuda-wgmma": "CUDA C++ wgmma"}[args.backend]
    print(f"\n=== INT2 decode attention benchmark "
          f"(head_dim={head_dim}, {tag}, backend={backend_name}) ===")

    # Effective GB/s table — INT2 columns use INT2 bytes (packed + scale/zero);
    # FlashInfer column uses fp16 bytes (K+V), measured on the same K/V shape.
    header = (
        f"{'shape':<30s} {'Tri ms':>9s} {'DSL ms':>9s} {'FI ms':>9s} "
        f"{'Tri GB/s':>10s} {'DSL GB/s':>10s} {'FI GB/s':>9s} {'speedup':>9s}"
    )
    print(header)
    d_min_speedup = float("inf")
    rows = []
    for (batch, q_heads, kv_heads, seq, splits) in shapes:
        tri, dsl, fi, tg, dg, fg = bench_decode(
            batch, q_heads, kv_heads, head_dim, seq, splits,
            backend=args.backend,
        )
        sp = tri / dsl if dsl > 0 else float("inf")
        d_min_speedup = min(d_min_speedup, sp)
        fi_ms_s = f"{fi:>9.4f}" if fi is not None else "       -"
        fi_gb_s = f"{fg:>9.1f}" if fg is not None else "       -"
        print(
            f"  B={batch} Hq={q_heads} Hk={kv_heads} S={seq:>5d}{'':<5s} "
            f"{tri:>9.4f} {dsl:>9.4f} {fi_ms_s} "
            f"{tg:>10.1f} {dg:>10.1f} {fi_gb_s} {sp:>8.2f}x"
        )
        rows.append((batch, q_heads, kv_heads, seq, tg, dg, fg, sp))

    # H100 HBM3 peak: 3350 GB/s. Print utilization for the largest shape.
    last = rows[-1]
    tg, dg, fg = last[4], last[5], last[6]
    print()
    print(f"H100 HBM3 peak: {H100_HBM3_PEAK_GBPS:.0f} GB/s. Util at S={last[3]}:")
    print(f"  Triton INT2:     {tg:>6.1f} GB/s "
          f"({100 * tg / H100_HBM3_PEAK_GBPS:>4.1f}%)")
    print(f"  CuteDSL INT2:    {dg:>6.1f} GB/s "
          f"({100 * dg / H100_HBM3_PEAK_GBPS:>4.1f}%)")
    if fg is not None:
        print(f"  FlashInfer fp16: {fg:>6.1f} GB/s "
              f"({100 * fg / H100_HBM3_PEAK_GBPS:>4.1f}%)  [reference]")
    else:
        print("  FlashInfer fp16: [not available]")

    print(f"\nmin decode speedup across {tag} shapes: "
          f"{d_min_speedup:.4f}x  (quantize min: {q_min_speedup:.4f}x)")

    if args.assert_speedup is not None:
        # Decode is the optimization target; quantize is DRAM-bound and saturates
        # at the same bandwidth in both kernels.
        gate = args.assert_speedup
        if d_min_speedup < gate:
            print(f"FAIL: min decode speedup {d_min_speedup:.4f}x < "
                  f"target {gate:.4f}x")
            sys.exit(1)
        print(f"PASS: decode speedup {d_min_speedup:.4f}x >= "
              f"target {gate:.4f}x on {tag} shapes")


if __name__ == "__main__":
    main()
