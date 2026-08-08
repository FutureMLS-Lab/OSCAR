#!/usr/bin/env python3
"""Bandwidth roofline for Qwen3-4B K3V3 decode attention on Rubin.

This models only the fused QK-softmax-PV attention kernel. It does not model
model-weight traffic, scheduler overhead, communication, or end-to-end tok/s.
"""

from __future__ import annotations

import argparse
import json


HARDWARE = (
    ("H100 SXM", 3.35, 1.979, False),
    ("B200 SXM", 8.0, 4.5, False),
    ("Rubin", 22.0, 8.75, True),
)


def project(
    context: int,
    *,
    layers: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    sink: int,
    recent: int,
    hbm_bytes_per_second: float,
    dense_fp8_flops_per_second: float,
    mma_m: int = 128,
    mma_n: int = 8,
    packed_query_rows: int | None = None,
    apply_lutb_b_tile_cap: bool = True,
) -> dict[str, float | int | str]:
    high_precision_tokens = min(context, sink + recent)
    lut_tokens = context - high_precision_tokens
    kv_values_per_token_layer = 2 * kv_heads * head_dim
    lut_bytes_per_value = 3.125 / 8

    lut_bytes_layer = lut_tokens * kv_values_per_token_layer * lut_bytes_per_value
    bf16_window_bytes_layer = (
        high_precision_tokens * kv_values_per_token_layer * 2
    )
    kv_bytes_layer = lut_bytes_layer + bf16_window_bytes_layer

    # QK and PV each perform one multiply and one add.
    attention_flops_layer = 4 * q_heads * head_dim * context
    arithmetic_intensity = attention_flops_layer / kv_bytes_layer
    bandwidth_roof_flops = arithmetic_intensity * hbm_bytes_per_second
    packed_query_rows = packed_query_rows or (q_heads // kv_heads)
    lutb_b_compute_cap = dense_fp8_flops_per_second * min(
        packed_query_rows / mma_m, 1.0
    )
    hypothetical_kv_a_compute_cap = dense_fp8_flops_per_second * min(
        packed_query_rows / mma_n, 1.0
    )
    dense_e4m3_m64_compute_cap = dense_fp8_flops_per_second * min(
        packed_query_rows / 64, 1.0
    )
    active_compute_cap = (
        lutb_b_compute_cap
        if apply_lutb_b_tile_cap
        else dense_fp8_flops_per_second
    )
    roof_flops = min(bandwidth_roof_flops, active_compute_cap)

    all_layer_bytes = kv_bytes_layer * layers
    all_layer_flops = attention_flops_layer * layers
    bandwidth_token_rate = hbm_bytes_per_second / all_layer_bytes
    compute_token_rate = active_compute_cap / all_layer_flops
    roof_token_rate = min(bandwidth_token_rate, compute_token_rate)
    bf16_bytes_layer = context * kv_values_per_token_layer * 2
    bf16_bandwidth_token_rate = hbm_bytes_per_second / (
        bf16_bytes_layer * layers
    )
    bf16_bandwidth_roof_flops = (
        attention_flops_layer / bf16_bytes_layer * hbm_bytes_per_second
    )

    return {
        "context": context,
        "bf16_window_tokens": high_precision_tokens,
        "effective_kv_bits_per_value": (
            high_precision_tokens * 16 + lut_tokens * 3.125
        )
        / context,
        "kv_megabytes_per_layer": kv_bytes_layer / 1e6,
        "attention_gflop_per_token_all_layers": all_layer_flops / 1e9,
        "arithmetic_intensity_flop_per_byte": arithmetic_intensity,
        "byte_only_roofline_tflop_per_second": bandwidth_roof_flops / 1e12,
        "packed_query_rows": packed_query_rows,
        "mma_m": mma_m,
        "mma_n": mma_n,
        "lutb_b_tile_utilization": min(packed_query_rows / mma_m, 1.0),
        "lutb_b_compute_cap_tflop_per_second": lutb_b_compute_cap / 1e12,
        "hypothetical_kv_a_compute_cap_tflop_per_second": (
            hypothetical_kv_a_compute_cap / 1e12
        ),
        "dense_e4m3_m64_compute_cap_tflop_per_second": (
            dense_e4m3_m64_compute_cap / 1e12
        ),
        "byte_only_over_operand_corrected": bandwidth_roof_flops / roof_flops,
        "limiting_resource": (
            "HBM"
            if bandwidth_roof_flops < active_compute_cap
            else "Tensor Core tile occupancy"
        ),
        "roofline_tflop_per_second": roof_flops / 1e12,
        "roofline_attention_tokens_per_second": roof_token_rate,
        "roofline_attention_latency_us": 1e6 / roof_token_rate,
        "bf16_roofline_tflop_per_second": bf16_bandwidth_roof_flops / 1e12,
        "bf16_roofline_attention_tokens_per_second": (
            bf16_bandwidth_token_rate
        ),
        "roofline_speedup_over_bf16": (
            roof_token_rate / bf16_bandwidth_token_rate
        ),
        "practical_60pct_hbm_tflop_per_second": roof_flops * 0.60 / 1e12,
        "practical_80pct_hbm_tflop_per_second": roof_flops * 0.80 / 1e12,
        "no_gqa_reuse_tflop_per_second": bandwidth_roof_flops
        / (q_heads / kv_heads)
        / 1e12,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", type=int, nargs="+", default=[8192, 32768])
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--sink", type=int, default=64)
    parser.add_argument("--recent", type=int, default=256)
    parser.add_argument("--hbm-tb-s", type=float, default=22.0)
    parser.add_argument(
        "--dense-fp8-pflop-s",
        type=float,
        default=8.75,
        help="Dense ceiling; half of NVIDIA's 17.5 PFLOP/s sparse figure.",
    )
    parser.add_argument("--mma-m", type=int, default=128)
    parser.add_argument("--mma-n", type=int, default=8)
    parser.add_argument(
        "--packed-query-rows",
        type=int,
        default=None,
        help="Rows sharing one LUT-B cache tile; defaults to the GQA ratio.",
    )
    parser.add_argument("--ignore-operand-tile-cap", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--compare-hardware", action="store_true")
    args = parser.parse_args()

    if args.compare_hardware:
        rows = []
        for context in args.contexts:
            for name, bandwidth, fp8, native_lutb in HARDWARE:
                row = project(
                    context,
                    layers=args.layers,
                    q_heads=args.q_heads,
                    kv_heads=args.kv_heads,
                    head_dim=args.head_dim,
                    sink=args.sink,
                    recent=args.recent,
                    hbm_bytes_per_second=bandwidth * 1e12,
                    dense_fp8_flops_per_second=fp8 * 1e15,
                    mma_m=args.mma_m,
                    mma_n=args.mma_n,
                    packed_query_rows=args.packed_query_rows,
                    apply_lutb_b_tile_cap=native_lutb
                    and not args.ignore_operand_tile_cap,
                )
                row["gpu"] = name
                row["native_lutb"] = native_lutb
                rows.append(row)
    else:
        rows = [
            project(
                context,
                layers=args.layers,
                q_heads=args.q_heads,
                kv_heads=args.kv_heads,
                head_dim=args.head_dim,
                sink=args.sink,
                recent=args.recent,
                hbm_bytes_per_second=args.hbm_tb_s * 1e12,
                dense_fp8_flops_per_second=args.dense_fp8_pflop_s * 1e15,
                mma_m=args.mma_m,
                mma_n=args.mma_n,
                packed_query_rows=args.packed_query_rows,
                apply_lutb_b_tile_cap=not args.ignore_operand_tile_cap,
            )
            for context in args.contexts
        ]
    if args.json:
        print(json.dumps(rows, indent=2))
        return

    if args.compare_hardware:
        print(
            "| Context | GPU | BF16 TFLOP/s | BF16 tok/s | "
            "K3 projected TFLOP/s | K3 projected tok/s | Native LUT-B |"
        )
        print("|---:|---|---:|---:|---:|---:|---:|")
        for row in rows:
            print(
                f"| {row['context'] // 1024}K | {row['gpu']} "
                f"| {row['bf16_roofline_tflop_per_second']:.1f} "
                f"| {row['bf16_roofline_attention_tokens_per_second']:.0f} "
                f"| {row['roofline_tflop_per_second']:.1f} "
                f"| {row['roofline_attention_tokens_per_second']:.0f} "
                f"| {'Yes' if row['native_lutb'] else 'No'} |"
            )
        return

    print(
        "| Context | Effective KV bits | KV MB/layer | FLOP/token | AI | "
        "Operand-corrected TFLOP/s | 60-80% projected TFLOP/s | vs BF16 | "
        "Attention-only tok/s |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['context'] // 1024}K "
            f"| {row['effective_kv_bits_per_value']:.3f} "
            f"| {row['kv_megabytes_per_layer']:.3f} "
            f"| {row['attention_gflop_per_token_all_layers']:.3f}G "
            f"| {row['arithmetic_intensity_flop_per_byte']:.2f} "
            f"| {row['roofline_tflop_per_second']:.1f} "
            f"| {row['practical_60pct_hbm_tflop_per_second']:.1f}-"
            f"{row['practical_80pct_hbm_tflop_per_second']:.1f} "
            f"| {row['roofline_speedup_over_bf16']:.2f}x "
            f"| {row['roofline_attention_tokens_per_second']:.0f} |"
        )


if __name__ == "__main__":
    main()
