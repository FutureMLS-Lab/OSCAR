#!/usr/bin/env python3
"""HBM capacity ceiling for a growing low-bit MLA latent cache."""

from __future__ import annotations

import argparse
import json


def context_limit(
    *,
    per_gpu_budget_bytes: float,
    layers: int,
    latent_dim: int,
    rope_dim: int,
    lut_bits: float,
    bf16_window: int,
    kv_shard_factor: int,
) -> float:
    body_bytes = latent_dim * lut_bits / 8 + rope_dim * 2
    window_bytes = (latent_dim + rope_dim) * 2
    shared_budget = per_gpu_budget_bytes * kv_shard_factor
    fixed_window_bytes = layers * bf16_window * window_bytes
    if shared_budget <= fixed_window_bytes:
        return shared_budget / (layers * window_bytes)
    return bf16_window + (
        shared_budget - fixed_window_bytes
    ) / (layers * body_bytes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-count", type=int, default=8)
    parser.add_argument("--hbm-gb", type=float, default=288.0)
    parser.add_argument("--weight-storage-tb", type=float, default=1.56)
    parser.add_argument("--mla-layers", type=int, default=24)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--rope-dim", type=int, default=64)
    parser.add_argument("--lut-bits", type=float, default=3.125)
    parser.add_argument("--bf16-window", type=int, default=320)
    parser.add_argument(
        "--reserve-gb",
        type=float,
        nargs="+",
        default=[0.0, 9.3, 14.27367168, 20.0, 30.0],
    )
    parser.add_argument(
        "--kv-shard-factor",
        type=int,
        default=1,
        help=(
            "How many GPUs shard one cache. Ordinary TP with a replicated "
            "MLA latent uses 1; context-parallel cache sharding may use more."
        ),
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.gpu_count < 1 or args.kv_shard_factor < 1:
        raise ValueError("GPU and KV shard counts must be positive")
    weight_per_gpu_gb = args.weight_storage_tb * 1000 / args.gpu_count
    ideal_free_gb = args.hbm_gb - weight_per_gpu_gb
    if ideal_free_gb <= 0:
        raise ValueError("Weights do not fit in the requested GPU count")

    body_bytes_per_layer_token = (
        args.latent_dim * args.lut_bits / 8 + args.rope_dim * 2
    )
    rows = []
    for reserve_gb in args.reserve_gb:
        budget_gb = ideal_free_gb - reserve_gb
        if budget_gb <= 0:
            continue
        limit = context_limit(
            per_gpu_budget_bytes=budget_gb * 1e9,
            layers=args.mla_layers,
            latent_dim=args.latent_dim,
            rope_dim=args.rope_dim,
            lut_bits=args.lut_bits,
            bf16_window=args.bf16_window,
            kv_shard_factor=args.kv_shard_factor,
        )
        rows.append(
            {
                "reserve_gb_per_gpu": reserve_gb,
                "kv_budget_gb_per_gpu": budget_gb,
                "context_tokens": limit,
            }
        )

    result = {
        "gpu_count": args.gpu_count,
        "hbm_gb_per_gpu": args.hbm_gb,
        "aggregate_hbm_tb": args.gpu_count * args.hbm_gb / 1000,
        "weight_storage_tb": args.weight_storage_tb,
        "weight_gb_per_gpu": weight_per_gpu_gb,
        "ideal_free_gb_per_gpu": ideal_free_gb,
        "mla_layers": args.mla_layers,
        "body_bytes_per_layer_token": body_bytes_per_layer_token,
        "bytes_per_body_token_all_layers": (
            body_bytes_per_layer_token * args.mla_layers
        ),
        "kv_shard_factor": args.kv_shard_factor,
        "scenarios": rows,
    }
    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("| Reserve/GPU | KV budget/GPU | Context ceiling |")
    print("|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['reserve_gb_per_gpu']:.1f} GB "
            f"| {row['kv_budget_gb_per_gpu']:.1f} GB "
            f"| {row['context_tokens'] / 1e6:.2f}M |"
        )


if __name__ == "__main__":
    main()
