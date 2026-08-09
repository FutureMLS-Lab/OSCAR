#!/usr/bin/env python3
"""Merge block-sharded Kimi PPL or KL results."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * probability)]


def load_shards(directory: Path) -> list[dict]:
    paths = sorted(directory.glob("block*.json"))
    if not paths:
        raise FileNotFoundError(f"No shards under {directory}")
    return [json.loads(path.read_text()) for path in paths]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--kind", choices=("ppl", "kl"), required=True)
    args = parser.parse_args()

    shards = load_shards(args.root / f"{args.kind}_shards" / args.mode)
    block_ids = sorted(
        {
            int(block_id)
            for shard in shards
            for block_id in (
                shard.get("block_ids")
                or [item["block_id"] for item in shard.get("block_results", [])]
            )
        }
    )
    if block_ids != list(range(16)):
        raise ValueError(f"Expected block ids 0..15, got {block_ids}")

    if args.kind == "ppl":
        if not all(shard.get("complete") for shard in shards):
            raise ValueError("At least one PPL shard is incomplete")
        total_nll = sum(float(shard["total_nll"]) for shard in shards)
        scored_tokens = sum(int(shard["scored_tokens"]) for shard in shards)
        block_results = sorted(
            (
                item
                for shard in shards
                for item in shard.get("block_results", [])
            ),
            key=lambda item: int(item["block_id"]),
        )
        output = {
            "model": f"Kimi-K3-{args.mode}",
            "dataset": "wikitext-2-raw-v1/test",
            "block_size": 8192,
            "score_tail_tokens": 2048,
            "num_blocks": 16,
            "scored_tokens": scored_tokens,
            "total_nll": total_nll,
            "mean_nll": total_nll / scored_tokens,
            "ppl": math.exp(total_nll / scored_tokens),
            "block_results": block_results,
            "complete": True,
            "block_ids": block_ids,
        }
        destination = args.root / "ppl" / args.mode / "ppl.json"
    else:
        values = [
            float(value)
            for shard in shards
            for value in shard["values"]
        ]
        if len(values) != 16 * 16:
            raise ValueError(f"Expected 256 KL samples, got {len(values)}")
        output = {
            "kind": "topk_bucketed_forward_kl",
            "protocol": "wikitext2_8k_tail16_top50_v1",
            "model": f"Kimi-K3-{args.mode}",
            "dataset": "wikitext-2-raw-v1/test",
            "block_tokens": 8192,
            "num_blocks": 16,
            "block_ids": block_ids,
            "positions_per_block": 16,
            "top_k": 50,
            "num_samples": len(values),
            "mean_kl_nats": statistics.fmean(values),
            "median_kl_nats": statistics.median(values),
            "p95_kl_nats": percentile(values, 0.95),
            "max_kl_nats": max(values),
            "values": values,
        }
        destination = args.root / "kl_8k" / args.mode / "kl.json"

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "values"}))


if __name__ == "__main__":
    main()
