#!/usr/bin/env python3
"""Summarize the two Kimi-K3 latent-K3V3 quality rows."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


METHODS = ("oscar_offline_lm_qp", "offline_lm_qp")


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def gpqa(root: Path, method: str) -> dict | None:
    seed_scores = []
    for seed in range(3):
        shards = [
            read_json(
                root
                / "gpqa"
                / method
                / f"seed{seed}"
                / f"shard{shard}"
                / "metrics.json"
            )
            for shard in range(4)
        ]
        if any(value is None for value in shards):
            return None
        examples = sum(int(value["num_examples"]) for value in shards)
        score = (
            sum(
                float(value["score"]) * int(value["num_examples"])
                for value in shards
            )
            / examples
        )
        seed_scores.append(score)
    return {
        "seeds": seed_scores,
        "mean": statistics.fmean(seed_scores),
        "stdev": statistics.stdev(seed_scores),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = {
        "model": "moonshotai/Kimi-K3",
        "target": "512-d MLA latent in 24 Gated-MLA layers",
        "native_kl_teacher_only": True,
        "methods": {},
    }
    for method in METHODS:
        ppl = read_json(args.root / "ppl" / method / "ppl.json")
        kl = read_json(args.root / "kl_8k" / method / "kl.json")
        output["methods"][method] = {
            "ppl": (
                float(ppl["ppl"])
                if ppl is not None and bool(ppl.get("complete"))
                else None
            ),
            "kl50": (
                float(kl["mean_kl_nats"])
                if kl is not None and "mean_kl_nats" in kl
                else None
            ),
            "gpqa": gpqa(args.root, method),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
