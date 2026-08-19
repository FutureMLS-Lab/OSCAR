#!/usr/bin/env python3
"""Aggregate Qwen3.5/Gemma4 K3V3 quality runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

MODELS = ("qwen3.5-35b-a3b", "gemma4-12b-it")
METHODS = ("oscar_offline_lm_qp", "offline_lm_qp")


def load_json(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def gpqa_summary(root: Path, method: str):
    scores = []
    for seed in range(3):
        metrics = [
            load_json(root / "gpqa" / method / f"seed{seed}" / f"shard{i}" / "metrics.json")
            for i in range(4)
        ]
        if any(item is None for item in metrics):
            return None
        examples = sum(int(item["num_examples"]) for item in metrics)
        scores.append(
            sum(float(item["score"]) * int(item["num_examples"]) for item in metrics)
            / examples
        )
    return {
        "seeds": scores,
        "mean": statistics.fmean(scores),
        "stdev": statistics.stdev(scores),
    }


def kl_value(payload):
    if payload is None:
        return None
    for key in ("mean_kl_nats", "mean_kl", "kl_mean", "mean"):
        if key in payload:
            return float(payload[key])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = {"models": {}}
    for model in MODELS:
        model_root = args.root / model
        model_result = {}
        for method in METHODS:
            ppl = load_json(model_root / "ppl" / method / "ppl.json")
            kl = load_json(model_root / "kl" / method / "kl.json")
            model_result[method] = {
                "gpqa": gpqa_summary(model_root, method),
                "ppl": float(ppl["ppl"]) if ppl and ppl.get("complete") else None,
                "kl50": kl_value(kl),
            }
        summary["models"][model] = model_result
    summary["models"]["kimi-k3"] = {
        "status": (
            "not_applicable: 69 KDA + 24 Gated-MLA layers do not expose the "
            "standard K/V cache targeted by K3V3; official weights are 1.56 TB"
        )
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
