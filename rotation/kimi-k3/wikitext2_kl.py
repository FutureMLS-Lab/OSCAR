#!/usr/bin/env python3
"""Estimate cross-model KL on fixed BF16-generated continuations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import requests
import torch


def post_generate(base_url: str, payload: dict, timeout: int) -> dict:
    response = requests.post(
        base_url.rstrip("/") + "/generate",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    result = response.json()
    return result[0] if isinstance(result, list) else result


def logprob_values(values) -> list[float]:
    result = []
    for value in values:
        if value is None:
            continue
        logprob = value[0] if isinstance(value, (list, tuple)) else value
        if logprob is not None:
            result.append(float(logprob))
    return result


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    if count < 1 or count > length:
        raise ValueError(f"num_samples must be in [1, {length}], got {count}")
    if count == 1:
        return [0]
    return [round(i * (length - 1) / (count - 1)) for i in range(count)]


def write_json(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n")


def generate_reference(args, blocks: list[list[int]]) -> None:
    block_ids = evenly_spaced_indices(len(blocks), args.num_samples)
    state = {
        "protocol": "bf16-sampled-kl-v1",
        "reference_model": args.model,
        "dataset": "wikitext-2-raw-v1/test",
        "num_samples": args.num_samples,
        "prompt_tokens": args.prompt_tokens,
        "new_tokens": args.new_tokens,
        "server_random_seed": 0,
        "samples": [],
    }
    for sample_id, block_id in enumerate(block_ids):
        prompt_ids = blocks[block_id][: args.prompt_tokens]
        result = post_generate(
            args.base_url,
            {
                "input_ids": prompt_ids,
                "sampling_params": {
                    "temperature": 1.0,
                    "max_new_tokens": args.new_tokens,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "return_text_in_logprobs": False,
                "logprob_start_len": -1,
            },
            args.timeout,
        )
        output_ids = [int(token) for token in result["output_ids"]]
        output_logprobs = logprob_values(result["meta_info"]["output_token_logprobs"])
        if len(output_ids) != args.new_tokens:
            raise ValueError(
                f"Sample {sample_id}: got {len(output_ids)} output tokens, "
                f"expected {args.new_tokens}"
            )
        if len(output_logprobs) != len(output_ids):
            raise ValueError(
                f"Sample {sample_id}: got {len(output_logprobs)} logprobs "
                f"for {len(output_ids)} output tokens"
            )
        state["samples"].append(
            {
                "sample_id": sample_id,
                "block_id": block_id,
                "prompt_ids": prompt_ids,
                "output_ids": output_ids,
                "reference_logprobs": output_logprobs,
            }
        )
        write_json(args.output, state)
        print(
            f"reference_sample={sample_id + 1}/{args.num_samples} "
            f"tokens={len(output_ids)}",
            flush=True,
        )
    print(json.dumps({k: v for k, v in state.items() if k != "samples"}))


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def score_candidate(args) -> None:
    reference = json.loads(args.reference.read_text())
    all_logratios = []
    sample_results = []
    for sample in reference["samples"]:
        output_ids = sample["output_ids"]
        input_ids = sample["prompt_ids"] + output_ids
        result = post_generate(
            args.base_url,
            {
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 0,
                },
                "return_logprob": True,
                "return_text_in_logprobs": False,
                # Only the fixed continuation is compared. Starting one token
                # before it avoids materializing full-vocabulary logits for
                # the long prompt.
                "logprob_start_len": len(sample["prompt_ids"]) - 1,
                "top_logprobs_num": 0,
            },
            args.timeout,
        )
        candidate_logprobs = logprob_values(
            result["meta_info"]["input_token_logprobs"]
        )[-len(output_ids) :]
        reference_logprobs = sample["reference_logprobs"]
        if len(candidate_logprobs) != len(reference_logprobs):
            raise ValueError(
                f"Sample {sample['sample_id']}: candidate returned "
                f"{len(candidate_logprobs)} scores, expected "
                f"{len(reference_logprobs)}"
            )
        logratios = [
            candidate - baseline
            for candidate, baseline in zip(candidate_logprobs, reference_logprobs)
        ]
        all_logratios.extend(logratios)
        sample_results.append(
            {
                "sample_id": sample["sample_id"],
                "block_id": sample["block_id"],
                "tokens": len(logratios),
                "direct_kl_pq": -sum(logratios) / len(logratios),
                "k3_kl_pq": sum(math.expm1(value) - value for value in logratios)
                / len(logratios),
                "logprob_mae": sum(abs(value) for value in logratios) / len(logratios),
            }
        )
        print(
            f"score_sample={len(sample_results)}/{len(reference['samples'])}",
            flush=True,
        )

    token_count = len(all_logratios)
    result = {
        "protocol": reference["protocol"],
        "reference_model": reference["reference_model"],
        "candidate_model": args.model,
        "dataset": reference["dataset"],
        "num_samples": len(reference["samples"]),
        "tokens": token_count,
        "direct_kl_pq": -sum(all_logratios) / token_count,
        "k3_kl_pq": sum(math.expm1(value) - value for value in all_logratios)
        / token_count,
        "mean_logq_minus_logp": sum(all_logratios) / token_count,
        "logprob_mae": sum(abs(value) for value in all_logratios) / token_count,
        "logprob_rmse": math.sqrt(
            sum(value * value for value in all_logratios) / token_count
        ),
        "logprob_abs_p95": quantile([abs(value) for value in all_logratios], 0.95),
        "logprob_abs_max": max(abs(value) for value in all_logratios),
        "samples": sample_results,
        "reference_path": str(args.reference),
    }
    write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("reference", "score"), required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--blocks-path", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    if args.mode == "reference":
        if args.blocks_path is None:
            parser.error("--blocks-path is required in reference mode")
        state = torch.load(args.blocks_path, map_location="cpu")
        generate_reference(args, state["blocks"])
    else:
        if args.reference is None:
            parser.error("--reference is required in score mode")
        score_candidate(args)


if __name__ == "__main__":
    main()
