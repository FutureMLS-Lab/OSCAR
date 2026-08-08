#!/usr/bin/env python3
"""Top-50 bucketed KL at 16 tail positions of each 8K WikiText block."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
from pathlib import Path

import requests
import torch


def post(endpoint: str, payload: dict, timeout: int) -> dict:
    response = requests.post(endpoint, json=payload, timeout=timeout)
    response.raise_for_status()
    value = response.json()
    return value[0] if isinstance(value, list) else value


def nonempty_tail(rows: list, count: int, *, name: str) -> list:
    result = [row for row in rows if row]
    if len(result) < count:
        raise ValueError(f"{name}: expected {count} rows, got {len(result)}")
    return result[-count:]


def teacher_block(
    endpoint: str,
    block_id: int,
    input_ids: list[int],
    *,
    positions: int,
    top_k: int,
    timeout: int,
) -> list[dict]:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
        "return_logprob": True,
        "logprob_start_len": max(len(input_ids) - positions - 1, 0),
        "top_logprobs_num": top_k,
    }
    result = post(endpoint, payload, timeout)
    rows = nonempty_tail(
        result["meta_info"]["input_top_logprobs"],
        positions,
        name=f"teacher block {block_id}",
    )
    samples = []
    for tail_index, row in enumerate(rows, start=-len(rows)):
        token_ids = [int(item[1]) for item in row]
        logprobs = [float(item[0]) for item in row]
        if len(token_ids) != top_k or len(set(token_ids)) != top_k:
            raise ValueError(
                f"Teacher block {block_id} tail {tail_index} returned "
                f"{len(token_ids)} non-distinct top tokens"
            )
        samples.append(
            {
                "block_id": block_id,
                "tail_index": tail_index,
                "token_ids": token_ids,
                "teacher_logprobs": logprobs,
            }
        )
    return samples


def bucketed_forward_kl(sample: dict, student: dict[int, float]) -> float:
    token_ids = [int(value) for value in sample["token_ids"]]
    teacher_logprobs = [float(value) for value in sample["teacher_logprobs"]]
    if not set(token_ids).issubset(student):
        raise ValueError("Student response omitted requested teacher token ids")
    student_logprobs = [student[token_id] for token_id in token_ids]
    teacher_probs = [math.exp(value) for value in teacher_logprobs]
    student_probs = [math.exp(value) for value in student_logprobs]
    teacher_tail = max(1.0 - sum(teacher_probs), 1e-12)
    student_tail = max(1.0 - sum(student_probs), 1e-12)
    value = sum(
        probability * (teacher_lp - student_lp)
        for probability, teacher_lp, student_lp in zip(
            teacher_probs,
            teacher_logprobs,
            student_logprobs,
        )
    )
    value += teacher_tail * math.log(teacher_tail / student_tail)
    return max(value, 0.0)


def student_block(
    endpoint: str,
    block_id: int,
    input_ids: list[int],
    samples: list[dict],
    *,
    timeout: int,
) -> list[float]:
    token_ids = sorted(
        {
            int(token_id)
            for sample in samples
            for token_id in sample["token_ids"]
        }
    )
    payload = {
        "input_ids": input_ids,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
        "return_logprob": True,
        "logprob_start_len": max(len(input_ids) - len(samples) - 1, 0),
        "top_logprobs_num": 0,
        "token_ids_logprob": token_ids,
    }
    result = post(endpoint, payload, timeout)
    rows = nonempty_tail(
        result["meta_info"]["input_token_ids_logprobs"],
        len(samples),
        name=f"student block {block_id}",
    )
    values = []
    for sample, row in zip(samples, rows):
        student = {int(item[1]): float(item[0]) for item in row}
        values.append(bucketed_forward_kl(sample, student))
    return values


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * probability)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--blocks-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-reference", type=Path)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-blocks", type=int, default=16)
    parser.add_argument("--block-start", type=int, default=0)
    parser.add_argument(
        "--block-end",
        type=int,
        default=0,
        help="Exclusive block id; 0 means --max-blocks.",
    )
    parser.add_argument("--positions-per-block", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()

    state = torch.load(args.blocks_path, map_location="cpu", weights_only=True)
    blocks = [
        [int(value) for value in block]
        for block in state["blocks"][: args.max_blocks]
    ]
    if any(len(block) != 8192 for block in blocks):
        raise ValueError("Kimi KL protocol requires 8192-token blocks")
    block_end = args.block_end or len(blocks)
    if not 0 <= args.block_start < block_end <= len(blocks):
        raise ValueError(
            f"Invalid block range [{args.block_start}, {block_end}) "
            f"for {len(blocks)} blocks"
        )
    selected_block_ids = list(range(args.block_start, block_end))
    endpoint = args.base_url.rstrip("/") + "/generate"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.teacher_reference is None:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers
        ) as executor:
            futures = [
                executor.submit(
                    teacher_block,
                    endpoint,
                    block_id,
                    block,
                    positions=args.positions_per_block,
                    top_k=args.top_k,
                    timeout=args.timeout,
                )
                for block_id, block in enumerate(blocks)
                if block_id in selected_block_ids
            ]
            samples = [
                sample
                for future in futures
                for sample in future.result()
            ]
        output = {
            "kind": "bf16_topk_teacher_reference",
            "protocol": "wikitext2_8k_tail16_top50_v1",
            "model": args.model,
            "dataset": "wikitext-2-raw-v1/test",
            "block_tokens": 8192,
            "num_blocks": len(blocks),
            "block_ids": selected_block_ids,
            "positions_per_block": args.positions_per_block,
            "top_k": args.top_k,
            "num_samples": len(samples),
            "samples": samples,
        }
    else:
        teacher = json.loads(args.teacher_reference.read_text())
        if teacher.get("protocol") != "wikitext2_8k_tail16_top50_v1":
            raise ValueError("Teacher reference uses a different KL protocol")
        if int(teacher["top_k"]) != args.top_k:
            raise ValueError("Teacher reference top-K does not match")
        grouped: dict[int, list[dict]] = {}
        for sample in teacher["samples"]:
            block_id = int(sample["block_id"])
            if block_id in selected_block_ids:
                grouped.setdefault(block_id, []).append(sample)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers
        ) as executor:
            futures = [
                executor.submit(
                    student_block,
                    endpoint,
                    block_id,
                    blocks[block_id],
                    grouped[block_id],
                    timeout=args.timeout,
                )
                for block_id in sorted(grouped)
            ]
            values = [
                value
                for future in futures
                for value in future.result()
            ]
        output = {
            "kind": "topk_bucketed_forward_kl",
            "protocol": teacher["protocol"],
            "model": args.model,
            "teacher": str(args.teacher_reference),
            "dataset": teacher["dataset"],
            "block_tokens": teacher["block_tokens"],
            "num_blocks": teacher["num_blocks"],
            "block_ids": sorted(grouped),
            "positions_per_block": teacher["positions_per_block"],
            "top_k": args.top_k,
            "num_samples": len(values),
            "mean_kl_nats": statistics.fmean(values),
            "median_kl_nats": statistics.median(values),
            "p95_kl_nats": percentile(values, 0.95),
            "max_kl_nats": max(values),
            "values": values,
        }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in output.items()
                if key not in {"samples", "values"}
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
