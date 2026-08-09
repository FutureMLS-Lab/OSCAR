#!/usr/bin/env python3
"""Evaluate WikiText-2 perplexity through an SGLang /generate endpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import requests
import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def load_or_create_blocks(args) -> list[list[int]]:
    if args.blocks_path.exists():
        state = torch.load(args.blocks_path, map_location="cpu")
        if state["block_size"] != args.block_size:
            raise ValueError(
                f"Cached block size is {state['block_size']}, expected {args.block_size}"
            )
        return state["blocks"]

    if args.text_path is not None:
        corpus = args.text_path.read_text()
    else:
        dataset = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test",
            cache_dir=str(args.dataset_cache),
        )
        corpus = "\n\n".join(dataset["text"])
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,
    )
    token_ids = tokenizer(
        corpus,
        add_special_tokens=False,
    ).input_ids
    usable = len(token_ids) // args.block_size * args.block_size
    blocks = [
        token_ids[i : i + args.block_size]
        for i in range(0, usable, args.block_size)
    ]
    args.blocks_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "dataset": "wikitext-2-raw-v1/test",
            "block_size": args.block_size,
            "tokenizer_path": args.tokenizer_path,
            "num_source_tokens": len(token_ids),
            "blocks": blocks,
        },
        args.blocks_path,
    )
    return blocks


def extract_input_logprobs(response: dict) -> list[float]:
    meta = response["meta_info"]
    values = meta["input_token_logprobs"]
    logprobs = []
    for value in values:
        logprob = value[0] if isinstance(value, (list, tuple)) else value
        if logprob is not None:
            logprobs.append(float(logprob))
    return logprobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks-path", type=Path, required=True)
    parser.add_argument("--text-path", type=Path)
    parser.add_argument(
        "--dataset-cache",
        type=Path,
        default=Path("/shared/huggingface/datasets"),
    )
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument(
        "--score-tail-tokens",
        type=int,
        default=0,
        help="If positive, score only this many tokens at the end of each block.",
    )
    parser.add_argument(
        "--max-blocks",
        type=int,
        default=0,
        help="0 evaluates every complete test block.",
    )
    parser.add_argument("--block-start", type=int, default=0)
    parser.add_argument(
        "--block-end",
        type=int,
        default=0,
        help="Exclusive source block index; 0 means the end of the corpus.",
    )
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    source_blocks = load_or_create_blocks(args)
    source_num_blocks = len(source_blocks)
    block_end = args.block_end or source_num_blocks
    if not 0 <= args.block_start < block_end <= source_num_blocks:
        raise ValueError(
            f"Invalid block range [{args.block_start}, {block_end}) "
            f"for {source_num_blocks} blocks"
        )
    blocks = source_blocks[args.block_start:block_end]
    if args.max_blocks:
        blocks = blocks[: args.max_blocks]
        block_end = args.block_start + len(blocks)
    if not blocks:
        raise ValueError("No complete WikiText-2 blocks were produced")
    if args.score_tail_tokens < 0 or args.score_tail_tokens > args.block_size:
        raise ValueError(
            "score_tail_tokens must be in [0, block_size], got "
            f"{args.score_tail_tokens}"
        )

    total_nll = 0.0
    scored_tokens = 0
    block_results = []
    next_block = 0
    recovered_prefix = None
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        if (
            previous.get("complete") is False
            and previous.get("model") == args.model
            and previous.get("block_size") == args.block_size
            and previous.get("num_blocks") == len(blocks)
            and previous.get("score_tail_tokens", 0) == args.score_tail_tokens
            and previous.get("block_start", 0) == args.block_start
            and previous.get("block_end", len(blocks)) == block_end
        ):
            total_nll = float(previous["total_nll"])
            scored_tokens = int(previous["scored_tokens"])
            block_results = previous.get("block_results", [])
            next_block = int(previous["next_block"])
            recovered_prefix = previous.get("recovered_prefix")
            print(
                f"resuming_from_block={next_block}/{len(blocks)} "
                f"running_ppl={math.exp(total_nll / scored_tokens):.6f}",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    endpoint = args.base_url.rstrip("/") + "/generate"
    for block_id in range(next_block, len(blocks)):
        input_ids = blocks[block_id]
        logprob_start_len = (
            max(0, len(input_ids) - args.score_tail_tokens - 1)
            if args.score_tail_tokens
            else 0
        )
        payload = {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 1,
            },
            "return_logprob": True,
            "logprob_start_len": logprob_start_len,
            "top_logprobs_num": 0,
        }
        request = requests.post(endpoint, json=payload, timeout=args.timeout)
        request.raise_for_status()
        response = request.json()
        if isinstance(response, list):
            response = response[0]
        logprobs = extract_input_logprobs(response)
        expected = len(input_ids) - logprob_start_len - 1
        if len(logprobs) != expected:
            raise ValueError(
                f"Block {block_id}: received {len(logprobs)} scored tokens, "
                f"expected {expected}"
            )
        block_nll = -sum(logprobs)
        total_nll += block_nll
        scored_tokens += len(logprobs)
        source_block_id = args.block_start + block_id
        block_results.append(
            {
                "block_id": source_block_id,
                "tokens": len(logprobs),
                "mean_nll": block_nll / len(logprobs),
            }
        )
        print(
            f"block={block_id + 1}/{len(blocks)} "
            f"running_ppl={math.exp(total_nll / scored_tokens):.6f}",
            flush=True,
        )
        partial_result = {
            "model": args.model,
            "dataset": "wikitext-2-raw-v1/test",
            "block_size": args.block_size,
            "score_tail_tokens": args.score_tail_tokens,
            "num_blocks": len(blocks),
            "source_num_blocks": source_num_blocks,
            "block_start": args.block_start,
            "block_end": block_end,
            "scored_tokens": scored_tokens,
            "total_nll": total_nll,
            "mean_nll": total_nll / scored_tokens,
            "ppl": math.exp(total_nll / scored_tokens),
            "blocks_path": str(args.blocks_path),
            "block_results": block_results,
            "complete": False,
            "next_block": block_id + 1,
            "recovered_prefix": recovered_prefix,
        }
        args.output.write_text(json.dumps(partial_result, indent=2) + "\n")

    result = {
        "model": args.model,
        "dataset": "wikitext-2-raw-v1/test",
        "block_size": args.block_size,
        "score_tail_tokens": args.score_tail_tokens,
        "num_blocks": len(blocks),
        "source_num_blocks": source_num_blocks,
        "block_start": args.block_start,
        "block_end": block_end,
        "scored_tokens": scored_tokens,
        "total_nll": total_nll,
        "mean_nll": total_nll / scored_tokens,
        "ppl": math.exp(total_nll / scored_tokens),
        "blocks_path": str(args.blocks_path),
        "block_results": block_results,
        "complete": True,
        "next_block": len(blocks),
        "recovered_prefix": recovered_prefix,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "block_results"}))


if __name__ == "__main__":
    main()
