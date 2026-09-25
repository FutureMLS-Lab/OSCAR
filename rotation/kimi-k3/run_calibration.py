#!/usr/bin/env python3
"""Send deterministic WikiText-2 tokens through a Q/K/V-dump server."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import requests
import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--blocks-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--dataset-cache")
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--request-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    try:
        state = torch.load(args.blocks_path, map_location="cpu")
    except FileNotFoundError:
        dataset = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test",
            cache_dir=args.dataset_cache,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path, trust_remote_code=True
        )
        all_ids = tokenizer(
            "\n\n".join(dataset["text"]), add_special_tokens=False
        ).input_ids
        block_size = 2048
        usable = len(all_ids) // block_size * block_size
        state = {
            "dataset": "wikitext-2-raw-v1/test",
            "block_size": block_size,
            "tokenizer_path": args.tokenizer_path,
            "num_source_tokens": len(all_ids),
            "blocks": [
                all_ids[i : i + block_size] for i in range(0, usable, block_size)
            ],
        }
        blocks_path = Path(args.blocks_path)
        blocks_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, blocks_path)
    token_ids = [token for block in state["blocks"] for token in block][: args.tokens]
    if len(token_ids) != args.tokens:
        raise ValueError(
            f"Requested {args.tokens} calibration tokens, found {len(token_ids)}"
        )
    endpoint = args.base_url.rstrip("/") + "/generate"
    generated_text = None
    for start in range(0, len(token_ids), args.request_tokens):
        chunk = token_ids[start : start + args.request_tokens]
        response = requests.post(
            endpoint,
            json={
                "input_ids": chunk,
                "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
            },
            timeout=args.timeout,
        )
        response.raise_for_status()
        result = response.json()
        generated_text = (
            result[0]["text"] if isinstance(result, list) else result["text"]
        )
        print(
            f"calibration_chunk={start // args.request_tokens + 1}/"
            f"{math.ceil(len(token_ids) / args.request_tokens)} "
            f"tokens={len(chunk)}",
            flush=True,
        )
    print(f"calibration_tokens={len(token_ids)} generated_text={generated_text!r}")


if __name__ == "__main__":
    main()
