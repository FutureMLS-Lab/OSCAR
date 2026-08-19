"""Stage: sample completions from the reference server for every prompt."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from rich.progress import Progress

from .config import RunCfg
from .engine import make_client


async def run_generate(cfg: RunCfg, prompts_path: Path, out_path: Path) -> int:
    prompts = [json.loads(line) for line in prompts_path.open()]
    client = make_client(cfg.reference, cfg.runtime.ref.url, concurrency=cfg.scoring.concurrency)
    gcfg = cfg.generation

    async def one(rec: dict) -> dict:
        res = await client.generate(
            rec["input_ids"],
            temperature=gcfg.temperature,
            top_p=gcfg.top_p,
            max_new_tokens=gcfg.max_new_tokens,
            # reproducible but distinct randomness per prompt
            sampling_seed=gcfg.seed + rec["pid"],
        )
        return {
            "pid": rec["pid"],
            "prompt_ids": rec["input_ids"],
            "output_ids": res.output_ids,
            "text": res.text,
            "finish_reason": res.finish_reason,
            "decode_logprobs": res.decode_logprobs,
        }

    try:
        results = []
        with Progress() as progress:
            task = progress.add_task("generate (ref)", total=len(prompts))
            for coro in asyncio.as_completed([one(p) for p in prompts]):
                results.append(await coro)
                progress.advance(task)
    finally:
        await client.close()

    results.sort(key=lambda r: r["pid"])
    # Keep empty generations too (score contributes no rows for them) so the
    # count of prompts the model immediately EOS'd on is visible, not hidden.
    empty = sum(1 for r in results if len(r["output_ids"]) == 0)
    if empty:
        print(f"note: {empty}/{len(results)} prompts produced zero tokens")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # temp + atomic rename: a crash mid-write must not leave a partial
    # generations.jsonl that a later stage would read as complete
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")
    tmp.replace(out_path)
    return len(results)
