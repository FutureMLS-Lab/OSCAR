"""Stage: prefill-score prompt+generation sequences against one server.

Both the reference and every candidate go through this same code path, so the
distributions being compared come from identical (prefill) kernels.

Scoring covers ALL positions: we score from index 1 to the end of the full
`prompt + generation` sequence (position 0 has no preceding context and is
unscoreable). `pos` in the output is the ABSOLUTE index in the full sequence, so
metrics can bucket by `pos // bucket_size` directly.

Rows are written incrementally with a ParquetWriter so peak memory is
O(one sequence), not O(all positions). Row order is completion order, not sorted;
`metrics.compare_scores` sorts by (pid, pos) before joining, so order here does
not matter.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rich.progress import Progress

from .config import ModelSpec, RunCfg, ServerRuntime
from .engine import make_client

# First scoreable position in the full sequence. Position 0 is the leading token
# (no preceding context), so scoring starts at 1 → all remaining positions.
SCORE_FROM = 1

SCORE_SCHEMA = pa.schema(
    [
        ("pid", pa.int32()),
        ("pos", pa.int32()),  # ABSOLUTE position in the full prompt+generation sequence
        ("token_id", pa.int32()),
        ("logprob", pa.float32()),  # exact logprob of the actual token
        ("top_ids", pa.list_(pa.int32())),
        ("top_logprobs", pa.list_(pa.float32())),
    ]
)


async def run_score(
    cfg: RunCfg, spec: ModelSpec, runtime: ServerRuntime, generations_path: Path, out_path: Path
) -> int:
    if cfg.scoring.score_chunk_size and cfg.scoring.score_chunk_size > 0:
        raise NotImplementedError(
            "chunked scoring (scoring.score_chunk_size > 0) is deferred; the default "
            "one-shot path is used. See docs/precomputed-generations.md, "
            '"Logprob response payload".'
        )
    gens = [json.loads(line) for line in generations_path.open()]
    client = make_client(spec, runtime.url, concurrency=cfg.scoring.concurrency)
    top_k = cfg.scoring.top_logprobs_num  # resolved to an int at run start

    async def one(rec: dict) -> list[dict]:
        full_ids = rec["prompt_ids"] + rec["output_ids"]
        if len(full_ids) <= SCORE_FROM:
            return []
        scores = await client.score(full_ids, prompt_len=SCORE_FROM, top_k=top_k)
        # The scored positions must be exactly full_ids[SCORE_FROM:]. A systematic
        # shift would hit ref and cand identically, so the ref-vs-cand alignment
        # check in compare_scores cannot catch it — assert it here.
        expected = full_ids[SCORE_FROM:]
        if len(scores) != len(expected):
            raise RuntimeError(
                f"pid {rec['pid']}: scored {len(scores)} positions for {len(expected)} "
                f"scoreable tokens"
            )
        if [s.token_id for s in scores] != expected:
            raise RuntimeError(f"pid {rec['pid']}: scored token ids don't match the sequence")
        return [
            {
                "pid": rec["pid"],
                "pos": SCORE_FROM + i,  # absolute position in the full sequence
                "token_id": s.token_id,
                "logprob": s.logprob,
                "top_ids": s.top_ids,
                "top_logprobs": s.top_logprobs,
            }
            for i, s in enumerate(scores)
        ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    writer = pq.ParquetWriter(tmp, SCORE_SCHEMA)
    total = 0
    try:
        with Progress() as progress:
            task = progress.add_task(f"score ({spec.model_path})", total=len(gens))
            for coro in asyncio.as_completed([one(g) for g in gens]):
                rows = await coro
                if rows:
                    writer.write_table(pa.Table.from_pylist(rows, schema=SCORE_SCHEMA))
                    total += len(rows)
                progress.advance(task)
    finally:
        writer.close()
        await client.close()
    # temp + atomic rename: a crash must not leave a partial parquet behind
    tmp.replace(out_path)
    return total
