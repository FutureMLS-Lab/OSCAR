"""Drive startup OSCAR calibration from the tokenizer process: pick the
prompt source, render and pack prompts, arm the collector, run
prefill-only requests, and finalize before the server accepts traffic."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import logging
import os
import random
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.mem_cache.oscar_calibration import prompt_ids_sha256
from sglang.srt.mem_cache.oscar_rotation_paths import oscar_calibration_required
from sglang.srt.runtime_context import get_schedule

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)

_DEFAULT_GPQA_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
)
_GPQA_QUERY_TEMPLATE = (
    "Answer the following multiple choice question. The last line of your "
    "response should be of the following format: 'Answer: $LETTER' (without "
    "quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n"
    "{Question}\n\n"
    "A) {A}\n"
    "B) {B}\n"
    "C) {C}\n"
    "D) {D}"
)
_GPQA_FIELDS = (
    "Question",
    "Correct Answer",
    "Incorrect Answer 1",
    "Incorrect Answer 2",
    "Incorrect Answer 3",
)


def _calibration_cache_dir() -> Path:
    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    return hf_home / "oscar-calibration"


def _write_atomically(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def resolve_oscar_calibration_prompts_path() -> str:
    """The configured prompt file, or the GPQA-Diamond CSV downloaded once
    into the HF cache and rendered to JSONL."""
    local_path = envs.SGLANG_OSCAR_CALIBRATION_PROMPTS_PATH.get()
    if local_path:
        return local_path
    csv_path = _calibration_cache_dir() / "gpqa_diamond.csv"
    if not csv_path.is_file():
        with urllib.request.urlopen(_DEFAULT_GPQA_URL, timeout=60) as response:
            _write_atomically(csv_path, response.read())
        logger.info("Downloaded default GPQA calibration source to %s", csv_path)
    return str(materialize_gpqa_csv_jsonl(csv_path))


def materialize_gpqa_csv_jsonl(csv_path: Path) -> Path:
    """Render an original GPQA CSV to chat-message JSONL with a fixed
    option shuffle; cached by content digest."""
    source = csv_path.read_bytes()
    digest = hashlib.sha256(source + b"\0oscar-gpqa-jsonl-v1-seed0").hexdigest()[:20]
    output_path = _calibration_cache_dir() / f"gpqa_diamond_seed0_{digest}.jsonl"
    if output_path.is_file():
        return output_path

    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not set(_GPQA_FIELDS).issubset(rows[0]):
        raise ValueError(
            f"{csv_path} is not an original GPQA CSV with fields {sorted(_GPQA_FIELDS)}"
        )
    rng = random.Random(0)
    lines = []
    for row in rows:
        choices = [row[name] for name in _GPQA_FIELDS[1:]]
        shuffled = [choices[i] for i in rng.sample(range(4), 4)]
        prompt = _GPQA_QUERY_TEMPLATE.format(
            Question=row["Question"], A=shuffled[0], B=shuffled[1], C=shuffled[2], D=shuffled[3]
        )
        record = {"messages": [{"role": "user", "content": prompt}]}
        lines.append(json.dumps(record, ensure_ascii=False))
    _write_atomically(output_path, ("\n".join(lines) + "\n").encode("utf-8"))
    logger.info("Materialized GPQA calibration JSONL at %s", output_path)
    return output_path


def load_oscar_calibration_messages(path: str) -> list[list[dict]]:
    """JSONL records of ``{"messages": [...]}`` (a bare string is one user
    turn); a ``.csv`` path is treated as an original GPQA CSV."""
    prompt_path = Path(path)
    if not prompt_path.is_file():
        raise FileNotFoundError(f"OSCAR calibration prompt source does not exist: {prompt_path}")
    if prompt_path.suffix.lower() == ".csv":
        prompt_path = materialize_gpqa_csv_jsonl(prompt_path)
    messages = []
    with prompt_path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if isinstance(value, str):
                value = {"messages": [{"role": "user", "content": value}]}
            prompt_messages = value.get("messages") if isinstance(value, dict) else None
            if not isinstance(prompt_messages, list) or not prompt_messages:
                raise ValueError(
                    f"{prompt_path}:{line_no} must contain a non-empty 'messages' list"
                )
            messages.append(prompt_messages)
    if not messages:
        raise ValueError(f"OSCAR calibration prompt file is empty: {prompt_path}")
    return messages


def render_oscar_prompt_ids(
    tokenizer, messages: list[list[dict]], *, max_token_budget: int
) -> list[list[int]]:
    """Apply the chat template to whole prompts until the budget is full."""
    if tokenizer is None:
        raise ValueError("OSCAR startup calibration requires tokenizer initialization")
    selected = []
    selected_tokens = 0
    for prompt_messages in messages:
        ids = tokenizer.apply_chat_template(
            prompt_messages, tokenize=True, add_generation_prompt=True, return_dict=False
        )
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            if len(ids) != 1:
                raise ValueError("One JSONL record must render to exactly one prompt")
            ids = ids[0]
        ids = list(map(int, ids))
        if not ids:
            continue
        if selected_tokens + len(ids) > max_token_budget:
            break
        selected.append(ids)
        selected_tokens += len(ids)
        if selected_tokens == max_token_budget:
            break
    if not selected:
        raise ValueError(
            "No complete OSCAR calibration prompt fits within the configured "
            f"{max_token_budget}-token limit"
        )
    return selected


def pack_oscar_prompt_batches(
    input_ids: list[list[int]],
    *,
    max_batch_size: int,
    chunked_prefill_size: int | None,
    page_size: int,
) -> list[list[tuple[int, list[int]]]]:
    """Group prompts so each batch prefills in one page-rounded chunk; a
    prompt split across chunks would be observed twice at the boundary."""
    chunk_limit = (
        chunked_prefill_size
        if chunked_prefill_size is not None and chunked_prefill_size > 0
        else None
    )
    batches: list[list[tuple[int, list[int]]]] = []
    current: list[tuple[int, list[int]]] = []
    current_tokens = 0
    for index, ids in enumerate(input_ids):
        rounded_tokens = ((len(ids) + page_size - 1) // page_size) * page_size
        if chunk_limit is not None and rounded_tokens > chunk_limit:
            raise ValueError(
                "Each OSCAR calibration prompt must fit in one page-rounded "
                "prefill chunk. Increase --chunked-prefill-size or shorten prompts."
            )
        if current and (
            len(current) >= max_batch_size
            or (chunk_limit is not None and current_tokens + rounded_tokens > chunk_limit)
        ):
            batches.append(current)
            current = []
            current_tokens = 0
        current.append((index, ids))
        current_tokens += rounded_tokens
    if current:
        batches.append(current)
    return batches


def _calibration_request(packed_batch: list[tuple[int, list[int]]]) -> GenerateReqInput:
    # One extra_key per prompt keeps the radix cache from sharing the chat
    # template prefix between prompts; a shared prefix would be prefilled
    # once and the collector would fall short of the token budget.
    return GenerateReqInput(
        input_ids=[ids for _, ids in packed_batch],
        sampling_params={"temperature": 0.0, "max_new_tokens": 0},
        extra_key=[f"oscar-calibration-{index}" for index, _ in packed_batch],
        stream=False,
        log_metrics=False,
        no_logs=True,
    )


async def run_oscar_startup_calibration(tokenizer_manager: TokenizerManager) -> None:
    """Fit and install OSCAR rotations; a no-op unless this launch is calibrating."""
    if not oscar_calibration_required():
        return
    max_token_budget = envs.SGLANG_OSCAR_CALIBRATION_TOKENS.get()
    batch_size = envs.SGLANG_OSCAR_CALIBRATION_BATCH_SIZE.get()
    timeout = envs.SGLANG_OSCAR_CALIBRATION_TIMEOUT.get()
    if max_token_budget <= 0 or batch_size <= 0 or timeout <= 0:
        raise ValueError(
            "SGLANG_OSCAR_CALIBRATION_TOKENS, SGLANG_OSCAR_CALIBRATION_BATCH_SIZE, "
            "and SGLANG_OSCAR_CALIBRATION_TIMEOUT must be positive"
        )
    prompt_messages = load_oscar_calibration_messages(resolve_oscar_calibration_prompts_path())
    input_ids = render_oscar_prompt_ids(
        tokenizer_manager.tokenizer, prompt_messages, max_token_budget=max_token_budget
    )
    selected_token_count = sum(map(len, input_ids))
    prompt_batches = pack_oscar_prompt_batches(
        input_ids,
        max_batch_size=batch_size,
        chunked_prefill_size=get_schedule().chunked_prefill_size,
        page_size=get_schedule().page_size,
    )
    prompt_hash = prompt_ids_sha256(input_ids)
    started_at = time.perf_counter()
    logger.info(
        "Starting OSCAR startup calibration: prompts=%d tokens=%d max_tokens=%d batches=%d",
        len(input_ids), selected_token_count, max_token_budget, len(prompt_batches),
    )

    async def _run() -> None:
        start = await tokenizer_manager.oscar_calibration_control(
            action="start", prompt_sha256=prompt_hash, token_budget=selected_token_count
        )
        if not start.success:
            raise RuntimeError(f"Failed to arm OSCAR calibration: {start.message}")
        for packed_batch in prompt_batches:
            async for _ in tokenizer_manager.generate_request(_calibration_request(packed_batch)):
                pass
        idle_flush = await tokenizer_manager.flush_cache(timeout_s=min(float(timeout), 300.0))
        if not idle_flush.success:
            raise RuntimeError(
                f"Failed to reach an idle scheduler after OSCAR calibration: {idle_flush.message}"
            )
        final = await tokenizer_manager.oscar_calibration_control(action="finalize")
        if not final.success:
            raise RuntimeError(f"Failed to finalize OSCAR calibration: {final.message}")
        if final.captured_tokens != selected_token_count:
            raise RuntimeError(
                "OSCAR calibration finalized with an unexpected token count: "
                f"{final.captured_tokens} != {selected_token_count}"
            )
        logger.info(
            "OSCAR startup calibration completed: prompts=%d tokens=%d elapsed=%.1fs sha256=%s",
            len(input_ids), selected_token_count, time.perf_counter() - started_at, prompt_hash,
        )
        envs.SGLANG_OSCAR_CALIBRATION_ACTIVE.set(False)

    await asyncio.wait_for(_run(), timeout=timeout)
