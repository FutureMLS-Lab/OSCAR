"""Sample prompts from a HF dataset and tokenize them with the reference
model's chat template.

Output: prompts.jsonl with {pid, text, input_ids}. input_ids are what both
servers receive, so tokenization is done exactly once, here.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .config import RunCfg

_TEXT_COLUMNS = ("prompt", "question", "instruction", "text")


def _extract_text(row: dict, column: str | None) -> str | None:
    if column is not None:
        val = row.get(column)
    else:
        val = None
        for c in _TEXT_COLUMNS:
            if isinstance(row.get(c), str) and row[c].strip():
                val = row[c]
                break
        if val is None:
            # chat-style rows: ultrachat uses "messages", lmsys-chat-1m and
            # WildChat use "conversation"
            turns = row.get("messages") or row.get("conversation")
            if isinstance(turns, list):
                for msg in turns:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        val = msg.get("content")
                        break
    if isinstance(val, str) and val.strip():
        return val.strip()
    return None


def load_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def build_prompts(cfg: RunCfg, out_path: Path) -> int:
    from datasets import load_dataset

    pcfg = cfg.prompts
    tok = load_tokenizer(cfg.reference.model_path)
    ds = load_dataset(pcfg.dataset, split=pcfg.split)

    rng = random.Random(pcfg.seed)
    order = list(range(len(ds)))
    rng.shuffle(order)

    records = []
    skipped_long = skipped_empty = 0
    for idx in order:
        if len(records) >= pcfg.n:
            break
        text = _extract_text(ds[idx], pcfg.column)
        if text is None:
            skipped_empty += 1
            continue
        input_ids = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,  # newer transformers returns a BatchEncoding by default
            **pcfg.chat_template_kwargs,
        )
        if len(input_ids) > pcfg.max_prompt_tokens:
            skipped_long += 1
            continue
        records.append({"pid": len(records), "text": text, "input_ids": list(input_ids)})

    if len(records) < pcfg.n:
        raise RuntimeError(
            f"only {len(records)}/{pcfg.n} usable prompts in {pcfg.dataset}:{pcfg.split} "
            f"(skipped {skipped_long} too-long, {skipped_empty} empty/unparseable)"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # temp + atomic rename: a crash mid-write must not leave a partial file
    # that a later stage would read as complete
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    tmp.replace(out_path)
    return len(records)


def check_tokenizers_match(ref_model_path: str, cand_model_path: str) -> list[str]:
    """Token-level comparison requires identical tokenization. Returns a list of
    mismatch descriptions (empty = OK)."""
    if ref_model_path == cand_model_path:
        return []
    ref = load_tokenizer(ref_model_path)
    cand = load_tokenizer(cand_model_path)
    problems = []
    if len(ref) != len(cand):
        problems.append(f"vocab size differs: {len(ref)} vs {len(cand)}")
    if ref.get_vocab() != cand.get_vocab():
        problems.append("vocab contents differ")
    probe = "Hello, world! 123 éèê 你好 def f(x):\n  return x*2"
    if ref.encode(probe) != cand.encode(probe):
        problems.append("probe string encodes differently")
    return problems
