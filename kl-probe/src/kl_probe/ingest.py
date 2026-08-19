"""Stage: assemble teacher-forcing sequences from PRECOMPUTED reference
generations (allow_generations: false).

Instead of sampling prompts and generating, we are handed the reference model's
own completions (e.g. an HLE response cache). Each row provides
(system, question, reasoning, response); we reconstruct the exact token sequence
the model processed through the chat template and emit the same
`generations.jsonl` the generate stage produces, so `score`/`metrics` downstream
are untouched.

Assembly recipe (VERIFIED against ground-truth completion_tokens, MiniMax-M3):
  msgs = [{system}, {user: question}, {assistant: response, reasoning_content: reasoning}]
  full   = apply_chat_template(msgs, add_generation_prompt=False)   # strip trailing '\n'
  prompt = apply_chat_template(msgs[:2], add_generation_prompt=True) # adaptive thinking
The template inserts the special tokens, the <thinking_instructions> block, and
wraps reasoning in <mm:think></mm:think>; hand-concatenation would be off-
distribution. See docs/precomputed-generations.md.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

from .config import RunCfg
from .prompts import load_tokenizer

# Original HLE system prompts, verbatim from the generator
# jheo_evals/scripts/run-hle-m3-host.py (centerforaisafety/hle predict.py).
# The cache omits the system prompt; select by answer_type (complete bijection).
SYSTEM_EXACT_ANSWER = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your final answer}\n"
    "Exact Answer: {your succinct, final answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)
SYSTEM_MC = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your answer choice}\n"
    "Answer: {your chosen answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)


def _apply(tok, messages, add_generation_prompt):
    # transformers 5.x returns an Encoding unless return_dict=False.
    return tok.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=False,
    )


def _assemble(tok, messages, newline_ids, eos_id):
    """(prompt_ids, output_ids) for a [system, user, assistant] message list.
    prompt_ids = the context up to the assistant header; output_ids = the
    assistant turn with the trailing '\\n' formatting token stripped so it ends
    on eos."""
    prompt_ids = _apply(tok, messages[:2], add_generation_prompt=True)
    full = _apply(tok, messages, add_generation_prompt=False)
    # template appends eos + '\n'; drop the trailing newline formatting token(s)
    while full and full[-1] in newline_ids:
        full = full[:-1]
    if eos_id is not None and full and full[-1] != eos_id:
        print(f"warning: assembled sequence does not end on eos (last id {full[-1]})")
    if full[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError("assembled prompt is not a prefix of the full sequence")
    return prompt_ids, full[len(prompt_ids) :]


def _hle_response_cache_rows(icfg):
    """Yield (messages, finish_reason, text) for each response_cache/*.json."""
    paths = sorted(glob.glob(os.path.join(icfg.source, "*.json")))
    if not paths:
        raise RuntimeError(f"no *.json files under ingest.source {icfg.source!r}")
    for path in paths:
        with open(path) as f:
            row = json.load(f)
        answer_type = row.get(icfg.answer_type_field)
        system = SYSTEM_EXACT_ANSWER if answer_type == "exactMatch" else SYSTEM_MC
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": row.get(icfg.question_field, "")},
            {
                "role": "assistant",
                "content": row.get(icfg.response_field) or "",
                "reasoning_content": row.get(icfg.reasoning_field) or "",
            },
        ]
        yield messages, row.get(icfg.finish_reason_field), row.get(icfg.response_field) or ""


def _chatml_messages_rows(icfg):
    """Yield (messages, finish_reason, text) for each row of a messages jsonl
    (e.g. hle_chatml.jsonl): row.messages = [system, user, assistant] with
    assistant.reasoning_content already separated."""
    with open(icfg.source) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages = row["messages"]
            roles = [m.get("role") for m in messages]
            if roles != ["system", "user", "assistant"]:
                raise RuntimeError(f"expected [system,user,assistant] messages, got {roles}")
            text = messages[2].get("content") or ""
            yield messages, row.get(icfg.finish_reason_field), text


def _pg19_rows(icfg, tok):
    """Yield (prompt_ids, output_ids, text, finish_reason) per PG-19 book.

    PG-19 is raw long-form book text — no chat structure and no completion
    boundary — so it bypasses the chat template entirely. Each book is tokenized
    once; a leading token (BOS if the tokenizer has one, else the first book
    token) becomes the unscored position-0 context (score.py scores from index 1),
    and the book tokens become output_ids. max_score_tokens then caps each book to
    one fixed-length chunk. finish_reason is a constant 'length' (there is no stop
    condition to filter on — set require_finish_reason: null in the config)."""
    from datasets import load_dataset

    ds = load_dataset(icfg.source, split=icfg.split, streaming=True)
    bos_id = getattr(tok, "bos_token_id", None)
    for row in ds:
        text = (row.get(icfg.text_field) or "").strip()
        if not text:
            continue
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) < 2:
            continue
        if bos_id is not None:
            prompt_ids, output_ids = [bos_id], ids
        else:
            prompt_ids, output_ids = ids[:1], ids[1:]
        yield prompt_ids, output_ids, text[:200], "length"


_ROW_SOURCES = {
    "hle_response_cache": _hle_response_cache_rows,
    "chatml_messages": _chatml_messages_rows,
}

# Formats that yield already-tokenized (prompt_ids, output_ids, text,
# finish_reason) directly (raw-text corpora), skipping the chat-template path.
_TOKEN_SOURCES = {
    "pg19": _pg19_rows,
}


def build_ingest(cfg: RunCfg, out_path: Path) -> int:
    icfg = cfg.ingest
    tok = load_tokenizer(cfg.reference.model_path)
    newline_ids = set(tok.encode("\n", add_special_tokens=False))
    eos_id = getattr(tok, "eos_token_id", None)

    records = []
    dropped_finish = truncated = 0

    # Normalize both source shapes to (prompt_ids, output_ids, text, finish_reason).
    # Token sources (pg19) yield those directly; chat sources yield messages that
    # go through the finish_reason filter + chat-template assembly first.
    if icfg.format in _TOKEN_SOURCES:
        assembled = _TOKEN_SOURCES[icfg.format](icfg, tok)
    else:

        def _assembled():
            nonlocal dropped_finish
            for messages, finish_reason, text in _ROW_SOURCES[icfg.format](icfg):
                if (
                    icfg.require_finish_reason is not None
                    and finish_reason != icfg.require_finish_reason
                ):
                    dropped_finish += 1
                    continue
                prompt_ids, output_ids = _assemble(tok, messages, newline_ids, eos_id)
                yield prompt_ids, output_ids, text, finish_reason

        assembled = _assembled()

    for prompt_ids, output_ids, text, finish_reason in assembled:
        if icfg.max_score_tokens is not None and len(output_ids) > icfg.max_score_tokens:
            output_ids = output_ids[: icfg.max_score_tokens]
            truncated += 1
        if not output_ids:
            continue
        records.append(
            {
                "pid": len(records),
                "prompt_ids": prompt_ids,
                "output_ids": output_ids,
                "text": text,
                "finish_reason": finish_reason,
            }
        )
        if icfg.max_prompts is not None and len(records) >= icfg.max_prompts:
            break

    if not records:
        raise RuntimeError("ingest produced zero usable sequences")

    # Honesty (the finish_reason filter is permanent; the caps are temporary
    # smoke-test scaffolding): surface what was dropped/capped.
    if dropped_finish:
        print(
            f"ingest: dropped {dropped_finish} rows with finish_reason != "
            f"{icfg.require_finish_reason!r} (e.g. cap-truncated traces)"
        )
    if icfg.max_prompts is not None:
        print(f"ingest: capped to max_prompts={icfg.max_prompts} (SMOKE — temporary cap)")
    if truncated:
        print(
            f"ingest: truncated {truncated} sequences to max_score_tokens="
            f"{icfg.max_score_tokens} (SMOKE — temporary cap)"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    tmp.replace(out_path)
    return len(records)
