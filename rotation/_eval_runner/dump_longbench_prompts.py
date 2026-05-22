#!/usr/bin/env python3
"""Send long-context LongBench prompts to a sglang dump server (max_tokens=1)
to trigger the DUMP_KVCACHE hook with LongBench-style activations.

This is the Phase 1d calibration-data producer for the OSCAR rotation phase.
The KV-cache statistics from 4K-28K narrative/multi-doc contexts should be
better matched to the LongBench-E evaluation distribution than the GPQA
calibration was (~500-token MCQ prompts).

Subsets prioritised for length: gov_report_e (long govt reports),
2wikimqa_e and hotpotqa_e (multi-passage QA).

Server-side env vars expected:
    DUMP_KVCACHE=true
    DUMP_KVCACHE_TOKENS=<budget>   (script keeps sending until budget hit)

Usage:
  python dump_longbench_prompts.py \
    --model Qwen/Qwen3-8B \
    --base-url http://127.0.0.1:31050/v1 \
    --num-threads 4 \
    --num-prompts 25
"""

import argparse
import json
import os
import random
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


# Prompt templates copied verbatim from THUDM/LongBench/config/dataset2prompt.json
# (Identical to OScaR-KV-Quant's, identical to ours in eval_longbench_e.py.)
DATASET2PROMPT = {
    "qasper": (
        "You are given a scientific article and a question. Answer the question as "
        "concisely as you can, using a single phrase or sentence if possible. If the "
        "question cannot be answered based on the information in the article, write "
        "\"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", "
        "or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n "
        "Answer the question based on the above article as concisely as you can, using a "
        "single phrase or sentence if possible. If the question cannot be answered based "
        "on the information in the article, write \"unanswerable\". If the question is a "
        "yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any "
        "explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary of the "
        "report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\n"
        "Summary:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all news. \n\n"
        "News:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer and "
        "do not output any other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do "
        "not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer and "
        "do not output any other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do "
        "not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the "
        "following question based on the above text, only give me the answer and do not "
        "output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine "
        "which paragraph the abstract is from.\n\n{context}\n\nThe following is an "
        "abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the "
        "abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph "
        "2\", etc.\n\nThe answer is: "
    ),
}


def _build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--num-prompts", type=int, default=25,
                   help="Total long prompts to send across subsets. "
                   "Server-side DUMP_KVCACHE_TOKENS will auto-stop sooner.")
    p.add_argument("--num-threads", type=int, default=4,
                   help="Concurrent prompts. Long contexts -> low concurrency.")
    p.add_argument("--max-input-len", type=int, default=28000,
                   help="Truncate each prompt to this many tokens "
                   "(first-half + last-half), matching LongBench pred.py.")
    p.add_argument("--min-context-len", type=int, default=8000,
                   help="Only keep examples whose raw `length` field is >= this.")
    p.add_argument("--max-tokens", type=int, default=1,
                   help="1 is enough — only the prefill pass triggers the dump.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-chat", type=lambda x: x.lower() in ("1", "true", "yes"),
                   default=True,
                   help="Match the eval client's chat-mode behavior so the "
                        "calibration KV-cache distribution matches the eval.")
    return p


def _truncate_prompt(prompt, tokenizer, max_len):
    """Token-level first-half + last-half truncation (LongBench pred.py)."""
    ids = tokenizer(prompt, truncation=False, return_tensors=None)["input_ids"]
    if len(ids) <= max_len:
        return prompt, len(ids)
    half = max_len // 2
    head = tokenizer.decode(ids[:half], skip_special_tokens=True)
    tail = tokenizer.decode(ids[-half:], skip_special_tokens=True)
    return head + tail, max_len


def _load_subset(zip_path, subset_e):
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        with z.open(f"data/{subset_e}.jsonl") as f:
            for line in f:
                rows.append(json.loads(line))
    return rows


def _build_prompts(num_prompts, min_ctx, max_input_len, tokenizer, seed):
    """Pick ~num_prompts long-context prompts across 4 subsets."""
    print(f"[dump] fetching THUDM/LongBench data.zip", flush=True)
    zip_path = hf_hub_download(
        repo_id="THUDM/LongBench", filename="data.zip",
        repo_type="dataset",
        cache_dir=os.environ.get("HF_DATASETS_CACHE")
        or os.environ.get("HF_HOME"),
    )
    rng = random.Random(seed)

    # Broaden subset coverage — Phase 1d v1 used only 4 subsets which left
    # 9 of the 13 LongBench-E datasets without any representation in the
    # calibration distribution. Expand to 7 subsets covering the main task
    # types (long-doc QA, multi-doc QA, single-doc QA, summarization, and
    # passage retrieval) so the learned rotation generalises across the
    # evaluation distribution.
    base_weights = [
        ("gov_report_e",        "gov_report",        3),   # long-doc summarization
        ("multi_news_e",        "multi_news",        3),   # multi-doc summarization
        ("qasper_e",            "qasper",            3),   # scientific-QA, long doc
        ("multifieldqa_en_e",   "multifieldqa_en",   3),   # multi-field QA
        ("hotpotqa_e",          "hotpotqa",          3),   # multi-hop QA
        ("2wikimqa_e",          "2wikimqa",          3),   # multi-hop wiki QA
        ("passage_retrieval_en_e", "passage_retrieval_en", 2),  # precise retrieval
    ]
    total_w = sum(w for _, _, w in base_weights)
    subset_split = []
    assigned = 0
    for i, (subset_e, base, w) in enumerate(base_weights):
        if i == len(base_weights) - 1:
            n = max(1, num_prompts - assigned)
        else:
            n = max(1, int(num_prompts * w / total_w))
        subset_split.append((subset_e, base, n))
        assigned += n

    prompts = []
    for subset_e, base, want in subset_split:
        rows = _load_subset(zip_path, subset_e)
        # Sort by descending raw length, then filter to min_ctx+
        rows.sort(key=lambda r: -int(r["length"]))
        eligible = [r for r in rows if int(r["length"]) >= min_ctx]
        # If a subset doesn't have enough min_ctx+ examples, fall back to the
        # longest available examples (sorted by descending length) — better
        # to include slightly-shorter examples from a thin subset than to
        # skip that subset's distribution entirely.
        if len(eligible) < want:
            print(f"[dump]   {subset_e}: only {len(eligible)} examples "
                  f">= {min_ctx} tokens, want {want} — falling back to "
                  f"longest-{want} regardless of min_ctx", flush=True)
            eligible = rows
        # Take longest `want` (with a touch of randomness via seed shuffle of ties).
        rng.shuffle(eligible)  # tiebreak — but list is already sorted-ish so this is mild
        eligible.sort(key=lambda r: -int(r["length"]))
        pick = eligible[:want]
        print(f"[dump]   {subset_e}: {len(rows)} examples, "
              f"{sum(1 for r in rows if int(r['length']) >= min_ctx)} >= {min_ctx} tokens, picking {len(pick)}",
              flush=True)

        tpl = DATASET2PROMPT[base]
        for r in pick:
            raw = tpl.format(context=r["context"], input=r.get("input", ""))
            trunc, ntok = _truncate_prompt(raw, tokenizer, max_input_len)
            prompts.append({
                "subset": subset_e,
                "prompt": trunc,
                "tokens": ntok,
            })
    rng.shuffle(prompts)
    return prompts


def _send_one(client, model, prompt, temperature, top_p, top_k, max_tokens,
              use_chat):
    try:
        if use_chat:
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body={
                    "top_k": top_k,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
        else:
            client.completions.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body={"top_k": top_k},
            )
        return "ok"
    except Exception as e:
        return f"err: {e!r}"


def main():
    args = _build_argparser().parse_args()
    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=900.0)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    prompts = _build_prompts(args.num_prompts, args.min_context_len,
                             args.max_input_len, tokenizer, args.seed)
    total_tokens = sum(p["tokens"] for p in prompts)
    print(f"[dump] sending {len(prompts)} LongBench prompts at "
          f"max_tokens={args.max_tokens} (sum tokens={total_tokens})",
          flush=True)
    print(f"[dump]   use_chat={args.use_chat}", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.num_threads) as ex:
        futs = [
            ex.submit(_send_one, client, args.model, p["prompt"],
                      0.0, 1.0, 40, args.max_tokens, args.use_chat)
            for p in prompts
        ]
        n_ok = n_err = 0
        for i, f in enumerate(as_completed(futs)):
            r = f.result()
            if r == "ok":
                n_ok += 1
            else:
                n_err += 1
                if n_err <= 5:
                    print(f"  prompt {i}: {r}", flush=True)
    print(f"[dump] done in {time.time()-t0:.1f}s  ok={n_ok}  err={n_err}",
          flush=True)


if __name__ == "__main__":
    main()
