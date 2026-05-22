#!/usr/bin/env python
"""LongBench-E eval client for a running SGLang OpenAI-compatible server.

- Downloads 13 LongBench-E datasets from HuggingFace `THUDM/LongBench`.
- Truncates each prompt to --max-input-len tokens via first-half + last-half
  (matches OScaR-KV-Quant's pred.py).
- Greedy decoding (temperature=0).
- Scores each dataset with the LongBench scorer_e bucketed metrics
  (0-4k / 4-8k / 8k+) and computes a single per-dataset score = mean of the
  three buckets that have samples, matching the published LongBench leaderboard.
- Writes per-dataset predictions to <output-dir>/pred/<dataset>.jsonl and a
  summary scores file to <output-dir>/result.json.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import string
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import openai
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


DATASETS_E = [
    "qasper_e",
    "multifieldqa_en_e",
    "hotpotqa_e",
    "2wikimqa_e",
    "gov_report_e",
    "multi_news_e",
    "trec_e",
    "triviaqa_e",
    "samsum_e",
    "passage_count_e",
    "passage_retrieval_en_e",
    "lcc_e",
    "repobench-p_e",
]

# Prompts (from THUDM/LongBench config/dataset2prompt.json) keyed by base name.
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
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the "
        "following question based on the above text, only give me the answer and do not "
        "output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer and "
        "do not output any other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do "
        "not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer and "
        "do not output any other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do "
        "not output any other words.\n\nQuestion: {input}\nAnswer:"
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
    "trec": (
        "Please determine the type of the question below. Here are some examples of "
        "questions.\n\n{context}\n{input}"
    ),
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the answer and do "
        "not output any other words. The following are some examples.\n\n{context}\n\n"
        "{input}"
    ),
    "samsum": (
        "Summarize the dialogue into a few short sentences. The following are some "
        "examples.\n\n{context}\n\n{input}"
    ),
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia. Some of them may be "
        "duplicates. Please carefully read these paragraphs and determine how many unique "
        "paragraphs there are after removing duplicates. In other words, how many "
        "non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the "
        "final count of unique paragraphs after removing duplicates. The output format "
        "should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer "
        "is: "
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine "
        "which paragraph the abstract is from.\n\n{context}\n\nThe following is an "
        "abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the "
        "abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph "
        "2\", etc.\n\nThe answer is: "
    ),
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": (
        "Please complete the code given below. \n{context}{input}Next line of code:\n"
    ),
}

# Max generation tokens per dataset (THUDM/LongBench dataset2maxlen.json).
DATASET2MAXLEN = {
    "qasper": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "gov_report": 512,
    "multi_news": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "lcc": 64,
    "repobench-p": 64,
}


# ---------- LongBench metric implementations (copy of OScaR-KV-Quant/longbench_metrics.py).

def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score(prediction: list[str], ground_truth: list[str]) -> float:
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction: str, ground_truth: str, **_kw: Any) -> float:
    np_pred = normalize_answer(prediction).split()
    np_gt = normalize_answer(ground_truth).split()
    return f1_score(np_pred, np_gt)


def count_score(prediction: str, ground_truth: str, **_kw: Any) -> float:
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    right = sum(1 for n in numbers if str(n) == str(ground_truth))
    return float(right / len(numbers))


def retrieval_score(prediction: str, ground_truth: str, **_kw: Any) -> float:
    matches = re.findall(r"Paragraph (\d+)", ground_truth)
    gt_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    right = sum(1 for n in numbers if str(n) == str(gt_id))
    return float(right / len(numbers))


def classification_score(prediction: str, ground_truth: str, **kwargs: Any) -> float:
    em_match_list: list[str] = []
    all_classes = kwargs["all_classes"]
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in list(em_match_list):
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        return 1.0 / len(em_match_list)
    return 0.0


def _rouge_l_f(prediction: str, ground_truth: str) -> float:
    """ROUGE-L F1 (LCS-based), matching the `rouge` package's behavior.

    `rouge.Rouge` lower-cases, strips, splits on whitespace, then computes LCS-F1
    with beta=1.2 (per `rouge`'s default). We replicate that here so we don't
    need the package as a dep.
    """
    pred = prediction.lower().split()
    ref = ground_truth.lower().split()
    if not pred or not ref:
        return 0.0
    m, n = len(pred), len(ref)
    # Standard LCS DP.
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i - 1] == ref[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    precision = lcs / m
    recall = lcs / n
    beta = 1.2
    denom = recall + beta * beta * precision
    if denom == 0:
        return 0.0
    return ((1 + beta * beta) * precision * recall) / denom


# Prefer the official `rouge` pip package (what OScaR-KV-Quant uses verbatim).
# Falls back to the self-written impl only if rouge isn't installed — but the
# fallback under-counts by 7-13 points on summarisation, so result.json means
# stop being directly comparable to OScaR-KV-Quant's published numbers. The
# canonical eval setup pre-installs `rouge`+`fuzzywuzzy` in the `oscar` conda env.
try:
    from rouge import Rouge as _OfficialRouge
    _ROUGE_OBJ = _OfficialRouge()
    _HAVE_OFFICIAL_ROUGE = True
except ImportError:
    _ROUGE_OBJ = None
    _HAVE_OFFICIAL_ROUGE = False

try:
    from fuzzywuzzy import fuzz as _official_fuzz
    _HAVE_OFFICIAL_FUZZ = True
except ImportError:
    _official_fuzz = None
    _HAVE_OFFICIAL_FUZZ = False


def rouge_score(prediction: str, ground_truth: str, **_kw: Any) -> float:
    if _HAVE_OFFICIAL_ROUGE:
        try:
            return _ROUGE_OBJ.get_scores([prediction], [ground_truth], avg=True)[
                "rouge-l"
            ]["f"]
        except Exception:
            return 0.0
    try:
        return _rouge_l_f(prediction, ground_truth)
    except Exception:
        return 0.0


def _fuzz_ratio(a: str, b: str) -> float:
    """fuzzywuzzy.fuzz.ratio: 100 * 2 * matching_blocks / (len(a) + len(b)).

    Uses the official `fuzzywuzzy` package when available (what
    OScaR-KV-Quant's `longbench_metrics.py` calls); otherwise falls back to
    a difflib.SequenceMatcher.ratio() approximation.
    """
    if _HAVE_OFFICIAL_FUZZ:
        return _official_fuzz.ratio(a, b)
    import difflib

    return difflib.SequenceMatcher(None, a, b).ratio() * 100


def code_sim_score(prediction: str, ground_truth: str, **_kw: Any) -> float:
    all_lines = prediction.lstrip("\n").split("\n")
    pred = ""
    for line in all_lines:
        if "`" not in line and "#" not in line and "//" not in line:
            pred = line
            break
    return _fuzz_ratio(pred, ground_truth) / 100


DATASET2METRIC = {
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "gov_report": rouge_score,
    "multi_news": rouge_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "passage_count": count_score,
    "passage_retrieval_en": retrieval_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}


def scorer_e(dataset: str, predictions: list[str], answers: list[list[str]],
             lengths: list[int], all_classes: list[str] | None) -> dict[str, float]:
    """LongBench-E bucketed scorer: 0-4k / 4-8k / 8k+, mean of available buckets."""
    buckets: dict[str, list[float]] = {"0-4k": [], "4-8k": [], "8k+": []}
    metric = DATASET2METRIC[dataset]
    for pred, gts, length in zip(predictions, answers, lengths):
        if dataset in ("trec", "triviaqa", "samsum"):
            pred = pred.lstrip("\n").split("\n")[0]
        score = 0.0
        for gt in gts:
            score = max(score, metric(pred, gt, all_classes=all_classes))
        if length < 4000:
            buckets["0-4k"].append(score)
        elif length < 8000:
            buckets["4-8k"].append(score)
        else:
            buckets["8k+"].append(score)
    out: dict[str, float] = {}
    for k, vals in buckets.items():
        out[k] = round(100 * float(np.mean(vals)), 2) if vals else float("nan")
    available = [v for v in out.values() if not (isinstance(v, float) and v != v)]  # filter NaN
    out["mean"] = round(float(np.mean(available)), 2) if available else float("nan")
    return out


# ---------- Truncation + inference.

def truncate_prompt(prompt: str, tokenizer: AutoTokenizer, max_len: int) -> str:
    """Token-level first-half + last-half truncation (LongBench pred.py)."""
    ids = tokenizer(prompt, truncation=False, return_tensors=None)["input_ids"]
    if len(ids) <= max_len:
        return prompt
    half = max_len // 2
    head = tokenizer.decode(ids[:half], skip_special_tokens=True)
    tail = tokenizer.decode(ids[-half:], skip_special_tokens=True)
    return head + tail


# LongBench's pred.py skips the chat template for these few-shot ICL tasks
# whose prompt already contains the demonstrations in raw form. Matches
# https://github.com/THUDM/LongBench/blob/main/LongBench/pred.py
RAW_COMPLETION_DATASETS = {"trec", "triviaqa", "samsum", "lsht", "lcc",
                           "repobench-p"}


def call_completion(client: openai.OpenAI, model: str, prompt: str,
                    max_tokens: int, stop: list[str] | None,
                    retries: int = 4) -> str:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.completions.create(
                model=model,
                prompt=prompt,
                temperature=0.0,
                top_p=1.0,
                max_tokens=max_tokens,
                stop=stop,
                n=1,
            )
            return resp.choices[0].text
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"completions call failed after {retries} retries: {last_err}")


def call_chat(client: openai.OpenAI, model: str, prompt: str, max_tokens: int,
              retries: int = 4) -> str:
    """Chat-completions call with Qwen3 thinking explicitly disabled."""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                top_p=1.0,
                max_tokens=max_tokens,
                n=1,
                # Disable Qwen3's <think> generation — LongBench expects a
                # direct answer, and thinking would burn the budget.
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"chat call failed after {retries} retries: {last_err}")


def _load_longbench_e(dataset_e: str, data_zip_path: str) -> list[dict[str, Any]]:
    """Read a LongBench-E split directly from the bundled `data.zip`."""
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(data_zip_path) as z:
        with z.open(f"data/{dataset_e}.jsonl") as f:
            for line in f:
                rows.append(json.loads(line))
    return rows


def run_dataset(dataset_e: str, client: openai.OpenAI, model: str,
                tokenizer: AutoTokenizer, max_input_len: int,
                output_dir: Path, num_workers: int,
                data_zip_path: str, use_chat: bool) -> dict[str, Any]:
    base = dataset_e[:-2]  # strip "_e"
    prompt_tpl = DATASET2PROMPT[base]
    max_gen = DATASET2MAXLEN[base]
    pred_path = output_dir / "pred" / f"{dataset_e}.jsonl"
    pred_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[{dataset_e}] loading from {data_zip_path}", flush=True)
    ds = _load_longbench_e(dataset_e, data_zip_path)
    # Per LongBench pred.py, few-shot ICL datasets stay in raw completion mode
    # even when use_chat is enabled.
    ds_use_chat = use_chat and base not in RAW_COMPLETION_DATASETS
    mode = "chat" if ds_use_chat else "raw"
    print(f"[{dataset_e}] {len(ds)} examples, max_gen={max_gen}, mode={mode}",
          flush=True)

    # Resume: skip already-predicted indices.
    done_idx: set[int] = set()
    if pred_path.exists():
        with open(pred_path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    done_idx.add(int(row["index"]))
                except Exception:  # noqa: BLE001
                    continue
        print(f"[{dataset_e}] resuming, {len(done_idx)} already done", flush=True)

    # In raw-completion mode, triviaqa benefits from a hard newline stop.
    stop = ["\n"] if (not ds_use_chat and base == "triviaqa") else None

    items = [(i, ex) for i, ex in enumerate(ds) if i not in done_idx]

    def _do(idx_ex):
        idx, ex = idx_ex
        prompt = prompt_tpl.format(context=ex["context"], input=ex["input"])
        prompt = truncate_prompt(prompt, tokenizer, max_input_len)
        if ds_use_chat:
            pred = call_chat(client, model, prompt, max_gen)
        else:
            pred = call_completion(client, model, prompt, max_gen, stop)
        return {
            "index": idx,
            "pred": pred,
            "answers": ex["answers"],
            "all_classes": ex.get("all_classes"),
            "length": int(ex["length"]),
            "mode": mode,
        }

    # Append-only writer with newline flush each row, so we can resume.
    t0 = time.time()
    with open(pred_path, "a", encoding="utf-8") as fout, \
            cf.ThreadPoolExecutor(max_workers=num_workers) as pool:
        for i, row in enumerate(pool.map(_do, items)):
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            if (i + 1) % 5 == 0 or (i + 1) == len(items):
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-6)
                print(f"[{dataset_e}] {i+1}/{len(items)} done "
                      f"({rate:.2f}/s)", flush=True)

    # Score the full file (including resumed rows).
    preds: list[str] = []
    answers: list[list[str]] = []
    lengths: list[int] = []
    all_classes: list[str] | None = None
    with open(pred_path) as f:
        rows = [json.loads(l) for l in f]
    rows.sort(key=lambda r: r["index"])
    for r in rows:
        preds.append(r["pred"])
        answers.append(r["answers"])
        lengths.append(int(r["length"]))
        ac = r.get("all_classes")
        if ac and all_classes is None:
            all_classes = ac
    scores = scorer_e(base, preds, answers, lengths, all_classes)
    print(f"[{dataset_e}] scores={scores}", flush=True)
    return {"dataset": dataset_e, "n": len(rows), "scores": scores}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True,
                   help="OpenAI-compatible base url, e.g. http://127.0.0.1:31200/v1")
    p.add_argument("--model", required=True, help="HF model id (for both tokenizer + API)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--datasets", nargs="*", default=DATASETS_E,
                   help="LongBench-E subsets to run; default: all 13.")
    p.add_argument("--max-input-len", type=int, default=32768)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--use-chat", type=lambda x: x.lower() in ("1", "true", "yes"),
                   default=True,
                   help="Use chat-completions w/ chat template (default true). "
                        "Few-shot ICL subsets (trec/triviaqa/samsum/lsht/lcc/"
                        "repobench-p) always stay in raw-completion mode to "
                        "match LongBench/pred.py.")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    client = openai.OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=600.0)

    # Smoke-test the server (raw completions always works).
    print(f"[main] pinging {args.base_url} (use_chat={args.use_chat}) ...",
          flush=True)
    smoke = call_completion(client, args.model, "Hello, ", 4, None)
    print(f"[main] server OK, smoke={smoke!r}", flush=True)

    # Download the LongBench data.zip once (it's the only file in
    # THUDM/LongBench that contains the _e JSONLs). HF redirects this to
    # zai-org/LongBench under the hood.
    print("[main] fetching LongBench data.zip ...", flush=True)
    data_zip_path = hf_hub_download(
        repo_id="THUDM/LongBench", filename="data.zip",
        repo_type="dataset",
        cache_dir=os.environ.get("HF_DATASETS_CACHE")
        or os.environ.get("HF_HOME"),
    )
    print(f"[main] data.zip = {data_zip_path}", flush=True)

    all_results: list[dict[str, Any]] = []
    for ds in args.datasets:
        if ds not in DATASETS_E:
            print(f"[main] WARNING: {ds} not in DATASETS_E, skipping", flush=True)
            continue
        try:
            res = run_dataset(ds, client, args.model, tokenizer,
                              args.max_input_len, output_dir, args.num_workers,
                              data_zip_path, args.use_chat)
        except Exception as e:  # noqa: BLE001
            print(f"[main] {ds} failed: {e!r}", flush=True)
            res = {"dataset": ds, "error": repr(e)}
        all_results.append(res)
        # Persist running summary after each dataset.
        summary = _build_summary(all_results)
        with open(output_dir / "results.json", "w") as f:
            json.dump(summary, f, indent=2)

    # Final print.
    summary = _build_summary(all_results)
    with open(output_dir / "result.json", "w") as f:
        json.dump(summary, f, indent=2)
    _print_table(summary)
    return 0


def _build_summary(all_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a flat-friendly summary.

    Top-level fields make automated comparison easy:
      - `mean`: overall LongBench-E mean (float, 2 decimals).
      - `per_dataset`: {name: per-dataset mean float}.
      - `per_dataset_buckets`: {name: {0-4k, 4-8k, 8k+, mean}} for inspection.
    """
    per_dataset_mean: dict[str, float] = {}
    per_dataset_buckets: dict[str, dict[str, float]] = {}
    errors: dict[str, str] = {}
    means: list[float] = []
    for r in all_results:
        ds = r["dataset"]
        if "scores" not in r:
            errors[ds] = r.get("error", "unknown")
            continue
        per_dataset_buckets[ds] = r["scores"]
        m = r["scores"].get("mean")
        if isinstance(m, float) and not (m != m):
            per_dataset_mean[ds] = m
            means.append(m)
    overall = round(float(np.mean(means)), 2) if means else float("nan")
    if _HAVE_OFFICIAL_ROUGE and _HAVE_OFFICIAL_FUZZ:
        metric_impl = (
            "official rouge==1.0.1 + fuzzywuzzy==0.18.0 "
            "(matches OScaR-KV-Quant's longbench_metrics.py)"
        )
    else:
        metric_impl = (
            "self-written ROUGE-L (LCS-based) + difflib.SequenceMatcher fallback "
            f"(rouge_pip_installed={_HAVE_OFFICIAL_ROUGE}, "
            f"fuzz_pip_installed={_HAVE_OFFICIAL_FUZZ})"
        )
    out: dict[str, Any] = {
        "mean": overall,
        "longbench_e_mean": overall,  # back-compat with prior key.
        "metric_impl": metric_impl,
        "per_dataset": per_dataset_mean,
        "per_dataset_buckets": per_dataset_buckets,
        "datasets_scored": len(means),
    }
    if errors:
        out["errors"] = errors
    return out


def _print_table(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 78, flush=True)
    print(f"LongBench-E summary  (mean over {summary['datasets_scored']} datasets = "
          f"{summary['mean']})", flush=True)
    print("=" * 78, flush=True)
    fmt = "{:<26} {:>8} {:>8} {:>8} {:>8}"
    print(fmt.format("dataset", "0-4k", "4-8k", "8k+", "mean"), flush=True)
    print("-" * 62, flush=True)
    for name in DATASETS_E:
        sc = summary["per_dataset_buckets"].get(name)
        if not sc or "mean" not in sc:
            print(fmt.format(name, "-", "-", "-", "ERR"), flush=True)
            continue

        def _fmt(v):
            if isinstance(v, float) and v != v:
                return "-"
            return f"{v:.2f}" if isinstance(v, float) else str(v)
        print(fmt.format(name, _fmt(sc.get("0-4k")), _fmt(sc.get("4-8k")),
                         _fmt(sc.get("8k+")), _fmt(sc.get("mean"))), flush=True)
    print("=" * 78, flush=True)


if __name__ == "__main__":
    sys.exit(main())
