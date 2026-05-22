#!/usr/bin/env python
"""OCRBench eval client for a running SGLang OpenAI-compatible VL server.

Hits /v1/chat/completions with the standard OpenAI vision payload
(image + text prompt), greedy decoding, max 100 new tokens.

Scores out of 1000 by category, matching OCRBench's official rule
(https://github.com/Yuliang-Liu/MultimodalOCR/blob/main/OCRBench/example.py).
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

import openai
from datasets import load_dataset


# OCRBench's 10-category scoring buckets (total = 1000 pts).
CATEGORY_BUDGETS = {
    "Regular Text Recognition": 50,
    "Irregular Text Recognition": 50,
    "Artistic Text Recognition": 50,
    "Handwriting Recognition": 50,
    "Digit String Recognition": 50,
    "Non-Semantic Text Recognition": 50,
    "Scene Text-centric VQA": 200,
    "Doc-oriented VQA": 200,
    "Key Information Extraction": 200,
    "Handwritten Mathematical Expression Recognition": 100,
}


def _to_png_b64(img) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def call_vl_server(client: openai.OpenAI, model: str, question: str,
                   image_b64: str, max_tokens: int = 100,
                   retries: int = 4) -> str:
    payload = [
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        {"type": "text", "text": question},
    ]
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": payload}],
                temperature=0.0,
                top_p=1.0,
                max_tokens=max_tokens,
                n=1,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"VL server call failed after {retries} retries: {last_err}")


def _match(predict: str, answer: str | list[str], is_hme: bool) -> int:
    """Substring containment per OCRBench example.py."""
    answers = answer if isinstance(answer, list) else [answer]
    if is_hme:
        norm_p = predict.strip().replace("\n", " ").replace(" ", "")
        for a in answers:
            if a.strip().replace("\n", " ").replace(" ", "") in norm_p:
                return 1
        return 0
    norm_p = predict.lower().strip().replace("\n", " ")
    for a in answers:
        if a.lower().strip().replace("\n", " ") in norm_p:
            return 1
    return 0


def run_eval(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "predictions.jsonl"

    print(f"[main] loading {args.dataset_repo} ...", flush=True)
    ds = load_dataset(args.dataset_repo, split=args.split)
    print(f"[main] {len(ds)} examples", flush=True)

    client = openai.OpenAI(base_url=args.base_url, api_key=args.api_key,
                           timeout=600.0)

    # Resume.
    done: set[int] = set()
    if pred_path.exists():
        with open(pred_path) as f:
            for line in f:
                try:
                    done.add(int(json.loads(line)["index"]))
                except Exception:  # noqa: BLE001
                    continue
        print(f"[main] resuming, {len(done)} already done", flush=True)

    items = [(i, ds[i]) for i in range(len(ds)) if i not in done]

    def _do(idx_ex):
        idx, ex = idx_ex
        img_b64 = _to_png_b64(ex["image"])
        pred = call_vl_server(client, args.model, ex["question"], img_b64,
                              max_tokens=args.max_tokens)
        return {
            "index": idx,
            "dataset": ex["dataset"],
            "question": ex["question"],
            "question_type": ex["question_type"],
            "answer": list(ex["answer"]) if isinstance(ex["answer"], list)
                       else [ex["answer"]],
            "predict": pred,
        }

    t0 = time.time()
    with open(pred_path, "a", encoding="utf-8") as fout, \
            cf.ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        for i, row in enumerate(pool.map(_do, items)):
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            if (i + 1) % 25 == 0 or (i + 1) == len(items):
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-6)
                print(f"[main] {i+1}/{len(items)} done ({rate:.2f}/s)", flush=True)

    # Score the full file.
    with open(pred_path) as f:
        rows = [json.loads(l) for l in f]
    rows.sort(key=lambda r: r["index"])

    per_cat = {k: 0 for k in CATEGORY_BUDGETS}
    counts_per_cat: dict[str, int] = {k: 0 for k in CATEGORY_BUDGETS}
    per_dataset_pass: dict[str, int] = {}
    per_dataset_total: dict[str, int] = {}
    for r in rows:
        cat = r["question_type"]
        ds_name = r["dataset"]
        is_hme = cat == "Handwritten Mathematical Expression Recognition"
        result = _match(r["predict"], r["answer"], is_hme)
        if cat in per_cat:
            per_cat[cat] += result
            counts_per_cat[cat] += 1
        per_dataset_pass[ds_name] = per_dataset_pass.get(ds_name, 0) + result
        per_dataset_total[ds_name] = per_dataset_total.get(ds_name, 0) + 1

    final_score = sum(per_cat.values())  # 1000-point sum
    summary = {
        "score": final_score,
        "total_score": final_score,
        "final_score": final_score,
        "category_scores": per_cat,
        "category_counts": counts_per_cat,
        "category_budgets": CATEGORY_BUDGETS,
        "per_dataset": {
            ds: {"pass": per_dataset_pass[ds], "total": per_dataset_total[ds]}
            for ds in sorted(per_dataset_total)
        },
        "n_examples": len(rows),
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    # Mirror to result.json (singular) to match the longbench_e_*/result.json
    # convention used elsewhere in the task tree.
    with open(output_dir / "result.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78, flush=True)
    print(f"OCRBench  final score = {final_score}/1000   (n={len(rows)})",
          flush=True)
    print("=" * 78, flush=True)
    fmt = "{:<55} {:>6} / {:>5}"
    for cat, budget in CATEGORY_BUDGETS.items():
        print(fmt.format(cat, per_cat[cat], budget), flush=True)
    print("=" * 78, flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True,
                   help="OpenAI-compatible base url for the VL server.")
    p.add_argument("--model", required=True, help="HF model id (e.g. Qwen/Qwen3-VL-8B-Instruct)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dataset-repo", default="echo840/OCRBench")
    p.add_argument("--split", default="test")
    p.add_argument("--max-tokens", type=int, default=100)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--api-key", default="EMPTY")
    args = p.parse_args()

    return run_eval(args)


if __name__ == "__main__":
    sys.exit(main())
