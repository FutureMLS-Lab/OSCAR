#!/usr/bin/env python3
"""Send OCRBench multimodal prompts (image + text) to a sglang dump server
(max_tokens=1) to trigger the DUMP_KVCACHE hook with VL activations.

Phase 2 E2 calibration-data producer for the OSCAR rotation phase.
The KV-cache statistics from VL prompts (with vision tokens prepended via
the model's image processor) should be matched to the OCRBench evaluation
distribution, where Doc-oriented VQA and Key Information Extraction were the
biggest deficit at INT2 g=128.

Subsets prioritised: Doc-oriented VQA + Key Information Extraction (the
heavy-context, fine-grained-reading categories).

Server-side env vars expected:
    DUMP_KVCACHE=true
    DUMP_KVCACHE_TOKENS=<budget>   (server stops dumping at budget)

Usage:
  python dump_ocrbench_prompts.py \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --base-url http://127.0.0.1:31060/v1 \
    --num-threads 4 \
    --num-prompts 150
"""

import argparse
import base64
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image


# Bias the dump distribution toward the categories with the biggest INT2
# accuracy deficit (Doc-oriented VQA: 40/200 on VL-8B at g=128).
# Weights are roughly proportional to OCRBench's own category budgets plus a
# 2x boost on doc/KIE to oversample fine-grained reading patterns.
CATEGORY_WEIGHTS = {
    "Doc-oriented VQA": 4,                                # 200 budget x2 boost
    "Key Information Extraction": 3,                      # 200 budget
    "Scene Text-centric VQA": 2,                          # 200 budget
    "Handwritten Mathematical Expression Recognition": 1, # 100 budget
    "Regular Text Recognition": 1,                        # 50 budget
    "Irregular Text Recognition": 1,
    "Artistic Text Recognition": 1,
    "Handwriting Recognition": 1,
    "Digit String Recognition": 1,
    "Non-Semantic Text Recognition": 1,
}


def _to_png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--num-prompts", type=int, default=150,
                   help="Number of (image, question) prompts to send. "
                   "Server-side DUMP_KVCACHE_TOKENS will auto-stop earlier.")
    p.add_argument("--num-threads", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset-repo", default="echo840/OCRBench")
    return p


def _load_ocrbench():
    print("[dump] fetching echo840/OCRBench parquet", flush=True)
    pq_path = hf_hub_download(
        repo_id="echo840/OCRBench",
        filename="data/test-00000-of-00001.parquet",
        repo_type="dataset",
    )
    return pd.read_parquet(pq_path)


def _pick_prompts(df: pd.DataFrame, num_prompts: int, seed: int):
    rng = pd.Series(range(len(df))).sample(frac=1.0, random_state=seed).values
    # Normalize weights into per-category sample counts.
    total_w = sum(CATEGORY_WEIGHTS.values())
    picks = []
    assigned = 0
    cats = list(CATEGORY_WEIGHTS.items())
    for i, (cat, w) in enumerate(cats):
        if i == len(cats) - 1:
            want = max(1, num_prompts - assigned)
        else:
            want = max(1, int(round(num_prompts * w / total_w)))
        cat_df = df[df["question_type"] == cat]
        if len(cat_df) == 0:
            continue
        take = cat_df.sample(n=min(want, len(cat_df)), random_state=seed)
        for _, row in take.iterrows():
            picks.append(row)
        assigned += len(take)
        print(f"[dump]   {cat}: picking {len(take)}/{len(cat_df)}", flush=True)
    # Shuffle so worker threads see a mixed-distribution stream.
    rng2 = pd.Series(range(len(picks))).sample(frac=1.0, random_state=seed).values
    return [picks[i] for i in rng2]


def _send_one(client, model, image_b64, question, max_tokens):
    payload = [
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        {"type": "text", "text": question},
    ]
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": payload}],
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_tokens,
            n=1,
        )
        return "ok"
    except Exception as e:  # noqa: BLE001
        return f"err: {e!r}"


def main():
    args = _build_argparser().parse_args()
    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key=args.api_key,
                    timeout=900.0)

    df = _load_ocrbench()
    print(f"[dump] OCRBench rows: {len(df)}", flush=True)

    picks = _pick_prompts(df, args.num_prompts, args.seed)
    print(f"[dump] sending {len(picks)} OCRBench prompts at "
          f"max_tokens={args.max_tokens}", flush=True)

    # Pre-encode images on the main thread so worker threads don't fight over
    # PIL/PNG encoding on the same image.
    encoded = []
    for row in picks:
        img = Image.open(io.BytesIO(row["image"]["bytes"]))
        b64 = _to_png_b64(img)
        encoded.append({
            "image_b64": b64,
            "question": row["question"],
            "question_type": row["question_type"],
            "dataset": row["dataset"],
        })

    t0 = time.time()
    n_ok = n_err = 0
    with ThreadPoolExecutor(max_workers=args.num_threads) as ex:
        futs = [
            ex.submit(_send_one, client, args.model, p["image_b64"],
                      p["question"], args.max_tokens)
            for p in encoded
        ]
        for i, f in enumerate(as_completed(futs)):
            r = f.result()
            if r == "ok":
                n_ok += 1
            else:
                n_err += 1
                if n_err <= 5:
                    print(f"  prompt {i}: {r}", flush=True)
            if (i + 1) % 25 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-6)
                print(f"[dump] {i+1}/{len(encoded)} done ({rate:.2f}/s)",
                      flush=True)

    print(f"[dump] done in {time.time()-t0:.1f}s  ok={n_ok}  err={n_err}",
          flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
