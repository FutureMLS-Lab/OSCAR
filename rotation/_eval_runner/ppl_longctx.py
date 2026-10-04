#!/usr/bin/env python3
"""Long-context perplexity through a running server, scored past the BF16 window.

GPQA-198 moves by several points between seeds, so it cannot rank transform
variants that differ by one point. Teacher-forced NLL over long windows can:
WikiText-2 test is concatenated, tokenized once and cut into non-overlapping
windows of ``--seq-len`` tokens; every window goes through the real serving
path (chunked prefill, HP window, flush, INT2 reads) and the server returns
prompt log-probs. Only tokens at positions ``>= --score-from`` count, so the
score reflects attention over quantized history rather than the BF16 recent
window (the Qwen3-8B recipe keeps 2048 recent tokens in BF16; the default
scores positions 4096 and later). ppl = exp(total_nll / scored_tokens).

  python ppl_longctx.py --port 31057 --model Qwen/Qwen3-8B --seq-len 32768 \\
      --score-from 4096 --out RUN_DIR/ppl.json
"""
import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests


def load_tokens(model: str, text_file: str | None):
    from transformers import AutoTokenizer

    if text_file:
        text = open(text_file, encoding="utf-8").read()
        source = text_file
    else:
        from datasets import load_dataset

        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")  # datasets>=4 needs the namespaced id
        text = "\n\n".join(ds["text"])
        source = "wikitext-2-raw-v1/test"
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    ids = tok(text, return_tensors=None, add_special_tokens=False)["input_ids"]
    print(f"[ppl] {source}: {len(text)} chars -> {len(ids)} tokens ({model})", flush=True)
    return ids


def score_window(base_url: str, input_ids, score_from: int, retries: int = 3):
    payload = {
        "input_ids": input_ids,
        "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    for attempt in range(retries):
        try:
            r = requests.post(f"{base_url}/generate", json=payload, timeout=3600)
            r.raise_for_status()
            lps = r.json()["meta_info"]["input_token_logprobs"]  # [[logprob, token_id, ...], ...]
            if len(lps) < len(input_ids) - 1:
                raise RuntimeError(f"short logprobs: {len(lps)} for a window of {len(input_ids)}")
            # entry i is the log-prob of token i given tokens < i; the first is None
            vals = [(i, x[0]) for i, x in enumerate(lps) if x[0] is not None]
            scored = [v for i, v in vals if i >= score_from]
            if not scored:
                raise RuntimeError("no scored tokens: --score-from is past the window")
            nll = -sum(scored)
            if not math.isfinite(nll):
                raise RuntimeError("non-finite log-prob in the response")
            return nll, len(scored)
        except Exception as e:  # noqa: BLE001 - retried, then raised
            if attempt == retries - 1:
                raise
            print(f"[ppl] retry {attempt + 1} after error: {e}", flush=True)
            time.sleep(5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--seq-len", type=int, default=32768)
    ap.add_argument("--score-from", type=int, default=4096)
    ap.add_argument("--max-windows", type=int, default=0, help="0 = every full window")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--text-file", default=None, help="score this file instead of WikiText-2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base = f"http://{a.host}:{a.port}"
    ids = load_tokens(a.model, a.text_file)
    n_win = len(ids) // a.seq_len
    if a.max_windows:
        n_win = min(n_win, a.max_windows)
    if n_win == 0:
        raise SystemExit(f"text has {len(ids)} tokens, fewer than one window of {a.seq_len}")
    windows = [ids[i * a.seq_len : (i + 1) * a.seq_len] for i in range(n_win)]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        outs = list(ex.map(lambda w: score_window(base, w, a.score_from), windows))
    nll = sum(o[0] for o in outs)
    ntok = sum(o[1] for o in outs)
    res = {
        "ppl": math.exp(nll / ntok),
        "nll_per_token": nll / ntok,
        "scored_tokens": ntok,
        "windows": n_win,
        "seq_len": a.seq_len,
        "score_from": a.score_from,
        "sec": time.time() - t0,
        "model": a.model,
    }
    print(f"[ppl] seq_len={a.seq_len} score_from={a.score_from}: ppl={res['ppl']:.4f} "
          f"over {n_win} windows / {ntok} scored tokens ({res['sec']:.0f}s)", flush=True)
    print("PPL_RESULT " + json.dumps(res), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
