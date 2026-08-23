#!/usr/bin/env python3
"""GPQA-Diamond driver for the Kimi-K3 OSCAR INT2 server.

Why this exists instead of run_simple_eval.py:

  * No --reasoning-parser matches K3, so the raw `content` still carries the
    model's structural tokens:
        <think text> <|close|>think<|sep|> <|open|>response<|sep|>
        <answer text> <|close|>response<|sep|> <|close|>message<|sep|>
    simple-evals runs `re.search(Answer:\\s*([A-D]))` over that whole string and
    takes the FIRST match, which is routinely a letter the model floated
    mid-thought and later rejected.  Here the answer is read from the response
    section only, and the whole-string reading is kept as a sensitivity check.

  * Every question is appended and fsynced to JSONL the moment it returns, so an
    eviction leaves a scorable prefix of a fixed question order rather than
    nothing.

  * Dispatch is CONTINUOUS, not chunk-synchronous.  The earlier version ran
    pool.map over fixed chunks of `threads`, which barriers at each chunk: as a
    chunk drains, concurrency decays threads -> ... -> 1 before the next chunk
    starts.  That is where the 20-question baseline's decode batches at bs=3
    came from (40% of its steps), and bs=3 was outside that run's captured graph
    set [1,2,4], so those steps were padded graph replays -- the known padding
    defect.  Keeping `threads` requests in flight holds the decode batch at
    `threads` for all but the tail, and the server is launched with every bs in
    1..threads captured so even the tail is never padded.

Question order is a seed-0 shuffle of all 198 diamond rows; the per-question
choice permutation is keyed by the row's original index, so it does not depend
on how many questions end up being run.  A prefix of 40 is therefore a subset of
a prefix of 80 and both are reproducible.
"""

import argparse
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request

# Verbatim simple_evals.common.QUERY_TEMPLATE_MULTICHOICE (inlined so this
# script needs no simple-evals import and no network at start-up).
TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

SYSTEM_MESSAGE = "You are a helpful assistant."

# The project's relaxed variant of ANSWER_PATTERN_MULTICHOICE: \s* rather than
# openai's newer [ \t]*, so "Answer:\n C" from a thinking model still matches.
ANSWER_RE = re.compile(r"(?i)Answer\s*:\s*\$?\\?\(?([A-D])\)?\$?")

RESP_OPEN = "<|open|>response<|sep|>"
THINK_CLOSE = "<|close|>think<|sep|>"
STRUCT_RE = re.compile(r"<\|[^|>]*\|>")


def split_sections(raw):
    """(think, response, think_closed, has_response_section)."""
    think_closed = THINK_CLOSE in raw
    if RESP_OPEN in raw:
        think, resp = raw.rsplit(RESP_OPEN, 1)
        return think, _cut(resp), think_closed, True
    if think_closed:
        think, resp = raw.rsplit(THINK_CLOSE, 1)
        return think, _cut(resp), True, False
    return raw, "", False, False


def _cut(s):
    """Drop everything from the first structural token onward, then scrub."""
    i = s.find("<|")
    if i >= 0:
        s = s[:i]
    return STRUCT_RE.sub("", s).strip()


def extract(text):
    """Last 'Answer: X' in `text`, or None."""
    m = ANSWER_RE.findall(text or "")
    return m[-1].upper() if m else None


def load_examples(csv_path):
    import pandas
    df = pandas.read_csv(csv_path)
    rows = [r.to_dict() for _, r in df.iterrows()]
    order = random.Random(0).sample(range(len(rows)), len(rows))
    out = []
    for orig in order:
        row = rows[orig]
        # Keyed by original row index -> independent of how many we run.
        perm = random.Random(10000 + orig).sample(range(4), 4)
        base = [row["Correct Answer"], row["Incorrect Answer 1"],
                row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        choices = [base[i] for i in perm]
        out.append({
            "orig_idx": orig,
            "question": row["Question"],
            "choices": choices,
            "correct": "ABCD"[choices.index(row["Correct Answer"])],
        })
    return out


def build_prompt(ex):
    return TEMPLATE.format(Question=ex["question"], A=ex["choices"][0],
                           B=ex["choices"][1], C=ex["choices"][2],
                           D=ex["choices"][3])


def ask(url, prompt, args):
    msgs = []
    if args.system:
        msgs.append({"role": "system", "content": args.system})
    msgs.append({"role": "user", "content": prompt})
    body = {"model": "m",
            "messages": msgs,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p}
    if args.top_k > 0:
        body["top_k"] = args.top_k
    data = json.dumps(body).encode()
    last = None
    for attempt in range(args.retries + 1):
        try:
            req = urllib.request.Request(url, data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=args.http_timeout) as r:
                j = json.load(r)
            ch = j["choices"][0]
            return {"raw": ch["message"].get("content") or "",
                    "finish_reason": ch.get("finish_reason"),
                    "usage": j.get("usage") or {},
                    "attempts": attempt + 1, "error": None}
        except Exception as e:                       # noqa: BLE001
            last = "%s: %s" % (type(e).__name__, e)
            if attempt < args.retries:
                time.sleep(min(30, 2 ** attempt))
    return {"raw": "", "finish_reason": None, "usage": {},
            "attempts": args.retries + 1, "error": last}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="/shared/gpqa_diamond.csv")
    p.add_argument("--system", default=SYSTEM_MESSAGE,
                   help="system message; empty string sends none (K3's template "
                        "may not carry a system role)")
    p.add_argument("--out", required=True, help="results JSONL (appended)")
    p.add_argument("--port", type=int, default=31255)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=198)
    p.add_argument("--threads", type=int, default=8,
                   help="requests kept in flight; the server must capture every "
                        "bs in 1..threads (--cuda-graph-bs) so no decode batch, "
                        "including the tail, is ever a padded graph replay")
    # 16384, not the project's usual 32768: K3's GPQA answers run ~850 tokens
    # (measured), so this is ~19x the median while bounding the single-stream
    # worst case to ~14 min at the measured 19 tok/s.  The cap-hit count is
    # reported, so if it is 0 the choice is immaterial.
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=-1,
                   help="<=0 disables top_k (Moonshot's K3 recipe is temp 1.0 / top_p 0.95)")
    p.add_argument("--deadline-s", type=int, default=25200,
                   help="stop dispatching new chunks after this many seconds")
    p.add_argument("--http-timeout", type=int, default=7200)
    p.add_argument("--retries", type=int, default=3)
    args = p.parse_args()

    url = "http://127.0.0.1:%d/v1/chat/completions" % args.port
    exs = load_examples(args.csv)
    todo = list(range(args.start, min(args.end, len(exs))))

    # Resume: skip shuffled positions already present in the output file.
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                done.add(json.loads(line)["pos"])
            except Exception:                        # noqa: BLE001
                pass
    todo = [i for i in todo if i not in done]
    print("[gpqa] %d questions to run (%d already in %s), threads=%d "
          "max_tokens=%d temp=%s top_p=%s top_k=%s"
          % (len(todo), len(done), args.out, args.threads, args.max_tokens,
             args.temperature, args.top_p, args.top_k), flush=True)

    fh = open(args.out, "a")
    lock = threading.Lock()
    t_start = time.time()
    n_ok = n_ans = 0

    from multiprocessing.pool import ThreadPool

    def run_one(pos):
        ex = exs[pos]
        prompt = build_prompt(ex)
        t0 = time.time()
        r = ask(url, prompt, args)
        dt = time.time() - t0
        think, resp, think_closed, has_resp = split_sections(r["raw"])
        got = extract(resp)
        full = STRUCT_RE.sub(" ", r["raw"])
        usage = r["usage"] or {}
        ct = usage.get("completion_tokens")
        rec = {
            "pos": pos, "orig_idx": ex["orig_idx"], "correct": ex["correct"],
            "extracted": got,
            "score": 1 if got == ex["correct"] else 0,
            "extracted_fullraw": extract(full),
            "score_fullraw": 1 if extract(full) == ex["correct"] else 0,
            "has_response_section": has_resp, "think_closed": think_closed,
            "finish_reason": r["finish_reason"],
            "completion_tokens": ct, "prompt_tokens": usage.get("prompt_tokens"),
            "hit_max_tokens": bool(ct is not None and ct >= args.max_tokens),
            "wall_s": round(dt, 1), "attempts": r["attempts"], "error": r["error"],
            "think_chars": len(think), "resp_chars": len(resp),
            "raw": r["raw"],
        }
        with lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return rec

    stop = threading.Event()

    def run_guarded(pos):
        # Past the deadline, drain in-flight work but start nothing new, so the
        # run always ends on completed questions rather than truncated ones.
        if stop.is_set():
            return None
        return run_one(pos)

    n = 0
    toks = 0
    pool = ThreadPool(args.threads)
    try:
        # imap_unordered keeps `threads` requests in flight continuously instead
        # of barriering every `threads` questions.
        for rec in pool.imap_unordered(run_guarded, todo):
            if rec is None:
                continue
            n += 1
            n_ok += rec["score"]
            n_ans += rec["extracted"] is not None
            toks += rec["completion_tokens"] or 0
            el = time.time() - t_start
            print("[gpqa] %3d/%3d done pos=%d  running_score=%.3f answered=%d/%d  "
                  "tok=%d agg_tok/s=%.1f  elapsed=%.0fs eta=%.0fs"
                  % (n, len(todo), rec["pos"], n_ok / max(1, n), n_ans, n,
                     toks, toks / max(1.0, el), el,
                     el / max(1, n) * (len(todo) - n)),
                  flush=True)
            if not stop.is_set() and el > args.deadline_s:
                print("[gpqa] DEADLINE reached, draining in-flight and stopping",
                      flush=True)
                stop.set()
    finally:
        pool.close()
        pool.join()
        fh.close()
    print("[gpqa] DISPATCH_DONE n=%d elapsed=%.0fs" % (n, time.time() - t_start),
          flush=True)


if __name__ == "__main__":
    main()
