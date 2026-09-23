#!/usr/bin/env python3
"""Measure DECODE speed at a 64K context against a running sglang server.

Answers one question: once 64K tokens are already in the KV cache, how fast does
generation run under INT2 OSCAR against the BF16 baseline?

Prefill is deliberately NOT the headline. It is reported, because it is real
cost, but it measures a different thing: on Blackwell the INT2 prefill falls
back to SDPA (FA3 does not support the int2 path), so the prefill ratio is a
statement about that fallback. The decode ratio is a statement about the KV
cache -- reading 64K tokens of it every step, which is what a 2-bit cache exists
to make cheaper. The two can and do point in opposite directions.

  decode_s_per_token = (total - ttft) / (completion_tokens - 1)

ttft is subtracted, so prefill cost cannot leak into the decode number, and the
first token is excluded because its latency is the prefill's, not a decode
step's.

Prompts are sent as raw `input_ids`, not text:
  * the prefill length is then EXACT, not "whatever the tokenizer produced";
  * no tokenizer needs to be loaded for a model this script never sees;
  * each repetition draws fresh ids, so the radix cache cannot serve rep 2+ from
    rep 1's pages and turn a prefill benchmark into a cache-hit benchmark.

TTFT is taken from the streaming response (first chunk), which for a 64K prompt
and a short generation is dominated by prefill -- exactly the quantity asked
about. Total wall time is reported alongside it so a decode-side regression
cannot hide inside a prefill win.
"""
import argparse, json, random, statistics, sys, time
import urllib.request


def one(port, ids, max_new, timeout):
    body = json.dumps({
        "input_ids": ids,
        "sampling_params": {"max_new_tokens": max_new, "temperature": 0.0},
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    last = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            if not raw.strip() or not raw.startswith(b"data:"):
                continue
            if ttft is None:
                ttft = time.perf_counter() - t0
            chunk = raw[len(b"data:"):].strip()
            if chunk != b"[DONE]":
                last = chunk
    total = time.perf_counter() - t0
    meta = {}
    if last:
        try:
            meta = json.loads(last).get("meta_info", {}) or {}
        except Exception:
            pass
    return ttft, total, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--prefill-tokens", type=int, default=65536)
    # Long enough that per-step time dominates the measurement rather than the
    # scheduler's first-batch overhead.
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rng = random.Random(0)
    # Mid-vocabulary ids only: id 0..999 are special/reserved in most of these
    # tokenizers and a stray EOS would end the prefill early, silently
    # benchmarking a shorter context than the one named in the results.
    def ids():
        return [rng.randint(1000, 20000) for _ in range(a.prefill_tokens)]

    for i in range(a.warmup):
        try:
            one(a.port, ids(), 4, a.timeout)
        except Exception as e:
            print(f"[bench] warmup {i} failed: {e}", flush=True)

    ttfts, totals, ptoks, decs = [], [], [], []
    for i in range(a.reps):
        try:
            ttft, total, meta = one(a.port, ids(), a.max_new_tokens, a.timeout)
        except Exception as e:
            print(f"[bench] rep {i} FAILED: {e}", flush=True)
            continue
        if ttft is None:
            print(f"[bench] rep {i}: no streamed chunk", flush=True)
            continue
        ntok = meta.get("completion_tokens") or 0
        if ntok < 2:
            # Nothing to time. Some models emit EOS immediately on random-token
            # input; a decode rate from one token would be the prefill's latency
            # wearing a decode label.
            print(f"[bench] rep {i}: only {ntok} completion token(s), no decode "
                  f"measurement possible", flush=True)
            continue
        dec = (total - ttft) / (ntok - 1)
        ttfts.append(ttft)
        totals.append(total)
        decs.append(dec)
        ptoks.append(meta.get("prompt_tokens"))
        print(f"[bench] rep {i}: ttft={ttft:.3f}s total={total:.3f}s "
              f"prompt_tokens={meta.get('prompt_tokens')} completion={ntok} "
              f"decode={dec*1000:.2f} ms/tok ({1.0/dec:.1f} tok/s)", flush=True)

    if not decs:
        print("[bench] no repetition produced a usable decode measurement", flush=True)
        sys.exit(1)

    # Report the median, not the mean: a single scheduler hiccup on a shared
    # cluster otherwise moves a 3-rep mean more than the effect being measured.
    res = {
        "label": a.label,
        "prefill_tokens": a.prefill_tokens,
        "max_new_tokens": a.max_new_tokens,
        "reps_ok": len(ttfts),
        "prompt_tokens_seen": ptoks,
        "ttft_s_median": statistics.median(ttfts),
        "ttft_s_all": ttfts,
        "total_s_median": statistics.median(totals),
        "total_s_all": totals,
        "decode_s_per_token_median": statistics.median(decs),
        "decode_s_per_token_all": decs,
        "decode_tok_per_s_median": 1.0 / statistics.median(decs),
    }
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[bench] {a.label}: decode={res['decode_s_per_token_median']*1000:.2f} ms/tok "
          f"({res['decode_tok_per_s_median']:.1f} tok/s)  "
          f"ttft_median={res['ttft_s_median']:.3f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
