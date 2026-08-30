#!/usr/bin/env python3
"""Is the packed arm slower because it truncates less, or because it degenerates?

The packed GLM-5.2 arm runs ~4x longer per question than either reference while
reporting *higher* decode throughput, which means it is emitting more tokens.
Two readings, opposite conclusions:

* **fewer truncations** -- the 2.92x pool lets requests the reference arms had to
  cut short or retract run to completion. That is the capacity win showing up as
  wall clock, and it belongs next to the 2.92x rather than being filed as a
  regression.
* **degenerate lengthening** -- the failure signature MiniMax-M2.7's head-tiling
  bug produced: 2x generation length, 3.3x truncation, verbatim repetition. A
  new decode kernel is exactly where that lives.

Discriminator, on what is already on disk:

1. completion-token distribution on the **same question ids**, paired by rendered
   prompt (not by position -- the option permutation differs between a 198-item
   run and any subset, and position pairing matched 1 of 198 last time);
2. terminating vs capped: longer *and* answered is the good case, longer *and*
   sitting on the 32,768 cap is the bad one;
3. repetition by repeated 5-grams, gated on ``top_count >= 3`` before any ratio
   is believed, digits kept as tokens, prompt-supplied n-grams excluded.
   Deliberately **not** a letter-ratio or alpha-density test: those have produced
   false verdicts here in both directions (request-failure strings score as
   fluent prose, markdown-heavy correct answers score like garbage).

Usage:
    discriminate13.py PACKED_DIR REF_DIR [REF_DIR ...]
"""
from __future__ import annotations

import json
import os
import re
import statistics
import sys
from collections import Counter

MAX_TOKENS = 32768
WORD = re.compile(r"[A-Za-z0-9']+")
ANSWER = re.compile(r"Answer\s*:\s*\(?([A-D])\)?", re.IGNORECASE)


def load(d: str) -> list[dict]:
    p = os.path.join(d, "io_log.jsonl")
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def get_tokenizer():
    """GLM-5.2's own tokenizer, or None -> fall back to characters.

    Token counts are the unit the cap is expressed in, so a character proxy
    cannot say "this response hit the 32,768 budget"; it can only rank lengths.
    Which one was used is printed, because that decides what the truncation
    column means.
    """
    try:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            "zai-org/GLM-5.2-FP8", trust_remote_code=True
        )
    except Exception as e:  # noqa: BLE001
        print(f"[warn] no tokenizer ({type(e).__name__}: {e}); using characters")
        return None


def repeated_5grams(text: str, prompt: str) -> dict:
    """Verbatim-repetition evidence, with the gate that makes it trustworthy.

    ``top_count`` is the occurrence count of the most-repeated 5-gram. Below 3
    the ratio is noise -- ordinary prose repeats a 5-gram twice all the time
    ("as a result of the") -- so the ratio is reported but must not be read
    unless the gate is met. Prompt-supplied n-grams are excluded: quoting the
    question back is not a loop.
    """
    toks = WORD.findall(text.lower())
    if len(toks) < 20:
        return {"top_count": 0, "rep_frac": 0.0, "gate": False, "n_tok": len(toks)}
    ptoks = WORD.findall(prompt.lower())
    pgrams = {tuple(ptoks[i : i + 5]) for i in range(max(len(ptoks) - 4, 0))}
    grams = [tuple(toks[i : i + 5]) for i in range(len(toks) - 4)]
    c = Counter(g for g in grams if g not in pgrams)
    if not c:
        return {"top_count": 0, "rep_frac": 0.0, "gate": False, "n_tok": len(toks)}
    top_gram, top_count = c.most_common(1)[0]
    repeated_positions = sum(v for v in c.values() if v >= 2)
    return {
        "top_count": top_count,
        "top_gram": " ".join(top_gram),
        "rep_frac": repeated_positions / max(len(grams), 1),
        "gate": top_count >= 3,
        "n_tok": len(toks),
    }


def summarize(name: str, rows: list[dict], tok) -> dict:
    recs = []
    for r in rows:
        resp = r.get("response") or ""
        prompt = r["messages"][0]["content"]
        n = len(tok.encode(resp)) if tok else len(resp)
        m = ANSWER.search(resp)
        rep = repeated_5grams(resp[-8000:], prompt)
        recs.append({
            "prompt": prompt,
            "n": n,
            "chars": len(resp),
            "answered": bool(m),
            "letter": m.group(1).upper() if m else None,
            # With a real tokenizer the cap is exact (the reference arms' capped
            # responses re-encoded to 32,760-32,776). With characters this is a
            # rank, not a cap test, so it is reported separately.
            "capped": (n >= MAX_TOKENS - 64) if tok else None,
            **{f"rep_{k}": v for k, v in rep.items()},
        })
    return {"name": name, "recs": recs}


def dist(vals):
    if not vals:
        return "-"
    s = sorted(vals)
    return (f"med {statistics.median(s):,.0f} | mean {statistics.mean(s):,.0f} | "
            f"p90 {s[int(.9*(len(s)-1))]:,.0f} | max {s[-1]:,.0f}")


def main() -> None:
    dirs = sys.argv[1:]
    if len(dirs) < 2:
        print(__doc__)
        return
    tok = get_tokenizer()
    print(f"length unit: {'GLM-5.2 tokens' if tok else 'characters'}\n")

    arms = [summarize(os.path.basename(d.rstrip('/')), load(d), tok) for d in dirs]
    by_prompt = [{r["prompt"]: r for r in a["recs"]} for a in arms]
    common = set(by_prompt[0])
    for m in by_prompt[1:]:
        common &= set(m)
    print(f"paired on rendered prompt: {len(common)} common ids "
          f"(arm sizes {[len(a['recs']) for a in arms]})")
    if not common:
        print("NO COMMON PROMPTS -- option permutation differs; cannot pair.")
        return

    print("\n=== completion length, terminating vs capped (matched ids only) ===")
    hdr = f"{'arm':<22} {'n':>3} {'length':<52} {'answered':>9} {'capped':>7}"
    print(hdr)
    print("-" * len(hdr))
    for a, m in zip(arms, by_prompt):
        sel = [m[p] for p in common]
        ans = sum(r["answered"] for r in sel)
        cap = sum(bool(r["capped"]) for r in sel)
        print(f"{a['name']:<22} {len(sel):>3} {dist([r['n'] for r in sel]):<52} "
              f"{ans:>4}/{len(sel):<4} {cap:>7}")

    print("\n=== verbatim repetition (repeated 5-grams, prompt n-grams excluded) ===")
    print(f"{'arm':<22} {'gate(top>=3)':>13} {'max top_count':>14} "
          f"{'rep_frac med':>13} {'worst example':<44}")
    for a, m in zip(arms, by_prompt):
        sel = [m[p] for p in common]
        gated = [r for r in sel if r["rep_gate"]]
        worst = max(sel, key=lambda r: r["rep_top_count"])
        print(f"{a['name']:<22} {len(gated):>6}/{len(sel):<6} "
              f"{max(r['rep_top_count'] for r in sel):>14} "
              f"{statistics.median([r['rep_rep_frac'] for r in sel]):>13.3f} "
              f"{str(worst.get('rep_top_gram'))[:42]:<44}")

    print("\n=== per-question pairing against the first reference ===")
    ref = by_prompt[1]
    pk = by_prompt[0]
    ids = sorted(common, key=lambda p: pk[p]["n"])
    longer = sum(pk[p]["n"] > ref[p]["n"] for p in common)
    both_ans = sum(pk[p]["answered"] and ref[p]["answered"] for p in common)
    only_pk = sum(pk[p]["answered"] and not ref[p]["answered"] for p in common)
    only_ref = sum(ref[p]["answered"] and not pk[p]["answered"] for p in common)
    agree = sum(1 for p in common
                if pk[p]["answered"] and ref[p]["answered"]
                and pk[p]["letter"] == ref[p]["letter"])
    print(f"  packed longer than ref on {longer}/{len(common)} questions")
    print(f"  both answered {both_ans}; only packed {only_pk}; only ref {only_ref}")
    print(f"  of the {both_ans} both-answered, same letter on {agree} "
          f"({100.0*agree/max(both_ans,1):.1f}%)")
    print("\n  per-question (sorted by packed length):")
    print(f"    {'packed_len':>10} {'ref_len':>9} {'pk_ans':>6} {'ref_ans':>7} "
          f"{'pk_top5g':>8} {'ref_top5g':>9}")
    for p in ids:
        print(f"    {pk[p]['n']:>10,} {ref[p]['n']:>9,} "
              f"{('Y' if pk[p]['answered'] else '.'):>6} "
              f"{('Y' if ref[p]['answered'] else '.'):>7} "
              f"{pk[p]['rep_top_count']:>8} {ref[p]['rep_top_count']:>9}")


if __name__ == "__main__":
    main()
