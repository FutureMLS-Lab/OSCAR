#!/usr/bin/env python3
"""Locate where a degenerate response stops being coherent.

The length/repetition pass says *how many* responses are loops. This says
*when* each one turns, which is what distinguishes two very different stories:

* a loop that starts near the beginning is a bad prompt/quantization
  interaction that a longer BF16 window would not touch;
* a loop that starts only after thousands of generated tokens is cumulative
  degradation, and the token position where it turns is the number a window
  or budget change has to beat.

Method: slide a window over the response's token stream and report the first
window whose distinct-5-gram fraction collapses below ``--threshold`` and stays
below it for ``--persist`` consecutive windows. Requiring persistence matters --
a single low-distinctness window is common in legitimate text (a table, a
repeated formula, an enumeration) and treating it as onset put the estimate
thousands of tokens too early on the first pass.

Same guards as length_repetition.py: digits kept, no letter-ratio heuristic.
Prompt n-grams are not excluded here because the quantity of interest is where
self-repetition begins, and a response that quotes the question back at token
20000 is still coherent at token 20000.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

TOKEN = re.compile(r"[A-Za-z0-9_\\^{}=+\-*/().,;:$]+")
N = 5


def onset(toks: list[str], win: int, step: int, thresh: float,
          persist: int) -> int | None:
    run = 0
    for start in range(0, max(1, len(toks) - win), step):
        chunk = toks[start : start + win]
        grams = [tuple(chunk[i : i + N]) for i in range(len(chunk) - N + 1)]
        if len(grams) < 20:
            continue
        frac = len(Counter(grams)) / len(grams)
        if frac < thresh:
            run += 1
            if run >= persist:
                return start - (persist - 1) * step
        else:
            run = 0
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--window", type=int, default=400)
    ap.add_argument("--step", type=int, default=200)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--persist", type=int, default=3)
    ap.add_argument("--show", type=int, default=0,
                    help="print this many loop excerpts")
    a = ap.parse_args()

    for rd in a.run_dirs:
        d = Path(rd)
        io = d / "io_log.jsonl"
        if not io.is_file():
            continue
        onsets, lens, shown = [], [], 0
        capped_loops = 0
        for line in io.open(errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            resp = rec.get("response") or ""
            toks = TOKEN.findall(resp)
            o = onset(toks, a.window, a.step, a.threshold, a.persist)
            if o is None:
                continue
            capped_loops += 1
            onsets.append(o)
            lens.append(len(toks))
            if shown < a.show:
                shown += 1
                seg = " ".join(toks[o : o + 60])
                print(f"--- {d.name} onset_word={o} total_words={len(toks)} "
                      f"finish={rec.get('finish_reason')}\n    {seg[:300]}")
        if onsets:
            onsets.sort()
            lens.sort()
            frac = [o / l for o, l in zip(sorted(onsets), sorted(lens)) if l]
            print(f"{d.name:<16} loops={capped_loops:3d} "
                  f"onset_words p10={onsets[len(onsets)//10]:6d} "
                  f"p50={onsets[len(onsets)//2]:6d} "
                  f"p90={onsets[9*len(onsets)//10]:6d} "
                  f"| total_words p50={lens[len(lens)//2]:6d} "
                  f"| onset/total p50={sorted(frac)[len(frac)//2]:.2f}")
        else:
            print(f"{d.name:<16} loops=0")


if __name__ == "__main__":
    main()
