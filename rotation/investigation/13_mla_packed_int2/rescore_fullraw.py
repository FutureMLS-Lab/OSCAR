#!/usr/bin/env python3
"""Re-score a finished GPQA run offline, strict vs fullraw, against ground truth.

The n=198 paired comparison said packed INT2 is 7.57 pp below BF16 and that the
gap is real (McNemar p~0.005) -- but also that only **4** of the 24
disagreements are cases where both arms produced an answer and disagreed. The
other 20 are parse failures, 16 of them INT2's, and 13 of those ran past 100k
characters, i.e. hit the 32768-token cap without ever emitting "Answer: X".

If that reading is right, scoring on the whole response instead of requiring a
well-formed final line should recover most of the gap. If it is wrong, the gap
survives and the loss really is in the reasoning.

Ground truth is reconstructed rather than read from a results file, because the
run dirs keep only prompts, responses and an aggregate score. simple_evals
seeds `random.Random(0)`, and with num_examples unset it draws no sample -- it
only permutes -- so the permutation sequence is reproducible exactly. Questions
are matched to the reconstructed rows by the QUESTION TEXT carried in the
prompt, not by row order, so a reordering cannot silently misalign the key.

Two scoring modes:
  strict  -- `re.search(ANSWER_PATTERN_MULTICHOICE, text)`, i.e. what the
             harness itself did. Reproduced here first as a check: if this does
             not match the reported score, the ground-truth reconstruction is
             wrong and nothing else in the output means anything.
  fullraw -- the LAST occurrence anywhere in the text. A truncated response
             often states its conclusion mid-stream and then keeps reasoning.
"""
from __future__ import annotations

import csv
import json
import os
import random
import re
import sys

ANSWER_PATTERN = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"
CSV = os.environ.get("GPQA_CSV", "/shared/gpqa_diamond.csv")


def build_key() -> dict:
    """question text -> correct letter, replicating gpqa_eval.GPQAEval."""
    with open(CSV, newline="") as f:
        rows = [r for r in csv.DictReader(f)]
    rng = random.Random(0)
    # num_examples unset in the runs being scored, so NO rng.sample() call --
    # inserting one here would consume different rng state and shift every
    # permutation that follows.
    rows = [r | {"permutation": rng.sample(range(4), 4)} for r in rows]
    key = {}
    for r in rows:
        choices = [
            r["Correct Answer"],
            r["Incorrect Answer 1"],
            r["Incorrect Answer 2"],
            r["Incorrect Answer 3"],
        ]
        choices = [choices[i] for i in r["permutation"]]
        key[r["Question"].strip()] = "ABCD"[choices.index(r["Correct Answer"])]
    return key


def load(path: str) -> list:
    out = []
    for ln in open(path):
        ln = ln.strip()
        if not ln:
            continue
        r = json.loads(ln)
        out.append((r["messages"][0]["content"], r.get("response", "") or ""))
    return out


def match_question(prompt: str, key: dict) -> str | None:
    """Find which dataset question this prompt embeds.

    Substring containment rather than equality: the prompt wraps the question
    in instructions and the shuffled choices.
    """
    for q, ans in key.items():
        if q and q in prompt:
            return ans
    return None


def score(path: str, key: dict, label: str) -> dict:
    rows = load(path)
    strict = fullraw = 0
    unmatched = 0
    recovered = []
    for prompt, text in rows:
        truth = match_question(prompt, key)
        if truth is None:
            unmatched += 1
            continue
        m = re.search(ANSWER_PATTERN, text)
        s = 1 if (m and m.group(1).upper() == truth) else 0
        all_m = re.findall(ANSWER_PATTERN, text)
        f = 1 if (all_m and all_m[-1].upper() == truth) else 0
        strict += s
        fullraw += f
        if f and not s:
            recovered.append(len(text))
    n = len(rows) - unmatched
    return {
        "label": label, "n": n, "unmatched": unmatched,
        "strict": strict, "fullraw": fullraw,
        "recovered": len(recovered),
        "recovered_mean_chars": (sum(recovered) / len(recovered)) if recovered else 0,
    }


def main() -> int:
    key = build_key()
    print(f"ground truth reconstructed for {len(key)} questions")
    arms = [
        ("/shared/mlapacked_glm_full198/io_log.jsonl", "INT2 packed (shipped defaults)", 0.747475),
        ("/shared/gpqa_glm52_bf16_198/io_log.jsonl", "BF16", 0.823232),
    ]
    res = []
    for path, label, reported in arms:
        if not os.path.exists(path):
            print(f"  MISSING {path}")
            continue
        r = score(path, key, label)
        r["reported"] = reported
        res.append(r)
        got = r["strict"] / max(1, r["n"])
        ok = abs(got - reported) < 1e-6
        print(f"\n  {label}")
        print(f"    matched {r['n']}/{r['n'] + r['unmatched']} questions"
              + (f"  ({r['unmatched']} UNMATCHED)" if r["unmatched"] else ""))
        print(f"    strict  {r['strict']:>3}/{r['n']} = {100 * got:.4f}%"
              f"   reported {100 * reported:.4f}%   "
              f"{'REPRODUCED' if ok else '<-- MISMATCH, key reconstruction is wrong'}")
        print(f"    fullraw {r['fullraw']:>3}/{r['n']} = "
              f"{100 * r['fullraw'] / max(1, r['n']):.4f}%")
        print(f"    recovered by fullraw: {r['recovered']} "
              f"(mean {r['recovered_mean_chars']:.0f} chars)")

    if len(res) == 2:
        a, b = res
        ds = (b["strict"] - a["strict"]) / a["n"] * 100
        df = (b["fullraw"] - a["fullraw"]) / a["n"] * 100
        print(f"\n  BF16 - INT2 gap:  strict {ds:+.2f} pp   fullraw {df:+.2f} pp")
        if abs(ds) > 1e-9:
            print(f"  fullraw closes {100 * (1 - df / ds):.0f}% of the strict gap")
        print("\n  A strict score that does not reproduce the reported number "
              "invalidates everything above it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
