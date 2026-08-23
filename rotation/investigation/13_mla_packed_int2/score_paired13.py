#!/usr/bin/env python3
"""Paired GPQA scoring on the matched ids only, with exact McNemar.

Three rules this obeys, each of which has cost this project a wrong number
before:

* **Pair by rendered prompt, never by position.** ``io_log`` is written in
  *completion* order, so its prefix is the questions that finished fastest.
  Position-pairing a partial run against a full one matched 1 of 198 last time,
  and length survivorship read 12 pp high on Kimi-K3.
* **Score only the ids all arms share.** A running arm's own aggregate is a
  survivorship average, not a score.
* **McNemar on the discordant pairs**, with the paired SE ``sqrt(b+c)/n`` --
  the per-arm binomial SE (about +-3 pp here) overstates the uncertainty of a
  paired difference.

The correct letter is recovered from the rendered prompt itself: simple_evals
permutes the options per question, so the letter is only meaningful against the
option text that request actually saw. Matching the CSV's ``Correct Answer``
text against the A)-D) block in the prompt is what makes a partial run
comparable to a completed one at all.

Usage:
    score_paired13.py gpqa_diamond.csv PACKED_DIR REF_DIR [REF_DIR ...]
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from math import comb

OPT = re.compile(r"^([A-D])\)\s*(.+?)\s*$", re.MULTILINE)
ANSWER = re.compile(r"Answer\s*:\s*\(?([A-D])\)?", re.IGNORECASE)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def key_from_csv(path: str) -> dict:
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[norm(row["Correct Answer"])] = row["Question"]
    return out


def correct_letter(prompt: str, correct_texts: set) -> str | None:
    """Which of A-D this request's option block put the correct answer at."""
    opts = OPT.findall(prompt)
    hit = [L for L, txt in opts if norm(txt) in correct_texts]
    return hit[0] if len(hit) == 1 else None


def load(d: str) -> dict:
    out = {}
    with open(os.path.join(d, "io_log.jsonl")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            p = r["messages"][0]["content"]
            m = ANSWER.search(r.get("response") or "")
            out[p] = {"letter": m.group(1).upper() if m else None,
                      "answered": bool(m), "chars": len(r.get("response") or ""),
                      # Completion order. The growth curve needs a STABLE
                      # ordering: sorting the matched ids by prompt text makes
                      # "the first 60" a different set every time the run grows,
                      # so the same n reported -3.33 one hour and -6.67 the next
                      # purely from membership churn. Ordering by the base arm's
                      # completion index makes the prefix monotone -- ids only
                      # get appended -- which is what "has it settled as the run
                      # progressed" actually means.
                      "ord": len(out)}
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main() -> None:
    if len(sys.argv) < 4:
        print(__doc__)
        return
    args = [a for a in sys.argv[1:] if a != "--growth"]
    growth = "--growth" in sys.argv
    csv_path, *dirs = args
    correct_texts = set(key_from_csv(csv_path))
    arms = [(os.path.basename(d.rstrip("/")), load(d)) for d in dirs]

    common = set(arms[0][1])
    for _, m in arms[1:]:
        common &= set(m)
    scored = {p: correct_letter(p, correct_texts) for p in common}
    base_m = arms[0][1]
    usable = sorted((p for p, L in scored.items() if L),
                    key=lambda p: base_m[p]["ord"])
    print(f"matched ids: {len(common)}   with a recoverable key: {len(usable)}"
          f"   (arm sizes {[len(m) for _, m in arms]})")
    if not usable:
        print("no scorable matched ids")
        return

    print("\n=== score on the matched ids ===")
    print(f"{'arm':<24} {'correct':>9} {'score':>8} {'answered':>9} {'mean chars':>11}")
    res = {}
    for name, m in arms:
        ok = sum(1 for p in usable if m[p]["letter"] == scored[p])
        ans = sum(1 for p in usable if m[p]["answered"])
        ch = sum(m[p]["chars"] for p in usable) / len(usable)
        res[name] = [m[p]["letter"] == scored[p] for p in usable]
        print(f"{name:<24} {ok:>4}/{len(usable):<4} "
              f"{100.0*ok/len(usable):>7.2f}% {ans:>4}/{len(usable):<4} {ch:>11,.0f}")

    base = arms[0][0]
    print(f"\n=== paired against {base} (exact McNemar) ===")
    for name, _ in arms[1:]:
        a, b_ = res[base], res[name]
        b = sum(1 for x, y in zip(a, b_) if x and not y)   # base right, other wrong
        c = sum(1 for x, y in zip(a, b_) if y and not x)
        n = len(usable)
        delta = 100.0 * (sum(a) - sum(b_)) / n
        se = 100.0 * ((b + c) ** 0.5) / n
        print(f"  {base} - {name}: delta {delta:+.2f} pp  paired SE +-{se:.2f} pp  "
              f"discordant {b}/{c}  exact p = {mcnemar_exact(b, c):.4f}")

    if growth:
        # Is the number still moving? Every wrong call in this investigation has
        # been a partial-n signal read as a result -- 21/21 agreement that became
        # -12.20 at 41, a 1/3 transparency gap that became 1/12, a +9.09 that was
        # a code-pin confound. A delta resting on five discordant pairs needs to
        # be shown settling, not just reported.
        print("\n=== delta vs prefix size, in the base arm's COMPLETION order "
              "(has it settled?) ===")
        print(f"{'n':>5}  " + "  ".join(f"{nm[:14]:>14}" for nm, _ in arms[1:])
              + "   discordant")
        step = max(10, len(usable) // 8)
        for k in range(step, len(usable) + 1, step):
            row, disc = [], []
            for name, _ in arms[1:]:
                a, b_ = res[base][:k], res[name][:k]
                b = sum(1 for x, y in zip(a, b_) if x and not y)
                c = sum(1 for x, y in zip(a, b_) if y and not x)
                row.append(f"{100.0*(sum(a)-sum(b_))/k:>+13.2f}")
                disc.append(f"{b}/{c}")
            print(f"{k:>5}  " + "  ".join(row) + "   " + " ".join(disc))

    print("\nCAVEAT, and it is load-bearing: these are the ids that finished "
          "first, so the set is biased toward SHORT generations. INT2's cost in "
          "this project grows with generation length, so agreement here does not "
          "license a claim about the full 198.")


if __name__ == "__main__":
    main()
