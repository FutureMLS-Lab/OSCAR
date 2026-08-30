#!/usr/bin/env python3
"""Re-score a GPQA run from ``io_log.jsonl`` without re-running the model.

``run_simple_eval.py`` only writes ``metrics.json`` when the whole 198-example
sweep finishes, so a long-budget arm that is still on its last few responses
reports no score at all. This reproduces simple_evals' GPQAEval grading exactly
and scores whatever is present, which is what makes a 95K-budget arm readable
before its two slowest loops time out.

Exactness matters in two places:

* The example list and option permutations are rebuilt with the same
  ``random.Random(0)`` draw order as ``GPQAEval.__init__`` (no ``num_examples``,
  ``n_repeats=1``), so ``permutation`` -- and therefore the correct letter -- is
  identical. Records are matched to examples by the *rendered prompt*, so a
  mismatch is impossible rather than merely unlikely.

* Grading uses simple_evals' own ``ANSWER_PATTERN_MULTICHOICE``, which is
  stricter than it looks: ``Answer[ \\t]*:[ \\t]*\\$?([A-D])\\$?`` does not
  tolerate markdown, so "Answer: **B**" scores ZERO in the official harness.
  A looser "answered" count is reported alongside it precisely so that gap is
  visible instead of being quietly folded into the score -- on a model that
  bolds its final answer, the two differ, and only the strict one is the score.

The denominator is always the full 198. A partial arm's score is therefore a
lower bound on its final score, and ``scored_of`` says how many were present.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas

CSV = "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_{variant}.csv"

QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

STRICT = re.compile(r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?")
LOOSE = re.compile(r"(?i)Answer\s*:\s*\**\s*([A-D])\b")


def build_key_to_answer(variant: str) -> dict[str, str]:
    df = pandas.read_csv(CSV.format(variant=variant))
    examples = [row.to_dict() for _, row in df.iterrows()]
    rng = random.Random(0)
    # No num_examples sampling and n_repeats == 1, matching the scored arms.
    examples = [e | {"permutation": rng.sample(range(4), 4)} for e in examples]
    out: dict[str, str] = {}
    for row in examples:
        choices = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        choices = [choices[i] for i in row["permutation"]]
        correct = "ABCD"[choices.index(row["Correct Answer"])]
        prompt = QUERY_TEMPLATE_MULTICHOICE.format(
            Question=row["Question"], A=choices[0], B=choices[1],
            C=choices[2], D=choices[3],
        )
        out[prompt.strip()] = correct
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--variant", default="diamond")
    ap.add_argument("--total", type=int, default=198)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    key = build_key_to_answer(a.variant)
    rows = []
    for rd in a.run_dirs:
        d = Path(rd)
        io = d / "io_log.jsonl"
        if not io.is_file():
            continue
        n = correct = strict_ans = loose_ans = unmatched = 0
        capped_correct = capped = 0
        for line in io.open(errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = rec.get("messages") or []
            prompt = ""
            for m in msgs:
                if isinstance(m, dict) and m.get("role") == "user":
                    prompt = str(m.get("content", ""))
                    break
            gold = key.get(prompt.strip())
            n += 1
            if gold is None:
                unmatched += 1
                continue
            resp = rec.get("response") or ""
            ms = STRICT.search(resp)
            ml = LOOSE.search(resp)
            if ms:
                strict_ans += 1
            if ml:
                loose_ans += 1
            # Case-SENSITIVE, deliberately. simple_evals extracts with a
            # ``(?i)`` regex but then compares ``extracted_answer ==
            # correct_answer`` against "ABCD", so a response ending "answer: b"
            # is extracted as "b" and scored ZERO by the official harness.
            # Upper-casing here would silently score such a run higher than the
            # number everyone else quotes.
            ok = bool(ms and ms.group(1) == gold)
            if ok:
                correct += 1
            if rec.get("finish_reason") == "length":
                capped += 1
                if ok:
                    capped_correct += 1
        rows.append({
            "arm": d.name,
            "responses": n,
            "unmatched_prompts": unmatched,
            "correct": correct,
            "score_over_total": round(100.0 * correct / a.total, 2),
            "scored_of": f"{n}/{a.total}",
            "answered_strict": strict_ans,
            "answered_loose": loose_ans,
            "markdown_loss": loose_ans - strict_ans,
            "capped": capped,
            "capped_correct": capped_correct,
        })

    if a.json:
        print(json.dumps(rows, indent=2))
        return
    hdr = ("arm", "n/198", "correct", "score%", "ans_str", "ans_loose",
           "md_loss", "capped", "unmatch")
    print(("{:<16}" + "{:>10}" * (len(hdr) - 1)).format(*hdr))
    for r in rows:
        print(("{:<16}" + "{:>10}" * (len(hdr) - 1)).format(
            r["arm"][:16], r["scored_of"], r["correct"], r["score_over_total"],
            r["answered_strict"], r["answered_loose"], r["markdown_loss"],
            r["capped"], r["unmatched_prompts"]))


if __name__ == "__main__":
    main()
