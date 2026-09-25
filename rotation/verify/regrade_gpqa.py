#!/usr/bin/env python3
"""Re-score a GPQA io_log, and say how much of the score is extraction.

simple_evals greps the response for a literal ``Answer: $LETTER``
(ANSWER_PATTERN_MULTICHOICE) and scores 0 for anything else. Kimi-K3 frequently
concludes with "So the answer is **(D)**" instead, which scores zero however
right it is -- and how often it complies swings with the seed, so three runs of
one identical config scored 18.75, 39.58 and 93.75. Before any of that gets
read as a KV-cache result, the extraction loss has to be separated from the
model being wrong.

Gold labels come from the GPQA csv, matched to each prompt by its rendered
option texts, NOT by replaying simple_evals' Random(0) permutation -- an
earlier attempt did replay it, drifted out of step, and reported 0% correct.

The STRICT column must reproduce metrics.json. If it does not, this script is
wrong and its LOOSE column means nothing, so it refuses to print one without
the other.

  python3 regrade_gpqa.py --csv gpqa_diamond.csv --run NAME=path/to/io_log.jsonl ...
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys

STRICT = re.compile(r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?")

# Ways K3 actually signs off, measured from its own outputs. Each must anchor
# on a word that commits to a choice -- "option B" counts, a bare "B)" in the
# middle of prose does not, or every restatement of the choice list matches.
LOOSE = [
    re.compile(r"(?i)\b(?:the\s+)?answer\s+is\s*:?\s*\**\(?([A-D])\)?\**"),
    re.compile(r"(?i)\bcorrect\s+(?:choice|option|answer)\s+is\s*:?\s*\**\(?([A-D])\)?\**"),
    re.compile(r"(?i)\bchoose\s*\**\(?([A-D])\)?\**"),
    re.compile(r"(?i)\boption\s*\**\(?([A-D])\)?\**\s*$"),
    re.compile(r"\\boxed\{\s*\(?([A-D])\)?\s*\}"),
]

OPT = re.compile(r"^([A-D])\)\s*(.*)$")

# Kimi-K3 emits XTML channels. Generation starts INSIDE the think channel (the
# prompt template ends there), so there is no opening think marker; the visible
# answer is what sits between these two.
K3_OPEN_RESPONSE = "<|open|>response<|sep|>"
K3_CLOSE_RESPONSE = "<|close|>response<|sep|>"


def _k3_response_channel(text: str):
    """What ``--reasoning-parser kimi_k3`` leaves as `content`, or None.

    Without that parser the fork hands the grader thinking AND answer
    concatenated, and simple_evals takes the FIRST ``Answer:`` match -- so an
    intermediate answer inside the reasoning wins over the final one. 38/43 of
    our K3 responses carry more than one ``Answer:`` line; upstream's carry
    zero, because its server stripped the thinking before the grader saw it.
    That is a scoring difference with nothing to do with the KV cache, and it
    was most of an apparent 91.67-vs-95.83 gap.

    Returns None when there are no channel markers, which is the case for text
    a server already parsed -- those are graded as-is.
    """
    i = text.find(K3_OPEN_RESPONSE)
    if i < 0:
        return None
    j = text.find(K3_CLOSE_RESPONSE, i)
    return text[i + len(K3_OPEN_RESPONSE) : j if j >= 0 else len(text)]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def _prompt_options(prompt: str):
    """The A)..D) option texts as rendered into the prompt."""
    out = {}
    cur = None
    for line in prompt.splitlines():
        m = OPT.match(line.strip())
        if m:
            cur = m.group(1)
            out[cur] = m.group(2)
        elif cur and line.strip():
            out[cur] += " " + line.strip()
        elif not line.strip():
            cur = None
    return out


def _gold_index(csv_path: pathlib.Path):
    """normalised correct-answer text -> set of incorrect texts, per row."""
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(
                (
                    _norm(row["Question"]),
                    _norm(row["Correct Answer"]),
                )
            )
    return rows


def _gold_letter(prompt: str, rows) -> str | None:
    opts = _prompt_options(prompt)
    if len(opts) != 4:
        return None
    norm_opts = {k: _norm(v) for k, v in opts.items()}
    pq = _norm(prompt)

    # Resolve the csv row by its QUESTION text first, then read that row's
    # correct answer. Matching on option text against a flat set of every gold
    # string collapses at full-diamond scale: "4" is the correct answer to
    # several different questions, so three of four options come back marked
    # correct. That path was fine on a 48-question subset and left 23/193
    # prompts ambiguous on the full 198, which made the checker refuse the run.
    cands = [(q, c) for q, c in rows if q and q in pq]
    if cands:
        # Longest question wins: one stem can be a prefix of another.
        _q, correct = max(cands, key=lambda qc: len(qc[0]))
        for letter, text in norm_opts.items():
            if text == correct:
                return letter
        # One diamond question renders its options truncated: the csv answer
        # carries a " smiles: ..." tail that the prompt drops, so the option is
        # a 136-char prefix of a 236-char answer while the three distractors
        # diverge within 31 chars. Accept a unique long prefix; the length
        # floor keeps "4" from matching "42".
        pre = [
            letter
            for letter, text in norm_opts.items()
            if len(text) >= 20 and correct.startswith(text)
        ]
        return pre[0] if len(pre) == 1 else None

    # Prompt formatting we do not recognise: fall back to option text, and only
    # accept it when exactly one option matches exactly one row's answer.
    hits = {
        letter
        for (_q, correct) in rows
        for letter, text in norm_opts.items()
        if text == correct
    }
    return hits.pop() if len(hits) == 1 else None


def _extract(text: str, patterns, first: bool = True) -> str | None:
    """The choice a grader would read out of the response.

    simple_evals uses re.search, i.e. the FIRST match, not the last -- taking
    the last one scored bf 22.92% against a metrics.json of 18.75% and looked
    like a small rounding difference rather than the wrong rule.
    """
    best, best_pos = None, None
    for pat in patterns:
        for m in pat.finditer(text):
            if best_pos is None or (m.start() < best_pos if first else m.start() > best_pos):
                best, best_pos = m.group(1).upper(), m.start()
    return best


def grade(path: pathlib.Path, rows, channel: bool = False):
    n = strict_ok = loose_ok = no_strict = no_gold = 0
    recovered = lost = 0
    unwrapped = malformed = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        resp = rec.get("response")
        text = resp if isinstance(resp, str) else json.dumps(resp)
        if channel:
            only = _k3_response_channel(text)
            if only is not None:
                text = only
                unwrapped += 1
        prompt = rec["messages"][0]["content"]
        gold = _gold_letter(prompt, rows)
        n += 1
        if gold is None:
            no_gold += 1
            continue
        s = _extract(text, [STRICT])
        # Same first-match rule for loose, so the only thing that changes
        # between the two columns is which phrasings count as an answer.
        l = _extract(text, [STRICT] + LOOSE)
        strict_ok += s == gold
        loose_ok += l == gold
        if s is None:
            no_strict += 1
            if l == gold:
                recovered += 1
            else:
                lost += 1
    return dict(
        n=n,
        no_gold=no_gold,
        strict=strict_ok,
        loose=loose_ok,
        unparsed_strict=no_strict,
        recovered_correct=recovered,
        unparsed_and_wrong=lost,
        unwrapped=unwrapped,
        malformed=malformed,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=pathlib.Path)
    ap.add_argument("--run", action="append", required=True,
                    metavar="NAME=PATH", help="repeatable")
    ap.add_argument("--channel", action="store_true",
                    help="grade only Kimi-K3's response channel, i.e. what "
                         "--reasoning-parser kimi_k3 would hand the grader; "
                         "text with no channel markers is graded as-is")
    ap.add_argument("--expect", action="append", default=[],
                    metavar="NAME=SCORE",
                    help="metrics.json score the STRICT column must reproduce")
    args = ap.parse_args()

    rows = _gold_index(args.csv)
    expect = {}
    for e in args.expect:
        k, _, v = e.partition("=")
        expect[k] = float(v)

    print(f"{'run':6s} {'n':>3s} {'strict':>9s} {'loose':>9s} "
          f"{'unparsed':>9s} {'recovered':>10s} {'unparsed+wrong':>15s}")
    bad = False
    for spec in args.run:
        name, _, p = spec.partition("=")
        r = grade(pathlib.Path(p), rows, channel=args.channel)
        if r.get("malformed"):
            print(f"{name}: {r['malformed']} io_log line(s) were not valid JSON "
                  f"and are excluded", file=sys.stderr)
        if r["no_gold"]:
            print(f"{name}: {r['no_gold']}/{r['n']} prompts had no gold match -- "
                  f"cannot trust this run", file=sys.stderr)
            bad = True
            continue
        sf, lf = r["strict"] / r["n"], r["loose"] / r["n"]
        flag = ""
        if not args.channel and name in expect and abs(sf - expect[name]) > 1e-6:
            flag = f"  <-- MISMATCH vs metrics.json {expect[name]:.4f}"
            bad = True
        print(f"{name:6s} {r['n']:>3d} {r['strict']:>4d} {sf:6.2%} "
              f"{r['loose']:>4d} {lf:6.2%} {r['unparsed_strict']:>9d} "
              f"{r['recovered_correct']:>10d} {r['unparsed_and_wrong']:>15d}{flag}"
              + (f"   [channel-unwrapped {r['unwrapped']}/{r['n']}]"
                 if args.channel else ""))
    if bad:
        print("\nSTRICT did not reproduce metrics.json: the LOOSE column above "
              "is not trustworthy.", file=sys.stderr)
        return 1
    if args.channel:
        print("\nchannel mode: metrics.json came from the RAW text, so it is not "
              "reproduced here by design -- this is the like-for-like number "
              "against a server that ran --reasoning-parser kimi_k3")
    else:
        print("\nstrict reproduces metrics.json; loose = same responses, "
              "grader's regex replaced by the phrasings K3 actually uses")
    return 0


if __name__ == "__main__":
    sys.exit(main())
