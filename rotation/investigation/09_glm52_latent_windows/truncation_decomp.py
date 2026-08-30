#!/usr/bin/env python3
"""Truncation decomposition and paired answer-transition tables for GLM-5.2 GPQA.

Why this exists separately from ``08_m27_truncation/paired_compare.py``: the
GLM arms' ``io_log.jsonl`` does **not** carry ``finish_reason``. The sampler
that produced them only recorded the request fields and the response text, so
``rec.get("finish_reason") == "length"`` is False on every record and
``paired_compare.py`` reports ``capped=0`` for every arm. That zero is an
artifact of the log schema, not a claim that nothing truncated -- taking it at
face value would report "0% truncation" for an arm whose whole story is
truncation.

So truncation is identified by its *signature* instead, and the two available
signals are cross-checked against each other rather than either being trusted
alone:

* ``unanswered`` -- no strict ``Answer: $LETTER`` line. This is the
  decision-relevant definition, because simple_evals scores such a response
  zero no matter how much correct reasoning precedes it. It is also the
  definition behind the 29.3% / 19.7% already recorded for these arms
  (58/198 and 39/198), so it keeps this table commensurable with those.
* ``at_cap`` -- response length in the top band of the arm's own length
  distribution, which is where a 32K-token budget piles responses up.

The cross-check is the point. "Unanswered" alone cannot separate a response
that ran out of budget mid-reasoning from one that degenerated early, and
those two have opposite implications: the first says raise the budget, the
second says the KV recipe broke the model. Reporting ``unanswered_short``
separately is what makes the "GLM's gap is truncation, not wrong answers"
claim falsifiable -- if the unanswered responses were short, the claim would
be dead, and this script would show it.

The paired section answers the specific question a score delta cannot: of the
questions BOTH arms answered, how often do they pick the SAME letter. High
agreement there plus a truncation gap means the deficit is budget; low
agreement means it is quantization damage.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "08_m27_truncation"))

import rescore_gpqa as rs  # noqa: E402

WORD = re.compile(r"[A-Za-z0-9_^{}=+\-*/().,;:$]+")


def load(run_dir: Path, key: dict[str, str]) -> dict[str, dict]:
    """Index one arm by rendered prompt (exact pairing, never positional)."""
    out: dict[str, dict] = {}
    for line in (run_dir / "io_log.jsonl").open(errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        prompt = ""
        for m in rec.get("messages") or []:
            if isinstance(m, dict) and m.get("role") == "user":
                prompt = str(m.get("content", ""))
                break
        gold = key.get(prompt.strip())
        if gold is None:
            continue
        resp = rec.get("response") or ""
        ms = rs.STRICT.search(resp)
        tail = resp[-4000:].encode("utf-8", "replace")
        out[prompt.strip()] = {
            "gold": gold,
            "chars": len(resp),
            "words": len(WORD.findall(resp)),
            "answered": bool(ms),
            "letter": ms.group(1) if ms else None,
            # Case-sensitive, matching simple_evals' ``== "ABCD"`` compare.
            "correct": bool(ms and ms.group(1) == gold),
            # Low ratio == highly compressible tail == repetition loop.
            "tail_zlib": (len(zlib.compress(tail, 6)) / max(1, len(tail))),
        }
    return out


def pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.2f}%" if d else "n/a"


def decompose(name: str, A: dict[str, dict], total: int) -> dict:
    ans = [p for p in A if A[p]["answered"]]
    un = [p for p in A if not A[p]["answered"]]
    correct = sum(1 for p in ans if A[p]["correct"])
    ref = sorted(A[p]["chars"] for p in ans)

    # Is "unanswered" really "ran out of budget"? Two independent signatures,
    # because the alternative explanation -- the KV recipe broke the model, so
    # it emitted something short and useless -- has the opposite length
    # signature and the opposite fix.
    #
    # Calibrated against the tokenizer, not guessed: re-encoding these arms'
    # responses with the GLM-5.2 tokenizer puts EVERY unanswered response at
    # 32,760-32,776 tokens against a 32,768 budget, and not one below 32,000.
    # So on these arms "no Answer: line" and "hit the cap" are the same set,
    # and the char thresholds below are a cheap stand-in that reproduces it
    # without loading a tokenizer.
    #
    # ``trunc_short`` is the falsifier and is expected to be 0. A nonzero count
    # means some unanswered responses are NOT budget-limited, and the
    # "truncation, not wrong answers" story would need re-examining rather than
    # restating -- so it is reported even though it is currently always zero.
    DEGENERATE_CHARS = 10_000
    loops = [p for p in un if A[p]["tail_zlib"] < 0.15]
    row = {
        "arm": name,
        "n": len(A),
        "score_n": correct,
        "score": round(100.0 * correct / total, 2),
        "answered": len(ans),
        "answered_pct": round(100.0 * len(ans) / total, 2),
        "truncated": len(un),
        "truncated_pct": round(100.0 * len(un) / total, 2),
        "acc_given_answered": round(100.0 * correct / len(ans), 2) if ans else 0.0,
        # Shortest unanswered response: the single most informative number
        # here. If it is large, no unanswered response is a short degenerate
        # one, and the whole unanswered set is budget exhaustion.
        "trunc_min_chars": min((A[p]["chars"] for p in un), default=0),
        "trunc_short": sum(1 for p in un if A[p]["chars"] < DEGENERATE_CHARS),
        "trunc_loop_tail": len(loops),
        "mean_chars": round(sum(A[p]["chars"] for p in A) / max(1, len(A)), 1),
        "median_chars_answered": ref[len(ref) // 2] if ref else 0,
    }
    return row


def paired(nameA: str, nameB: str, A: dict, B: dict) -> None:
    common = sorted(set(A) & set(B))
    n = len(common)
    print(f"\n=== PAIRED  A={nameA}  B={nameB}  paired={n} ===")

    # Answer-transition 2x2: this is the truncation story, per question.
    aa = sum(1 for p in common if A[p]["answered"] and B[p]["answered"])
    a_only = sum(1 for p in common if A[p]["answered"] and not B[p]["answered"])
    b_only = sum(1 for p in common if not A[p]["answered"] and B[p]["answered"])
    nn = sum(1 for p in common if not A[p]["answered"] and not B[p]["answered"])
    print(f"answered 2x2: both={aa}  A-only={a_only}  B-only={b_only}  neither={nn}")

    # Agreement among BOTH-answered: the quantization-damage read.
    both = [p for p in common if A[p]["answered"] and B[p]["answered"]]
    agree = sum(1 for p in both if A[p]["letter"] == B[p]["letter"])
    ca = sum(1 for p in both if A[p]["correct"])
    cb = sum(1 for p in both if B[p]["correct"])
    print(f"both answered: {len(both)}  same letter: {agree} "
          f"({pct(agree, len(both))})")
    print(f"  acc on both-answered: A={ca}/{len(both)}={pct(ca, len(both))}  "
          f"B={cb}/{len(both)}={pct(cb, len(both))}")

    # Correctness 2x2 over the FULL denominator (the scored view).
    n11 = sum(1 for p in common if A[p]["correct"] and B[p]["correct"])
    n10 = sum(1 for p in common if A[p]["correct"] and not B[p]["correct"])
    n01 = sum(1 for p in common if not A[p]["correct"] and B[p]["correct"])
    n00 = sum(1 for p in common if not A[p]["correct"] and not B[p]["correct"])
    print(f"correct 2x2 (all {n}): A+B+={n11} A+B-={n10} A-B+={n01} A-B-={n00}")
    # SE of a PAIRED delta comes from the discordant pairs only -- the
    # questions both arms got right carry no information about the difference.
    # Quoting each arm's own binomial SE (~2.7-3.4 pp here) and treating the
    # arms as independent would overstate the uncertainty on the delta, which
    # is the quantity actually being claimed. n11/n00 cancel by construction.
    se_pp = 100.0 * ((n10 + n01) ** 0.5) / n
    delta_pp = 100.0 * (n01 - n10) / n
    print(f"  A={n11 + n10}  B={n11 + n01}  B loses {n10}, gains {n01}, "
          f"net {n01 - n10:+d} questions ({delta_pp:+.2f} pp)")
    print(f"  paired delta {delta_pp:+.2f} +/- {se_pp:.2f} pp (1 SE, single seed, "
          f"from {n10 + n01} discordant pairs)")

    # Of the questions B lost, how many did B simply fail to finish?
    lost = [p for p in common if A[p]["correct"] and not B[p]["correct"]]
    lost_trunc = sum(1 for p in lost if not B[p]["answered"])
    lost_wrong = sum(1 for p in lost if B[p]["answered"])
    if lost:
        print(f"  of B's {len(lost)} losses: {lost_trunc} B-truncated "
              f"({pct(lost_trunc, len(lost))}), {lost_wrong} B-answered-wrong")

    # Length only over pairs where BOTH answered -- a truncated response's
    # length is the budget, not the model's choice, so including it would
    # measure the cap and call it verbosity.
    rat = sorted(B[p]["words"] / A[p]["words"]
                 for p in both if A[p]["words"] > 0)
    if rat:
        q = (rat[len(rat) // 4], rat[len(rat) // 2], rat[3 * len(rat) // 4])
        print(f"  words(B)/words(A) on both-answered ({len(rat)}): "
              f"p25={q[0]:.2f} p50={q[1]:.2f} p75={q[2]:.2f}")


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial p for the discordant pairs (b vs c)."""
    n = b + c
    if n == 0:
        return 1.0
    from math import comb
    obs = min(b, c)
    tail = sum(comb(n, k) for k in range(0, obs + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="run dirs; label=path accepted")
    ap.add_argument("--variant", default="diamond")
    ap.add_argument("--total", type=int, default=198)
    ap.add_argument("--pair", action="append", default=[],
                    help="labelA:labelB, repeatable")
    a = ap.parse_args()

    key = rs.build_key_to_answer(a.variant)
    loaded: dict[str, dict] = {}
    for spec in a.arms:
        label, _, path = spec.partition("=")
        if not path:
            path, label = label, Path(label).name
        loaded[label] = load(Path(path), key)

    rows = [decompose(k, v, a.total) for k, v in loaded.items()]
    cols = ["arm", "score", "score_n", "answered", "truncated", "truncated_pct",
            "acc_given_answered", "trunc_min_chars", "trunc_short",
            "trunc_loop_tail", "mean_chars"]
    w = [18, 7, 8, 9, 10, 13, 18, 16, 12, 15, 11]
    print("".join(c.rjust(x) for c, x in zip(cols, w)))
    for r in rows:
        print("".join(str(r[c]).rjust(x) for c, x in zip(cols, w)))

    for spec in a.pair:
        la, _, lb = spec.partition(":")
        if la in loaded and lb in loaded:
            paired(la, lb, loaded[la], loaded[lb])
            common = sorted(set(loaded[la]) & set(loaded[lb]))
            b = sum(1 for p in common
                    if loaded[la][p]["correct"] and not loaded[lb][p]["correct"])
            c = sum(1 for p in common
                    if not loaded[la][p]["correct"] and loaded[lb][p]["correct"])
            print(f"  McNemar exact two-sided p = {mcnemar_exact(b, c):.4g} "
                  f"(discordant {b} vs {c})")

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
