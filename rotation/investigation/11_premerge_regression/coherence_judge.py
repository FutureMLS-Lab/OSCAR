#!/usr/bin/env python3
"""Judge whether a generated sample is coherent English, without the checks
that have produced false verdicts on this project.

Explicitly NOT used:

  * letter ratio / alpha density. A request-failure string
    ("Error: connection reset by peer") is almost pure letters and scores as
    fluent prose; a correct markdown-heavy answer full of ``|``, ``$``, ``\\``
    and digits scores as garbage. Both verdicts have been made here.
  * ``top_gram / n_grams`` as a repetition ratio. When every n-gram occurs
    exactly once the max count is 1, so the ratio is 1/n_grams -- a number that
    shrinks with length and says nothing about repetition. A short terse answer
    then looks maximally "repetitive". Gate on ``top_count >= 2`` first; if the
    most common n-gram occurs once, there is no repetition to measure.

What is used, in order:

  1. hard non-text rejects: empty, or an obvious transport/HTTP error payload.
  2. digit soup: fraction of non-whitespace characters that are digits or
     ``.``/``,``. Gemma's failure mode was ``To find0.0.000.0.0.00...`` at
     digit_frac 0.97. Prose with numbers sits well under 0.35.
  3. repeated n-gram loop: word 8-grams, gated on ``top_count >= 2``, then
     measured as ``top_count * n / total_words`` -- the share of the text the
     single most repeated block occupies. A real loop drives this toward 1.0.
  4. word shape: the text must contain a reasonable number of dictionary-shaped
     word tokens (length 2-20, alphabetic, at least one vowel). This is what
     separates prose from token soup *without* counting raw letters, so
     ``0.0.000`` and ``\\u4e2d\\u6587\\u4e71\\u7801`` both fail while a two-word
     answer passes on the ``terse`` exemption.

A terse-but-real answer ("Answer: C") is PASS via the terse exemption: it is
short, has no digit soup and no loop. Terse is not garbage.

Usage:
    python3 coherence_judge.py io_log.jsonl [--field response] [--json]
    python3 coherence_judge.py --text "some sample"
"""
import argparse
import collections
import json
import re
import sys

WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")
VOWEL = re.compile(r"[aeiouyAEIOUY]")
# Transport / harness failure payloads that read as fluent English but are not
# model output. Matched only at the very start, and only when the whole sample
# is short, so a model legitimately discussing errors is not rejected.
ERROR_HEAD = re.compile(
    r"^\s*(?:error|traceback|exception|httpx?\.|openai\.|requests\.|"
    r"connection\s+(?:reset|refused|error)|timeout|"
    r"internal\s+server\s+error|bad\s+gateway|service\s+unavailable|"
    r"\{\s*[\"']error[\"']|<html)",
    re.I,
)

NGRAM_N = 8
CHAR_NGRAM_N = 16
DIGIT_SOUP_MAX = 0.35
WORD_DENSITY_MIN = 6.0   # word tokens per 100 non-space chars; prose runs ~14-18
LOOP_SHARE_MAX = 0.50
MIN_WORDS_FOR_PROSE = 12
TERSE_MAX_CHARS = 400


def digit_frac(text: str) -> float:
    body = [c for c in text if not c.isspace()]
    if not body:
        return 1.0
    n = sum(1 for c in body if c.isdigit() or c in ".,")
    return n / len(body)


def _loop(units, n):
    """Return (top_count, share). share is meaningful only if top_count >= 2."""
    if len(units) < n * 2:
        return 0, 0.0
    grams = collections.Counter(
        tuple(units[i:i + n]) for i in range(len(units) - n + 1)
    )
    _gram, top = grams.most_common(1)[0]
    if top < 2:                       # THE GATE: nothing repeats, no ratio
        return top, 0.0
    return top, min(1.0, top * n / len(units))


def loop_stats(words):
    return _loop(words, NGRAM_N)


def char_loop_stats(text):
    """Word n-grams cannot see repetition in a script that has no spaces
    (CJK filler, or a run of punctuation). Fall back to character 16-grams on
    the whitespace-stripped body, with the same top_count >= 2 gate."""
    body = "".join(text.split())
    return _loop(body, CHAR_NGRAM_N)


def real_words(text):
    return [w for w in WORD.findall(text) if 2 <= len(w) <= 20 and VOWEL.search(w)]


def judge(text: str) -> dict:
    t = text or ""
    out = {"chars": len(t)}
    if not t.strip():
        return {**out, "verdict": "FAIL", "reason": "empty response"}
    if ERROR_HEAD.match(t) and len(t) < 2000:
        return {**out, "verdict": "FAIL",
                "reason": "transport/harness error payload, not model output"}

    df = digit_frac(t)
    words = t.split()
    rw = real_words(t)
    top, share = loop_stats(words)
    ctop, cshare = char_loop_stats(t)
    out.update(digit_frac=round(df, 4), words=len(words), real_words=len(rw),
               top_ngram_count=top, loop_share=round(share, 4),
               top_char_ngram_count=ctop, char_loop_share=round(cshare, 4))

    # Digit soup needs BOTH tests. A high digit fraction alone rejects a
    # legitimate answer built around a numeric table -- the same class of false
    # verdict as scoring markdown as garbage. Real soup ("To find0.0.000.0.0")
    # is digit-dominated *and* has almost no word tokens; prose about numbers is
    # digit-dominated but still word-dense.
    body = "".join(t.split())
    word_density = 100.0 * len(rw) / max(1, len(body))
    out["word_density_per_100c"] = round(word_density, 2)
    if len(body) >= 40 and df > DIGIT_SOUP_MAX and word_density < WORD_DENSITY_MIN:
        return {**out, "verdict": "FAIL",
                "reason": (f"digit soup: {df:.2f} digits/.,  with only "
                           f"{word_density:.1f} word tokens per 100 chars")}
    if top >= 2 and share > LOOP_SHARE_MAX:
        return {**out, "verdict": "FAIL",
                "reason": (f"repeated {NGRAM_N}-gram occurs {top}x and covers "
                           f"{share:.0%} of the text")}
    if ctop >= 2 and cshare > LOOP_SHARE_MAX:
        return {**out, "verdict": "FAIL",
                "reason": (f"repeated {CHAR_NGRAM_N}-char block occurs {ctop}x and "
                           f"covers {cshare:.0%} of the text")}
    if len(rw) < MIN_WORDS_FOR_PROSE:
        # A bare option letter ("C") is a real, correct, terse answer. It has no
        # vowel so it is not a "word", and there is nothing in 1-3 characters
        # that could be incoherent.
        if len(t.strip()) <= 3:
            return {**out, "verdict": "PASS",
                    "reason": f"trivially short answer token ({t.strip()!r})"}
        # Terse exemption: short AND it actually contains word-shaped text.
        # Without the second half, 320 chars of space-free filler passes as
        # "terse" -- observed on a CJK-repetition sample.
        if len(t) <= TERSE_MAX_CHARS and len(rw) >= 1:
            return {**out, "verdict": "PASS",
                    "reason": f"terse but clean ({len(rw)} word tokens, no soup, no loop)"}
        return {**out, "verdict": "FAIL",
                "reason": f"{len(t)} chars but only {len(rw)} word-shaped tokens"}
    return {**out, "verdict": "PASS",
            "reason": f"{len(rw)} word tokens, digit_frac {df:.2f}, no repeated block"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--field", default="response")
    ap.add_argument("--text")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--show", type=int, default=1, help="print N sample excerpts")
    a = ap.parse_args()

    if a.text is not None:
        print(json.dumps(judge(a.text), indent=2))
        return
    if not a.path:
        ap.error("need a jsonl path or --text")

    verdicts, rows = collections.Counter(), []
    for line in open(a.path):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        r = rec.get(a.field) or ""
        j = judge(r)
        verdicts[j["verdict"]] += 1
        rows.append((j, r))

    n = len(rows)
    npass = verdicts["PASS"]
    summary = {
        "n": n,
        "pass": npass,
        "fail": verdicts["FAIL"],
        "pass_rate": round(npass / n, 4) if n else None,
        "fail_reasons": collections.Counter(
            j["reason"].split(":")[0] for j, _ in rows if j["verdict"] == "FAIL"
        ),
        "overall": "PASS" if n and npass / n >= 0.95 else "FAIL",
    }
    if a.json:
        print(json.dumps({**summary,
                          "fail_reasons": dict(summary["fail_reasons"])}, indent=2))
    else:
        print(f"n={n} pass={npass} fail={verdicts['FAIL']} "
              f"pass_rate={summary['pass_rate']} overall={summary['overall']}")
        for reason, c in summary["fail_reasons"].most_common():
            print(f"  FAIL {c:>4}  {reason}")
    # exemplars: the longest passing sample, then any failures
    good = [(j, r) for j, r in rows if j["verdict"] == "PASS"]
    good.sort(key=lambda x: -x[0]["chars"])
    for j, r in good[:a.show]:
        print(f"\n--- PASS sample ({j['reason']}) ---\n{r[:700]}")
    bad = [(j, r) for j, r in rows if j["verdict"] == "FAIL"]
    for j, r in bad[:a.show]:
        print(f"\n--- FAIL sample ({j['reason']}) ---\n{r[:700]}")
    sys.exit(0 if summary["overall"] == "PASS" else 1)


if __name__ == "__main__":
    main()
