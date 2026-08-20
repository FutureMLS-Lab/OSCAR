#!/usr/bin/env python3
"""Length and repetition statistics for the MiniMax-M2.7 INT2-vs-BF16 arms.

The question this answers is the one a score cannot: when an INT2 arm fails to
produce an extractable answer, is the response *coherent but too long for the
budget* (a) or *degenerate* (b)?  Those have different fixes, so they must be
separated before anything is refit or re-tuned.

Method notes, all of them the result of earlier false verdicts on this project:

* No letter-ratio / alpha-density check.  Those have produced four wrong
  verdicts here: error strings read as prose, markdown scored like garbage,
  letter-only n-grams invented loops, and terse correct answers have no words.
  The only repetition evidence used is n-gram multiplicity.
* Digits are kept as tokens.  Stripping them hid the arithmetic-loop failure
  mode, which is the most common one on a reasoning benchmark.
* n-grams that the *prompt* already contains are excluded, so quoting the
  question back (which every arm does) is not scored as a loop.
* ``top_count >= 3`` gates every ratio.  A 5-gram seen twice is prose; the
  ratio on a two-occurrence n-gram is noise.
* ``distinct_frac`` = distinct 5-grams / total 5-grams over the response tail.
  ~0.9 = coherent-but-long.  ~0.02 = a loop.

Capped-ness is read from the server's own ``finish_reason`` -- ``"length"`` is
truncation, ``"stop"`` is a model that chose to stop.  The io_log carries it on
every record, so no chars/token estimate is needed; the old
``runaway_unanswered_long`` proxy (chars > 2*max_tokens AND no answer) conflated
"long" with "truncated" and so could not tell (a) from (b) at all.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ANSWER = re.compile(r"(?i)Answer\s*:\s*\**\s*([A-D])\b")
# Keep digits.  Split on whitespace after dropping punctuation that only ever
# appears as formatting, so "x_1=2" stays one token rather than three.
TOKEN = re.compile(r"[A-Za-z0-9_\\^{}=+\-*/().,;:$]+")

N = 5
TAIL_CHARS = 20000


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text)


def ngrams(toks: list[str], n: int = N) -> list[tuple[str, ...]]:
    return [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]


def repetition(response: str, prompt: str) -> dict:
    """n-gram multiplicity on the response tail, prompt n-grams excluded."""
    tail = response[-TAIL_CHARS:]
    toks = tokens(tail)
    grams = ngrams(toks)
    if len(grams) < 50:
        return {"n_grams": len(grams), "distinct_frac": None, "top_count": 0,
                "top_share": None, "loop": False}
    prompt_grams = set(ngrams(tokens(prompt))) if prompt else set()
    grams = [g for g in grams if g not in prompt_grams]
    if len(grams) < 50:
        return {"n_grams": len(grams), "distinct_frac": None, "top_count": 0,
                "top_share": None, "loop": False}
    counts = Counter(grams)
    top_gram, top_count = counts.most_common(1)[0]
    distinct_frac = len(counts) / len(grams)
    # Share of the tail occupied by n-grams that repeat at least 3 times.
    repeated = sum(c for c in counts.values() if c >= 3)
    top_share = repeated / len(grams)
    # A loop needs BOTH a low distinct fraction and a genuinely multiple
    # top n-gram.  Either alone misfires: a short tail can look non-distinct,
    # and a high top_count can come from a legitimately repeated formula.
    loop = bool(distinct_frac < 0.35 and top_count >= 3)
    return {
        "n_grams": len(grams),
        "distinct_frac": round(distinct_frac, 4),
        "top_count": top_count,
        "top_share": round(top_share, 4),
        "top_gram": " ".join(top_gram)[:120],
        "loop": loop,
    }


def pct(xs: list[float], p: float):
    if not xs:
        return None
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(p / 100 * (len(s) - 1)))))
    return s[i]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--cap-slack", type=float, default=0.98,
                    help="completion_tokens >= slack*max_tokens counts as capped")
    a = ap.parse_args()

    rows = []
    for rd in a.run_dirs:
        d = Path(rd)
        io = d / "io_log.jsonl"
        if not io.is_file():
            continue
        n = answered = capped = loops = capped_loops = capped_coherent = 0
        have_tok = 0
        finish: Counter = Counter()
        answered_capped = 0
        toks_list: list[int] = []
        chars_list: list[int] = []
        distinct_capped: list[float] = []
        max_tokens = None
        examples: list[dict] = []
        for line in io.open(errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            resp = rec.get("response") or ""
            prompt = rec.get("prompt") or rec.get("question") or rec.get("messages") or ""
            if isinstance(prompt, list):
                prompt = " ".join(
                    str(m.get("content", "")) if isinstance(m, dict) else str(m)
                    for m in prompt
                )
            mt = rec.get("max_tokens")
            if mt:
                max_tokens = mt
            n += 1
            chars_list.append(len(resp))
            ct = rec.get("completion_tokens")
            if ct is None:
                usage = rec.get("usage") or {}
                ct = usage.get("completion_tokens")
            if ct is not None:
                have_tok += 1
                toks_list.append(int(ct))
            has_ans = bool(ANSWER.search(resp))
            if has_ans:
                answered += 1
            # Capped: the server's own finish_reason is authoritative.
            # Fall back to token counts, then to a chars estimate, only when a
            # log predates the finish_reason field.
            fr = rec.get("finish_reason")
            if fr is not None:
                is_capped = fr == "length"
                finish[str(fr)] += 1
            elif ct is not None and max_tokens:
                is_capped = int(ct) >= a.cap_slack * max_tokens
            elif max_tokens:
                is_capped = len(resp) > 2.0 * max_tokens
            else:
                is_capped = False
            rep = repetition(resp, prompt)
            if rep["loop"]:
                loops += 1
            if is_capped:
                capped += 1
                if has_ans:
                    answered_capped += 1
                if rep["distinct_frac"] is not None:
                    distinct_capped.append(rep["distinct_frac"])
                if rep["loop"]:
                    capped_loops += 1
                else:
                    capped_coherent += 1
                if len(examples) < 3:
                    examples.append({
                        "chars": len(resp), "completion_tokens": ct,
                        **{k: rep[k] for k in
                           ("distinct_frac", "top_count", "top_share", "top_gram")},
                    })
        row = {
            "arm": d.name,
            "responses": n,
            "answered": answered,
            "max_tokens": max_tokens,
            "have_token_counts": have_tok,
            "finish_reasons": dict(finish),
            "capped": capped,
            "capped_answered": answered_capped,
            "capped_loops": capped_loops,
            "capped_coherent": capped_coherent,
            "loops_any": loops,
            "tok_p50": pct(toks_list, 50),
            "tok_p90": pct(toks_list, 90),
            "tok_max": max(toks_list) if toks_list else None,
            "chars_p50": pct(chars_list, 50),
            "chars_p90": pct(chars_list, 90),
            "capped_distinct_frac_median": (
                round(pct(distinct_capped, 50), 4) if distinct_capped else None
            ),
            "capped_examples": examples,
        }
        rows.append(row)

    if a.json:
        print(json.dumps(rows, indent=2))
        return
    hdr = ("arm", "n", "ans", "capped", "cap_loop", "cap_coh", "loops",
           "tok_p50", "tok_p90", "ch_p50", "cap_distinct")
    print(("{:<14}" + "{:>9}" * (len(hdr) - 1)).format(*hdr))
    for r in rows:
        print(("{:<14}" + "{:>9}" * (len(hdr) - 1)).format(
            r["arm"][:14], r["responses"], r["answered"], r["capped"],
            r["capped_loops"], r["capped_coherent"], r["loops_any"],
            str(r["tok_p50"]), str(r["tok_p90"]), str(r["chars_p50"]),
            str(r["capped_distinct_frac_median"])))


if __name__ == "__main__":
    main()
