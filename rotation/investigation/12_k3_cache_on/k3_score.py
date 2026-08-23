#!/usr/bin/env python3
"""Score + garbling report for a k3_gpqa.py results JSONL.

A low score and a garbled server look identical if you only print accuracy, so
the answer-line rate, the max-token rate and the repetition/digit shape of the
text are reported next to the number.
"""

import collections
import json
import math
import re
import sys

STRUCT_RE = re.compile(r"<\|[^|>]*\|>")
ANSWER_LINE_RE = re.compile(r"(?im)^\s*Answer\s*:\s*\$?\\?\(?[A-D]\)?\$?\s*$")


def shape(text):
    """Garbling shape of one response: word-likeness, digit soup, repetition."""
    body = STRUCT_RE.sub(" ", text)
    words = re.findall(r"[A-Za-z][A-Za-z']*", body)
    alnum = [c for c in body if c.isalnum()]
    seq = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z']*|\d+", body)]
    grams = [tuple(seq[i:i + 5]) for i in range(max(0, len(seq) - 4))]
    top = collections.Counter(grams).most_common(1)
    return {
        "words": len(words),
        "mean_word_len": (sum(map(len, words)) / len(words)) if words else 0.0,
        "digit_frac": (sum(c.isdigit() for c in alnum) / len(alnum)) if alnum else 1.0,
        "top_5gram_frac": (top[0][1] / len(grams)) if grams else 0.0,
    }


def main(path, n_tails=3):
    recs = []
    for line in open(path):
        line = line.strip()
        if line:
            recs.append(json.loads(line))
    recs.sort(key=lambda r: r["pos"])
    n = len(recs)
    if not n:
        print("no records")
        return

    contiguous = 0
    for i, r in enumerate(recs):
        if r["pos"] != i:
            break
        contiguous = i + 1

    k = sum(r["score"] for r in recs)
    p = k / n
    se = math.sqrt(p * (1 - p) / n)
    k2 = sum(r["score_fullraw"] for r in recs)

    answered = sum(r["extracted"] is not None for r in recs)
    # K3 emits "Answer: C<|close|>response<|sep|>" with no trailing newline, so
    # the line anchor only works once the structural tokens become breaks.
    ansline = sum(bool(ANSWER_LINE_RE.search(STRUCT_RE.sub("\n", r["raw"]))) for r in recs)
    no_resp = sum(not r["has_response_section"] for r in recs)
    maxtok = sum(bool(r["hit_max_tokens"]) for r in recs)
    lenfin = sum(r["finish_reason"] == "length" for r in recs)
    errs = sum(bool(r["error"]) for r in recs)
    ct = sorted(r["completion_tokens"] or 0 for r in recs)
    disagree = sum(r["extracted"] != r["extracted_fullraw"] for r in recs)

    sh = [shape(r["raw"]) for r in recs]
    bad_shape = [
        recs[i]["pos"] for i, s in enumerate(sh)
        if not (2.2 <= s["mean_word_len"] <= 9.0)
        or s["digit_frac"] >= 0.55
        or (s["top_5gram_frac"] >= 0.06 and s["words"] > 40)
    ]

    print("=" * 78)
    print("GPQA-Diamond, first %d of a seed-0 shuffle  (contiguous prefix: %d)" % (n, contiguous))
    print("=" * 78)
    print("SCORE (answer read from K3 response section)")
    print("  correct                 %d / %d" % (k, n))
    print("  accuracy                %.4f  (%.1f%%)" % (p, 100 * p))
    print("  standard error          %.4f  (%.1f pp)" % (se, 100 * se))
    print("  95%% CI (normal)         %.1f%% .. %.1f%%" % (100 * (p - 1.96 * se), 100 * (p + 1.96 * se)))
    print("  sensitivity: same regex over the FULL raw string (think+response)")
    print("                          %d / %d = %.1f%%   (extraction disagreed on %d)"
          % (k2, n, 100 * k2 / n, disagree))
    print()
    print("GARBLING / HARNESS STATS")
    print("  total responses         %d" % n)
    print("  with an 'Answer: X' line (whole raw, own line)   %d  (%.1f%%)"
          % (ansline, 100 * ansline / n))
    print("  answer extracted from response section           %d  (%.1f%%)"
          % (answered, 100 * answered / n))
    print("  NO <|open|>response<|sep|> section at all        %d  (%.1f%%)"
          % (no_resp, 100 * no_resp / n))
    print("  ran to max_tokens (usage)                        %d  (%.1f%%)"
          % (maxtok, 100 * maxtok / n))
    print("  finish_reason == 'length'                        %d" % lenfin)
    print("  request errors after retries                     %d" % errs)
    print("  completion tokens  min/median/p90/max            %d / %d / %d / %d"
          % (ct[0], ct[n // 2], ct[int(0.9 * (n - 1))], ct[-1]))
    print("  mean digit_frac %.3f   mean top_5gram_frac %.3f   mean word_len %.2f"
          % (sum(s["digit_frac"] for s in sh) / n,
             sum(s["top_5gram_frac"] for s in sh) / n,
             sum(s["mean_word_len"] for s in sh) / n))
    print("  responses failing a garbling shape check         %d  %s"
          % (len(bad_shape), bad_shape[:20]))

    # Loop rate, reported separately from the shape check.  The failure that
    # matters here is the verbatim repetition loop that eats the whole token
    # budget: a response that both ran to the cap AND ends in a highly repeated
    # 5-gram.  The prior 20-question baseline had 5/20 of these.
    looped, looped_capped = [], []
    for i, r in enumerate(recs):
        capped = bool(r["hit_max_tokens"]) or r["finish_reason"] == "length"
        s = sh[i]
        is_loop = s["top_5gram_frac"] >= 0.06 and s["words"] > 40
        if is_loop:
            looped.append(r["pos"])
        if is_loop and capped:
            looped_capped.append(r["pos"])
    print("  LOOP RATE (repeated 5-gram, any length)          %d / %d = %.1f%%  %s"
          % (len(looped), n, 100.0 * len(looped) / n, looped[:20]))
    print("  loops that also ran to the cap                   %d / %d = %.1f%%  %s"
          % (len(looped_capped), n, 100.0 * len(looped_capped) / n, looped_capped[:20]))
    # Did the INT2 tier actually carry this workload?  Only the 64-token HP
    # prefix and the 256-token HP recent window stay bf16; everything else is
    # quantized.  A request whose whole context fits in those windows tests
    # nothing, so count those separately.
    HP = 64 + 256
    tot_ctx = quant_ctx = 0
    never = 0
    for r in recs:
        ctx = (r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0)
        tot_ctx += ctx
        quant_ctx += max(0, ctx - HP)
        if ctx <= HP:
            never += 1
    print()
    print("INT2 EXPOSURE (bf16 HP windows are prefix 64 + recent 256 = %d tokens)" % HP)
    print("  end-of-generation context, mean                  %.0f tokens" % (tot_ctx / n))
    print("  fraction of that context in the INT2 tier        %.1f%%"
          % (100 * quant_ctx / max(1, tot_ctx)))
    print("  requests that never left the bf16 windows        %d" % never)
    print()
    wrong = [r for r in recs if not r["score"]]
    print("WRONG-ANSWER BREAKDOWN (%d)" % len(wrong))
    print("  wrong letter extracted   %d" % sum(r["extracted"] is not None for r in wrong))
    print("  no answer extractable    %d" % sum(r["extracted"] is None for r in wrong))
    print("  distribution of extracted letters: %s"
          % dict(collections.Counter(str(r["extracted"]) for r in recs)))
    print("  distribution of correct letters:   %s"
          % dict(collections.Counter(r["correct"] for r in recs)))
    print()
    print("SAMPLE TAILS (last 320 chars of raw)")
    picks = []
    for r in recs:
        if r["extracted"] is None:
            picks.append(("NO-ANSWER", r))
    for r in recs:
        if r["hit_max_tokens"] and len(picks) < n_tails + 2:
            picks.append(("MAXTOK", r))
    for r in recs[:n_tails]:
        picks.append(("NORMAL", r))
    seen = set()
    for tag, r in picks[: n_tails + 6]:
        if r["pos"] in seen:
            continue
        seen.add(r["pos"])
        print("  --- pos=%d %s  correct=%s extracted=%s ctok=%s finish=%s"
              % (r["pos"], tag, r["correct"], r["extracted"],
                 r["completion_tokens"], r["finish_reason"]))
        print("      %s" % repr(r["raw"][-320:]))
    print("=" * 78)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3)
