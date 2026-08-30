#!/usr/bin/env python3
"""Where exactly does a Kimi-K3 GPQA response fail to terminate?

"78% of the error budget is non-termination" is only actionable if we know
*which* non-termination: a response that never closed its think block is a
different failure from one that closed it and then looped inside the answer,
and only the first is a plausible long-horizon reasoning failure.  This
classifies every record and cross-tabs the classes against score, so the same
table can be produced for the BF16 arm and subtracted.
"""

import collections
import json
import re
import sys

STRUCT_RE = re.compile(r"<\|[^|>]*\|>")
THINK_CLOSE = "<|close|>think<|sep|>"
RESP_OPEN = "<|open|>response<|sep|>"


def load(path, limit=0):
    recs = []
    for line in open(path):
        line = line.strip()
        if line:
            try:
                recs.append(json.loads(line))
            except Exception:  # noqa: BLE001
                pass
    recs.sort(key=lambda r: r["pos"])
    out = []
    for i, r in enumerate(recs):
        if r["pos"] != i:
            break
        out.append(r)
    return out[:limit] if limit else out


def ngram_stats(text, n=5):
    seq = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z']*|\d+", text)]
    grams = [tuple(seq[i:i + n]) for i in range(max(0, len(seq) - n + 1))]
    if not grams:
        return 0.0, 0, None
    c = collections.Counter(grams)
    top, cnt = c.most_common(1)[0]
    return cnt / len(grams), len(seq), " ".join(top)


def tail_loop_onset(text):
    """Fraction of the way through the text at which the dominant tail 5-gram
    first appears -- i.e. how early the model locked into the loop."""
    frac, _, top = ngram_stats(text[-4000:])
    if top is None or frac < 0.06:
        return None, None
    i = text.find(top.split()[0])
    return (i / max(1, len(text))), top


def classify(r):
    raw = r["raw"] or ""
    capped = bool(r.get("hit_max_tokens")) or r.get("finish_reason") == "length"
    closed = THINK_CLOSE in raw
    has_resp = bool(r.get("has_response_section"))
    got = r["extracted"] is not None
    if got:
        return "answered"
    if not capped:
        # Stopped on its own but no parsable answer.
        return "stopped-no-answer(closed)" if closed else "stopped-no-answer(open-think)"
    if not closed:
        return "capped-inside-think"
    if has_resp:
        return "capped-inside-response"
    return "capped-after-think-no-response"


def report(path, label, limit=0):
    recs = load(path, limit)
    n = len(recs)
    print("=" * 78)
    print("%s  n=%d (contiguous prefix)" % (label, n))
    print("=" * 78)

    cls = [classify(r) for r in recs]
    cnt = collections.Counter(cls)
    print("TERMINATION CLASSES")
    for k in ("answered", "capped-inside-think", "capped-after-think-no-response",
              "capped-inside-response", "stopped-no-answer(closed)",
              "stopped-no-answer(open-think)"):
        v = cnt.get(k, 0)
        sc = sum(recs[i]["score"] for i in range(n) if cls[i] == k)
        print("  %-34s %3d  (%4.1f%%)   correct within class: %d" % (k, v, 100.0 * v / n, sc))
    print()

    # Of the capped ones, how repetitive is the tail, and how early did it lock?
    capped = [(i, r) for i, r in enumerate(recs)
              if bool(r.get("hit_max_tokens")) or r.get("finish_reason") == "length"]
    loopy = 0
    onsets = []
    print("CAPPED RESPONSES (%d)" % len(capped))
    for i, r in capped:
        frac, ntok, top = ngram_stats((r["raw"] or "")[-4000:])
        if frac >= 0.06:
            loopy += 1
            o, _ = tail_loop_onset(r["raw"] or "")
            if o is not None:
                onsets.append(o)
    print("  tail is a verbatim repetition loop            %d / %d = %.0f%%"
          % (loopy, len(capped), 100.0 * loopy / max(1, len(capped))))
    if onsets:
        onsets.sort()
        print("  loop onset as a fraction of the response      p25 %.2f  med %.2f  p75 %.2f"
              % (onsets[len(onsets) // 4], onsets[len(onsets) // 2],
                 onsets[3 * len(onsets) // 4]))
    print("  closed think before being capped               %d / %d"
          % (sum(1 for _, r in capped if THINK_CLOSE in (r["raw"] or "")), len(capped)))
    print()

    # Length structure: the claim is a LONG-HORIZON failure, so the score should
    # fall off with generated length.
    print("SCORE vs COMPLETION LENGTH (is this a long-horizon effect?)")
    buckets = [(0, 2000), (2000, 5000), (5000, 10000), (10000, 20000), (20000, 30719), (30719, 10 ** 9)]
    for lo, hi in buckets:
        sub = [r for r in recs if lo <= (r.get("completion_tokens") or 0) < hi]
        if not sub:
            continue
        k = sum(r["score"] for r in sub)
        a = sum(r["extracted"] is not None for r in sub)
        print("  %6d-%-6d tok  n=%3d  score %3d/%-3d = %5.1f%%   answered %5.1f%%"
              % (lo, hi if hi < 10 ** 8 else 30720, len(sub), k, len(sub),
                 100.0 * k / len(sub), 100.0 * a / len(sub)))
    print()

    # Conditional accuracy is reported ONLY as a diagnostic, never as the score:
    # dropping the unanswered questions is survivorship bias by construction.
    ans = [r for r in recs if r["extracted"] is not None]
    if ans:
        print("DIAGNOSTIC ONLY (not the score): accuracy among answered = %d/%d = %.1f%%"
              % (sum(r["score"] for r in ans), len(ans),
                 100.0 * sum(r["score"] for r in ans) / len(ans)))
        print("  -> the gap between this and the scored number is entirely non-termination")
    print()

    print("TOP REPEATED PHRASES IN CAPPED RESPONSES (first 8)")
    shown = 0
    for i, r in capped:
        frac, _, top = ngram_stats((r["raw"] or "")[-4000:])
        if frac >= 0.06 and shown < 8:
            shown += 1
            print("  pos=%-4d ctok=%-6s frac=%.2f  %r"
                  % (r["pos"], r.get("completion_tokens"), frac, top[:110]))
    print()


if __name__ == "__main__":
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    label = sys.argv[3] if len(sys.argv) > 3 else sys.argv[1].split("/")[-1]
    report(sys.argv[1], label, limit)
