#!/usr/bin/env python3
"""Contiguous-prefix scorer for k3_gpqa.py result JSONLs, with a matched-scope
two-arm comparison mode.

Why not k3_score.py: that scorer scores EVERY record in the file and only
*reports* how long the contiguous prefix is.  With a live run that mixes
finished-fast questions and still-in-flight slow ones, scoring everything is
length-survivorship bias -- the fast questions are over-represented, and on this
run that reads ~12 pp high.  Here the default scope is the contiguous prefix
pos == 0..n-1, so no position with a gap before it can contribute.

Compare mode intersects the two arms' contiguous prefixes and scores both over
exactly that set, so the arms are never compared at different scopes.  (The
sibling M2.7 investigation lost weeks to an asymmetric comparison.)
"""

import argparse
import collections
import json
import math
import re
import sys

STRUCT_RE = re.compile(r"<\|[^|>]*\|>")


def load(path):
    recs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    pass
    # A relaunch appends; if a position somehow appears twice keep the first.
    seen, uniq = set(), []
    for r in sorted(recs, key=lambda r: r["pos"]):
        if r["pos"] in seen:
            continue
        seen.add(r["pos"])
        uniq.append(r)
    return uniq


def contiguous(recs):
    out = []
    for i, r in enumerate(recs):
        if r["pos"] != i:
            break
        out.append(r)
    return out


def shape(text):
    body = STRUCT_RE.sub(" ", text or "")
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


def metrics(recs, label):
    n = len(recs)
    if not n:
        return None
    k = sum(r["score"] for r in recs)
    p = k / n
    se = math.sqrt(p * (1 - p) / n)
    sh = [shape(r["raw"]) for r in recs]
    capped = [r["pos"] for r in recs
              if bool(r.get("hit_max_tokens")) or r.get("finish_reason") == "length"]
    loops = [recs[i]["pos"] for i, s in enumerate(sh)
             if s["top_5gram_frac"] >= 0.06 and s["words"] > 40]
    loop_capped = [recs[i]["pos"] for i, s in enumerate(sh)
                   if s["top_5gram_frac"] >= 0.06 and s["words"] > 40
                   and (bool(recs[i].get("hit_max_tokens"))
                        or recs[i].get("finish_reason") == "length")]
    ct = sorted((r.get("completion_tokens") or 0) for r in recs)
    wrong = [r for r in recs if not r["score"]]
    HP = 64 + 256
    tot_ctx = sum((r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0)
                  for r in recs)
    quant_ctx = sum(max(0, (r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0) - HP)
                    for r in recs)
    return {
        "label": label, "n": n, "correct": k, "acc": p, "se": se,
        "answered": sum(r["extracted"] is not None for r in recs),
        "no_resp": sum(not r.get("has_response_section") for r in recs),
        "capped": len(capped), "capped_pos": capped,
        "loops": len(loops), "loop_pos": loops,
        "loop_capped": len(loop_capped),
        "errors": sum(bool(r.get("error")) for r in recs),
        "ct_min": ct[0], "ct_p25": ct[max(0, int(0.25 * (n - 1)))],
        "ct_med": ct[n // 2], "ct_p75": ct[int(0.75 * (n - 1))],
        "ct_p90": ct[int(0.9 * (n - 1))], "ct_max": ct[-1],
        "ct_mean": sum(ct) / n,
        "wrong_letter": sum(r["extracted"] is not None for r in wrong),
        "no_answer": sum(r["extracted"] is None for r in wrong),
        "acc_fullraw": sum(r["score_fullraw"] for r in recs) / n,
        "mean_ctx": tot_ctx / n,
        "quant_frac": 100.0 * quant_ctx / max(1, tot_ctx),
        "recs": recs,
    }


def show(m):
    n = m["n"]
    print("=" * 74)
    print("%s  --  contiguous prefix, n=%d" % (m["label"], n))
    print("=" * 74)
    print("  score                    %d/%d = %.1f%%   SE %.1f pp   95%% CI %.1f..%.1f"
          % (m["correct"], n, 100 * m["acc"], 100 * m["se"],
             100 * (m["acc"] - 1.96 * m["se"]), 100 * (m["acc"] + 1.96 * m["se"])))
    print("  sensitivity (full raw)   %.1f%%" % (100 * m["acc_fullraw"]))
    print("  answered                 %d/%d = %.1f%%"
          % (m["answered"], n, 100.0 * m["answered"] / n))
    print("  hit the generation cap   %d/%d = %.1f%%"
          % (m["capped"], n, 100.0 * m["capped"] / n))
    print("  repetition loops         %d/%d = %.1f%%   (also capped: %d)"
          % (m["loops"], n, 100.0 * m["loops"] / n, m["loop_capped"]))
    print("  no response section      %d      request errors %d" % (m["no_resp"], m["errors"]))
    print("  completion tokens        min %d  p25 %d  med %d  p75 %d  p90 %d  max %d  mean %.0f"
          % (m["ct_min"], m["ct_p25"], m["ct_med"], m["ct_p75"], m["ct_p90"],
             m["ct_max"], m["ct_mean"]))
    print("  error budget             %d wrong letter + %d unextractable = %d"
          % (m["wrong_letter"], m["no_answer"], n - m["correct"]))
    if n - m["correct"]:
        print("                           -> %.0f%% of errors are non-termination"
              % (100.0 * m["no_answer"] / (n - m["correct"])))
    print("  mean end ctx %.0f tok, %.1f%% of it in the INT2 tier"
          % (m["mean_ctx"], m["quant_frac"]))


def compare(ma, mb):
    """McNemar + deltas over the matched set (both already restricted)."""
    a = {r["pos"]: r for r in ma["recs"]}
    b = {r["pos"]: r for r in mb["recs"]}
    both = sorted(set(a) & set(b))
    n = len(both)
    a01 = sum(1 for p in both if not a[p]["score"] and b[p]["score"])
    a10 = sum(1 for p in both if a[p]["score"] and not b[p]["score"])
    print()
    print("=" * 74)
    print("MATCHED-SCOPE COMPARISON  (%s vs %s), n=%d" % (ma["label"], mb["label"], n))
    print("=" * 74)
    ka = sum(a[p]["score"] for p in both)
    kb = sum(b[p]["score"] for p in both)
    pa, pb = ka / n, kb / n
    # SE of the paired difference from the discordant pairs (McNemar).
    d = a10 + a01
    se_d = math.sqrt(max(1e-12, (a10 + a01) - (a10 - a01) ** 2 / n)) / n if n else 0.0
    print("  %-22s %d/%d = %.1f%%" % (ma["label"], ka, n, 100 * pa))
    print("  %-22s %d/%d = %.1f%%" % (mb["label"], kb, n, 100 * pb))
    print("  delta (%s - %s)      %+.1f pp   paired SE %.1f pp"
          % (ma["label"], mb["label"], 100 * (pa - pb), 100 * se_d))
    print("  discordant pairs         %s-only %d / %s-only %d  (of %d)"
          % (ma["label"], a10, mb["label"], a01, d))
    if d:
        # exact two-sided binomial sign test on the discordant pairs
        from math import comb
        k = min(a10, a01)
        pv = min(1.0, 2 * sum(comb(d, i) for i in range(k + 1)) / (2 ** d))
        print("  exact McNemar p          %.3f" % pv)
    print()
    print("  FAILURE METRICS AT MATCHED SCOPE")
    hdr = "  %-24s %14s %14s %10s" % ("metric", ma["label"], mb["label"], "delta")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    def row(name, fa, fb, pct=True, fmt="%.1f"):
        va, vb = fa, fb
        if pct:
            print("  %-24s %13.1f%% %13.1f%% %+9.1f" % (name, va, vb, va - vb))
        else:
            print(("  %-24s %14" + fmt[1:] + " %14" + fmt[1:] + " %+9" + fmt[1:])
                  % (name, va, vb, va - vb))

    sub_a = metrics([a[p] for p in both], ma["label"])
    sub_b = metrics([b[p] for p in both], mb["label"])
    row("score", 100 * sub_a["acc"], 100 * sub_b["acc"])
    row("answered", 100.0 * sub_a["answered"] / n, 100.0 * sub_b["answered"] / n)
    row("hit generation cap", 100.0 * sub_a["capped"] / n, 100.0 * sub_b["capped"] / n)
    row("repetition loop rate", 100.0 * sub_a["loops"] / n, 100.0 * sub_b["loops"] / n)
    row("median completion tok", sub_a["ct_med"], sub_b["ct_med"], pct=False, fmt="%.0f")
    row("mean completion tok", sub_a["ct_mean"], sub_b["ct_mean"], pct=False, fmt="%.0f")
    row("p90 completion tok", sub_a["ct_p90"], sub_b["ct_p90"], pct=False, fmt="%.0f")
    print()
    print("  positions capped in %s only: %s"
          % (ma["label"], sorted(set(sub_a["capped_pos"]) - set(sub_b["capped_pos"]))))
    print("  positions capped in %s only: %s"
          % (mb["label"], sorted(set(sub_b["capped_pos"]) - set(sub_a["capped_pos"]))))
    print("  positions capped in BOTH:    %s"
          % sorted(set(sub_a["capped_pos"]) & set(sub_b["capped_pos"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="results JSONL(s); 2 files -> compare")
    ap.add_argument("--labels", default="")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap the scope at this many leading positions")
    ap.add_argument("--growth", action="store_true",
                    help="print score vs prefix size, to see whether it has settled")
    args = ap.parse_args()

    labels = args.labels.split(",") if args.labels else []
    ms = []
    for i, f in enumerate(args.files):
        recs = contiguous(load(f))
        lab = labels[i] if i < len(labels) else f.split("/")[-1]
        ms.append((lab, recs))

    if len(ms) == 2:
        # matched scope = shortest contiguous prefix common to both arms
        cap = min(len(ms[0][1]), len(ms[1][1]))
        if args.limit:
            cap = min(cap, args.limit)
        print("[scope] contiguous prefixes: %s=%d  %s=%d  -> matched scope n=%d"
              % (ms[0][0], len(ms[0][1]), ms[1][0], len(ms[1][1]), cap))
        a = metrics(ms[0][1][:cap], ms[0][0])
        b = metrics(ms[1][1][:cap], ms[1][0])
        show(a)
        print()
        show(b)
        compare(a, b)
        return

    lab, recs = ms[0]
    if args.limit:
        recs = recs[:args.limit]
    m = metrics(recs, lab)
    if m is None:
        print("no contiguous records in", args.files[0])
        return
    show(m)
    if args.growth:
        print()
        print("  SCORE vs PREFIX SIZE (is it still moving?)")
        for cut in list(range(10, len(recs), 10)) + [len(recs)]:
            sub = recs[:cut]
            kk = sum(r["score"] for r in sub)
            pp = kk / cut
            print("    n=%3d  %3d/%3d = %5.1f%%  SE %.1f pp"
                  % (cut, kk, cut, 100 * pp, 100 * math.sqrt(pp * (1 - pp) / cut)))


if __name__ == "__main__":
    main()
