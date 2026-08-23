#!/usr/bin/env python3
"""Verify that two Kimi-K3 result JSONLs are talking about the same questions.

k3_prefix_score.py pairs arms by `pos`, which is only valid if `pos` means the
same question in both files.  That is not free: `pos` is an index into a
seed-0 shuffle of the GPQA-diamond rows, and the per-question option
permutation is a second seed (10000 + orig_idx).  A subset run with a different
`num_examples` produces a DIFFERENT shuffle -- on the sibling GLM-5.2
investigation, pairing a 40-question subset against the 198-question run matched
exactly one row, and the two were nearly reported as a paired delta anyway.

So before any cross-run McNemar: check that every shared position agrees on both
orig_idx (same source row) and correct (same option permutation).  A mismatch
means the runs are not pairable and the comparison must be abandoned, not
patched.
"""

import json
import sys


def load(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:                                # noqa: BLE001
                continue
            out.setdefault(r["pos"], r)     # a relaunch appends; keep the first
    return out


def contiguous_len(d):
    n = 0
    while n in d:
        n += 1
    return n


def main(paths):
    arms = [(p.split("/")[-1], load(p)) for p in paths]
    for name, d in arms:
        print("%-28s records=%3d  contiguous prefix=%3d" % (name, len(d), contiguous_len(d)))
    ref_name, ref = arms[0]
    ok = True
    for name, d in arms[1:]:
        shared = sorted(set(ref) & set(d))
        bad_idx = [p for p in shared if ref[p]["orig_idx"] != d[p]["orig_idx"]]
        bad_key = [p for p in shared if ref[p]["correct"] != d[p]["correct"]]
        print("%-28s vs %-22s shared=%3d  orig_idx mismatches=%d  correct-letter mismatches=%d"
              % (name, ref_name, len(shared), len(bad_idx), len(bad_key)))
        if bad_idx or bad_key:
            ok = False
            print("   NOT PAIRABLE: first offending positions %s %s"
                  % (bad_idx[:5], bad_key[:5]))
    print("ALIGNMENT", "OK -- these runs share one question order and one option "
          "permutation, so pairing by pos is valid" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
