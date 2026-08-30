#!/usr/bin/env python3
"""One table for the investigation-10 arms.

Reads ``summary.json`` (written by investigation/06's analyze.py) for every
``g10_*`` run and prints the two read-outs that separate the healthy and broken
regimes far more sharply than the score does at n=48:

    repetitive_tail    0 in every healthy arm, 13-50 in every broken one
    resp_chars_median  ~31k healthy, ~61-76k broken

``decode_bs_hist`` and the captured set are printed next to every row, because
client concurrency silently selects which graph replays and comparing arms at
different concurrency has already produced two opposite fake results here.

Usage: python3 summarize10.py [runs_root]
"""
import json
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/shared/zz-m27-recheck/runs"

ORDER = [
    ("g10_base", "int2 conc16 mbs32", "positive control"),
    ("g10_c8", "int2 conc8  mbs32", "negative control"),
    ("g10_bf16", "bf16 conc16 mbs32", "is it INT2-specific?"),
    ("g10_bf16c8", "bf16 conc8  mbs32", "bf16 baseline"),
    ("g10_noovl", "int2 conc16 no-overlap", "scheduler/stream race"),
    ("g10_norad", "int2 conc16 no-radix", "radix x graph"),
    ("g10_mbs16", "int2 conc16 mbs16", "capture set / pool aliasing"),
    ("g10_mbs12", "int2 conc16 mbs12", "shipped workaround"),
]

hdr = (f"{'arm':<12} {'config':<24} {'score':>6} {'ans':>5} {'reptail':>8} "
       f"{'median':>8} {'top bs':>10}  note")
print(hdr)
print("-" * len(hdr))
for arm, cfg, note in ORDER:
    p = os.path.join(ROOT, arm, "summary.json")
    if not os.path.isfile(p):
        print(f"{arm:<12} {cfg:<24} {'--':>6} {'--':>5} {'--':>8} {'--':>8} "
              f"{'--':>10}  {note} (not finished)")
        continue
    try:
        d = json.load(open(p))
    except Exception as e:
        print(f"{arm:<12} {cfg:<24}  unreadable: {e}")
        continue
    hist = d.get("decode_bs_hist") or {}
    top = max(hist.items(), key=lambda kv: kv[1])[0] if hist else "?"
    tops = f"{top}x{hist.get(top, 0)}" if hist else "?"
    sc = d.get("score")
    print(f"{arm:<12} {cfg:<24} "
          f"{(f'{100*sc:.1f}' if sc is not None else '--'):>6} "
          f"{str(d.get('answered', '--')):>5} "
          f"{str(d.get('repetitive_tail', '--')):>8} "
          f"{str(d.get('resp_chars_median', '--')):>8} "
          f"{tops:>10}  {note}")
