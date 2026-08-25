#!/usr/bin/env python3
"""Score a K3 results.jsonl without misreading the schema.

Written after a near-miss: ``correct`` in these records is the correct ANSWER
LETTER ('B'), not a boolean, so ``sum(1 for r in rs if r["correct"])`` counts
every record and reports 100%. The scored field is ``score`` (0/1), and
``extracted`` is what the model actually answered.

The same read also hid that every one of those records was a failed request --
``error: URLError Connection refused``, ``raw: ''`` -- because a dead server
produces rows that look like results. So this refuses to print an accuracy
until it has told you how many rows are errors, and it reports the scored
denominator rather than the answered one: conditioning on "the ones that came
back" is survivorship bias, and it flatters exactly the runs that are broken.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter


def load(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    a = p.parse_args()

    for path in a.paths:
        try:
            rs = load(path)
        except FileNotFoundError:
            print(f"{path}: missing")
            continue
        n = len(rs)
        if not n:
            print(f"{path}: empty")
            continue
        err = [r for r in rs if r.get("error")]
        empty = [r for r in rs if not (r.get("raw") or "")]
        scored = sum(int(r.get("score") or 0) for r in rs)

        print(f"\n{path}")
        print(f"  rows            {n}")
        print(f"  errors          {len(err)}"
              + (f"   e.g. {str(err[0]['error'])[:70]}" if err else ""))
        print(f"  empty raw       {len(empty)}")
        # Denominator is every row, including the failures. A run whose server
        # died scores 0, which is the honest number -- not "no data".
        print(f"  SCORE           {scored}/{n} = {100.0 * scored / n:.2f}%")
        if err or empty:
            print(f"  !! {max(len(err), len(empty))} of {n} rows produced no "
                  f"model output. This is an infrastructure failure, not an "
                  f"accuracy result -- do not report the percentage above as "
                  f"a quantization number.")
        hit = sum(1 for r in rs if r.get("hit_max_tokens"))
        if hit:
            print(f"  hit max tokens  {hit}")
        ex = Counter(str(r.get("extracted")) for r in rs)
        print(f"  extracted       {dict(ex.most_common(6))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
