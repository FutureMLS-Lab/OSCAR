#!/usr/bin/env python3
"""Per-question paired comparison of two arms of the same GPQA run.

A score difference cannot say whether an arm is worse everywhere or badly wrong
on a few items, and a mean length cannot say whether one arm rambles on every
question or blows up on a handful. Both matter here, because the whole question
is whether INT2's deficit is a per-question quantization cost or a truncation
artifact concentrated in the tail.

Matching is by rendered prompt, so the pairing is exact rather than positional
-- the arms are separate runs with their own request ordering, and pairing by
line number would silently compare different questions.

Reported per pair:

* ``len_ratio`` quartiles over questions where BOTH arms terminated normally.
  Restricting to cleanly-terminated pairs is the point: a truncated response's
  length is the budget, not the model's choice, so including capped responses
  would measure the cap and call it verbosity.
* the 2x2 correctness table, which separates "B is worse on questions A also
  got wrong" from "B loses questions A got right".
* ``both_clean_acc``, accuracy on the subset where both terminated. This is the
  cleanest available read on quality with truncation removed. It is still a
  conditioned subset, so it is reported next to the full-denominator score and
  never instead of it.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import rescore_gpqa as rs

TOKEN = re.compile(r"[A-Za-z0-9_\\^{}=+\-*/().,;:$]+")


def load(run_dir: Path, key: dict[str, str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    io = run_dir / "io_log.jsonl"
    for line in io.open(errors="replace"):
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
        prompt = prompt.strip()
        gold = key.get(prompt)
        if gold is None:
            continue
        resp = rec.get("response") or ""
        ms = rs.STRICT.search(resp)
        out[prompt] = {
            "gold": gold,
            "words": len(TOKEN.findall(resp)),
            "capped": rec.get("finish_reason") == "length",
            # Case-sensitive, matching simple_evals (see rescore_gpqa).
            "correct": bool(ms and ms.group(1) == gold),
            "answered": bool(ms),
        }
    return out


def quart(xs):
    if not xs:
        return (None, None, None)
    s = sorted(xs)
    return (s[len(s) // 4], s[len(s) // 2], s[3 * len(s) // 4])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_a")
    ap.add_argument("arm_b")
    ap.add_argument("--variant", default="diamond")
    a = ap.parse_args()

    key = rs.build_key_to_answer(a.variant)
    A = load(Path(a.arm_a), key)
    B = load(Path(a.arm_b), key)
    common = sorted(set(A) & set(B))
    print(f"A={Path(a.arm_a).name}  B={Path(a.arm_b).name}  paired={len(common)}")

    both_clean = [p for p in common if not A[p]["capped"] and not B[p]["capped"]]
    ratios = [
        B[p]["words"] / A[p]["words"]
        for p in both_clean
        if A[p]["words"] > 0
    ]
    q = quart(ratios)
    print(f"both terminated normally: {len(both_clean)}")
    if q[0] is not None:
        print(f"  words(B)/words(A) quartiles: "
              f"p25={q[0]:.2f} p50={q[1]:.2f} p75={q[2]:.2f}")
        print(f"  B longer than A on {sum(1 for r in ratios if r > 1)}/{len(ratios)}")
    aw = quart([A[p]["words"] for p in both_clean])
    bw = quart([B[p]["words"] for p in both_clean])
    print(f"  words A p25/p50/p75 = {aw}")
    print(f"  words B p25/p50/p75 = {bw}")

    n11 = sum(1 for p in common if A[p]["correct"] and B[p]["correct"])
    n10 = sum(1 for p in common if A[p]["correct"] and not B[p]["correct"])
    n01 = sum(1 for p in common if not A[p]["correct"] and B[p]["correct"])
    n00 = sum(1 for p in common if not A[p]["correct"] and not B[p]["correct"])
    print(f"correctness 2x2 over all {len(common)} paired: "
          f"A+B+={n11} A+B-={n10} A-B+={n01} A-B-={n00}")
    print(f"  A correct={n11 + n10}  B correct={n11 + n01}  "
          f"B loses {n10}, gains {n01}")

    ca = sum(1 for p in both_clean if A[p]["correct"])
    cb = sum(1 for p in both_clean if B[p]["correct"])
    if both_clean:
        print(f"both_clean_acc: A={ca}/{len(both_clean)}={100*ca/len(both_clean):.1f}%  "
              f"B={cb}/{len(both_clean)}={100*cb/len(both_clean):.1f}%")
    print(f"capped: A={sum(1 for p in common if A[p]['capped'])} "
          f"B={sum(1 for p in common if B[p]['capped'])}")


if __name__ == "__main__":
    main()
