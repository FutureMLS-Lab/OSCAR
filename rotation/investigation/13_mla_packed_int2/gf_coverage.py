#!/usr/bin/env python3
"""Which token positions does each pass of the group-factored path actually cover?

The equivalence gate says the group-factored decode mismatches the production
kernel at bs=8 seq=4096 and bs=4 seq=600, rel ~0.28 where a passing shape sits
at 0.003, and its own bisect line is consistent every time:

    NO-ARENA gf vs production: MATCH -> fault is exclusion/window

So the kernel arithmetic is right and the fault is in *which tokens each pass
claims*. The two-pass split is only correct if the passes partition the sequence:
every position covered exactly once. Reading the two predicates side by side is
how I got the wrong answer twice, so this enumerates them instead.

The packed pass walks the whole sequence and DROPS a position when
``r >= 0 and owner == slot``. The window pass walks only ``[0, P_TOK)`` and
``[tail_start, seq)`` and KEEPS a position under the same predicate. Those are
complementary only if no position outside the window ranges satisfies the
predicate -- if one does, the packed pass drops it and the window pass never
looks at it, and the token vanishes from the attention entirely.

Reports, per request in the batch:
  * positions dropped by packed but not covered by window  (VANISHED)
  * positions kept by both                                  (DOUBLE COUNTED)
  * the window range the kernel computes, for comparison

Pure index arithmetic against the same fixture ``build()`` produces, so it runs
on CPU and needs no GPU.
"""
from __future__ import annotations

import argparse


def coverage(bs: int, seq: int, windows: int = 576, p_tok: int = 64):
    r_tok = windows - p_tok
    problems = []
    for b in range(bs):
        base = b * seq

        # --- the fixture, mirroring bench_kernel.build() ---
        n_pre = min(p_tok, seq)
        tail_lo = max(seq - r_tok, n_pre)
        owned = set(range(n_pre)) | set(range(tail_lo, seq))  # positions with a ring row

        # --- the kernel's window range, mirroring _fwd_hp_window_stage1 ---
        tail_start = max(seq - r_tok, p_tok)
        n_window = min(p_tok, seq) + max(seq - tail_start, 0)
        win = set()
        for i in range(n_window):
            if i < min(p_tok, seq):
                win.add(i)
            else:
                win.add(tail_start + (i - min(p_tok, seq)))

        # --- the packed pass drops every owned position, wherever it is ---
        dropped = owned
        vanished = sorted(dropped - win)
        doubled = sorted(win - owned)   # window keeps it, packed did not drop it

        if vanished or doubled:
            problems.append((b, base, vanished, doubled))
    return problems, dict(p_tok=p_tok, r_tok=r_tok)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=576)
    ap.add_argument("--p-tok", type=int, default=64)
    a = ap.parse_args()

    # The gate's shapes: the two that mismatch first, then the ones that pass.
    shapes = [(8, 4096), (4, 600), (8, 256), (1, 100), (2, 140)]
    print(f"{'bs':>3} {'seq':>6}  {'vanished':>9} {'doubled':>8}  verdict")
    print("-" * 52)
    for bs, seq in shapes:
        problems, cfg = coverage(bs, seq, a.windows, a.p_tok)
        nv = sum(len(v) for _, _, v, _ in problems)
        nd = sum(len(d) for _, _, _, d in problems)
        verdict = ("partition is clean" if not (nv or nd)
                   else "NOT A PARTITION -- tokens are lost or counted twice")
        print(f"{bs:>3} {seq:>6}  {nv:>9} {nd:>8}  {verdict}")
        if problems:
            b, base, v, d = problems[0]
            if v:
                print(f"      req {b}: {len(v)} positions dropped by packed and "
                      f"never covered by window, e.g. {v[:6]}")
            if d:
                print(f"      req {b}: {len(d)} positions the window claims but "
                      f"packed did not drop, e.g. {d[:6]}")
    print(f"\nP_TOK={a.p_tok} R_TOK={a.windows - a.p_tok}")
    print("A shape that partitions cleanly here but still mismatches on GPU "
          "points at the owner tags rather than the ranges.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
