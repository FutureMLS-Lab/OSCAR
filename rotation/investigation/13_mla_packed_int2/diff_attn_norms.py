#!/usr/bin/env python3
"""Locate the layer at which two K3 forward paths diverge.

Both arms are traced with SGLANG_OSCAR_TRACE_ATTN_OUT=1, which logs the norm of
the attention output of every full-attention layer on every TP rank. Comparing
those norms layer by layer separates two very different diagnoses:

  * disagreement at the FIRST full-attention layer -> a math error inside that
    layer's operators, which is directly localisable
  * agreement early and drift later -> an accumulation effect whose cause is
    somewhere else entirely

The trace only fires where the host actually executes Python, so a CUDA-graph
replay records nothing. For MLA models sglang dispatches the MHA-style path for
prefill and the absorbed path only for decode, so an uncaptured run yields
`path=mha` for BOTH arms -- which is precisely what makes them comparable: the
same operator sequence reading two different cache layouts. If the norms agree
there, the latent materialisation is fine and the fault is in the absorbed
decode path; if they disagree, the fault is upstream in the latent itself.

Norms are averaged over TP ranks. A rank count that differs between arms means
the runs are not comparable and is reported rather than silently averaged away.
"""
from __future__ import annotations

import argparse
import collections
import math
import re
import sys

RE_LAYER = re.compile(r"layer=(\d+)")
RE_NORM = re.compile(r"norm=([0-9.eE+-]+)")
RE_PATH = re.compile(r"path=([a-z-]+)")
# shape=(1, 1536) contains a space, so \S+ silently matches nothing here.
RE_SHAPE = re.compile(r"shape=\(([^)]*)\)")


def grab(path: str):
    per_layer = collections.defaultdict(list)
    paths = collections.Counter()
    shapes = collections.Counter()
    try:
        fh = open(path, errors="ignore")
    except FileNotFoundError:
        return None, paths, shapes
    with fh:
        for line in fh:
            if "ATTN-OUT" not in line:
                continue
            m_l, m_n = RE_LAYER.search(line), RE_NORM.search(line)
            if not (m_l and m_n):
                continue
            per_layer[int(m_l.group(1))].append(float(m_n.group(1)))
            m_p = RE_PATH.search(line)
            if m_p:
                paths[m_p.group(1)] += 1
            m_s = RE_SHAPE.search(line)
            if m_s:
                shapes[m_s.group(1)] += 1
    return per_layer, paths, shapes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    # 2% tolerates TP-rank ordering and bf16 reduction noise; 50% is well past
    # anything a precision effect produces.
    ap.add_argument("--ok", type=float, default=0.02)
    ap.add_argument("--diverge", type=float, default=0.50)
    args = ap.parse_args()

    A, pa, sa = grab(args.a)
    B, pb, sb = grab(args.b)
    if A is None or B is None:
        print("missing log: %s%s" % (args.a if A is None else "",
                                     args.b if B is None else ""))
        return 2
    if not A or not B:
        print("no ATTN-OUT lines in %s -- the trace never fired (captured "
              "decode? wrong forward method instrumented?)"
              % (args.a if not A else args.b))
        return 2

    print("%-10s paths=%s shapes=%s" % (args.label_a, dict(pa), dict(sa)))
    print("%-10s paths=%s shapes=%s" % (args.label_b, dict(pb), dict(sb)))

    hdr = "%4s %4s %11s %4s %11s %9s  %s" % (
        "L", "n1", "mean1", "n2", "mean2", "ratio", "verdict")
    print("\n" + hdr)
    print("-" * len(hdr))

    first = None
    rels = []
    for k in sorted(set(A) | set(B)):
        a, b = A.get(k, []), B.get(k, [])
        m1 = sum(a) / len(a) if a else math.nan
        m2 = sum(b) / len(b) if b else math.nan
        verdict, ratio = "", math.nan
        if a and b and m1:
            ratio = m2 / m1
            rel = abs(m2 - m1) / max(abs(m1), 1e-9)
            rels.append((k, rel))
            verdict = ("OK" if rel < args.ok else
                       "drift" if rel < args.diverge else "DIVERGE")
            if verdict != "OK" and first is None:
                first = k
        elif not a or not b:
            verdict = "MISSING in %s" % (args.label_a if not a
                                         else args.label_b)
        print("%4d %4d %11.4f %4d %11.4f %9.3f  %s"
              % (k, len(a), m1, len(b), m2, ratio, verdict))

    print()
    if first is None:
        print("no divergence above %.0f%% -- the two paths agree at every "
              "traced layer, so the fault is NOT in the traced path"
              % (100 * args.ok))
    else:
        traced = sorted(set(A) & set(B))
        where = ("the FIRST traced full-attention layer -> a math error in "
                 "this layer's operators"
                 if first == traced[0] else
                 "not the first traced layer (%d agreed) -> accumulation, "
                 "cause is upstream of the divergence" % traced[0])
        print("first divergent layer: L%d -- %s" % (first, where))
    if rels:
        worst = max(rels, key=lambda x: x[1])
        print("worst relative gap: L%d  %.1f%%" % (worst[0], 100 * worst[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
