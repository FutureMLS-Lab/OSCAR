#!/usr/bin/env python3
"""GLM-5.2 lossless check: aggregate bench_matrix/{bf16,oscar} into mean±std
per benchmark + Δ and a Welch t so the OSCAR≈BF16 (无损) claim is quantified."""
import json
import math
import os
import statistics as st

BASE = os.path.join(os.path.dirname(__file__), "..", "GLM-5.2-FP8", "bench_matrix")
BENCHES = [("HumanEval", "humaneval"), ("GPQA-Diamond", "gpqa"),
           ("AIME 2025", "aime25"), ("MATH500 (200)", "math500")]


def seeds(method, bench):
    out = []
    for s in range(8):
        p = os.path.join(BASE, method, bench, f"seed{s}", "metrics.json")
        if os.path.exists(p):
            try:
                out.append(float(json.load(open(p))["score"]))
            except Exception:
                pass
    return out


def fmt(vals):
    if not vals:
        return "—"
    m = 100 * st.mean(vals)
    sd = 100 * (st.pstdev(vals) if len(vals) > 1 else 0.0)
    return f"{m:.1f} ± {sd:.1f} (n={len(vals)})"


def welch(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = st.variance(a), st.variance(b)
    se = math.sqrt(va / len(a) + vb / len(b))
    return (st.mean(b) - st.mean(a)) / se if se > 0 else None


print("## GLM-5.2-FP8 — BF16 vs OSCAR INT2 latent (pure Rcov·P·Hblock + LM)\n")
print("| Benchmark | BF16 | OSCAR | Δ (pp) | Welch t |")
print("|---|---|---|---|---|")
for label, key in BENCHES:
    bf, osc = seeds("bf16", key), seeds("oscar", key)
    d = f"{100*(st.mean(osc)-st.mean(bf)):+.1f}" if bf and osc else "—"
    t = welch(bf, osc)
    print(f"| {label} | {fmt(bf)} | {fmt(osc)} | {d} | {f'{t:.2f}' if t is not None else '—'} |")
print("\nper-seed:")
for label, key in BENCHES:
    print(f"  {key:10} bf16={[round(x,4) for x in seeds('bf16',key)]} "
          f"oscar={[round(x,4) for x in seeds('oscar',key)]}")
