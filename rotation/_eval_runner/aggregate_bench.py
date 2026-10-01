#!/usr/bin/env python3
"""Aggregate the Qwen3.5 benchmark matrix into mean +/- std over seeds and emit
a README-ready table. Reads:
  rotation/<model>/bench_matrix/<method>/<bench>/seed<N>/metrics.json
"""
import json
import os
import statistics as st

BASE = os.path.join(os.path.dirname(__file__), "..")  # rotation/
MODELS = [("Qwen3.5-4B", "qwen3.5-4B"), ("Qwen3.5-35B-A3B", "qwen3.5-35B-A3B")]
METHODS = [("BF16", "bf16"), ("OSCAR", "oscar_lloydmax")]  # published OSCAR = Lloyd-Max
# Per-model best OSCAR config (2-bit, sink64/recent256): 4B = uniform,
# 35B-A3B = Lloyd-Max (recovers long-horizon AIME; ≥BF16 on GPQA/HumanEval).
OSCAR_DIR = {"qwen3.5-4B": "oscar", "qwen3.5-35B-A3B": "oscar_lloydmax"}
BENCHES = [("GPQA-Diamond", "gpqa"), ("HumanEval", "humaneval"),
           ("AIME 2025", "aime25"), ("MATH500", "math500")]
SEEDS = list(range(8))  # picks up all available seeds (35B AIME has 8)


def scores(mdir, method, bench):
    out = []
    for s in SEEDS:
        p = os.path.join(BASE, mdir, "bench_matrix", method, bench, f"seed{s}", "metrics.json")
        if os.path.exists(p):
            try:
                out.append(float(json.load(open(p)).get("score")))
            except Exception:
                pass
    return out


def cell(vals):
    if not vals:
        return "—", None
    m = 100 * st.mean(vals)
    sd = 100 * st.pstdev(vals) if len(vals) > 1 else 0.0
    return f"{m:.1f} ± {sd:.1f}", m


def main():
    for label, mdir in MODELS:
        print(f"\n### {label}  (mean ± std over {len(SEEDS)} seeds, %)\n")
        print("| Benchmark | BF16 | OSCAR | Δ (pp) | seeds |")
        print("|---|---|---|---|---|")
        for blabel, bkey in BENCHES:
            bf = scores(mdir, "bf16", bkey)
            os_ = scores(mdir, OSCAR_DIR[mdir], bkey)
            bf_s, bf_m = cell(bf)
            os_s, os_m = cell(os_)
            delta = f"{os_m - bf_m:+.1f}" if (bf_m is not None and os_m is not None) else "—"
            print(f"| {blabel} | {bf_s} | {os_s} | {delta} | bf16={len(bf)}/3 oscar={len(os_)}/3 |")
        # raw per-seed dump
        print("\n<details><summary>per-seed</summary>\n")
        for blabel, bkey in BENCHES:
            print(f"  {blabel:14} bf16={[round(x,4) for x in scores(mdir,'bf16',bkey)]}  "
                  f"oscar={[round(x,4) for x in scores(mdir,OSCAR_DIR[mdir],bkey)]}")
        print("\n</details>")


if __name__ == "__main__":
    main()
