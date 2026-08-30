#!/usr/bin/env python3
"""Roll the per-seed analyze.py rows into the per-arm report.

Usage: aggregate.py RUNS_DIR arm [arm ...]

Prints, per arm, exactly the eight columns the task asks for plus the evidence
columns, then the arm-vs-arm deltas with a Welch t-test. With n=3 per arm the
test is weak by construction and is printed so it can be read as "no detectable
difference", not as proof of equality.
"""
import glob
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

INV = Path(__file__).resolve().parent


def rows(base, arm):
    out = []
    for s in sorted(glob.glob(f"{base}/{arm}/seed*")):
        if not Path(s).is_dir():
            continue
        r = subprocess.run([sys.executable, str(INV / "analyze.py"), s, "--json"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            out.append(json.loads(r.stdout))
    return out


def welch(a, b):
    """Welch t-test, two-sided, normal approximation for the p-value."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None, None
    va, vb = statistics.variance(a), statistics.variance(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return None, None
    t = (statistics.mean(a) - statistics.mean(b)) / se
    # normal-approximation two-sided p; with 4 dof this is optimistic, so a
    # non-significant result here is safely non-significant.
    p = math.erfc(abs(t) / math.sqrt(2))
    return t, p


def main():
    base = sys.argv[1]
    arms = sys.argv[2:]
    data = {}
    for arm in arms:
        rs = rows(base, arm)
        if not rs:
            print(f"### {arm}: NO DATA")
            continue
        data[arm] = rs
        sc = [r["score"] * 100 for r in rs]
        wl = [r["seed_wall_s"] for r in rs]
        hit = [round(r["cache_hit_rate"] * 100, 3) for r in rs]
        print(f"### {arm}  (n={len(rs)})")
        print(f"  score            {statistics.mean(sc):.2f} +/- "
              f"{statistics.stdev(sc):.2f}   seeds {[round(x, 2) for x in sc]}")
        print(f"  answered         {sum(r['answered'] for r in rs)}/"
              f"{sum(r['responses'] for r in rs)}   per-seed "
              f"{[r['answered'] for r in rs]}")
        print(f"  unclosed <think> {[r['unclosed_think'] for r in rs]}  "
              f"(structurally 0: open tag is in the chat template)")
        print(f"  no </think>      {[r['no_think_close'] for r in rs]}  "
              f"(the metric that carries the information)")
        print(f"  runaway          {[r['runaway'] for r in rs]}  "
              f"({rs[0].get('runaway_definition')})")
        print(f"  cache hit rate   {statistics.mean(hit):.3f}%  per-seed {hit}  "
              f"cached_tok {[r['prefill_cached_tokens'] for r in rs]}")
        print(f"  wall clock       {statistics.mean(wl):.0f}s  per-seed {wl}")
        print(f"  decode samples   {[r['decode_steps'] for r in rs]}  "
              f"padded_replay {[r['decode_steps_padded_replay'] for r in rs]}  "
              f"frac {[r['padded_replay_fraction'] for r in rs]}  "
              f"noncaptured_eager {[r['noncaptured_but_eager'] for r in rs]}")
        print(f"  captured bs      {rs[0]['captured_bs']}")
        print(f"  config           radix_off={rs[0]['server_disable_radix_cache']} "
              f"strategy={rs[0]['mamba_scheduler_strategy']} "
              f"overlap_off={rs[0]['disable_overlap_schedule']} "
              f"group={rs[0]['server_quant_group_size']} "
              f"lloyd_max={rs[0].get('lloyd_max')} "
              f"P={rs[0].get('hp_prefix_tokens')} R={rs[0].get('hp_recent_tokens')} "
              f"kv_layers={rs[0].get('full_attn_layers')}")
        print(f"  resp chars med   {[r['resp_chars_median'] for r in rs]}")
        print()

    print("=== deltas (Welch t, normal-approx p; n=3 per arm) ===")
    keys = list(data)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            sa = [r["score"] * 100 for r in data[a]]
            sb = [r["score"] * 100 for r in data[b]]
            t, p = welch(sa, sb)
            d = statistics.mean(sa) - statistics.mean(sb)
            ts = f"t={t:+.2f} p={p:.3f}" if t is not None else "t=n/a"
            print(f"  {a:5s} - {b:5s}  score {d:+.2f} pp   {ts}")
            wa = [r["seed_wall_s"] for r in data[a]]
            wb = [r["seed_wall_s"] for r in data[b]]
            wd = statistics.mean(wa) / statistics.mean(wb) - 1.0
            tw, pw = welch(wa, wb)
            tws = f"t={tw:+.2f} p={pw:.3f}" if tw is not None else "t=n/a"
            print(f"  {' ' * 13}  wall  {wd*100:+.1f}%      {tws}")


if __name__ == "__main__":
    main()
