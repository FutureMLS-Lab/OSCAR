#!/usr/bin/env python3
"""Arm x seed table for the Qwen3.5-35B-A3B prefix-cache A/B.

Usage: table.py RUNS_DIR arm [arm ...]

Columns are the eight the project holds every model to, plus the evidence
columns that say whether the thing under test actually happened. ``padrep`` is
the joint condition (graph replayed at a non-captured size), not "bs was not
captured" -- see analyze.py.
"""
import json
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = Path(sys.argv[1])
ARMS = sys.argv[2:]

HDR = ("arm/seed", "score", "answ", "noClose", "runaway", "hit%", "wall_s",
       "radix", "grp", "strategy", "ovlpOFF", "padrep", "ncEager", "viol", "p0")
W = (13, 7, 6, 8, 8, 8, 8, 7, 5, 13, 8, 7, 8, 6, 6)


def row(d):
    r = subprocess.run([sys.executable, str(HERE / "analyze.py"), str(d), "--json"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return {"error": r.stderr.strip()[-300:]}
    return json.loads(r.stdout)


print("".join(h.ljust(x) for h, x in zip(HDR, W)))
allrows = {}
allwall = {}
allhit = {}
for arm in ARMS:
    ad = RUNS / arm
    if not ad.is_dir():
        print(f"{arm}: MISSING")
        continue
    scores, walls, hits = [], [], []
    for s in sorted(p for p in ad.glob("seed*") if p.is_dir()):
        r = row(s)
        if "error" in r:
            print(f"{arm}/{s.name}: ERROR {r['error']}")
            continue
        sc = r.get("score")
        if sc is not None:
            scores.append(sc * 100 if sc <= 1.0 else sc)
        if r.get("seed_wall_s"):
            walls.append(r["seed_wall_s"])
        if r.get("cache_hit_rate") is not None:
            hits.append(r["cache_hit_rate"] * 100)
        cells = [
            f"{arm}/{s.name.replace('seed', 's')}",
            f"{sc*100:.2f}" if isinstance(sc, float) else str(sc),
            r.get("answered"), r.get("no_think_close"), r.get("runaway"),
            f"{(r.get('cache_hit_rate') or 0)*100:.3f}",
            r.get("seed_wall_s"),
            r.get("server_disable_radix_cache"),
            r.get("server_quant_group_size"),
            r.get("mamba_scheduler_strategy"),
            r.get("disable_overlap_schedule"),
            r.get("decode_steps_padded_replay"),
            r.get("noncaptured_but_eager"),
            r.get("audit_violation_total"),
            r.get("hp_prefix_page0_ever_allocated"),
        ]
        print("".join(str(c).ljust(x) for c, x in zip(cells, W)))
    allrows[arm] = scores
    allwall[arm] = walls
    allhit[arm] = hits

print()
for arm, sc in allrows.items():
    if not sc:
        continue
    m = statistics.mean(sc)
    sd = statistics.stdev(sc) if len(sc) > 1 else 0.0
    wl = allwall.get(arm) or [0]
    ht = allhit.get(arm) or [0]
    print(f"{arm:13s} n={len(sc)}  score={m:.2f} +/- {sd:.2f}  "
          f"seeds={[round(x, 2) for x in sc]}  "
          f"wall_mean={statistics.mean(wl):.0f}s  hit%={statistics.mean(ht):.3f}")
