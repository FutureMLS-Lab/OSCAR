#!/usr/bin/env python3
"""Build the eight-item acceptance table for MiniMax-M2.7 across arms.

Takes a list of "<label>=<run_dir>" pairs, runs the same extraction as
analyze.py, and prints one row per arm plus the two corruption-signature
counters (garbled head / repetitive tail) that distinguish bug-1 damage from
ordinary truncation.
"""
import json
import re
import subprocess
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORD = re.compile(r"[A-Za-z]{3,}")


def garbled_head(t: str) -> bool:
    """First 400 chars are not English prose.

    Bug 1 lands on *early* positions, so a request whose position-0 KV was
    clobbered is incoherent from its first token -- distinguishable from a
    normal cap-truncation, which is coherent throughout.
    """
    h = t[:400]
    if len(h) < 80:
        return True
    letters = sum(c.isalpha() and c.isascii() for c in h)
    return len(WORD.findall(h)) < 12 or letters / len(h) < 0.45


def repetitive_tail(t: str) -> bool:
    b = t[-8000:].encode("utf-8", "replace")
    if len(b) < 512:
        return False
    return len(zlib.compress(b, 6)) / len(b) < 0.12


def main():
    rows = []
    for spec in sys.argv[1:]:
        label, _, path = spec.partition("=")
        d = Path(path)
        out = subprocess.run(
            [sys.executable, str(HERE / "analyze.py"), str(d), "--arm", label, "--json"],
            capture_output=True, text=True,
        )
        try:
            r = json.loads(out.stdout)
        except json.JSONDecodeError:
            print(f"!! {label}: analyze failed: {out.stdout[-300:]} {out.stderr[-300:]}")
            continue
        io = d / "io_log.jsonl"
        gh = rt = 0
        if io.is_file():
            for line in io.open():
                line = line.strip()
                if not line:
                    continue
                try:
                    resp = json.loads(line).get("response") or ""
                except json.JSONDecodeError:
                    continue
                gh += garbled_head(resp)
                rt += repetitive_tail(resp)
        r["garbled_head"] = gh
        r["repetitive_tail_n"] = rt
        rows.append(r)

    cols = [
        ("arm", 26), ("score", 8), ("answered", 9), ("responses", 5),
        ("cache_hit_rate", 8), ("cuda_graph_max_bs", 6),
        ("padded_replay_fraction", 8), ("garbled_head", 7),
        ("repetitive_tail_n", 7), ("decode_write_into_hp_prefix", 8),
        ("kv_content_changed", 8), ("hp_prefix_alloc_min_page", 8),
        ("mixed_kv_prefix_tokens", 6), ("mixed_kv_recent_tokens", 7),
        ("max_tokens", 8),
    ]
    hdr = {"padded_replay_fraction": "padfrac", "decode_write_into_hp_prefix": "DWIHP",
           "kv_content_changed": "KVCHG", "hp_prefix_alloc_min_page": "minpage",
           "mixed_kv_prefix_tokens": "sink", "mixed_kv_recent_tokens": "recent",
           "cuda_graph_max_bs": "maxbs", "cache_hit_rate": "hitrate",
           "repetitive_tail_n": "reptail", "garbled_head": "garble",
           "max_tokens": "budget", "responses": "n"}
    print("  ".join(f"{hdr.get(c, c):<{w}}" for c, w in cols))
    print("  ".join("-" * w for _, w in cols))
    for r in rows:
        cells = []
        for c, w in cols:
            v = r.get(c)
            if isinstance(v, float):
                v = f"{v:.4f}"
            cells.append(f"{str(v if v is not None else '-'):<{w}}")
        print("  ".join(cells))
    print()
    for r in rows:
        print(f"{r['arm']}: captured_bs={r.get('captured_bs')} "
              f"decode_bs_hist={r.get('decode_bs_hist')}")


if __name__ == "__main__":
    main()
