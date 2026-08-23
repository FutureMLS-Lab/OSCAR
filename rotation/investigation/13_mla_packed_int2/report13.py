#!/usr/bin/env python3
"""Read the deployment facts back out of a run directory's own logs.

Every column here is read from what the *server* reported, never from what the
job intended to configure. Two of this project's published GLM-5.2 rows were
wrong because of exactly that gap -- an archived arm whose Lloyd-Max setting
could not be recovered, and a "prefix cache off" column that three runs
contradicted at 18-21% hit rate.

``max_total_num_tokens`` is the load-bearing one: if the packed arm has not
moved off the BF16 arm's number, the storage change did not land, whatever the
score says.

    python3 report13.py /shared/mlapacked_gpqa_a /shared/gpqa_glm52_w512_r4 ...
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path


def _first(pat, text, cast=str, default=None):
    m = re.search(pat, text)
    return cast(m.group(1)) if m else default


def read_run(d: Path) -> dict:
    log = d / "server.log"
    text = log.read_text(errors="ignore") if log.exists() else ""
    out: dict = {"run": d.name}

    out["pool"] = _first(r"max_total_num_tokens=(\d+)", text, int)
    out["kv_dtype"] = _first(r"kv_cache_dtype='([a-z0-9_]+)'", text)
    out["radix_off"] = _first(r"disable_radix_cache=(\w+)", text)
    out["max_running"] = _first(r"max_running_requests=(\d+)", text, int)
    out["captured"] = _first(r"Capture cuda graph bs (\[[^\]]*\])", text)
    out["packed"] = "yes" if "[MLAPacked]" in text else "no"
    m = re.search(r"\[MLAPacked\] packed latent storage: (\d+) B/token/layer", text)
    out["bytes_per_token"] = int(m.group(1)) if m else None
    m = re.search(r"worst_rel=([0-9.eE+-]+) over (\d+) writes", text)
    if m:
        out["selfcheck"] = f"worst_rel={float(m.group(1)):.2e} over {m.group(2)} writes"

    # CUDA graph coverage: replay only happens when the step says graph True,
    # and padding only exists in a replay. A non-captured size running eager is
    # not padded -- counting it as such once inverted this project's central
    # conclusion (99.5% reported where the truth was 0%).
    g_true = len(re.findall(r"cuda graph: True", text))
    g_false = len(re.findall(r"cuda graph: False", text))
    out["graph_true"], out["graph_false"] = g_true, g_false
    out["graph_cov"] = (
        f"{100.0 * g_true / (g_true + g_false):.1f}%" if g_true + g_false else "-"
    )
    hist = Counter(int(x) for x in re.findall(r"#running-req: (\d+)", text))
    out["decode_bs_hist"] = dict(hist.most_common(6))
    if out["captured"]:
        cap = set(json.loads(out["captured"]))
        out["bs_outside_captured"] = {k: v for k, v in hist.items() if k not in cap}

    cached = [int(x) for x in re.findall(r"#cached-token: (\d+)", text)]
    new = [int(x) for x in re.findall(r"#new-token: (\d+)", text)]
    if cached and new and sum(cached) + sum(new):
        out["cache_hit"] = f"{100.0 * sum(cached) / (sum(cached) + sum(new)):.2f}%"
    tp = [float(x) for x in re.findall(r"gen throughput \(token/s\): ([0-9.]+)", text)]
    if tp:
        tp_s = sorted(tp)
        out["decode_tok_s_median"] = round(tp_s[len(tp_s) // 2], 1)
        out["decode_tok_s_max"] = round(tp_s[-1], 1)

    ev = d / "eval.log"
    if ev.exists():
        et = ev.read_text(errors="ignore")
        out["score"] = _first(r"gpqa/score\s*\|\s*([0-9.]+)", et, float)
    io = d / "io_log.jsonl"
    if io.exists():
        out["n"] = sum(1 for _ in io.open())

    cfg = d / "config.env"
    if cfg.exists():
        for line in cfg.read_text().splitlines():
            if line.startswith(("HEAD=", "SGLANG_MIXED_KV_RECENT_TOKENS=",
                                "SGLANG_MIXED_KV_PREFIX_TOKENS=", "MLA_PACKED=",
                                "LLOYD_MAX=", "NUM_WORKERS=", "MLA_GROUP_SIZE=")):
                k, _, v = line.partition("=")
                out[k.lower()] = v
    return out


COLS = [
    ("run", "run"), ("packed", "packed"), ("score", "score"), ("n", "n"),
    ("pool", "max_total_num_tokens"), ("bytes_per_token", "B/tok/layer"),
    ("decode_tok_s_median", "decode tok/s (med)"),
    ("graph_cov", "graph coverage"), ("cache_hit", "cache hit"),
    ("kv_dtype", "kv dtype"), ("radix_off", "disable_radix_cache"),
    ("num_workers", "concurrency"),
    ("sglang_mixed_kv_recent_tokens", "recent"),
]


def main() -> None:
    rows = [read_run(Path(p)) for p in sys.argv[1:]]
    if not rows:
        print(__doc__)
        return
    w = [max(len(h), *(len(str(r.get(k, "-"))) for r in rows)) for k, h in COLS]
    print(" | ".join(h.ljust(x) for (_, h), x in zip(COLS, w)))
    print("-+-".join("-" * x for x in w))
    for r in rows:
        print(" | ".join(str(r.get(k, "-")).ljust(x) for (k, _), x in zip(COLS, w)))
    print()
    for r in rows:
        print(f"{r['run']}:")
        for k in ("captured", "decode_bs_hist", "bs_outside_captured",
                  "graph_true", "graph_false", "selfcheck", "head"):
            if r.get(k) is not None:
                print(f"    {k}: {r[k]}")


if __name__ == "__main__":
    main()
