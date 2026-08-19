#!/usr/bin/env python3
"""Build the per-bucket KL ladder page for a kl-probe run.

Two modes:

  analyze  print the run-level numbers metrics.json does not compute — token-weighted
           mean KL per arm and PAIRED bootstrap CIs over pid. Read these before writing
           any prose: they are what the findings cards must be based on.

  build    fill assets/ladder.html.tmpl with the run's data plus the copy you supply,
           and write a self-contained page ready for the Artifact tool.

Why paired: every arm in a run is scored on the identical token set, so prompt-difficulty
variance cancels in the difference. Per-arm marginal CIs are ~3x wider and overlap for
everything, which reads as "underpowered" and is wrong. Compare arms, never CIs.

Usage
-----
    python build_ladder.py analyze --run runs/<name>/<stamp>
    python build_ladder.py analyze --run runs/<name>/<stamp> --floor 0.00042
    python build_ladder.py build   --run runs/<name>/<stamp> --out page.html \
        --title "..." --h1 "..." --intro intro.html --findings cards.json \
        [--extra extra.html] [--footer footer.html] [--meta meta.json]

cards.json is a list of 1-4 objects: {"tone": key|good|flat|warn, "lbl": ..., "big": ...,
"sub": ...}. `sub` may contain inline HTML. tone drives the accent stripe: key = headline,
good = a confirmed win, flat = a null result, warn = a caveat the reader must apply.
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parent / "assets" / "ladder.html.tmpl"
B = 4000
SEED = 0


# ----------------------------------------------------------------- run loading
def load_run(run: Path):
    metrics = json.loads((run / "metrics.json").read_text())
    return metrics, list(metrics["candidates"])


def per_pid(path: Path):
    """(sum kl, count) per pid — the unit the bootstrap resamples."""
    t = pq.read_table(path, columns=["pid", "kl"])
    pid = t["pid"].to_numpy()
    kl = t["kl"].to_numpy().astype(np.float64)
    u = np.unique(pid)
    idx = np.searchsorted(u, pid)
    s = np.zeros(u.size)
    c = np.zeros(u.size)
    np.add.at(s, idx, kl)
    np.add.at(c, idx, 1)
    return s, c, u.size


def bootstrap(run: Path, names: list[str]):
    agg, npid = {}, None
    for n in names:
        p = run / f"tokens_{n}.parquet"
        if not p.is_file():
            continue
        s, c, npid = per_pid(p)
        agg[n] = (s, c)
    if not agg:
        raise SystemExit(f"no tokens_*.parquet in {run} — run the metrics stage first")
    rng = np.random.default_rng(SEED)
    I = rng.integers(0, npid, size=(B, npid))
    boot = {n: s[I].sum(1) / c[I].sum(1) for n, (s, c) in agg.items()}
    point = {n: s.sum() / c.sum() for n, (s, c) in agg.items()}
    return point, boot, npid


# --------------------------------------------------------------------- analyze
def cmd_analyze(args):
    run = Path(args.run)
    _metrics, names = load_run(run)
    point, boot, npid = bootstrap(run, names)
    order = sorted(point, key=point.get)
    floor = args.floor

    print(f"\n{len(point)} arms, {npid} prompts, B={B}, noise floor {floor}")
    print(f"{'arm':30s} {'mean KL':>9s}   {'marginal 95% CI (do NOT rank on this)':>38s}")
    for n in order:
        b = boot[n]
        print(f"{n:30s} {point[n]:9.5f}   [{np.percentile(b, 2.5):.5f},{np.percentile(b, 97.5):.5f}]")

    print(f"\npaired differences vs {order[0]} (best):")
    for n in order[1:]:
        d = boot[n] - boot[order[0]]
        lo, hi = np.percentile(d, [2.5, 97.5])
        obs = point[n] - point[order[0]]
        print(f"  {n:30s} d={obs:+.5f} CI[{lo:+.5f},{hi:+.5f}] {verdict(obs, lo, hi, floor)}")

    if args.pairs:
        print("\nrequested contrasts:")
        for spec in args.pairs:
            a, b_ = spec.split(":", 1)
            if a not in boot or b_ not in boot:
                print(f"  {spec}: unknown arm")
                continue
            d = boot[a] - boot[b_]
            lo, hi = np.percentile(d, [2.5, 97.5])
            obs = point[a] - point[b_]
            print(f"  {a:26s} - {b_:26s} d={obs:+.5f} CI[{lo:+.5f},{hi:+.5f}] {verdict(obs, lo, hi, floor)}")


def verdict(obs, lo, hi, floor):
    if abs(obs) < floor:
        return "TIE (inside noise floor)"
    return "SEP" if (lo > 0 or hi < 0) else "overlap"


# ----------------------------------------------------------------- build payload
def arms_table(repo_root: Path) -> dict:
    """Layer sets from scripts/make_skip_variant.py — the single source of truth."""
    p = repo_root / "scripts" / "make_skip_variant.py"
    if not p.is_file():
        return {}
    spec = importlib.util.spec_from_file_location("_msv", p)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        return dict(getattr(mod, "ARMS", {}))
    except Exception:
        return {}


def fmt_layers(ls: list[int]) -> str:
    """Collapse a sorted layer list to runs: [3,4,5,12,13] -> '3-5,12-13'."""
    if not ls:
        return ""
    out, start, prev = [], ls[0], ls[0]
    for x in ls[1:] + [None]:
        if x == prev + 1:
            prev = x
            continue
        out.append(str(start) if start == prev else f"{start}–{prev}")
        if x is None:
            break
        start = prev = x
    return ",".join(out)


def build_payload(run: Path, repo_root: Path, meta_over: dict) -> dict:
    metrics, names = load_run(run)
    ARMS = arms_table(repo_root)
    cands = []
    for i, n in enumerate(names):
        m = metrics["candidates"][n]
        bs = m["aggregate"]["buckets"]
        tot = sum(b["num_tokens"] for b in bs)
        w = lambda f: sum(f(b) * b["num_tokens"] for b in bs) / tot  # noqa: E731
        overall = {
            "b": -1, "n": tot,
            "kl": round(w(lambda b: b["kl"]["mean"]), 5),
            "p99": round(w(lambda b: b["kl"]["p99"]), 4),
            "mx": round(max(b["kl"]["max"] for b in bs), 3),
            "t1": round(100 * w(lambda b: b["top1_agreement"]), 2),
            "t5": round(100 * w(lambda b: b["overlap5"]), 2),
        }
        buckets = [overall] + [
            {"b": b["bucket"], "n": b["num_tokens"], "kl": round(b["kl"]["mean"], 5),
             "p99": round(b["kl"]["p99"], 4), "mx": round(b["kl"]["max"], 3),
             "t1": round(100 * b["top1_agreement"], 2), "t5": round(100 * b["overlap5"], 2)}
            for b in bs
        ]
        path = m["model_path"]
        ov = meta_over.get(n, {})
        layers = ARMS.get(n)
        k = ov.get("k", len(layers) if layers else _k_from_name(n))
        cands.append({
            "name": n,
            "full": ov.get("full", Path(path).name),
            "hue": ov.get("hue", round(360 * i / max(1, len(names)))),
            "origin": ov.get("origin", "in-house" if "mxfp8skip" in Path(path).name else "hf"),
            "k": k,
            "layers": ov.get("layers", fmt_layers(layers) if layers else "—"),
            "new": int(ov.get("new", 0)),
            "shape": ov.get("shape", ""),
            # Free-text second line under the model name. Defaults in the template to the
            # MiniMax-M3 phrasing ("K=… · MXFP8 layers …"); set it in --meta for a study
            # whose arms differ along some other axis (e.g. quantized surface, not layers).
            "sub": ov.get("sub", ""),
            "buckets": buckets,
        })
    first = metrics["candidates"][names[0]]["aggregate"]["buckets"]
    bmeta = [{"bucket": -1, "pos_range": "all positions"}] + [
        {"bucket": b["bucket"], "pos_range": b["pos_range"]} for b in first]
    return {
        "ref": Path(metrics["reference"]).name,
        "run": run.name,
        "bucket_size": metrics["bucket_size"],
        "nbuckets": len(bmeta) - 1,
        "buckets": bmeta,
        "candidates": cands,
    }


def _k_from_name(n: str) -> int:
    m = re.search(r"mxfp8skip(\d+)-", n)
    return int(m.group(1)) if m else 0


# --------------------------------------------------------------------- findings
TONES = {"key", "good", "flat", "warn"}


def render_findings(cards: list[dict]) -> str:
    if not cards:
        return '<div class="findings"></div>'
    out = ['<div class="findings">']
    for c in cards:
        tone = c.get("tone", "key")
        if tone not in TONES:
            raise SystemExit(f"tone must be one of {sorted(TONES)}, got {tone!r}")
        out.append(f'  <div class="f {tone}">')
        out.append(f'    <div class="lbl">{html.escape(c["lbl"])}</div>')
        out.append(f'    <div class="big">{html.escape(c["big"])}</div>')
        out.append(f'    <div class="sub">{c["sub"]}</div>')  # inline HTML allowed
        out.append("  </div>")
    out.append("</div>")
    return "\n".join(out)


# ------------------------------------------------------------------------ build
def cmd_build(args):
    run = Path(args.run)
    repo_root = Path(args.repo_root) if args.repo_root else run.parents[2]
    meta = json.loads(Path(args.meta).read_text()) if args.meta else {}
    payload = build_payload(run, repo_root, meta)

    def read(opt, default=""):
        if not opt:
            return default
        p = Path(opt)
        return p.read_text() if p.is_file() else opt

    cards = json.loads(Path(args.findings).read_text()) if args.findings else []
    tmpl = TEMPLATE.read_text()
    page = (tmpl
            .replace("{{TITLE}}", html.escape(args.title))
            .replace("{{H1}}", html.escape(args.h1 or args.title))
            .replace("{{INTRO}}", read(args.intro, '  <p class="note"></p>'))
            .replace("{{FINDINGS}}", render_findings(cards))
            .replace("{{EXTRA}}", read(args.extra))
            .replace("{{FOOTER}}", read(args.footer, "<footer></footer>"))
            .replace("{{DATA}}", json.dumps(payload, separators=(",", ":"))))
    for tok in ("{{TITLE}}", "{{H1}}", "{{INTRO}}", "{{FINDINGS}}", "{{EXTRA}}", "{{FOOTER}}", "{{DATA}}"):
        assert tok not in page, f"unfilled placeholder {tok}"
    for bad in ("<!doctype", "<html", "<body"):
        assert bad not in page.lower(), f"{bad} must not appear — the Artifact tool wraps the file"
    Path(args.out).write_text(page)
    print(f"wrote {args.out}  ({len(page):,} bytes, {len(payload['candidates'])} arms, "
          f"{payload['nbuckets']} buckets)")
    print("publish with the Artifact tool: file_path=<out>, favicon, description")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="token-weighted means + paired bootstrap CIs")
    a.add_argument("--run", required=True)
    a.add_argument("--floor", type=float, default=0.00042,
                   help="noise floor; gaps below it are reported as TIE (default: the "
                        "0.00042 measured on the K=11 one-layer arm pair)")
    a.add_argument("--pairs", nargs="*", default=[], metavar="A:B",
                   help="extra contrasts to print, e.g. armA:armB")
    a.set_defaults(func=cmd_analyze)

    b = sub.add_parser("build", help="render the page")
    b.add_argument("--run", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--title", required=True)
    b.add_argument("--h1")
    b.add_argument("--intro", help="HTML file or literal string (a <p class=note> block)")
    b.add_argument("--findings", help="JSON file: list of 1-4 cards")
    b.add_argument("--extra", help="HTML file for an extra <section> above the tables")
    b.add_argument("--footer", help="HTML file for the <footer> block")
    b.add_argument("--meta", help="JSON file: per-arm overrides (layers/full/origin/new/shape/hue)")
    b.add_argument("--repo-root", help="defaults to two levels above the run dir")
    b.set_defaults(func=cmd_build)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
