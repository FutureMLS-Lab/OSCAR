"""`probe` — compare two or more SGLang/vLLM-served models token-distribution-wise.

    probe run -c configs/qwen3.6-27b.yaml           # end-to-end (all candidates)
    probe run -c ... --null                          # ref vs itself (pipeline sanity check)
    probe servers up|down|logs -c ...                # manage the docker servers only
    probe prompts|generate|ingest|score|metrics -c . # individual stages

One reference is scored once and every candidate is compared against it. When
allow_generations is false, the reference generations are ingested from a
dataset instead of generated. Each invocation writes to a fresh
runs/<run_name>/<datetime>/ directory.

Design rationale for the pipeline decisions below lives in
docs/precomputed-generations.md.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from . import servers
from .config import RunCfg, get_top_k, load_config
from .engine import wait_healthy
from .generate import run_generate
from .ingest import build_ingest
from .metrics import (
    aggregate,
    compare_scores,
    prefill_vs_decode_diagnostic,
    worst_positions,
)
from .prompts import build_prompts, check_tokenizers_match
from .score import run_score

console = Console()


def _load(args) -> RunCfg:
    cfg = load_config(args.config)
    # Resolve scoring.top_logprobs_num to an int so scoring and the launcher
    # both see an int: an explicit int passes through; "auto" derives from the
    # reference generation_config top_k (fallback 40).
    tln = cfg.scoring.top_logprobs_num
    cfg.scoring.top_logprobs_num = get_top_k(cfg.reference.model_path) if tln == "auto" else tln
    if getattr(args, "null", False):
        cfg.candidates = [dataclasses.replace(cfg.reference, name="ref-null")]
        cfg.runtime.candidates = {
            "ref-null": dataclasses.replace(cfg.runtime.ref, base_url=cfg.runtime.ref.url)
        }
        cfg.run_name += "-null"
    return cfg


def stage_servers_up(cfg: RunCfg, only: str | None = None, cand_name: str | None = None) -> None:
    # `servers.up` prewarms the checkpoint into page cache then launches the
    # container detached (`compose up -d` returns as soon as it starts), so the
    # split below is meaningful: `prewarm+launch` ≈ safetensors read, and the
    # health wait ≈ in-container load (weight sharding, kernel autotune, CUDA
    # graph capture). These cold-starts dominate a run's wall clock, so time
    # each one to make the cost visible.
    console.print(f"[dim]server launch started {datetime.now():%H:%M:%S}[/dim]")
    t_start = time.monotonic()
    urls = servers.up(cfg, only=only, cand_name=cand_name)
    up_elapsed = time.monotonic() - t_start
    for url in urls:
        console.print(f"waiting for {url} …")
        t_wait = time.monotonic()
        asyncio.run(wait_healthy(url))
        ready = time.monotonic()
        console.print(
            f"[green]healthy:[/green] {url} "
            f"[dim](ready in {ready - t_start:.0f}s — "
            f"prewarm+launch {up_elapsed:.0f}s, health wait {ready - t_wait:.0f}s)[/dim]"
        )


def stage_ingest(cfg: RunCfg) -> Path:
    out = cfg.run_dir / "generations.jsonl"
    n = build_ingest(cfg, out)
    console.print(f"[green]ingested:[/green] {n} reference generations -> {out}")
    return out


def stage_prompts(cfg: RunCfg) -> Path:
    out = cfg.run_dir / "prompts.jsonl"
    n = build_prompts(cfg, out)
    console.print(f"[green]prompts:[/green] {n} -> {out}")
    return out


def stage_generate(cfg: RunCfg) -> Path:
    out = cfg.run_dir / "generations.jsonl"
    console.print(f"[dim]generation requests dispatched {datetime.now():%H:%M:%S}[/dim]")
    t0 = time.perf_counter()
    n = asyncio.run(run_generate(cfg, cfg.run_dir / "prompts.jsonl", out))
    console.print(
        f"[green]generations:[/green] {n} -> {out} [dim]({time.perf_counter() - t0:.0f}s)[/dim]"
    )
    return out


def stage_score(cfg: RunCfg, spec, runtime, out: Path) -> Path:
    console.print(f"[dim]scoring {spec.name}: requests dispatched {datetime.now():%H:%M:%S}[/dim]")
    t0 = time.perf_counter()
    n = asyncio.run(run_score(cfg, spec, runtime, cfg.run_dir / "generations.jsonl", out))
    dt = time.perf_counter() - t0
    rate = f", {n / dt:,.0f} pos/s" if dt > 0 else ""
    console.print(
        f"[green]scored {spec.name}:[/green] {n} positions -> {out} [dim]({dt:.0f}s{rate})[/dim]"
    )
    return out


def _make_generations(cfg: RunCfg) -> None:
    """Produce generations.jsonl — by ingest or by generating on the ref."""
    if cfg.allow_generations:
        stage_servers_up(cfg, only="ref")
        stage_prompts(cfg)
        stage_generate(cfg)
    else:
        stage_ingest(cfg)


def _resolve_reuse_dir(cfg: RunCfg, value: str) -> Path:
    """Turn a --reuse value into a concrete prior-run dir. An explicit path is
    used as-is; the sentinel 'auto' (bare --reuse) picks the most recent run
    under runs/<run_name>/ (excluding this run) that holds generations.jsonl."""
    if value != "auto":
        return Path(value)
    base = Path(cfg.output_dir) / cfg.run_name
    prior = sorted(
        p
        for p in base.glob("*")
        if p.is_dir() and p != cfg.run_dir and (p / "generations.jsonl").is_file()
    )
    if not prior:
        raise SystemExit(f"--reuse: no prior run under {base} has generations.jsonl")
    return prior[-1]


def _reuse_scores(cfg: RunCfg, src: Path) -> set[str]:
    """Copy reusable artifacts from a prior run dir into this run and return the
    set of model names ('ref' and/or candidate names) whose scores were reused,
    so the caller skips re-scoring them and only computes what's missing.

    Requires src/generations.jsonl — the token sequences everything is scored
    against; it is copied in and reused verbatim (so any freshly-scored model
    aligns with the reused ones). Reuses scores_ref.parquet and any
    scores_<cand>.parquet already present. When src/metrics.json exists we verify
    the reference checkpoint, scoring.top_logprobs_num, and each reused
    candidate's model_path match this config — refusing on a ref/top-k mismatch
    and skipping (re-scoring) a candidate whose checkpoint changed. Artifacts are
    copied (not referenced) so this run stays a self-contained run dir."""
    gens = src / "generations.jsonl"
    if not gens.is_file():
        raise SystemExit(f"--reuse {src}: missing generations.jsonl")
    prior: dict = {}
    meta = src / "metrics.json"
    if meta.is_file():
        prior = json.loads(meta.read_text())
        if prior.get("reference") not in (None, cfg.reference.model_path):
            raise SystemExit(
                f"--reuse: source reference {prior['reference']!r} != config "
                f"reference {cfg.reference.model_path!r} — scores are not comparable"
            )
        if prior.get("top_logprobs_num") not in (None, cfg.scoring.top_logprobs_num):
            raise SystemExit(
                f"--reuse: source top_logprobs_num {prior['top_logprobs_num']} != "
                f"config {cfg.scoring.top_logprobs_num} — top-k support differs"
            )
    shutil.copy2(gens, cfg.run_dir / "generations.jsonl")
    console.print(f"[green]reusing[/green] from {src} [dim](generations + existing scores)[/dim]")
    reused: set[str] = set()
    if (src / "scores_ref.parquet").is_file():
        shutil.copy2(src / "scores_ref.parquet", cfg.run_dir / "scores_ref.parquet")
        reused.add("ref")
        console.print("  [green]reused ref[/green] [dim](skipping ref score)[/dim]")
    prior_cands = prior.get("candidates", {}) if isinstance(prior, dict) else {}
    for cand in cfg.candidates:
        sp = src / f"scores_{cand.name}.parquet"
        if not sp.is_file():
            continue
        prior_mp = prior_cands.get(cand.name, {}).get("model_path")
        if prior_mp not in (None, cand.model_path):
            console.print(
                f"  [yellow]not reusing {cand.name}[/yellow]: source model_path "
                f"{prior_mp!r} != {cand.model_path!r} — will re-score"
            )
            continue
        shutil.copy2(sp, cfg.run_dir / f"scores_{cand.name}.parquet")
        reused.add(cand.name)
        console.print(f"  [green]reused {cand.name}[/green] [dim](skipping score)[/dim]")
    return reused


def _candidate_metrics(cfg: RunCfg, cand, tokenizer_note: str) -> dict:
    run_dir = cfg.run_dir
    tokens = compare_scores(
        run_dir / "scores_ref.parquet",
        run_dir / f"scores_{cand.name}.parquet",
        run_dir / f"tokens_{cand.name}.parquet",
    )
    diagnostics = {"tokenizer_check": tokenizer_note}
    # prefill-vs-decode floor only exists when we generated ourselves.
    if cfg.allow_generations:
        diagnostics["prefill_vs_decode"] = prefill_vs_decode_diagnostic(
            run_dir / "generations.jsonl", run_dir / "scores_ref.parquet"
        )
    return {
        "model_path": cand.model_path,
        "aggregate": aggregate(tokens, cfg.metrics.bucket_size),
        "worst_positions": worst_positions(
            cfg.reference.model_path,
            run_dir / f"tokens_{cand.name}.parquet",
            run_dir / "generations.jsonl",
            run_dir / "scores_ref.parquet",
            run_dir / f"scores_{cand.name}.parquet",
            cfg.metrics.worst_n,
        ),
        "diagnostics": diagnostics,
    }


def _write_metrics(cfg: RunCfg, per_candidate: dict) -> Path:
    metrics = {
        "run_name": cfg.run_name,
        "reference": cfg.reference.model_path,
        "allow_generations": cfg.allow_generations,
        "top_logprobs_num": cfg.scoring.top_logprobs_num,
        "bucket_size": cfg.metrics.bucket_size,
        "candidates": per_candidate,
    }
    out = cfg.run_dir / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    return out


def _print_buckets(cfg: RunCfg, per_candidate: dict) -> None:
    """End-of-run summary: one row per (candidate, bucket) vs the reference."""
    table = Table(
        title=f"{cfg.run_name}: per-bucket KL(ref‖candidate) — ref = {cfg.reference.model_path}"
    )
    table.add_column("candidate")
    table.add_column("bucket (pos)")
    table.add_column("mean KL", justify="right")
    table.add_column("p99 / max KL", justify="right")
    table.add_column("top-1 agree", justify="right")
    table.add_column("tokens", justify="right")
    table.add_column("seqs", justify="right")
    for name, m in per_candidate.items():
        for b in m["aggregate"]["buckets"]:
            kl = b["kl"]
            table.add_row(
                name,
                b["pos_range"],
                f"{kl['mean']:.5f}",
                f"{kl['p99']:.4f} / {kl['max']:.3f}",
                f"{b['top1_agreement']:.2%}",
                f"{b['num_tokens']:,}",
                f"{b['num_sequences']:,}",
            )
    console.print(table)
    console.print(
        f"[dim]full per-bucket + worst-position detail: {cfg.run_dir / 'metrics.json'}[/dim]"
    )


def _tokenizer_note(cfg: RunCfg) -> str:
    """Every candidate must share the reference tokenizer for token-level KL."""
    problems = []
    for cand in cfg.candidates:
        if cand.model_path == cfg.reference.model_path:
            continue
        for p in check_tokenizers_match(cfg.reference.model_path, cand.model_path):
            problems.append(f"[{cand.name}] {p}")
    if problems:
        raise SystemExit(
            "tokenizer mismatch vs reference — token-level comparison is invalid:\n  "
            + "\n  ".join(problems)
        )
    return "vocab + probe encodings identical across candidates"


def cmd_run(args) -> None:
    reuse = getattr(args, "reuse", None)
    if reuse and getattr(args, "null", False):
        raise SystemExit("--reuse and --null are mutually exclusive")
    cfg = _load(args)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[dim]run dir: {cfg.run_dir}[/dim]")
    tokenizer_note = _tokenizer_note(cfg)
    try:
        # `reused` holds model names ('ref' + candidate names) whose scores were
        # copied from a prior run; those are skipped below.
        reused: set[str] = set()
        if reuse:
            # Reuse a prior run's generations + any already-computed scores. Only
            # the models missing from that run are launched and scored.
            reused = _reuse_scores(cfg, _resolve_reuse_dir(cfg, reuse))
        else:
            # 1. generations.jsonl (ingest or generate on ref)
            _make_generations(cfg)
        # 2. score the reference ONCE unless it was reused, then free the GPUs.
        if "ref" not in reused:
            # ingest mode (and any --reuse without ref scores) must bring the ref
            # up here; generate mode already brought it up in _make_generations.
            if not cfg.allow_generations or reuse:
                stage_servers_up(cfg, only="ref")
            stage_score(cfg, cfg.reference, cfg.runtime.ref, cfg.run_dir / "scores_ref.parquet")
            # Free the ref GPUs before candidates — UNLESS a candidate reuses the ref
            # server (e.g. --null, or an external base_url pointing at it), in which
            # case tearing it down would kill the server that candidate scores against.
            ref_shared = any(
                cfg.runtime.candidates[c.name].url == cfg.runtime.ref.url for c in cfg.candidates
            )
            if cfg.runtime.ref.managed and not ref_shared:
                servers.down(cfg, only="ref")
        # 3. each candidate: (reuse | up -> score -> down) -> metrics vs ref
        per: dict = {}
        for cand in cfg.candidates:
            runtime = cfg.runtime.candidates[cand.name]
            if cand.name in reused:
                console.print(f"[dim]reusing {cand.name} scores (skipping launch + score)[/dim]")
            else:
                if runtime.managed:
                    stage_servers_up(cfg, only="cand", cand_name=cand.name)
                stage_score(cfg, cand, runtime, cfg.run_dir / f"scores_{cand.name}.parquet")
                if runtime.managed:
                    servers.down(cfg, only="cand", cand_name=cand.name)
            per[cand.name] = _candidate_metrics(cfg, cand, tokenizer_note)
            buckets = per[cand.name]["aggregate"]["buckets"]
            tot = sum(b["num_tokens"] for b in buckets)
            console.print(
                f"[bold green]{cand.name}: scored[/bold green] "
                f"({tot:,} tokens over {len(buckets)} bucket(s))"
            )
        _write_metrics(cfg, per)
        _print_buckets(cfg, per)
    finally:
        # --down must also fire when a stage fails — a dead run must not leave
        # servers squatting on the GPUs
        if args.down:
            servers.down(cfg)


def cmd_servers(args) -> None:
    cfg = _load(args)
    if args.action == "up":
        stage_servers_up(cfg, only=args.only, cand_name=args.cand)
    elif args.action == "down":
        servers.down(cfg, only=args.only, cand_name=args.cand)
    elif args.action == "logs":
        servers.logs(cfg, follow=args.follow, cand_name=args.cand)


def _score_target(cfg: RunCfg, model: str):
    """Resolve `--model` (ref | candidate name) to (spec, runtime, out_path)."""
    if model == "ref":
        return cfg.reference, cfg.runtime.ref, cfg.run_dir / "scores_ref.parquet"
    spec, runtime = servers._selected_cand(cfg, model)
    return spec, runtime, cfg.run_dir / f"scores_{model}.parquet"


def cmd_stage(args) -> None:
    cfg = _load(args)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[dim]run dir: {cfg.run_dir}[/dim]")
    if args.stage == "prompts":
        stage_prompts(cfg)
    elif args.stage == "ingest":
        stage_ingest(cfg)
    elif args.stage == "generate":
        stage_generate(cfg)
    elif args.stage == "score":
        role = args.model
        spec, runtime, out = _score_target(cfg, role)
        stage_score(cfg, spec, runtime, out)
    elif args.stage == "metrics":
        note = _tokenizer_note(cfg)
        per = {cand.name: _candidate_metrics(cfg, cand, note) for cand in cfg.candidates}
        _write_metrics(cfg, per)
        _print_buckets(cfg, per)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-c", "--config", required=True)
    p.add_argument(
        "--null",
        action="store_true",
        help="score the reference against itself (pipeline sanity check)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="probe", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="full pipeline")
    _add_common(p_run)
    p_run.add_argument("--down", action="store_true", help="stop servers when done")
    p_run.add_argument(
        "--reuse",
        nargs="?",
        const="auto",
        default=None,
        metavar="RUN_DIR",
        help="reuse a prior run's generations + already-computed scores (ref and "
        "candidates), scoring only the models missing from it. Bare --reuse uses "
        "the latest run under runs/<run_name>/; or pass a specific RUN_DIR. Must "
        "share the reference checkpoint and scoring.top_logprobs_num.",
    )
    p_run.set_defaults(func=cmd_run)

    p_srv = sub.add_parser("servers", help="manage the model servers")
    p_srv.add_argument("action", choices=["up", "down", "logs"])
    p_srv.add_argument(
        "--only",
        choices=["ref", "cand"],
        default=None,
        help="start/stop just one server (sequential staging)",
    )
    p_srv.add_argument(
        "--cand",
        default=None,
        help="candidate name to route through the 'cand' slot (default: first)",
    )
    p_srv.add_argument("-f", "--follow", action="store_true")
    _add_common(p_srv)
    p_srv.set_defaults(func=cmd_servers)

    for stage in ("prompts", "ingest", "generate", "score", "metrics"):
        p = sub.add_parser(stage, help=f"run only the {stage} stage")
        _add_common(p)
        if stage == "score":
            p.add_argument("--model", required=True, help="'ref' or a candidate name")
        p.set_defaults(func=cmd_stage, stage=stage)

    args = parser.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
