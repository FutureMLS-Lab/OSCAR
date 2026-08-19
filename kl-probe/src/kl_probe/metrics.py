"""Per-position distribution comparison and aggregation.

KL estimator: both distributions are RENORMALIZED over the reference model's
top-k support, then KL(ref~ || cand~) is computed there. Renormalizing over a
common support makes the estimate a true KL divergence (non-negative by Gibbs'
inequality) between the conditional distributions "given the next token is in
ref's top-k" — a well-defined proxy for the full-vocab KL as long as ref's
top-k covers most of the mass. The uncovered ref tail mass is reported next to
the headline number so truncation stays visible. For a ref-top-k token missing
from cand's list, cand's probability is clamped to cand's k-th (smallest
returned) probability; the actual next token's cand logprob is always exact
(SGLang returns it separately) and is patched into the support.

Metrics are reported per bucket: a bucket is `bucket_size` contiguous token
positions, keyed on the ABSOLUTE position in the sequence. Each bucket carries
num_tokens and num_sequences so length-attrition stays visible (later buckets
only contain the longer sequences).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

TOKENS_SCHEMA = pa.schema(
    [
        ("pid", pa.int32()),
        ("pos", pa.int32()),
        ("token_id", pa.int32()),
        ("kl", pa.float32()),
        ("top1_agree", pa.bool_()),
        ("overlap5", pa.float32()),
        ("overlap_k", pa.float32()),
        ("delta_logprob", pa.float32()),  # ref - cand, actual token
        ("ref_top1_rank_in_cand", pa.int32()),  # -1 = beyond cand's top-k
        ("ref_tail_mass", pa.float32()),
        ("ref_top1", pa.int32()),
        ("cand_top1", pa.int32()),
    ]
)


def _require_finite_logprobs(name: str, values) -> None:
    vals = list(values)
    if not vals:
        raise ValueError(f"{name} must contain at least one logprob")
    for i, lp in enumerate(vals):
        if not math.isfinite(float(lp)):
            raise ValueError(f"{name} contains non-finite logprob at index {i}: {lp!r}")


def truncated_kl(
    ref_top_ids: list[int],
    ref_top_logprobs: list[float],
    cand_top_ids: list[int],
    cand_top_logprobs: list[float],
    exact_cand: dict[int, float] | None = None,
) -> tuple[float, float]:
    """KL(ref~ || cand~) with both renormalized over ref's top-k support
    (non-negative). Returns (kl, ref_tail_mass)."""
    _require_finite_logprobs("ref_top_logprobs", ref_top_logprobs)
    _require_finite_logprobs("cand_top_logprobs", cand_top_logprobs)
    if exact_cand:
        for tid, lp in exact_cand.items():
            if not math.isfinite(float(lp)):
                raise ValueError(f"exact_cand contains non-finite logprob for token {tid}: {lp!r}")
    cand_map = dict(zip(cand_top_ids, cand_top_logprobs))
    if exact_cand:
        cand_map.update(exact_cand)
    cand_floor = min(cand_top_logprobs)
    p = [math.exp(lp) for lp in ref_top_logprobs]
    q = [math.exp(cand_map.get(tid, cand_floor)) for tid in ref_top_ids]
    p_sum, q_sum = sum(p), sum(q)
    kl = sum((pi / p_sum) * (math.log(pi / p_sum) - math.log(qi / q_sum)) for pi, qi in zip(p, q))
    return max(0.0, kl), max(0.0, 1.0 - p_sum)


def overlap_at(ref_ids: list[int], cand_ids: list[int], k: int) -> float:
    k = min(k, len(ref_ids), len(cand_ids))
    if k == 0:
        return float("nan")
    return len(set(ref_ids[:k]) & set(cand_ids[:k])) / k


def _sorted_rows(path: Path) -> list[dict]:
    table = pq.read_table(path).sort_by([("pid", "ascending"), ("pos", "ascending")])
    return table.to_pylist()


def compare_scores(ref_path: Path, cand_path: Path, out_path: Path) -> pa.Table:
    """Join the two score files position-by-position and emit per-token metrics.

    Both files are sorted by (pid, pos) first, since score.py writes rows in
    completion order (streaming). After sorting they align row-for-row."""
    ref = _sorted_rows(ref_path)
    cand = _sorted_rows(cand_path)
    if len(ref) != len(cand):
        raise RuntimeError(f"score row count mismatch: ref={len(ref)} cand={len(cand)}")

    rows = []
    for r, c in zip(ref, cand):
        if (r["pid"], r["pos"], r["token_id"]) != (c["pid"], c["pos"], c["token_id"]):
            raise RuntimeError(
                f"misaligned score rows: ref={r['pid']}/{r['pos']}/{r['token_id']} "
                f"cand={c['pid']}/{c['pos']}/{c['token_id']}"
            )
        for label, row in (("ref", r), ("cand", c)):
            if not math.isfinite(float(row["logprob"])):
                raise ValueError(
                    f"{label} score contains non-finite actual-token logprob at "
                    f"{row['pid']}/{row['pos']}: {row['logprob']!r}"
                )
        # ref top-k lists are sorted by logprob desc (sglang returns them ranked)
        kl, tail = truncated_kl(
            r["top_ids"],
            r["top_logprobs"],
            c["top_ids"],
            c["top_logprobs"],
            exact_cand={c["token_id"]: c["logprob"]},
        )
        ref_top1 = r["top_ids"][0]
        rank = c["top_ids"].index(ref_top1) if ref_top1 in c["top_ids"] else -1
        rows.append(
            {
                "pid": r["pid"],
                "pos": r["pos"],
                "token_id": r["token_id"],
                "kl": kl,
                "top1_agree": ref_top1 == c["top_ids"][0],
                "overlap5": overlap_at(r["top_ids"], c["top_ids"], 5),
                "overlap_k": overlap_at(r["top_ids"], c["top_ids"], len(r["top_ids"])),
                "delta_logprob": r["logprob"] - c["logprob"],
                "ref_top1_rank_in_cand": rank,
                "ref_tail_mass": tail,
                "ref_top1": ref_top1,
                "cand_top1": c["top_ids"][0],
            }
        )
    table = pa.Table.from_pylist(rows, schema=TOKENS_SCHEMA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    pq.write_table(table, tmp)
    tmp.replace(out_path)
    return table


def _percentiles(x: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(np.max(x)),
    }


def _bucket_metrics(kl, agree, overlap5, overlap_k, adl, tail, pids) -> dict:
    """The metric set computed over one bucket's positions."""
    return {
        "num_tokens": int(len(kl)),
        "num_sequences": int(len(np.unique(pids))),
        "kl": _percentiles(kl),
        "top1_agreement": float(np.mean(agree)),
        "overlap5": float(np.nanmean(overlap5)),
        "overlap_k": float(np.nanmean(overlap_k)),
        "delta_logprob_actual_mean": float(np.mean(adl)),
        "mean_ref_tail_mass": float(np.mean(tail)),
    }


def aggregate(tokens: pa.Table, bucket_size: int = 4096) -> dict:
    """Per-bucket metrics only. A bucket is `bucket_size` contiguous ABSOLUTE
    positions (bucket = pos // bucket_size); each bucket carries its own KL
    percentiles, agreement/overlap, and token/sequence counts."""
    if tokens.num_rows == 0:
        raise RuntimeError("no compared positions — did every prompt produce an empty generation?")
    kl = tokens["kl"].to_numpy()
    pids = tokens["pid"].to_numpy()
    pos = tokens["pos"].to_numpy()
    agree = tokens["top1_agree"].to_numpy()
    overlap5 = tokens["overlap5"].to_numpy()
    overlap_k = tokens["overlap_k"].to_numpy()
    adl = np.abs(tokens["delta_logprob"].to_numpy())
    tail = tokens["ref_tail_mass"].to_numpy()

    bidx = pos // bucket_size
    buckets = []
    for b in np.unique(bidx):
        m = bidx == b
        entry = {"bucket": int(b), "pos_range": f"{b * bucket_size}-{(b + 1) * bucket_size - 1}"}
        entry.update(
            _bucket_metrics(kl[m], agree[m], overlap5[m], overlap_k[m], adl[m], tail[m], pids[m])
        )
        buckets.append(entry)
    return {"bucket_size": bucket_size, "buckets": buckets}


def worst_positions(
    reference_model_path: str,
    tokens_path: Path,
    generations_path: Path,
    ref_scores_path: Path,
    cand_scores_path: Path,
    n_worst: int = 20,
) -> list[dict]:
    """The highest-KL positions with decoded context and ref/cand top-3 — the
    'where did they diverge' drill-down. `pos` is absolute, so context is
    full_ids[pos-20:pos]."""
    if n_worst <= 0:
        return []
    from .prompts import load_tokenizer

    tok = load_tokenizer(reference_model_path)
    gens = {rec["pid"]: rec for rec in map(json.loads, generations_path.open())}
    tokens = pq.read_table(tokens_path).to_pylist()
    worst = sorted(tokens, key=lambda r: r["kl"], reverse=True)[:n_worst]
    want = {(r["pid"], r["pos"]) for r in worst}
    ref_rows = {
        (r["pid"], r["pos"]): r
        for r in pq.read_table(ref_scores_path).to_pylist()
        if (r["pid"], r["pos"]) in want
    }
    cand_rows = {
        (r["pid"], r["pos"]): r
        for r in pq.read_table(cand_scores_path).to_pylist()
        if (r["pid"], r["pos"]) in want
    }

    def top3(row: dict) -> list[dict]:
        pairs = list(zip(row["top_ids"], row["top_logprobs"]))[:3]
        return [{"token": tok.decode([tid]), "prob": math.exp(lp)} for tid, lp in pairs]

    out = []
    for r in worst:
        key = (r["pid"], r["pos"])
        gen = gens[r["pid"]]
        full = gen["prompt_ids"] + gen["output_ids"]
        pos = r["pos"]  # absolute
        out.append(
            {
                "pid": r["pid"],
                "pos": pos,
                "kl": r["kl"],
                "context": tok.decode(full[max(0, pos - 20) : pos]),
                "actual": tok.decode([r["token_id"]]),
                "ref_top3": top3(ref_rows[key]) if key in ref_rows else [],
                "cand_top3": top3(cand_rows[key]) if key in cand_rows else [],
            }
        )
    return out


def prefill_vs_decode_diagnostic(generations_path: Path, ref_scores_path: Path) -> dict:
    """Compare the ref model's decode-time logprobs (from generation) with its
    prefill-scored logprobs at the same positions. Large deltas mean the prefill
    and decode kernel paths disagree, which bounds how much of the measured
    ref-vs-cand gap could be kernel noise rather than quantization.

    Only meaningful when we generated the completion ourselves (decode_logprobs
    present); the caller gates this on allow_generations."""
    decode: dict[tuple[int, int], float] = {}
    for line in generations_path.open():
        rec = json.loads(line)
        for i, lp in enumerate(rec.get("decode_logprobs") or []):
            # decode position i is the (i+1)-th token; absolute pos = len(prompt)+i.
            decode[(rec["pid"], len(rec["prompt_ids"]) + i)] = lp
    scores = pq.read_table(ref_scores_path, columns=["pid", "pos", "logprob"]).to_pylist()
    deltas = [
        abs(row["logprob"] - decode[(row["pid"], row["pos"])])
        for row in scores
        if (row["pid"], row["pos"]) in decode
    ]
    arr = np.array(deltas) if deltas else np.array([0.0])
    return {
        "n": len(deltas),
        "mean_abs_delta": float(np.mean(arr)),
        "p99_abs_delta": float(np.percentile(arr, 99)),
        "max_abs_delta": float(np.max(arr)),
    }
