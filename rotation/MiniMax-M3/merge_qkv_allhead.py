#!/usr/bin/env python3
"""Merge per-TP-rank Q/K/V dumps into full-head tensors before fitting.

The dump hook writes one directory per rank::

    <dump>/layer_<id>/rank_<r>/{q,k,v}/<chunk>.pt

Each shard holds that rank's slice of the head axis, so a single rank sees only
``num_q_heads / tp`` query heads and ``num_kv_heads / tp`` KV heads. Fitting a
per-layer shared rotation on one shard produces a rotation no better than
Hadamard, and nothing in the fit reports it -- see this directory's README.

This concatenates the shards back along the head axis, in rank order, and writes
``<out>/layer_<id>/{q,k,v}/<chunk>.pt`` in the layout ``compute_kv_rotation.py``
expects.

KV heads are replicated across the ranks inside a TP group when
``num_kv_heads < tp``. Those duplicates are dropped: shards are compared and one
copy per distinct group is kept, so the merged K/V has exactly ``num_kv_heads``
heads. The comparison is exact -- if two shards that should be identical are
not, that is a real inconsistency and this raises rather than picking one.
"""

import argparse
import pathlib
import sys

import torch


def _chunks(d: pathlib.Path):
    return sorted(d.glob("*.pt"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)


def merge_one(rank_dirs, which, num_heads, chunk_name):
    """Concatenate one chunk across ranks along the head axis (dim=1)."""
    parts = []
    for rd in rank_dirs:
        p = rd / which / chunk_name
        if not p.exists():
            raise FileNotFoundError(f"missing shard: {p}")
        parts.append(torch.load(p, map_location="cpu"))

    if parts[0].dim() != 3:
        raise ValueError(
            f"expected [tokens, heads, head_dim] shards, got {tuple(parts[0].shape)}"
        )

    total = sum(p.shape[1] for p in parts)
    if total == num_heads:
        return torch.cat(parts, dim=1)

    # Replicated across the group: keep one copy per distinct shard.
    if total % num_heads != 0:
        raise ValueError(
            f"{which}: {len(parts)} shards x {parts[0].shape[1]} heads = {total}, "
            f"which is neither {num_heads} nor a multiple of it"
        )
    stride = total // num_heads
    kept = parts[::stride]
    for i, p in enumerate(parts):
        ref = kept[i // stride]
        if not torch.equal(p, ref):
            raise ValueError(
                f"{which} {chunk_name}: shard {i} was expected to duplicate shard "
                f"{(i // stride) * stride} but differs -- the dump is inconsistent"
            )
    return torch.cat(kept, dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, type=pathlib.Path,
                    help="directory holding layer_<id>/rank_<r>/")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--tp", type=int, required=True)
    ap.add_argument("--num-q-heads", type=int, required=True)
    ap.add_argument("--num-kv-heads", type=int, required=True)
    args = ap.parse_args()

    layers = sorted(args.dump.glob("layer_*"),
                    key=lambda p: int(p.name.split("_")[1]))
    if not layers:
        sys.exit(f"no layer_* directories under {args.dump}")

    for layer in layers:
        rank_dirs = [layer / f"rank_{r}" for r in range(args.tp)]
        missing = [d for d in rank_dirs if not d.is_dir()]
        if missing:
            sys.exit(
                f"{layer.name}: {len(missing)} of {args.tp} ranks missing "
                f"(e.g. {missing[0]}). Fitting on a partial dump silently "
                f"produces a Hadamard-quality rotation -- redump instead."
            )

        for which, n_heads in (("q", args.num_q_heads),
                               ("k", args.num_kv_heads),
                               ("v", args.num_kv_heads)):
            dst = args.out / layer.name / which
            dst.mkdir(parents=True, exist_ok=True)
            for chunk in _chunks(rank_dirs[0] / which):
                merged = merge_one(rank_dirs, which, n_heads, chunk.name)
                torch.save(merged, dst / chunk.name)

        sample = torch.load(args.out / layer.name / "k" / _chunks(args.out / layer.name / "k")[0].name,
                            map_location="cpu")
        print(f"{layer.name}: k {tuple(sample.shape)}", flush=True)

    print(f"merged {len(layers)} layers -> {args.out}")


if __name__ == "__main__":
    main()
