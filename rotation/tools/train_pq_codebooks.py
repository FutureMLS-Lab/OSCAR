#!/usr/bin/env python3
"""Train per-layer product-quantization codebooks (PQ, or two-stage RVQ) for
the unified pool's K or V tier from calibration dumps.

Inputs are the ``qkv_dumps/<dataset>/layer_<i>/{k,v}/*.pt`` chunks written by
``rotation/<model>/save_qkv_*.sh`` and the rotation checkpoint the server will
serve with; rows are rotated as ``x @ R`` exactly like the serving path, so a
codebook is bound to its rotation. The output loads through
``SGLANG_OSCAR_PQ_K_CODEBOOK`` / ``SGLANG_OSCAR_PQ_V_CODEBOOK``.

Example (Qwen3-8B, 1.0-bit K then 1.5-bit RVQ K and 1.0-bit V):

    python rotation/tools/train_pq_codebooks.py --dumps $D/qkv_dumps/gpqa \\
        --rotation $D/rotations/k_rotation_qqt_r_h_pbr.pt --tensor k \\
        --out codebooks/k_pq_n16_c256_d8.pt
    python rotation/tools/train_pq_codebooks.py ... --stage2-centroids 16 \\
        --out codebooks/k_rvq_n16_c256x16_d8.pt
    python rotation/tools/train_pq_codebooks.py --tensor v \\
        --rotation $D/rotations/v_rotation_sst_r_h_pbr.pt --out codebooks/v_pq_n16_c256_d8.pt
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import numpy as np
import torch


def _sqnr_db(x: torch.Tensor, xh: torch.Tensor) -> float:
    signal = float((x.double() ** 2).mean())
    error = float(((x.double() - xh.double()) ** 2).mean())
    return 10 * math.log10(signal / error) if error > 0 else float("inf")


def _kmeans(values: torch.Tensor, n_centroids: int, *, iters: int, seed: int) -> torch.Tensor:
    """Lloyd iterations from a random sample init; empty clusters keep their
    previous centroid."""
    generator = torch.Generator(device=values.device).manual_seed(seed)
    init = torch.randperm(values.shape[0], generator=generator, device=values.device)[:n_centroids]
    centroids = values[init].clone()
    value_norm = (values * values).sum(dim=1, keepdim=True)
    for _ in range(iters):
        distances = value_norm + (centroids * centroids).sum(dim=1)[None, :] - 2.0 * (values @ centroids.T)
        codes = distances.argmin(dim=1)
        sums = torch.zeros_like(centroids)
        sums.index_add_(0, codes, values)
        counts = torch.bincount(codes, minlength=n_centroids)
        nonempty = counts > 0
        centroids[nonempty] = sums[nonempty] / counts[nonempty, None]
    return centroids


def _encode_decode(values: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    distances = (
        (values * values).sum(dim=1, keepdim=True)
        + (codebook * codebook).sum(dim=1)[None, :]
        - 2.0 * (values @ codebook.T)
    )
    return codebook[distances.argmin(dim=1)]


def _load_rotations(path: str) -> dict[int, torch.Tensor]:
    state = torch.load(path, map_location="cpu")
    rotations = {}
    for key, entry in state["layers"].items():
        layer_id = int(entry.get("layer_id", key))
        rotations[layer_id] = entry["rotation"].float()
    return rotations


def _rotate(rows: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """``rows`` is ``[tokens, heads, head_dim]``; a V2 per-head rotation is
    ``[heads, head_dim, head_dim]``."""
    if rotation.dim() == 3:
        return torch.einsum("thd,hde->the", rows, rotation)
    return rows @ rotation


def _load_layer(dumps: str, layer_id: int, tensor: str, rotation, max_samples: int, head_dim: int):
    files = sorted(glob.glob(os.path.join(dumps, f"layer_{layer_id}", tensor, "*.pt")))
    if not files:
        return None
    parts = []
    for path in files:
        chunk = torch.load(path, map_location="cpu").float()
        if chunk.dim() == 2:
            chunk = chunk.reshape(chunk.shape[0], -1, head_dim)
        parts.append(_rotate(chunk, rotation).reshape(-1, head_dim))
    rows = torch.cat(parts)
    if rows.shape[0] > max_samples:
        rng = np.random.default_rng(42 + layer_id)
        rows = rows[torch.from_numpy(rng.choice(rows.shape[0], max_samples, replace=False))]
    return rows


def _train_layer(rows: torch.Tensor, *, n_sub: int, sub_dim: int, centroids: int, stage2: int, iters: int, seed: int, device):
    sub = rows.to(device).reshape(-1, n_sub, sub_dim)
    books1, recon1 = [], torch.empty_like(sub)
    for s in range(n_sub):
        book = _kmeans(sub[:, s, :], centroids, iters=iters, seed=seed + s)
        books1.append(book)
        recon1[:, s, :] = _encode_decode(sub[:, s, :], book)
    result = {"stage1": torch.stack(books1).cpu(), "sqnr_stage1": _sqnr_db(sub.reshape(len(rows), -1), recon1.reshape(len(rows), -1))}
    if stage2 > 0:
        residual = sub - recon1
        books2, recon2 = [], torch.empty_like(sub)
        for s in range(n_sub):
            book = _kmeans(residual[:, s, :], stage2, iters=iters, seed=seed + 1000 + s)
            books2.append(book)
            recon2[:, s, :] = _encode_decode(residual[:, s, :], book)
        result["stage2"] = torch.stack(books2).cpu()
        result["sqnr_rvq"] = _sqnr_db(sub.reshape(len(rows), -1), (recon1 + recon2).reshape(len(rows), -1))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dumps", required=True, help="qkv_dumps/<dataset> directory with layer_<i>/{k,v}/*.pt")
    parser.add_argument("--rotation", required=True, help="rotation checkpoint the server will use for this tensor")
    parser.add_argument("--tensor", choices=("k", "v"), default="k")
    parser.add_argument("--out", required=True)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--n-sub", type=int, default=16)
    parser.add_argument("--centroids", type=int, default=256)
    parser.add_argument("--stage2-centroids", type=int, default=0, help="> 0 trains a residual stage (RVQ)")
    parser.add_argument("--samples", type=int, default=50000, help="rows per layer used for k-means")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--layers", default="", help="comma-separated subset of layer ids (default: all dumped)")
    parser.add_argument("--model", default="", help="free-text provenance stored in the file")
    args = parser.parse_args()

    if args.head_dim % args.n_sub:
        parser.error("--head-dim must be divisible by --n-sub")
    for name, value in (("--centroids", args.centroids), ("--stage2-centroids", args.stage2_centroids)):
        if value > 256 or (value > 0 and value & (value - 1)):
            parser.error(f"{name} must be a power of two <= 256 (uint8 codes)")
    sub_dim = args.head_dim // args.n_sub
    rotations = _load_rotations(args.rotation)
    dumped = sorted(
        int(name.split("_")[1])
        for name in os.listdir(args.dumps)
        if name.startswith("layer_") and os.path.isdir(os.path.join(args.dumps, name))
    )
    layers = [int(x) for x in args.layers.split(",") if x] or dumped
    missing = [l for l in layers if l not in rotations]
    if missing:
        parser.error(f"rotation file has no entry for layers {missing}")

    stage1, stage2, sqnr1, sqnr_rvq = [], [], {}, {}
    for layer_id in layers:
        rows = _load_layer(args.dumps, layer_id, args.tensor, rotations[layer_id], args.samples, args.head_dim)
        if rows is None:
            print(f"layer {layer_id}: no {args.tensor} dumps", file=sys.stderr)
            return 1
        trained = _train_layer(
            rows, n_sub=args.n_sub, sub_dim=sub_dim, centroids=args.centroids,
            stage2=args.stage2_centroids, iters=args.iters, seed=42 + layer_id * 100, device=args.device,
        )
        stage1.append(trained["stage1"])
        sqnr1[layer_id] = trained["sqnr_stage1"]
        line = f"layer {layer_id:3d}: rows={rows.shape[0]:,} sqnr={trained['sqnr_stage1']:+.2f} dB"
        if args.stage2_centroids > 0:
            stage2.append(trained["stage2"])
            sqnr_rvq[layer_id] = trained["sqnr_rvq"]
            line += f" rvq={trained['sqnr_rvq']:+.2f} dB"
        print(line, flush=True)

    payload = {
        "layer_ids": layers,
        "n_sub": args.n_sub,
        "sub_dim": sub_dim,
        "model": args.model,
        "tensor": args.tensor,
        "rotation": os.path.basename(args.rotation).removesuffix(".pt"),
        "rotation_convention": "row_times_checkpoint_rotation",
        "data": os.path.abspath(args.dumps),
        "sqnr_per_layer": sqnr1,
        "sqnr_avg": float(np.mean(list(sqnr1.values()))),
    }
    if args.stage2_centroids > 0:
        payload.update(
            codebooks_stage1=torch.stack(stage1),
            codebooks_stage2=torch.stack(stage2),
            n_cents1=args.centroids,
            n_cents2=args.stage2_centroids,
            sqnr_rvq_per_layer=sqnr_rvq,
            sqnr_rvq_avg=float(np.mean(list(sqnr_rvq.values()))),
        )
    else:
        payload.update(codebooks_per_layer=torch.stack(stage1), n_centroids=args.centroids)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(payload, args.out)
    bits = 8.0 * args.n_sub * (2 if args.stage2_centroids > 0 else 1) / args.head_dim
    print(f"saved {args.out}: {len(layers)} layers, {bits:.2f} bits/value, avg sqnr {payload['sqnr_avg']:+.2f} dB"
          + (f" (rvq {payload['sqnr_rvq_avg']:+.2f} dB)" if args.stage2_centroids > 0 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
