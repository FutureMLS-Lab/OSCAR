#!/usr/bin/env python3
"""Deterministic offline comparison of MLA latent rotations.

For each rotation method we replicate the runtime fake-quant exactly
(rotate → groupwise Lloyd-Max INT2 → unrotate), measure the quantization
error Δc = dequant(c) − c on the calibration dump, and score it against the
*functional* latent Hessians derived from kv_b_proj:

    data_mse  = E‖Δc‖²                = tr(Cov_Δc)             (what data-cov PCA targets)
    score_err = E[(q̃·Δc)²]  (iso q)   = tr(H_score · Cov_Δc)   (impact on attention score)
    val_err   = E‖Δc·W_UV‖²            = tr(H_val   · Cov_Δc)   (impact on attention output)
    total     = score_err + α·val_err                          (the OSCAR-for-latent objective)

This separates "did we reduce the functional error" from GPQA noise. Lower is
better; we report each method relative to the no-rotation baseline (in dB:
10·log10(method/identity), so more negative = better).
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch

from compute_mla_upproj_rotation import KVBProjReader, _resolve_snapshot, latent_hessian

# ── Lloyd-Max INT2 constants — must match mla_int2_kv_pool.py ────────────────
_LM_THRESHOLDS = (-0.9810652732849121, 0.0, 0.9810652732849121)
_LM_CENTROIDS = (-1.5095585584640503, -0.4527800381183624, 0.4527800381183624, 1.5095585584640503)
_LM_SPAN = _LM_CENTROIDS[3] - _LM_CENTROIDS[0]
_LM_RATIO = 1.16


def fake_quant_int2_groupwise(x: torch.Tensor, group_size: int, lloyd_max: bool) -> torch.Tensor:
    orig_shape = x.shape
    x = x.reshape(-1, group_size).to(torch.float32)
    if lloyd_max:
        mean = x.mean(dim=-1, keepdim=True)
        diff = x - mean
        std = (diff.pow(2).mean(dim=-1, keepdim=True) + 1e-8).sqrt()
        z = diff / std
        t0, t1, t2 = _LM_THRESHOLDS
        q = ((z >= t0).to(torch.uint8) + (z >= t1).to(torch.uint8) + (z >= t2).to(torch.uint8))
        uniform_scale = (_LM_SPAN / 3.0) * _LM_RATIO * std
        uniform_zero = -_LM_CENTROIDS[0] / (_LM_SPAN / 3.0) - mean / uniform_scale
        x_deq = (q.to(torch.float32) - uniform_zero) * uniform_scale
    else:
        x_min = x.amin(dim=-1, keepdim=True)
        x_max = x.amax(dim=-1, keepdim=True)
        scale = torch.where((x_max - x_min).abs() > 1e-8, (x_max - x_min) / 3.0, torch.ones_like(x_min))
        q = ((x - x_min) / scale).round().clamp_(0.0, 3.0)
        x_deq = q * scale + x_min
    return x_deq.reshape(orig_shape)


def random_sign_hadamard(d: int) -> torch.Tensor:
    """Replicate _load_or_make_rotations('hadamard') from mla_int2_kv_pool.py."""
    rng = torch.Generator()
    rng.manual_seed(0)
    signs = torch.where(torch.rand(d, generator=rng) > 0.5, torch.ones(d), -torch.ones(d)).view(d, 1)
    base = torch.randn(d, d, generator=rng)
    R, _ = torch.linalg.qr(base)
    return (R * signs).contiguous()


def load_rotation(spec: str, d: int, layer_id: int) -> torch.Tensor | None:
    """spec: 'none' | 'hadamard' | <dir with layer_<id>.pt>."""
    if spec == "none":
        return None
    if spec == "hadamard":
        return random_sign_hadamard(d)
    p = Path(spec) / f"layer_{layer_id}.pt"
    if not p.exists():
        return None
    return torch.load(str(p), map_location="cpu").float()


def apply_fake_quant(c: torch.Tensor, R: torch.Tensor | None, group_size: int) -> torch.Tensor:
    x = c.to(torch.float32)
    if R is not None:
        x = x @ R
    xq = fake_quant_int2_groupwise(x, group_size, lloyd_max=True)
    if R is not None:
        xq = xq @ R.T
    return xq


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump-dir", default="/shared/glm51_fp8_c_kv_dump")
    ap.add_argument("--model-path", default="zai-org/GLM-5.1-FP8")
    ap.add_argument("--methods", nargs="+", required=True,
                    help="name=spec pairs, e.g. none=none hadamard=hadamard "
                         "datacov=/path/to/calib/rotations upproj=/path/to/upproj/rotations")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--kv-lora-rank", type=int, default=512)
    ap.add_argument("--qk-nope-head-dim", type=int, default=192)
    ap.add_argument("--v-head-dim", type=int, default=256)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4096, help="cap tokens/layer for speed")
    ap.add_argument("--layers", type=str, default=None, help="comma list; default all in dump")
    args = ap.parse_args()

    methods = []
    for m in args.methods:
        name, _, spec = m.partition("=")
        methods.append((name, spec))

    snapshot = _resolve_snapshot(args.model_path)
    reader = KVBProjReader(snapshot)

    dump = Path(args.dump_dir)
    layer_ids = sorted(int(p.stem.split("_")[1]) for p in dump.glob("layer_*.pt"))
    if args.layers:
        want = {int(x) for x in args.layers.split(",")}
        layer_ids = [l for l in layer_ids if l in want]

    d = args.kv_lora_rank
    # accumulators per method: [data_mse, score_err, val_err]
    acc = {name: torch.zeros(3, dtype=torch.float64) for name, _ in methods}
    n_layers = 0

    for lid in layer_ids:
        c = torch.load(str(dump / f"layer_{lid}.pt"), map_location="cpu").float().reshape(-1, d)
        if c.shape[0] > args.max_tokens:
            c = c[: args.max_tokens]
        w = reader.get_dequant(lid)
        _, h_score, h_val = latent_hessian(
            w, args.qk_nope_head_dim, args.v_head_dim, d, args.alpha
        )
        # normalize Hessians by trace so score/val are comparable across layers
        h_score = h_score / h_score.diagonal().sum()
        h_val = h_val / h_val.diagonal().sum()

        for name, spec in methods:
            R = load_rotation(spec, d, lid)
            cq = apply_fake_quant(c, R, args.group_size)
            dc = (cq - c).double()                       # [N, d]
            cov = dc.T @ dc / dc.shape[0]                 # [d, d]
            data_mse = cov.diagonal().sum()
            score_err = (h_score * cov).sum()             # tr(H_score · cov)
            val_err = (h_val * cov).sum()
            acc[name] += torch.stack([data_mse, score_err, val_err])
        n_layers += 1
        print(f"  layer {lid:2d} done ({c.shape[0]} tok)")

    print(f"\n=== Aggregate over {n_layers} layers (mean per layer) ===")
    base = acc[methods[0][0]] / n_layers
    hdr = f"{'method':12} {'data_mse':>12} {'score_err':>12} {'val_err':>12} {'total':>12}   (dB vs %s)" % methods[0][0]
    print(hdr)
    for name, _ in methods:
        v = acc[name] / n_layers
        total = v[1] + args.alpha * v[2]
        base_total = base[1] + args.alpha * base[2]
        db_total = 10 * math.log10((total / base_total).clamp(min=1e-30).item())
        db_score = 10 * math.log10((v[1] / base[1]).clamp(min=1e-30).item())
        db_val = 10 * math.log10((v[2] / base[2]).clamp(min=1e-30).item())
        print(f"{name:12} {v[0]:12.5g} {v[1]:12.5g} {v[2]:12.5g} {total:12.5g}   "
              f"score{db_score:+6.2f} val{db_val:+6.2f} total{db_total:+6.2f}")
    print("\nLower total = better. The OSCAR-for-latent objective is score_err + α·val_err;")
    print("data_mse is what the plain c_kv-covariance PCA minimizes (and is NOT the goal).")


if __name__ == "__main__":
    main()
