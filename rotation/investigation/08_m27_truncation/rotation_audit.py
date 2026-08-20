#!/usr/bin/env python3
"""Audit an OSCAR rotation checkpoint without needing the calibration dump.

Answers hypothesis H1 -- "the rotation is under-calibrated and is silently
serving at roughly Hadamard quality" -- from the checkpoint alone, because the
checkpoint stores the fitted covariance eigenvalues next to each layer's
composed rotation.

What is reported per layer, and why each one is here:

* ``ortho_err``   max|R Rᵀ − I|.  Catches a corrupted or non-orthogonal file.
  Note this is a weak test on its own: the composed rotation is a product of
  orthogonal factors, so a *useless* fit is still perfectly orthogonal.  A clean
  ortho_err is necessary, not sufficient, and has been mistaken for a pass here.

* ``eff_rank``    participation ratio (Σλ)²/Σλ² of the fitted eigenvalues.
  This is the real calibration-health signal.  Attention K/V second moments are
  strongly anisotropic, so a genuine fit on real tensors has eff_rank well below
  the head dimension.  A dump that captured only warmup tokens, or that was
  overwritten to near-noise, produces a near-flat spectrum -- eff_rank pinned at
  ~head_dim -- and the fit still "succeeds" and still loads.  That is the silent
  failure mode this file exists to detect.

* ``cond``        λmax/λmin, and ``top8_energy``.  Same signal, easier to read.

* ``rel_err_*``   rotate -> per-group asymmetric INT2 (with the runtime's clip
  ratio) -> unrotate, measured on a Gaussian surrogate drawn from the *fitted*
  covariance, for three rotations: the calibrated file, a pure Hadamard, and
  identity.  The project bar is K rel-err <= 0.1 at 2 bits.

  Read these as a *comparison*, not as the true serving error.  The surrogate is
  Gaussian, and OSCAR's whole job is fighting the heavy tails that a Gaussian
  does not have, so the absolute numbers are optimistic for every arm.  What
  survives the surrogate is the ranking: if calibrated does not beat Hadamard on
  the distribution it was itself fitted to, it is contributing nothing, and
  refitting it is not the lever.  The honest end-to-end check is the paired
  Hadamard-rotation eval arm, which is why that arm is run alongside this.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def build_hadamard(n: int) -> torch.Tensor:
    if n == 1:
        return torch.ones(1, 1, dtype=torch.float64)
    if n & (n - 1):
        lp = 1 << (n.bit_length() - 1)
        return torch.block_diag(build_hadamard(lp), build_hadamard(n - lp))
    h = build_hadamard(n // 2)
    return torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0) / math.sqrt(2)


def bit_reversal_perm(d: int) -> torch.Tensor:
    bits = (d - 1).bit_length()
    padded = 1 << bits
    order = [int(bin(i)[2:].zfill(bits)[::-1], 2) for i in range(padded)]
    return torch.tensor([i for i in order if i < d])


def make_br_perm_matrix(eigenvalues: torch.Tensor) -> torch.Tensor:
    """Exact copy of compute_kv_rotation.make_br_perm_matrix.

    Must be fed the eigenvalues in the order they were *stored* (ascending, as
    ``torch.linalg.eigh`` returns them), not sorted, or the recovered
    eigenvector factor is permuted and the whole audit is meaningless.
    """
    d = len(eigenvalues)
    sorted_idx = torch.argsort(eigenvalues, descending=True)
    br = bit_reversal_perm(d)
    perm = torch.zeros(d, dtype=torch.long)
    for i in range(d):
        perm[br[i]] = sorted_idx[i]
    return torch.eye(d, dtype=torch.float64)[:, perm]


def quant_int2_asym(x: torch.Tensor, group: int, clip: float) -> torch.Tensor:
    """Per-group asymmetric 2-bit round trip, matching the runtime's clip."""
    shape = x.shape
    xg = x.reshape(-1, shape[-1] // group, group).float()
    lo = xg.amin(-1, keepdim=True) * clip
    hi = xg.amax(-1, keepdim=True) * clip
    scale = (hi - lo).clamp(min=1e-8) / 3.0
    zero = -lo / scale
    q = (xg / scale + zero + 0.5).to(torch.int32).clamp(0, 3)
    return ((q.float() - zero) * scale).reshape(shape).to(x.dtype)


def rel_err(x: torch.Tensor, xh: torch.Tensor) -> float:
    return (
        torch.linalg.norm(xh - x) / torch.linalg.norm(x).clamp(min=1e-12)
    ).item()


def roundtrip(cov_sqrt: torch.Tensor, R: torch.Tensor, n: int,
              group: int, clip: float, gen: torch.Generator) -> float:
    """rel-err of rotate -> INT2 -> unrotate on x ~ N(0, cov)."""
    d = cov_sqrt.shape[0]
    z = torch.randn(n, d, generator=gen, dtype=torch.float64)
    x = z @ cov_sqrt.T
    y = x @ R                       # runtime applies x @ R
    yq = quant_int2_asym(y, group, clip)
    xh = yq @ R.T                   # R orthogonal -> inverse is transpose
    return rel_err(x, xh)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--clip", type=float, default=0.96)
    ap.add_argument("--samples", type=int, default=4096)
    ap.add_argument("--layers", type=int, default=0,
                    help="0 = all; else stride-sample this many layers")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    state = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    layers = state["layers"]
    keys = sorted(layers.keys(), key=lambda k: int(k))
    if a.layers and a.layers < len(keys):
        step = len(keys) / a.layers
        keys = [keys[int(i * step)] for i in range(a.layers)]

    gen = torch.Generator().manual_seed(0)
    rows = []
    for k in keys:
        entry = layers[k]
        R = entry["rotation"].double()
        lam = entry["eigenvalues"].double().clamp(min=0)
        d = R.shape[0]
        ortho = (R @ R.T - torch.eye(d, dtype=torch.float64)).abs().max().item()
        s = lam.sum().clamp(min=1e-30)
        eff_rank = (s * s / (lam * lam).sum().clamp(min=1e-30)).item()
        lam_sorted = torch.sort(lam, descending=True).values
        cond = (lam_sorted[0] / lam_sorted[-1].clamp(min=1e-30)).item()
        top8 = (lam_sorted[:8].sum() / s).item()
        # Gaussian surrogate with the fitted covariance.
        #
        # The stored matrix is the *composed* rotation R_eig · H · P, not the
        # eigenvector factor, so the covariance is NOT R diag(lam) Rᵀ. Using
        # that form makes the composed rotation look worse than identity (it
        # assumes the data is aligned with the composed basis, i.e. already
        # whitened and then de-whitened), which is how this bug announced
        # itself. Recover the eigenvector factor exactly instead -- H and P are
        # both reproducible, P deterministically from the stored eigenvalues:
        #     R = R_eig · H · P   =>   R_eig = R · Pᵀ · Hᵀ
        H = build_hadamard(d)
        P = make_br_perm_matrix(lam)
        R_eig = R @ P.T @ H.T
        cov = (R_eig * lam) @ R_eig.T
        cov = (cov + cov.T) / 2
        evals, evecs = torch.linalg.eigh(cov)
        cov_sqrt = evecs @ torch.diag(evals.clamp(min=0).sqrt()) @ evecs.T
        I = torch.eye(d, dtype=torch.float64)
        # Flatness of the per-coordinate variance after each rotation. The whole
        # point of the H factor is to equalise these; if the composed rotation
        # is doing its job this is ~1.0, and a value far above 1 means the
        # recovered factorisation (and so the audit) is wrong, not that the
        # rotation is bad.
        def flat(Rm):
            dv = torch.diagonal(Rm.T @ cov @ Rm)
            return (dv.max() / dv.mean().clamp(min=1e-30)).item()
        rows.append({
            "layer": int(k),
            "ortho_err": ortho,
            "eff_rank": round(eff_rank, 2),
            "head_dim": d,
            "cond": cond,
            "top8_energy": round(top8, 4),
            "var_peak_calibrated": round(flat(R), 3),
            "var_peak_hadamard": round(flat(H), 3),
            "var_peak_identity": round(flat(I), 3),
            "rel_err_calibrated": round(
                roundtrip(cov_sqrt, R, a.samples, a.group_size, a.clip, gen), 4),
            "rel_err_hadamard": round(
                roundtrip(cov_sqrt, H, a.samples, a.group_size, a.clip, gen), 4),
            "rel_err_identity": round(
                roundtrip(cov_sqrt, I, a.samples, a.group_size, a.clip, gen), 4),
        })

    summ = {
        "checkpoint": a.checkpoint,
        "objective": state.get("objective"),
        "format_version": state.get("format_version"),
        "num_layers_in_file": len(layers),
        "layers_audited": len(rows),
        "ortho_err_max": max(r["ortho_err"] for r in rows),
        "eff_rank_min": min(r["eff_rank"] for r in rows),
        "eff_rank_median": sorted(r["eff_rank"] for r in rows)[len(rows) // 2],
        "eff_rank_max": max(r["eff_rank"] for r in rows),
        "rel_err_calibrated_mean": round(
            sum(r["rel_err_calibrated"] for r in rows) / len(rows), 4),
        "rel_err_hadamard_mean": round(
            sum(r["rel_err_hadamard"] for r in rows) / len(rows), 4),
        "rel_err_identity_mean": round(
            sum(r["rel_err_identity"] for r in rows) / len(rows), 4),
    }

    if a.json:
        print(json.dumps({"summary": summ, "layers": rows}, indent=2))
        return
    print(json.dumps(summ, indent=2))
    print()
    hdr = ("layer", "ortho", "eff_rank", "cond", "top8",
           "vpk_cal", "vpk_had", "calib", "hadam", "ident")
    print(("{:>9}" * len(hdr)).format(*hdr))
    for r in rows:
        print(("{:>9d}{:>9.1e}{:>9.2f}{:>9.1e}{:>9.3f}"
               "{:>9.2f}{:>9.2f}{:>9.3f}{:>9.3f}{:>9.3f}").format(
            r["layer"], r["ortho_err"], r["eff_rank"], r["cond"],
            r["top8_energy"], r["var_peak_calibrated"], r["var_peak_hadamard"],
            r["rel_err_calibrated"], r["rel_err_hadamard"],
            r["rel_err_identity"]))


if __name__ == "__main__":
    main()
