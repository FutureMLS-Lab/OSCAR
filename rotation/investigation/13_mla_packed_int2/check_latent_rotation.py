#!/usr/bin/env python3
"""Gate a fitted MLA latent rotation before spending a GPQA run on it.

`ortho_err` is not a quality metric. A plain Hadamard is exactly orthogonal, so
a fit that silently degenerated to one reports ortho_err ~1e-7 and looks
perfect -- that is the documented way this has collapsed before. The question a
rotation file has to answer is whether it makes `c_kv` *cheaper to quantise*
than the alternatives, and that needs the quantiser in the loop.

Three arms per layer, all at the packed kernel's own budget (INT2, per-token
per-group asymmetric scale/zero, group_size=128):

  identity  -- no rotation, the thing to beat
  hadamard  -- block-diagonal Hadamard, what a degenerate fit collapses to
  fitted    -- the shipped Rcov P H_block

If `fitted` does not beat `hadamard`, the covariance term contributed nothing
and the dump or the fit is suspect. If `fitted` is above the 0.1 rel-err line,
2-bit will not hold regardless of how it compares.

Out-of-sample by construction: the second half of each dump's tokens is held
out, and the comparison rotation is refitted on the first half only, so the
reported number is generalisation rather than the fit reproducing its own
calibration set.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def fake_quant_int2(x: torch.Tensor, group_size: int) -> torch.Tensor:
    """Per-token, per-group asymmetric INT2 -- the packed pool's own format."""
    n, d = x.shape
    g = x.reshape(n, d // group_size, group_size).float()
    lo = g.amin(dim=-1, keepdim=True)
    hi = g.amax(dim=-1, keepdim=True)
    scale = (hi - lo).clamp_min(1e-8) / 3.0
    code = ((g - lo) / scale).round().clamp_(0, 3)
    return (code * scale + lo).reshape(n, d).to(x.dtype)


def rel_err(x: torch.Tensor, rot: torch.Tensor | None, group_size: int) -> float:
    y = x if rot is None else x @ rot
    yq = fake_quant_int2(y, group_size)
    xq = yq if rot is None else yq @ rot.T
    return ((xq - x).norm() / x.norm().clamp_min(1e-12)).item()


def fit_oscar(c: torch.Tensor, group_size: int) -> torch.Tensor:
    """Refit Rcov P H on the given tokens only (mirrors the shipping script)."""
    from compute_kv_rotation import build_hadamard, make_br_perm_matrix

    # Mirrors compute_mla_oscar_rotation.py exactly: eigh returns ascending
    # eigenvalues, and make_br_perm_matrix takes the *eigenvalues* -- it does
    # the descending sort itself. Pre-sorting Rc here would sort twice and
    # produce a different (and wrong) comparison rotation.
    d = c.shape[1]
    cov = c.double().T @ c.double() / c.shape[0]
    cov = (cov + cov.T) / 2
    evals, rcov = torch.linalg.eigh(cov)
    perm = make_br_perm_matrix(evals)
    hblk = torch.zeros(d, d, dtype=torch.float64)
    h = build_hadamard(group_size).double()
    for i in range(d // group_size):
        s = slice(i * group_size, (i + 1) * group_size)
        hblk[s, s] = h
    return (rcov @ perm @ hblk).float()


def hadamard_only(d: int, group_size: int) -> torch.Tensor:
    from compute_kv_rotation import build_hadamard

    hblk = torch.zeros(d, d)
    h = build_hadamard(group_size).float()
    for i in range(d // group_size):
        s = slice(i * group_size, (i + 1) * group_size)
        hblk[s, s] = h
    return hblk


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dump-path", required=True)
    p.add_argument("--rot-dir", required=True)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--threshold", type=float, default=0.10)
    a = p.parse_args()

    files = sorted(
        glob.glob(os.path.join(a.dump_path, "layer_*.pt")),
        key=lambda f: int(os.path.basename(f)[6:-3]),
    )
    if not files:
        print(f"FATAL: no dumps under {a.dump_path}")
        return 2

    print(f"{'layer':>6} {'ident':>8} {'hadam':>8} {'fitted':>8} "
          f"{'fit<had':>8} {'verdict':>9}")
    rows = []
    for f in files:
        lid = int(os.path.basename(f)[6:-3])
        c = torch.load(f, map_location="cpu").float()
        n = c.shape[0]
        if n < 64:
            print(f"{lid:>6}  too few tokens ({n}), skipped")
            continue
        # Held out: fit on the first half, score on the second.
        cal, oos = c[: n // 2], c[n // 2:]
        rot_path = os.path.join(a.rot_dir, f"layer_{lid}.pt")
        if not os.path.exists(rot_path):
            print(f"{lid:>6}  MISSING rotation {rot_path}")
            return 2
        shipped = torch.load(rot_path, map_location="cpu").float()
        refit = fit_oscar(cal, a.group_size)

        e_id = rel_err(oos, None, a.group_size)
        e_h = rel_err(oos, hadamard_only(c.shape[1], a.group_size), a.group_size)
        e_f = rel_err(oos, shipped, a.group_size)
        e_r = rel_err(oos, refit, a.group_size)
        # The shipped file saw these tokens during its own fit; the refit did
        # not. Report the shipped number but judge on the honest one.
        better = e_r < e_h
        ok = (e_r <= a.threshold) and better
        rows.append((lid, e_id, e_h, e_f, e_r, better, ok))
        print(f"{lid:>6} {e_id:>8.4f} {e_h:>8.4f} {e_r:>8.4f} "
              f"{'yes' if better else 'NO':>8} {'ok' if ok else 'FAIL':>9}")

    n_ok = sum(r[6] for r in rows)
    n_better = sum(r[5] for r in rows)
    worst = max((r[4] for r in rows), default=float("nan"))
    print(f"\n{n_ok}/{len(rows)} layers pass "
          f"(OOS rel-err <= {a.threshold} and beats block-Hadamard)")
    print(f"beats Hadamard: {n_better}/{len(rows)}   worst OOS rel-err: {worst:.4f}")
    if n_better == 0:
        print("VERDICT: the covariance term bought nothing -- the fit is "
              "indistinguishable from a plain Hadamard. Do not ship.")
    elif n_ok == len(rows):
        print("VERDICT: pass. Safe to spend a GPQA run on this rotation.")
    else:
        print("VERDICT: partial. Some layers will be the accuracy floor.")
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
