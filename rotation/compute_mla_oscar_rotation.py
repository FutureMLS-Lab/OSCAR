#!/usr/bin/env python3
"""Full-OSCAR latent rotation: Rcov · P · H_block (the composition the original
`compute_mla_kv_rotation.py` was *missing* — it emitted plain covariance eigvecs).

Per layer, from the calibration dump of `c_kv`:
  Rcov   = eigenvectors of the latent second moment (c_kvᵀ c_kv / N)
  P      = bit-reversal permutation of the eigenvalue-sorted dims
           → interleaves high/low-variance dims across the 128-groups (homogenizes)
  H_block= block-diagonal per-group (128×128) Hadamard (within-group incoherence)

R = Rcov @ P @ H_block  (orthogonal). The runtime applies `c@R` / `c_q@R.T`, so
this drops in as the rotation file — no runtime change. Beats plain Hadamard
(−0.4 dB OOS at the same 2-bit budget) and stacks with the HP subspace.
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
import torch
from compute_kv_rotation import build_hadamard, make_br_perm_matrix


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump-path", default="/shared/glm51_fp8_c_kv_dump")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--kv-lora-rank", type=int, default=512)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--max-tokens", type=int, default=8192)
    args = p.parse_args()
    d, gs = args.kv_lora_rank, args.group_size
    G = d // gs

    # block-diagonal per-group Hadamard
    h = build_hadamard(gs)
    Hblock = torch.zeros(d, d, dtype=torch.float64)
    for g in range(G):
        Hblock[g*gs:(g+1)*gs, g*gs:(g+1)*gs] = h

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.dump_path).glob("layer_*.pt"),
                   key=lambda x: int(x.stem.split("_")[1]))
    for f in files:
        lid = int(f.stem.split("_")[1])
        c = torch.load(str(f), map_location="cpu").float().reshape(-1, d)[:args.max_tokens].double()
        cov = c.T @ c / c.shape[0]
        cov = (cov + cov.T) / 2
        evals, Rc = torch.linalg.eigh(cov)            # ascending
        P = make_br_perm_matrix(evals)
        M = (Rc @ P @ Hblock).float().contiguous()
        err = (M.T @ M - torch.eye(d)).abs().max().item()
        torch.save(M, str(out / f"layer_{lid}.pt"))
        print(f"layer {lid:2d}: ortho_err={err:.1e}", flush=True)
    print(f"Done. {len(files)} rotations → {out}")


if __name__ == "__main__":
    main()
