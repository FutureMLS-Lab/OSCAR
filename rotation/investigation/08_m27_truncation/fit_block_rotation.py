#!/usr/bin/env python3
"""Fit a block-diagonal OSCAR K rotation that respects M2.7's partial-RoPE split.

M2.7 has ``head_dim=128`` but ``rotary_dim=64``, so within every K head dims
0-63 are position-dependent (they rotate with position) and dims 64-127 are
static content. The stock recipe applies ONE 128x128 rotation across both, which
mixes two subspaces with very different statistics, and then quantizes the
mixture with a single 2-bit scale.

The direct precedent that this is the wrong thing to do is MLA: GLM-5.2's latent
pool keeps ``k_pe`` -- the positional part -- entirely in BF16 and quantizes only
``c_kv``. M2.7's first 64 dims are the analogue and nothing currently protects
them.

This fits the same objective and the same composition as
``compute_kv_rotation.py --method qqt --composition r_h_pbr``, but INDEPENDENTLY
within each 64-dim block, then places the two blocks on the diagonal:

    R = blockdiag( R_a . H64 . P_a ,  R_b . H64 . P_b )

so the rotation can never move energy across the RoPE boundary. Everything else
is held identical to the baseline -- same dump, same qqt objective, same
bit budget, same quant group size -- so a score change is attributable to the
block structure alone.

Only K is treated. V is not RoPE'd at all, so it has no boundary to respect;
leaving the V rotation untouched keeps this a single-variable change.

Usage:
  python3 fit_block_rotation.py --dump-path <qkv_dumps/gpqa> \
      --out k_rotation_qqt_block64.pt [--num-layers 62] [--rotary-dim 64]
"""
from __future__ import annotations

import argparse
import glob
import math
import os

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
    order = [int(bin(i)[2:].zfill(bits)[::-1], 2) for i in range(1 << bits)]
    return torch.tensor([i for i in order if i < d])


def make_br_perm_matrix(eigenvalues: torch.Tensor) -> torch.Tensor:
    d = len(eigenvalues)
    sorted_idx = torch.argsort(eigenvalues, descending=True)
    br = bit_reversal_perm(d)
    perm = torch.zeros(d, dtype=torch.long)
    for i in range(d):
        perm[br[i]] = sorted_idx[i]
    return torch.eye(d, dtype=torch.float64)[:, perm]


def load_all(layer_dir: str, name: str) -> torch.Tensor:
    paths = sorted(glob.glob(os.path.join(layer_dir, name, "*.pt")),
                   key=lambda p: int(os.path.basename(p)[:-3]))
    if not paths:
        raise FileNotFoundError(f"no chunks in {layer_dir}/{name}")
    return torch.cat(
        [torch.load(p, map_location="cpu", weights_only=True).float().double()
         for p in paths],
        dim=0,
    )


def qqt_cov_block(q: torch.Tensor, kv_heads: int, sl: slice) -> torch.Tensor:
    """Q second moment restricted to a head-dim slice, averaged over KV groups.

    Mirrors compute_qqt: average over KV heads of the GQA group's Qᵀ Q, so the
    result is the same objective, just restricted to the block.
    """
    n_heads = q.shape[1]
    gqa = n_heads // kv_heads
    d = sl.stop - sl.start
    cov = torch.zeros(d, d, dtype=torch.float64)
    for h in range(kv_heads):
        qg = q[:, h * gqa : (h + 1) * gqa, sl].reshape(-1, d)
        cov += qg.T @ qg / qg.shape[0]
    cov /= kv_heads
    return (cov + cov.T) / 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-layers", type=int, default=62)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--rotary-dim", type=int, default=64)
    a = ap.parse_args()

    blocks = [slice(0, a.rotary_dim), slice(a.rotary_dim, a.head_dim)]
    result = {"format_version": 1, "objective": "qqt_block_r_h_pbr",
              "source_grouping": "layer", "layers": {}}

    for li in range(a.num_layers):
        ldir = os.path.join(a.dump_path, f"layer_{li}")
        q = load_all(ldir, "q")
        k = load_all(ldir, "k")
        kv_heads = k.shape[1]
        q = q.reshape(-1, q.shape[1], a.head_dim)

        mats, eigs = [], []
        for sl in blocks:
            d = sl.stop - sl.start
            cov = qqt_cov_block(q, kv_heads, sl)
            ev, evec = torch.linalg.eigh(cov)
            R = evec @ build_hadamard(d) @ make_br_perm_matrix(ev)
            mats.append(R)
            eigs.append(ev)
        R_full = torch.block_diag(*mats)
        err = (R_full @ R_full.T - torch.eye(a.head_dim, dtype=torch.float64)
               ).abs().max().item()
        assert err < 1e-8, f"layer {li} not orthogonal: {err:.2e}"
        # Sanity: the block structure must actually be block-diagonal, i.e. the
        # off-diagonal blocks are exactly zero. This is the whole point of the
        # file, so assert it rather than trusting block_diag.
        assert R_full[: a.rotary_dim, a.rotary_dim :].abs().max() == 0
        assert R_full[a.rotary_dim :, : a.rotary_dim].abs().max() == 0

        result["layers"][li] = {
            "layer_id": li,
            "rotation": R_full.float().contiguous(),
            "eigenvalues": torch.cat(eigs).float().contiguous(),
        }
        if li % 10 == 0 or li == a.num_layers - 1:
            eff = [float((e.sum() ** 2) / (e * e).sum()) for e in eigs]
            print(f"layer {li:2d}: tokens={q.shape[0]} kv_heads={kv_heads} "
                  f"ortho={err:.1e} eff_rank_rope={eff[0]:.1f}/{a.rotary_dim} "
                  f"eff_rank_nope={eff[1]:.1f}/{a.head_dim - a.rotary_dim}")

    torch.save(result, a.out)
    print(f"saved {a.out} ({os.path.getsize(a.out)} bytes, "
          f"{len(result['layers'])} layers)")


if __name__ == "__main__":
    main()
