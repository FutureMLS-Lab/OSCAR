#!/usr/bin/env python3
"""Does a Lloyd-Max codebook close the 2-bit gap on K3's rotated latent?

The packed arm runs SGLANG_LLOYD_MAX=0, i.e. uniform quantization: four levels
spread evenly between the group's min and max. That is the right choice for a
uniform distribution and the wrong one for a bell-shaped distribution, where
most of the mass sits near the centre and the extreme levels are spent on
outliers.

The OSCAR rotation is what makes this worth testing. Rcov P H decorrelates and
spreads energy across the group, which drives each coordinate toward Gaussian --
precisely the regime where a Lloyd-Max codebook, whose levels are placed at the
conditional means of a normal, beats a uniform grid. The rotation and the
codebook are complementary, and only one of them is currently switched on.

Measured against the threshold a 2-bit KV cache needs, 0.10 relative error out
of sample. The shipped rotation puts K3 at 0.35 (group 128) and 0.23 (group 32),
so uniform 2-bit fails by 2.3x at best. The question is whether the codebook is
worth that factor or only a few percent -- the answer decides whether lat2 is
reachable at all or whether K3 genuinely needs more bits.

Scored on a held-out half, because the rotation is fit on the data.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/work/oscar/sglang-research/python")

# The same constants the kernels use, so this cannot drift from what ships.
from sglang.srt.mem_cache.mla_int2_kv_pool import (  # noqa: E402
    _LM_CENTROIDS,
    _LM_RATIO,
    _LM_SPAN,
    _LM_THRESHOLDS,
)


def q_uniform(g: torch.Tensor, bits: int) -> torch.Tensor:
    lo = g.amin(-1, keepdim=True)
    rng = g.amax(-1, keepdim=True) - lo
    maxq = float((1 << bits) - 1)
    scale = torch.where(rng.abs() > 1e-8, rng / maxq, torch.ones_like(rng))
    q = torch.round((g - lo) / scale).clamp_(0, maxq)
    return q * scale + lo


def q_lloyd(g: torch.Tensor) -> torch.Tensor:
    """Two-bit Lloyd-Max for a normal, standardised per group.

    Four centroids at the conditional means of N(0,1), thresholds at the
    midpoints. Standardising per group is what makes the fixed codebook apply:
    the rotation equalises shape, not scale.
    """
    mean = g.mean(-1, keepdim=True)
    std = g.std(-1, keepdim=True).clamp_min(1e-8)
    z = (g - mean) / std
    t = torch.tensor(_LM_THRESHOLDS, dtype=g.dtype, device=g.device)
    idx = (z.unsqueeze(-1) >= t).sum(-1)
    c = torch.tensor(_LM_CENTROIDS, dtype=g.dtype, device=g.device)
    return c[idx] * std + mean


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-path", default="/shared/k3_c_kv_dump")
    ap.add_argument("--rot-dir", default="/shared/k3_latent_rot")
    ap.add_argument("--groups", default="32,64,128")
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--max-layers", type=int, default=6)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dump_path, "layer_*.pt")),
                   key=lambda p: int(os.path.basename(p)[6:-3]))
    step = max(1, len(files) // a.max_layers)
    files = files[::step][:a.max_layers]
    if not files:
        print(f"no dump under {a.dump_path}")
        return 2

    print(f"{'rot':>6} {'quant':>8} {'group':>6} {'bpe':>5} {'mean':>8} "
          f"{'worst':>8}  verdict")
    print("-" * 58)
    best = None
    for use_rot in (False, True):
        for gs in [int(x) for x in a.groups.split(",")]:
            for name in ("uniform", "lloyd"):
                errs = []
                for f in files:
                    c = torch.load(f, map_location="cpu")
                    if isinstance(c, dict):
                        c = c.get("c_kv", next(iter(c.values())))
                    c = c.reshape(-1, c.shape[-1]).float()
                    if use_rot:
                        li = int(os.path.basename(f)[6:-3])
                        rp = os.path.join(a.rot_dir, f"layer_{li}.pt")
                        if not os.path.exists(rp):
                            continue
                        r = torch.load(rp, map_location="cpu")
                        if isinstance(r, dict):
                            r = r.get("rotation", next(iter(r.values())))
                        c = c @ r.float()
                    x = c[c.shape[0] // 2:]           # held out
                    d = x.shape[-1]
                    g = x.reshape(-1, d // gs, gs)
                    deq = q_uniform(g, 2) if name == "uniform" else q_lloyd(g)
                    deq = deq.reshape(x.shape)
                    errs.append(((deq - x).norm() / x.norm()).item())
                if not errs:
                    continue
                mean, worst = sum(errs) / len(errs), max(errs)
                # uniform stores (scale, zero); lloyd stores (scale, zero) too,
                # so the amortized cost is identical at the same group size.
                bpe = 2 + 2 * 32 / gs
                ok = worst <= a.threshold
                if ok and (best is None or bpe < best[0]):
                    best = (bpe, name, gs, use_rot, worst)
                print(f"{str(use_rot):>6} {name:>8} {gs:>6} {bpe:>5.2f} "
                      f"{mean:>8.4f} {worst:>8.4f}  {'PASS' if ok else 'fail'}")
    print()
    if best:
        print(f"2 bits IS reachable: {best[1]} at group {best[2]}, rot={best[3]}, "
              f"{best[0]:.2f} bits/elem, worst {best[4]:.4f}")
    else:
        print(f"No 2-bit configuration reaches {a.threshold}. The codebook is not "
              f"the missing factor, and K3's latent needs more than two bits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
