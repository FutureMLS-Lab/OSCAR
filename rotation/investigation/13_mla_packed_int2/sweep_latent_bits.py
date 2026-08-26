#!/usr/bin/env python3
"""Bound what it would take to quantize K3's MLA latent, before writing any kernel.

check_latent_rotation says every K3 full-attention layer sits at 0.28-0.46
out-of-sample relative error under INT2 at group size 128, against the 0.10 a
2-bit KV cache needs. That is 3-5x over, and it explains the garbling directly:
the packed arm produced 36.5% non-ascii at 64 generated tokens while the BF16
latent arm produced 0.0%. So the fault is neither the forward path nor the
decode kernel -- it is that this tensor is not INT2-representable at this
grouping.

Before spending more time on rotations or kernels, bound the problem: sweep the
group size and the bit width on the dump already on disk. The two questions
that decide what to build next are

  * does a smaller group get INT2 under threshold, i.e. is this a grouping
    problem with a cheap fix (more scales, slightly more bits per element), or
  * does it take 3 or 4 bits regardless, i.e. is the latent genuinely
    high-entropy and no amount of rotation will make 2 bits work

Reported per configuration: mean and worst out-of-sample relative error across
layers, and the amortized bits per element including the per-group scale and
zero point -- because a smaller group buys accuracy with storage, and a
configuration that needs 4 bits/elem at group 32 is not cheaper than plain INT4.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))


def fake_quant(x: torch.Tensor, group_size: int, bits: int) -> torch.Tensor:
    """Asymmetric per-group uniform quantization, the pool's storage model."""
    d = x.shape[-1]
    g = x.reshape(-1, d // group_size, group_size)
    lo = g.amin(-1, keepdim=True)
    hi = g.amax(-1, keepdim=True)
    n = (1 << bits) - 1
    scale = (hi - lo).clamp_min(1e-8) / n
    q = ((g - lo) / scale).round().clamp_(0, n)
    return (q * scale + lo).reshape(x.shape)


def rel_err(x: torch.Tensor, rot: torch.Tensor | None,
            group_size: int, bits: int) -> float:
    y = x if rot is None else x @ rot
    e = fake_quant(y, group_size, bits)
    return ((e - y).norm() / y.norm().clamp_min(1e-12)).item()


def bits_per_elem(group_size: int, bits: int, scale_bits: int = 32) -> float:
    """Amortized cost: the payload plus one scale and one zero point per group."""
    return bits + 2.0 * scale_bits / group_size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-path", required=True)
    ap.add_argument("--rot-dir", default=None,
                    help="if given, evaluate the shipped rotation as well")
    ap.add_argument("--groups", default="32,64,128,256")
    ap.add_argument("--bits", default="2,3,4")
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--max-layers", type=int, default=8,
                    help="layers to sample; the spread across layers is small")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dump_path, "layer_*.pt")),
                   key=lambda p: int(os.path.basename(p)[6:-3]))
    if not files:
        print(f"no layer_*.pt under {a.dump_path}")
        return 2
    step = max(1, len(files) // a.max_layers)
    files = files[::step][:a.max_layers]

    groups = [int(x) for x in a.groups.split(",")]
    bits_list = [int(x) for x in a.bits.split(",")]

    # Load the shipped rotation per layer. Sweeping without it answers a
    # different question than the one that matters -- the rotation is worth
    # roughly 1.8x on this tensor at group 128, which is more than the gap
    # between two adjacent bit widths, so a table computed with rot=None can be
    # a full bit pessimistic.
    rots = {}
    if a.rot_dir:
        for f in files:
            li = int(os.path.basename(f)[6:-3])
            rp = os.path.join(a.rot_dir, f"layer_{li}.pt")
            if os.path.exists(rp):
                r = torch.load(rp, map_location="cpu")
                if isinstance(r, dict):
                    r = r.get("rotation", next(iter(r.values())))
                rots[f] = r.float()
        print(f"rotations loaded: {len(rots)}/{len(files)}"
              + ("" if len(rots) == len(files) else
                 "  !! missing rotations are scored UNROTATED"))

    cache = {}
    for f in files:
        c = torch.load(f, map_location="cpu")
        if isinstance(c, dict):
            c = c.get("c_kv", next(iter(c.values())))
        cache[f] = c.reshape(-1, c.shape[-1]).float()

    print(f"layers sampled: {[os.path.basename(f) for f in files]}\n")
    hdr = (f"{'rot':>5} {'bits':>4} {'group':>6} {'bpe':>6} {'mean':>8} "
           f"{'worst':>8}  verdict")
    print(hdr)
    print("-" * len(hdr))
    best = None
    modes = [("none", False)] + ([("fitted", True)] if rots else [])
    for label, use_rot in modes:
        for bits in bits_list:
            for gs in groups:
                errs = []
                for f in files:
                    c = cache[f]
                    if c.shape[-1] % gs:
                        errs = []
                        break
                    # Held out: scoring on the half the rotation was not fit on.
                    half = c.shape[0] // 2
                    errs.append(rel_err(c[half:],
                                        rots.get(f) if use_rot else None,
                                        gs, bits))
                if not errs:
                    continue
                mean, worst = sum(errs) / len(errs), max(errs)
                bpe = bits_per_elem(gs, bits)
                ok = worst <= a.threshold
                if ok and (best is None or bpe < best[0]):
                    best = (bpe, bits, gs, worst, label)
                print(f"{label:>5} {bits:>4} {gs:>6} {bpe:>6.2f} {mean:>8.4f} "
                      f"{worst:>8.4f}  {'PASS' if ok else 'fail'}")
    print()
    if best:
        print(f"cheapest configuration under threshold: {best[1]} bits at group "
              f"{best[2]}, rotation={best[4]} = {best[0]:.2f} bits/elem "
              f"(worst {best[3]:.4f})")
        print("Compare against bf16 at 16 bits/elem for the compression ratio "
              "this path can actually claim.")
    else:
        print(f"NO configuration in this sweep reaches worst <= {a.threshold}. "
              f"Rotation work cannot close a gap this size; the latent needs "
              f"more bits than any setting tried here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
