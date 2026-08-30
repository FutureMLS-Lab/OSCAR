#!/usr/bin/env python3
"""Round-trip REAL c_kv through the kernels serving actually uses.

test_bits.py validated quantize_pack/dequantize -- the reference kernels. The
serving path does not call them. It calls scatter_pack_rows on the write and
gather_dequant_rows on the read, which are the block-tiled variants, and those
are what the bit-width parameterization also touched. Validating the reference
pair and shipping the other pair is how a width knob can pass its test and still
garble a server, which is exactly what happened: 4-bit K3 produced 68.5%
non-ascii at 64 generated tokens where the offline sweep predicted 0.0694
relative error.

So: take the real dump, apply the shipped rotation exactly as the pool does on
write, scatter it into a codes/params buffer, gather it back, and compare
against both the input and the offline prediction. A serving-path error far
above the offline number localises the fault to these kernels rather than to
the choice of bit width.

Scattering to non-contiguous slots is deliberate. The pool writes rows at
allocator-chosen locations, so a kernel that is correct only for slot == row
index passes a contiguous test and fails in production.
"""
from __future__ import annotations

import glob
import os
import sys

import torch

sys.path.insert(0, "/work/oscar/sglang-research/python")

from sglang.QuantKernel.mla_latent_int2 import (  # noqa: E402
    gather_dequant_rows,
    scatter_pack_rows,
)

DUMP = os.environ.get("DUMP", "/shared/k3_c_kv_dump")
ROT = os.environ.get("ROT", "/shared/k3_latent_rot")
GS = int(os.environ.get("GS", "128"))
NROWS = int(os.environ.get("NROWS", "4096"))


def main() -> int:
    dev = "cuda"
    files = sorted(glob.glob(os.path.join(DUMP, "layer_*.pt")),
                   key=lambda p: int(os.path.basename(p)[6:-3]))
    if not files:
        print(f"no dump under {DUMP}")
        return 2
    rc = 0
    print(f"{'layer':>6} {'bits':>4} {'rot':>6} {'serving rel':>12} "
          f"{'offline rel':>12}  verdict")
    print("-" * 62)
    for f in files[::8][:3]:
        li = int(os.path.basename(f)[6:-3])
        c = torch.load(f, map_location="cpu")
        if isinstance(c, dict):
            c = c.get("c_kv", next(iter(c.values())))
        c = c.reshape(-1, c.shape[-1])[:NROWS].to(dev, torch.float32)
        rp = os.path.join(ROT, f"layer_{li}.pt")
        rot = None
        if os.path.exists(rp):
            r = torch.load(rp, map_location="cpu")
            if isinstance(r, dict):
                r = r.get("rotation", next(iter(r.values())))
            rot = r.to(dev, torch.float32)

        for use_rot in ([False, True] if rot is not None else [False]):
            x = c @ rot if use_rot else c
            for bits in (2, 4):
                n, D = x.shape
                pf = 8 // bits
                codes = torch.zeros((n, D // pf), dtype=torch.uint8, device=dev)
                params = torch.zeros((n, (D // GS) * 2), dtype=torch.float32,
                                     device=dev)
                # Non-contiguous, shuffled slots: production never writes rows
                # in order, and a kernel correct only for the identity mapping
                # would pass a contiguous test.
                slots = torch.randperm(n, device=dev).to(torch.int32)
                scatter_pack_rows(x, slots, codes, params, GS, False, bits)
                out = torch.empty((n, D), dtype=torch.float32, device=dev)
                gather_dequant_rows(slots, codes, params, out, GS, False, bits)
                serving = ((out - x).norm() / x.norm()).item()

                # offline prediction: the same uniform quantizer in torch
                g = x.reshape(-1, D // GS, GS)
                lo = g.amin(-1, keepdim=True)
                rng = g.amax(-1, keepdim=True) - lo
                mq = float((1 << bits) - 1)
                sc = torch.where(rng.abs() > 1e-8, rng / mq, torch.ones_like(rng))
                q = torch.round((g - lo) / sc).clamp_(0, mq)
                ref = (q * sc + lo).reshape(x.shape)
                offline = ((ref - x).norm() / x.norm()).item()

                ok = serving <= offline * 1.10 + 1e-4
                if not ok:
                    rc = 1
                print(f"{li:>6} {bits:>4} {str(use_rot):>6} {serving:>12.4f} "
                      f"{offline:>12.4f}  {'ok' if ok else 'SERVING PATH IS WORSE'}")
    print("\nVERDICT:", "PASS" if rc == 0 else "FAIL")
    if rc:
        print("The serving kernels do not reproduce the offline quantizer, so "
              "the fault is in scatter_pack_rows/gather_dequant_rows, not in "
              "the choice of bit width.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
