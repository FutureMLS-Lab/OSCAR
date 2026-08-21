#!/usr/bin/env python3
"""Where does M2.7's 2-bit K error land: the RoPE'd half or the static half?

M2.7 has ``head_dim=128`` but ``rotary_dim=64``. Dims 0-63 carry position, dims
64-127 carry static content, and the stock recipe puts ONE 128x128 rotation and
ONE 2-bit scale across both. If the static half has the larger dynamic range, the
shared scale is set by it and the positional half is quantized far more coarsely
than its own magnitude would warrant -- and positional resolution is exactly what
a 15K-token reasoning chain depends on.

This measures that directly on the real calibration dump, which is far cheaper
than a serving arm and decides whether a mixed-precision serving path is worth
implementing at all.

Reported per layer:

* ``rms_rope`` / ``rms_nope``  -- raw per-half magnitude of K before any
  rotation. If these differ a lot, a shared scale is already suspect.
* ``relerr_rope`` / ``relerr_nope`` -- relative error of each half AFTER a full
  rotate -> 2-bit -> unrotate round trip, measured back in the original
  coordinates. This is the number the hypothesis predicts will be lopsided.
* ``logit_relerr`` -- error in the actual q.k attention logits, which is what
  the model consumes. Decomposed into the contribution of each half, because a
  half can be badly quantized and still not matter if it carries little logit
  signal.
* the same three under a block-diagonal rotation, which cannot move energy
  across the boundary.

The logit metric is the one to weight: per-half reconstruction error is a proxy,
q.k is the quantity attention actually uses.
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


def quant_int2_asym(x: torch.Tensor, group: int, clip: float) -> torch.Tensor:
    shape = x.shape
    xg = x.reshape(-1, shape[-1] // group, group)
    lo = xg.amin(-1, keepdim=True) * clip
    hi = xg.amax(-1, keepdim=True) * clip
    scale = (hi - lo).clamp(min=1e-12) / 3.0
    zero = -lo / scale
    q = (xg / scale + zero + 0.5).floor().clamp(0, 3)
    return ((q - zero) * scale).reshape(shape)


def load_all(layer_dir: str, name: str, limit: int) -> torch.Tensor:
    paths = sorted(glob.glob(os.path.join(layer_dir, name, "*.pt")),
                   key=lambda p: int(os.path.basename(p)[:-3]))
    out, n = [], 0
    for p in paths:
        t = torch.load(p, map_location="cpu", weights_only=True).float().double()
        out.append(t)
        n += t.shape[0]
        if n >= limit:
            break
    return torch.cat(out, 0)[:limit]


def relerr(a: torch.Tensor, b: torch.Tensor) -> float:
    return (torch.linalg.norm(b - a) / torch.linalg.norm(a).clamp(min=1e-30)).item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-path", required=True)
    ap.add_argument("--rot", required=True, help="full 128x128 rotation .pt")
    ap.add_argument("--block-rot", default=None, help="block-diagonal .pt")
    ap.add_argument("--layers", default="0,10,20,31,41,51,61")
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--queries", type=int, default=256)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--clip", type=float, default=0.96)
    ap.add_argument("--rotary-dim", type=int, default=64)
    a = ap.parse_args()

    full = torch.load(a.rot, map_location="cpu", weights_only=False)["layers"]
    blk = (torch.load(a.block_rot, map_location="cpu", weights_only=False)["layers"]
           if a.block_rot else None)
    rd = a.rotary_dim
    scale = 1.0 / math.sqrt(128)

    print(f"{'layer':>6}{'rms_rope':>10}{'rms_nope':>10}"
          f"{'relerr_rope':>13}{'relerr_nope':>13}{'logit_rel':>11}"
          f"{'B_err_rope':>12}{'B_err_nope':>12}{'B_logit':>10}")
    agg = {}
    for li in [int(x) for x in a.layers.split(",")]:
        ldir = os.path.join(a.dump_path, f"layer_{li}")
        k = load_all(ldir, "k", a.tokens)      # [T, kv_heads, 128]
        q = load_all(ldir, "q", a.tokens)      # [T, n_heads, 128]
        kv_heads = k.shape[1]
        gqa = q.shape[1] // kv_heads

        rms_r = k[:, :, :rd].pow(2).mean().sqrt().item()
        rms_n = k[:, :, rd:].pow(2).mean().sqrt().item()

        row = {"rms_rope": rms_r, "rms_nope": rms_n}
        for tag, src in (("A", full), ("B", blk)):
            if src is None:
                continue
            R = src[li]["rotation"].double() if li in src else src[str(li)]["rotation"].double()
            # Round trip exactly as the runtime does it: rows @ R, quantize in
            # the rotated basis, then @ R.T on the way out.
            kq = (quant_int2_asym(k @ R, a.group_size, a.clip)) @ R.T
            row[f"{tag}_relerr_rope"] = relerr(k[:, :, :rd], kq[:, :, :rd])
            row[f"{tag}_relerr_nope"] = relerr(k[:, :, rd:], kq[:, :, rd:])
            # Attention logits for the last `queries` positions against all keys,
            # per KV group, exact vs quantized K.
            errs, sigs = [], []
            for h in range(kv_heads):
                qh = q[-a.queries :, h * gqa : (h + 1) * gqa, :].reshape(-1, 128)
                kh, khq = k[:, h, :], kq[:, h, :]
                lo = (qh @ kh.T) * scale
                lq = (qh @ khq.T) * scale
                errs.append(torch.linalg.norm(lq - lo).item() ** 2)
                sigs.append(torch.linalg.norm(lo).item() ** 2)
            row[f"{tag}_logit"] = math.sqrt(sum(errs) / max(sum(sigs), 1e-30))

        # Upper bound on the GLM-style "keep the positional half in BF16" fix,
        # and its mirror image, without implementing a mixed-precision serving
        # path. Each variant quantizes only ONE half (in that half's own rotated
        # basis, using that half's own scale) and leaves the other exact, so the
        # logit error it produces is the best such a split could possibly do.
        # Sub-blocks of the full rotation are not orthogonal on their own, so the
        # block-diagonal file is the right rotation to use for a half-only
        # scheme -- that is precisely the rotation a split-precision layout could
        # actually apply.
        if blk is not None:
            Rb = (blk[li]["rotation"] if li in blk
                  else blk[str(li)]["rotation"]).double()
            for tag, sl in (("ropeBF16", slice(rd, 128)),
                            ("nopeBF16", slice(0, rd))):
                # Quantize only `sl`; the complement stays exact.
                kq = k.clone()
                sub = k[:, :, sl] @ Rb[sl, sl]
                kq[:, :, sl] = quant_int2_asym(sub, sl.stop - sl.start,
                                               a.clip) @ Rb[sl, sl].T
                errs, sigs = [], []
                for h in range(kv_heads):
                    qh = q[-a.queries :, h * gqa : (h + 1) * gqa, :].reshape(-1, 128)
                    lo = (qh @ k[:, h, :].T) * scale
                    lq = (qh @ kq[:, h, :].T) * scale
                    errs.append(torch.linalg.norm(lq - lo).item() ** 2)
                    sigs.append(torch.linalg.norm(lo).item() ** 2)
                row[f"{tag}_logit"] = math.sqrt(sum(errs) / max(sum(sigs), 1e-30))
        print(f"{li:>6}{rms_r:>10.3f}{rms_n:>10.3f}"
              f"{row['A_relerr_rope']:>13.4f}{row['A_relerr_nope']:>13.4f}"
              f"{row['A_logit']:>11.4f}"
              + (f"{row['B_relerr_rope']:>12.4f}{row['B_relerr_nope']:>12.4f}"
                 f"{row['B_logit']:>10.4f}" if blk else ""))
        for kk, vv in row.items():
            agg.setdefault(kk, []).append(vv)

    print("\nmeans over audited layers:")
    for kk in sorted(agg):
        print(f"  {kk:<18} {sum(agg[kk]) / len(agg[kk]):.4f}")


if __name__ == "__main__":
    main()
