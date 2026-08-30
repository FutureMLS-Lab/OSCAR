#!/usr/bin/env python3
"""Round-trip the packed latent kernels at 2 and 4 bits against a torch reference.

Two things must hold, and they protect different risks.

At **2 bits** the result must be UNCHANGED. That path is validated -- it carries
GLM-5.2's 76.03 GPQA -- and the bit-width parameterization touched every kernel
in it. A torch reference reproduces the documented quantizer exactly (uniform:
q = round((x - min) / s) clamped, x_deq = q * scale + min), so any drift shows up
as a per-element mismatch rather than as a slightly worse benchmark six hours
later.

At **4 bits** the result must be CORRECT and must actually use the extra bits.
The specific trap: Lloyd-Max is a three-threshold codebook, so a width-blind
implementation emits codes in 0..3 while dequant reads a four-bit field. That
looks like "it runs" and produces roughly 2-bit error at twice the storage. So
this asserts on the observed code range, not only on the error.

Also checked: the packed buffer is D // (8 // bits) bytes wide, because the
kernel derives the latent dimension back out of that width -- at the wrong
packing factor it recovers half the dimension and every index downstream is
wrong with nothing to catch it.
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/work/oscar/sglang-research/python")

from sglang.QuantKernel.mla_latent_int2 import (  # noqa: E402
    dequantize,
    quantize_pack,
)


def ref_uniform(x: torch.Tensor, group_size: int, bits: int) -> torch.Tensor:
    """The documented uniform quantizer, in torch."""
    d = x.shape[-1]
    g = x.reshape(-1, d // group_size, group_size).float()
    lo = g.amin(-1, keepdim=True)
    rng = g.amax(-1, keepdim=True) - lo
    maxq = float((1 << bits) - 1)
    scale = torch.where(rng.abs() > 1e-8, rng / maxq, torch.ones_like(rng))
    q = torch.round((g - lo) / scale).clamp_(0, maxq)
    return (q * scale + lo).reshape(x.shape)


def unpack(codes: torch.Tensor, group_size: int, bits: int) -> torch.Tensor:
    pf = 8 // bits
    mask = (1 << bits) - 1
    c = codes.reshape(-1, group_size // pf).to(torch.int32)
    out = torch.empty((c.shape[0], group_size), dtype=torch.int32, device=c.device)
    for j in range(pf):
        out[:, j::pf] = (c >> (bits * j)) & mask
    return out


def main() -> int:
    torch.manual_seed(0)
    dev = "cuda"
    R, GS, N = 512, 128, 2048
    x = torch.randn(N, R, device=dev, dtype=torch.float32)
    rc = 0

    for bits in (2, 4):
        codes, params = quantize_pack(x, group_size=GS, lloyd_max=False, bits=bits)
        deq = dequantize(codes, params, (N, R), group_size=GS,
                         lloyd_max=False, bits=bits)
        ref = ref_uniform(x, GS, bits)

        want_w = GS // (8 // bits)
        got_w = codes.shape[-1]
        width_ok = got_w == want_w

        # The kernel recovers D from the byte width; check the round trip the
        # launcher actually performs.
        d_recovered = (R // GS) * got_w * (8 // bits)
        d_ok = d_recovered == R

        diff = (deq.reshape(N, R) - ref).abs()
        total = diff.numel()
        rel = (deq.reshape(N, R) - x).norm() / x.norm()

        q = unpack(codes, GS, bits)
        qmax, qmin = int(q.max()), int(q.min())
        uses_width = qmax == (1 << bits) - 1

        # Compare CODES, not dequantized floats. The two implementations divide
        # and round in different orders, so the reconstructed values differ in
        # the last fp32 ULP on a few percent of elements -- which an exact-float
        # comparison reports as a 2.5% failure while max |diff| stays at 4.8e-07,
        # seven orders below a quantization step. The invariant that matters is
        # that the same code was chosen; a real drift moves a code by a full
        # step, which is O(scale), not O(eps).
        g = x.reshape(-1, R // GS, GS).float()
        lo = g.amin(-1, keepdim=True)
        rng = g.amax(-1, keepdim=True) - lo
        maxq_t = float((1 << bits) - 1)
        sc = torch.where(rng.abs() > 1e-8, rng / maxq_t, torch.ones_like(rng))
        q_ref = torch.round((g - lo) / sc).clamp_(0, maxq_t).to(torch.int32)
        q_ref = q_ref.reshape(-1, GS)
        code_mismatch = int((q.reshape(-1, GS) != q_ref).sum())
        step = sc.mean().item()

        print(f"\n=== bits={bits} ===")
        print(f"  codes width      {got_w} (want {want_w})  {'ok' if width_ok else 'WRONG'}")
        print(f"  D recovered      {d_recovered} (want {R})  {'ok' if d_ok else 'WRONG'}")
        print(f"  code mismatches  {code_mismatch}/{total} = "
              f"{100.0*code_mismatch/total:.4f}%")
        print(f"  max |diff|       {diff.max().item():.3e}  "
              f"(one quantization step is ~{step:.3f})")
        print(f"  code range       [{qmin}, {qmax}]  "
              f"{'ok' if uses_width else 'DOES NOT USE THE FULL WIDTH'}")
        print(f"  round-trip rel   {rel.item():.4f}")

        if not (width_ok and d_ok and uses_width):
            rc = 1
        # The documented limit is ~0.010% of codes differing by one step on raw
        # c_kv, from quotients that land within half a ULP of a .5 boundary.
        if code_mismatch > total * 0.001:
            print(f"  !! {100.0*code_mismatch/total:.3f}% of CODES differ from the "
                  f"reference -- that is drift, not rounding")
            rc = 1
        # Independently: no reconstructed value may be off by anything
        # approaching a quantization step.
        if diff.max().item() > 0.01 * step:
            print("  !! a reconstructed value is off by an appreciable fraction "
                  "of a quantization step")
            rc = 1

    # One extra bit halves the step, so two extra bits should cut the error by
    # roughly 4x. A width-blind 4-bit path that emits 2-bit codes would land
    # near 1.0 here and pass every check above except the code range.
    c2, p2 = quantize_pack(x, group_size=GS, lloyd_max=False, bits=2)
    c4, p4 = quantize_pack(x, group_size=GS, lloyd_max=False, bits=4)
    e2 = (dequantize(c2, p2, (N, R), group_size=GS, bits=2).reshape(N, R) - x).norm()
    e4 = (dequantize(c4, p4, (N, R), group_size=GS, bits=4).reshape(N, R) - x).norm()
    ratio = (e2 / e4).item()
    print(f"\nerror ratio 2-bit / 4-bit = {ratio:.2f}  (expect ~4)")
    if not (2.5 < ratio < 6.0):
        print("  !! the extra bits are not doing what extra bits do")
        rc = 1

    print("\nVERDICT:", "PASS" if rc == 0 else "FAIL")
    return rc


if __name__ == "__main__":
    sys.exit(main())
