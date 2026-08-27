#!/usr/bin/env python3
"""Is the ported mHC numerically the reference implementation?

Ports of a formula are exactly where "it looks right" is worthless. The mHC
mapping has four places a faithful-looking rewrite silently diverges:

  * ``comb`` is softmaxed and then Sinkhorn-projected, and the FIRST Sinkhorn
    step normalises columns only while the remaining iters-1 do rows-then-
    columns. A symmetric loop is off by half a step and still converges to a
    doubly-stochastic matrix -- just a different one.
  * ``post`` is ``2 * sigmoid``, range [0, 2]. Plain sigmoid halves every
    block's contribution and the model still produces text.
  * ``pre`` carries ``+ eps``; ``comb`` carries it too, in two places.
  * the whole mapping runs in float32 under an UNWEIGHTED RMS norm.

So compare against the actual transformers module, same weights, same input,
rather than eyeballing the port. Run with the reference package importable:

    PYTHONPATH=<extracted transformers 5.16.1>:<sglang python> python3 hc_equiv.py
"""
from __future__ import annotations

import os
import sys

import torch


def main() -> int:
    torch.manual_seed(0)
    ref_path = os.environ.get("REF_TF", "")
    if ref_path and ref_path not in sys.path:
        sys.path.insert(0, ref_path)

    try:
        from transformers.models.glm5_next.modeling_glm5_next import (
            Glm5NextTextHyperConnection,
        )
        import transformers
    except Exception as e:  # noqa: BLE001
        print(f"cannot import the reference: {type(e).__name__}: {str(e)[:200]}")
        print("set REF_TF to a directory containing transformers>=5.16.1")
        return 2
    print(f"reference transformers {transformers.__version__}")

    from sglang.srt.layers.hyper_connection import HyperConnection, hc_writeback

    H, D, T = 4, 128, 7           # small hidden: this tests maths, not throughput
    EPS, ITERS, NEPS = 1e-6, 20, 1e-5

    class Cfg:
        hc_mult, hc_eps, hc_sinkhorn_iters = H, EPS, ITERS
        hidden_size, rms_norm_eps = D, NEPS

    ref = Glm5NextTextHyperConnection(Cfg()).eval()
    mine = HyperConnection(D, H, EPS, ITERS, NEPS).eval()

    # Same weights in both. Deliberately not near-zero: a scale of 0 makes the
    # sigmoid branch constant and hides sign/ordering errors.
    with torch.no_grad():
        for p_ref, p_mine, init in (
            (ref.fn, mine.fn, torch.randn(ref.fn.shape) * 0.05),
            (ref.base, mine.base, torch.randn(ref.base.shape)),
            (ref.scale, mine.scale, torch.randn(ref.scale.shape) * 2.0),
        ):
            p_ref.copy_(init)
            p_mine.copy_(init)

    # Reference expects [B, S, H, D] and flattens start_dim=2; sglang hands a
    # flat [tokens, H, D]. Feed the reference [1, T, H, D] and mine [T, H, D],
    # which is exactly the shape difference the port has to survive.
    streams = torch.randn(T, H, D)
    with torch.no_grad():
        r_post, r_comb, r_coll = ref(streams.unsqueeze(0))
        m_post, m_comb, m_coll = mine(streams)

    ok = True
    for name, a, b in (
        ("post", r_post.squeeze(0), m_post),
        ("comb", r_comb.squeeze(0), m_comb),
        ("collapsed", r_coll.squeeze(0), m_coll),
    ):
        d = (a - b).abs().max().item()
        rel = d / max(a.abs().max().item(), 1e-9)
        good = rel < 1e-6
        ok &= good
        print(f"  {name:<10} max|d| {d:.3e}  rel {rel:.3e}  "
              f"{'MATCH' if good else 'MISMATCH'}")

    # Doubly stochastic, as the manifold constraint claims. If this fails the
    # Sinkhorn loop is wrong even when it matches a reference that is also wrong.
    rs = m_comb.sum(-1)
    cs = m_comb.sum(-2)
    print(f"  comb row sums  in [{rs.min():.6f}, {rs.max():.6f}]")
    print(f"  comb col sums  in [{cs.min():.6f}, {cs.max():.6f}]")

    # And the write-back, since that is where a transpose is easy to drop.
    y = torch.randn(T, D)
    with torch.no_grad():
        mine_wb = hc_writeback(m_post, m_comb, y, streams)
        ref_wb = r_post.squeeze(0).unsqueeze(-1) * y.unsqueeze(-2) + torch.matmul(
            r_comb.squeeze(0).transpose(-1, -2), streams
        )
    d = (ref_wb - mine_wb).abs().max().item()
    rel = d / max(ref_wb.abs().max().item(), 1e-9)
    ok &= rel < 1e-6
    print(f"  writeback  max|d| {d:.3e}  rel {rel:.3e}  "
          f"{'MATCH' if rel < 1e-6 else 'MISMATCH'}")

    # A negative control: if the port were the "obvious" implementation --
    # plain sigmoid for post, no Sinkhorn -- would this test have caught it?
    # A test that passes against a wrong implementation proves nothing.
    with torch.no_grad():
        naive_post = m_post / 2.0                      # sigmoid instead of 2*sigmoid
        naive_comb = torch.softmax(
            torch.randn(T, H, H), dim=-1)              # softmax, no Sinkhorn
    print(f"\n  [control] plain-sigmoid post would differ by rel "
          f"{((r_post.squeeze(0) - naive_post).abs().max() / r_post.abs().max()).item():.3e}")
    print(f"  [control] no-Sinkhorn comb col sums in "
          f"[{naive_comb.sum(-2).min():.3f}, {naive_comb.sum(-2).max():.3f}] "
          f"(doubly stochastic would be ~1.0)")

    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
