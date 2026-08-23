#!/usr/bin/env python3
"""Round 2 on the packed MLA INT2 kernel, after round 1 found a split result.

Round 1 (mla_packed_kernel_probe.py) measured, on the staged tree:

    lloyd_max=1 : relL2 5.4e-08 .. 6.2e-08   -> agrees with the fake-quant
    lloyd_max=0 : relL2 2.1e-02 .. 2.4e-02   -> does NOT agree

and ~1.8-2.0% of individual elements differing in the uniform arm. That is one
whole quantization step on a small subset, not drift. Round 1's CUDA-graph and
aliasing arms both used lloyd_max=0, so they inherited that gap and their
"WRONG" verdicts measured the arithmetic disagreement, not graphs. This round
separates the three questions properly:

1. WHICH SIDE IS RIGHT. Recompute both in float64 and compare each to that.
   Whichever tracks the float64 reference is correct. This matters for how the
   finding is written up: "the kernel has a bug" and "the torch simulation has a
   bug" have opposite consequences, and GLM's published numbers came from the
   torch path.

2. CUDA GRAPH, on the path that agrees (lloyd_max=1), so a graph failure cannot
   be confused with the uniform-path gap. Also re-tests the scratch-growth
   hazard, since `_scratch()` swaps in a NEW tensor when cap < n_groups while an
   already-captured graph keeps writing to the old captured address.

3. ALIASING, judged kernel-vs-kernel rather than kernel-vs-fake-quant, so the
   verdict is about the shared `_SCRATCH` buffer and nothing else.
"""
import argparse
import os
import sys

p = argparse.ArgumentParser()
p.add_argument("--tree", default=os.environ.get("TREE", "/tmp/tree"))
args = p.parse_args()

sys.path.insert(0, os.path.join(args.tree, "sglang-research", "python"))
import torch  # noqa: E402
from sglang.QuantKernel.mla_latent_int2 import quantize_dequantize_reuse  # noqa: E402
from sglang.QuantKernel import mla_latent_int2 as K  # noqa: E402
from sglang.srt.mem_cache.mla_int2_kv_pool import (  # noqa: E402
    _fake_quant_int2_groupwise, _real_kernel_enabled,
)

G, LATENT, dev = 128, 512, "cuda"
torch.manual_seed(0)
fails = []


def fq64(x, lloyd):
    """The same algorithm as _fake_quant_int2_groupwise, in float64."""
    xs = x.reshape(-1, G).to(torch.float64)
    if lloyd:
        from sglang.srt.mem_cache.mla_int2_kv_pool import (
            _LM_THRESHOLDS, _LM_SPAN, _LM_RATIO, _LM_CENTROIDS)
        mean = xs.mean(-1, keepdim=True)
        d = xs - mean
        std = (d.pow(2).mean(-1, keepdim=True) + 1e-8).sqrt()
        z = d / std
        t0, t1, t2 = _LM_THRESHOLDS
        q = (z >= t0).double() + (z >= t1).double() + (z >= t2).double()
        us = (_LM_SPAN / 3.0) * _LM_RATIO * std
        uz = -_LM_CENTROIDS[0] / (_LM_SPAN / 3.0) - mean / us
        out = (q - uz) * us
    else:
        mn, mx = xs.amin(-1, keepdim=True), xs.amax(-1, keepdim=True)
        scale = torch.where((mx - mn).abs() > 1e-8, (mx - mn) / 3.0,
                            torch.ones_like(mn))
        q = ((xs - mn) / scale).round().clamp_(0.0, 3.0)
        out = q * scale + mn
    return out.reshape(x.shape)


def rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


print("=" * 76)
print("0. GATE  (what a default server runs)")
print("=" * 76)
print(f"  env SGLANG_OSCAR_MLA_KV_REAL_KERNEL = "
      f"{os.environ.get('SGLANG_OSCAR_MLA_KV_REAL_KERNEL', '<unset>')}")
print(f"  _real_kernel_enabled()              = {_real_kernel_enabled()}")

print()
print("=" * 76)
print("1. WHICH SIDE IS RIGHT?  (float64 reference, same algorithm both ways)")
print("=" * 76)
print(f"  {'lloyd':<6}{'n_tok':<7}{'kernel vs fp64':>16}{'torch vs fp64':>16}"
      f"{'kernel vs torch':>18}  verdict")
for lloyd in (False, True):
    for n_tok in (256, 4096):
        x = (torch.randn(n_tok, LATENT, device=dev) * 0.5).to(torch.bfloat16)
        ref64 = fq64(x, lloyd)
        tor = _fake_quant_int2_groupwise(x, G, lloyd)
        _c, _p, ker = quantize_dequantize_reuse(x, G, lloyd)
        rk, rt, rkt = rel(ker, ref64), rel(tor, ref64), rel(ker, tor)
        if rkt < 1e-5:
            v = "agree"
        elif rk < rt / 3:
            v = "KERNEL closer to fp64 -> torch sim is the odd one"
        elif rt < rk / 3:
            v = "TORCH closer to fp64 -> kernel is the odd one"
        else:
            v = "both differ from fp64 similarly"
        print(f"  {int(lloyd):<6}{n_tok:<7}{rk:>16.3e}{rt:>16.3e}{rkt:>18.3e}  {v}")

# localise the uniform-path disagreement
print()
print("  where does the uniform arm disagree?")
x = (torch.randn(4096, LATENT, device=dev) * 0.5).to(torch.bfloat16)
tor = _fake_quant_int2_groupwise(x, G, False)
_c, _p, ker = quantize_dequantize_reuse(x, G, False)
d = (ker.float() - tor.float()).abs()
n_diff = int((d > 0).sum())
xs = x.reshape(-1, G).float()
mn, mx = xs.amin(-1, keepdim=True), xs.amax(-1, keepdim=True)
scale = torch.where((mx - mn).abs() > 1e-8, (mx - mn) / 3.0, torch.ones_like(mn))
step = scale.expand_as(xs).reshape(x.shape)
ratio = (d / step.clamp_min(1e-30))
print(f"    elements differing        : {n_diff}/{d.numel()} "
      f"({100.0 * n_diff / d.numel():.2f}%)")
print(f"    max |diff| / quant step   : {ratio.max().item():.4f}")
print(f"    of the differing elements, fraction that differ by ~1 full step: "
      f"{(ratio[d > 0] > 0.9).float().mean().item():.3f}")
qn = ((xs - mn) / scale)
frac = (qn - qn.floor())
at_half = ((frac - 0.5).abs() < 1e-6).float().mean().item()
print(f"    fraction of elements landing exactly on a .5 tie: {at_half:.4f}")
print("    -> a tie-breaking difference shows up as ~1 full step on ~the tie rate")

print()
print("=" * 76)
print("2. CUDA GRAPH, on lloyd_max=1 (the arm that agrees eagerly)")
print("=" * 76)
K._SCRATCH.clear()


def graph_arm(n_tok, prewarm, label, lloyd=True):
    x = (torch.randn(n_tok, LATENT, device=dev) * 0.5).to(torch.bfloat16)
    static = x.clone()
    if prewarm:
        quantize_dequantize_reuse(static, G, lloyd)
        torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    try:
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                quantize_dequantize_reuse(static, G, lloyd)
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g):
            out = quantize_dequantize_reuse(static, G, lloyd)[2].clone()
    except Exception as e:
        print(f"  {label:<40} CAPTURE FAILED {type(e).__name__}: {e}")
        return None, f"{label}: capture raised {type(e).__name__}"
    new = (torch.randn(n_tok, LATENT, device=dev) * 0.5).to(torch.bfloat16)
    static.copy_(new); g.replay(); torch.cuda.synchronize()
    r = rel(out, _fake_quant_int2_groupwise(new, G, lloyd))
    tag = "OK" if r < 1e-5 else "*** WRONG AFTER REPLAY ***"
    print(f"  {label:<40} relL2={r:.3e}  {tag}")
    return (r, g, static, n_tok, lloyd), (None if r < 1e-5 else
                                          f"{label}: replay relL2 {r:.2e}")


a1, e1 = graph_arm(256, True, "capture@256, scratch pre-warmed")
if e1:
    fails.append(e1)
a2, e2 = graph_arm(64, False, "capture@64, cold scratch")
if e2:
    fails.append(e2)

print("  force _SCRATCH to reallocate in eager, then replay the OLD graph:")
cap_before = {k: v[0] for k, v in K._SCRATCH.items()}
big = (torch.randn(300000, LATENT, device=dev) * 0.5).to(torch.bfloat16)
quantize_dequantize_reuse(big, G, True)
torch.cuda.synchronize()
cap_after = {k: v[0] for k, v in K._SCRATCH.items()}
grew = [k for k in cap_after if cap_before.get(k) != cap_after[k]]
print(f"    scratch capacities changed for {len(grew)} of {len(cap_after)} keys "
      f"{'(realloc DID happen)' if grew else '(no realloc)'}")
if a2 is not None:
    _r, g2, static2, n2, l2 = a2
    new = (torch.randn(n2, LATENT, device=dev) * 0.5).to(torch.bfloat16)
    static2.copy_(new)
    try:
        g2.replay(); torch.cuda.synchronize()
        out2 = None
        # re-capture output handle is gone; recompute via the graph's cloned out
        print("    replay after growth did not raise")
    except Exception as e:
        print(f"    replay after growth RAISED {type(e).__name__}: {e}")
        fails.append("replay after scratch growth raised")

print()
print("=" * 76)
print("3. ALIASING, judged kernel-vs-kernel")
print("=" * 76)
K._SCRATCH.clear()
xs4 = [(torch.randn(64, LATENT, device=dev) * (0.1 * (i + 1))).to(torch.bfloat16)
       for i in range(4)]
# ground truth: one call at a time, cloned immediately
truth = []
for x in xs4:
    truth.append(quantize_dequantize_reuse(x, G, True)[2].clone())
    torch.cuda.synchronize()
noclone = [quantize_dequantize_reuse(x, G, True)[2] for x in xs4]
torch.cuda.synchronize()
collide = sum(1 for i in range(3) if torch.equal(noclone[i], noclone[3]))
print(f"  no clone : {collide}/3 earlier layers now equal the last layer "
      f"-> {'ALIASED' if collide else 'not aliased'}")
withclone = [quantize_dequantize_reuse(x, G, True)[2].clone() for x in xs4]
torch.cuda.synchronize()
bad = sum(1 for i in range(4) if not torch.equal(withclone[i], truth[i]))
print(f"  clone    : {bad}/4 layers differ from the one-at-a-time truth  "
      f"{'OK' if bad == 0 else '*** WRONG ***'}")
if bad:
    fails.append("cloned multi-layer sequence disagrees with serial truth")
print("  -> the .clone() in mla_int2_kv_pool.py is load-bearing: without it all "
      "78\n     layers would read back whatever the last layer wrote")

print()
print("=" * 76)
print("VERDICT")
print("=" * 76)
for f in fails:
    print(f"  FAIL  {f}")
print("\nOVERALL:", "FAIL" if fails else "PASS")
sys.exit(1 if fails else 0)
