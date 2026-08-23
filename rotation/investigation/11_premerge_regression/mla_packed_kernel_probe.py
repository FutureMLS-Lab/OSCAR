#!/usr/bin/env python3
"""Does the packed MLA INT2 kernel (a) match the fake-quant, (b) survive CUDA
graph capture/replay, and (c) actually save memory?

Context. `SGLANG_OSCAR_MLA_KV_REAL_KERNEL` (environ.py, EnvBool(False)) gates
`_real_kernel_enabled()` in mla_int2_kv_pool.py. With it unset -- which is how
every published GLM-5.2 number was produced -- the write path calls
`_fake_quant_int2_groupwise` (pure torch). With it set, the path calls
`quantize_dequantize_reuse` from QuantKernel/mla_latent_int2.py.

Three things this probe pins down, all of which are claims a reader of the PR
would otherwise have to take on faith:

1. AGREEMENT. The kernel is said to match the simulation to relL2 ~3.6e-07. The
   committed test (tests/test_mla_kernel.py) hardcodes
   `/home/charlie/.../work/oscar/sglang-research/python` on sys.path, a path that
   does not exist in a clean checkout, so that number is not reproducible as
   shipped. Re-measure it here.

2. CUDA GRAPH. `quantize_dequantize_reuse` returns *views into module-level
   scratch* (`_SCRATCH` dict in mla_latent_int2.py). `_scratch()` reallocates
   whenever `cap < n_groups`, replacing the dict entry with a NEW tensor. Under
   graph capture that is a concrete hazard: a graph captured before a realloc
   still writes to the old captured address, while the dict -- and therefore any
   later eager call -- points somewhere else. The caller's `.clone()` is what
   stops all 78 layers aliasing one buffer inside a single forward. So:
     2a. confirm the aliasing is real (drop the clone -> layers collide)
     2b. capture/replay across a scratch GROWTH boundary and check correctness

3. MEMORY. `_real_kernel_enabled`'s own docstring says it does NOT save memory:
   "the codes are unpacked straight back into the BF16 pool, because the MLA
   attention path has no INT2 read support." The returned tensor is `out`, the
   dequantized values; `codes`/`params` are discarded by the caller. Verify that
   the packed codes are genuinely thrown away, i.e. this is a *write-path speed*
   change, not a KV-footprint change.
"""
import argparse
import os
import sys

p = argparse.ArgumentParser()
p.add_argument("--tree", default=os.environ.get("TREE", "/tmp/tree"))
p.add_argument("--rot", default=None, help="a GLM-5.2 layer_*.pt to use as real c_kv geometry")
args = p.parse_args()

sys.path.insert(0, os.path.join(args.tree, "sglang-research", "python"))
import torch  # noqa: E402
from sglang.QuantKernel.mla_latent_int2 import (  # noqa: E402
    quantize_dequantize_reuse, quantize_pack, dequantize, bytes_per_value,
)
from sglang.QuantKernel import mla_latent_int2 as K  # noqa: E402
from sglang.srt.mem_cache.mla_int2_kv_pool import (  # noqa: E402
    _fake_quant_int2_groupwise, _real_kernel_enabled,
)

G = 128
LATENT = 512          # GLM-5.2 kv_lora_rank
dev = "cuda"
torch.manual_seed(0)
fails = []

print("=" * 74)
print("0. GATE STATE  (what a default server actually runs)")
print("=" * 74)
print(f"  SGLANG_OSCAR_MLA_KV_REAL_KERNEL in env : "
      f"{os.environ.get('SGLANG_OSCAR_MLA_KV_REAL_KERNEL', '<unset>')}")
print(f"  _real_kernel_enabled()                 : {_real_kernel_enabled()}")
print("  -> when False the write path is _fake_quant_int2_groupwise (torch)")

print()
print("=" * 74)
print("1. AGREEMENT vs the fake-quant simulation")
print("=" * 74)
for lloyd in (False, True):
    for n_tok in (8, 1000, 4096):
        x = (torch.randn(n_tok, LATENT, device=dev) * 0.5).to(torch.bfloat16)
        ref = _fake_quant_int2_groupwise(x, G, lloyd)
        _c, _p, got = quantize_dequantize_reuse(x, G, lloyd)
        r32, g32 = ref.float(), got.float()
        rel = ((r32 - g32).norm() / r32.norm().clamp_min(1e-9)).item()
        exact = int((r32 == g32).sum()), r32.numel()
        tag = "OK" if rel < 1e-5 else "*** MISMATCH ***"
        print(f"  lloyd={int(lloyd)} n_tok={n_tok:<5} relL2={rel:.3e}  "
              f"bit-exact {exact[0]}/{exact[1]}  {tag}")
        if rel >= 1e-5:
            fails.append(f"agreement lloyd={lloyd} n_tok={n_tok} relL2={rel:.2e}")

print()
print("=" * 74)
print("2a. ALIASING -- is the caller's .clone() load-bearing?")
print("=" * 74)
# emulate 78 layers writing through the reuse path in one forward, no clone
xs = [(torch.randn(64, LATENT, device=dev) * (0.1 * (i + 1))).to(torch.bfloat16)
      for i in range(4)]
no_clone = [quantize_dequantize_reuse(x, G, False)[2] for x in xs]
torch.cuda.synchronize()
collided = sum(1 for t in no_clone[:-1]
               if torch.equal(t.float(), no_clone[-1].float()))
print(f"  without .clone(): {collided} of {len(no_clone)-1} earlier 'layers' now "
      f"equal the LAST layer's values")
print("  -> confirms the returned tensor is a view into shared scratch; the "
      "clone in\n     mla_int2_kv_pool.py is required for correctness, not "
      "defensive style")
if collided == 0:
    print("  NOTE: no collision observed -- either scratch grew between calls or "
          "the\n        buffers are not shared for this shape; the clone is then "
          "belt-and-braces")

with_clone = [quantize_dequantize_reuse(x, G, False)[2].clone() for x in xs]
torch.cuda.synchronize()
bad = sum(1 for i, t in enumerate(with_clone)
          if not torch.allclose(t.float(),
                                _fake_quant_int2_groupwise(xs[i], G, False).float(),
                                atol=0, rtol=0))
print(f"  with .clone():    {bad} of {len(xs)} layers wrong  "
      f"{'OK' if bad == 0 else '*** WRONG ***'}")
if bad:
    fails.append("cloned multi-layer sequence still wrong")

print()
print("=" * 74)
print("2b. CUDA GRAPH capture / replay, including a scratch GROWTH boundary")
print("=" * 74)
K._SCRATCH.clear()


def graph_arm(n_tok, warm_first, label):
    """Capture a graph around the packed path at n_tok, then replay it."""
    x = (torch.randn(n_tok, LATENT, device=dev) * 0.5).to(torch.bfloat16)
    static_in = x.clone()
    if warm_first:                    # let scratch reach final size before capture
        quantize_dequantize_reuse(static_in, G, False)
        torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    pool_before = len(K._SCRATCH)
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                quantize_dequantize_reuse(static_in, G, False)
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g):
            out = quantize_dequantize_reuse(static_in, G, False)[2].clone()
    except Exception as e:
        print(f"  {label:<38} CAPTURE FAILED: {type(e).__name__}: {e}")
        return None, f"{label}: capture raised {type(e).__name__}"
    # replay against fresh input values
    new = (torch.randn(n_tok, LATENT, device=dev) * 0.5).to(torch.bfloat16)
    static_in.copy_(new)
    g.replay()
    torch.cuda.synchronize()
    ref = _fake_quant_int2_groupwise(new, G, False).float()
    rel = ((out.float() - ref).norm() / ref.norm().clamp_min(1e-9)).item()
    tag = "OK" if rel < 1e-5 else "*** WRONG AFTER REPLAY ***"
    print(f"  {label:<38} relL2={rel:.3e}  scratch_keys "
          f"{pool_before}->{len(K._SCRATCH)}  {tag}")
    return (rel, g), (None if rel < 1e-5 else f"{label}: replay relL2 {rel:.2e}")


r1, e1 = graph_arm(256, True, "capture@256 (scratch pre-warmed)")
if e1:
    fails.append(e1)
# now force a growth AFTER a graph was captured: capture small, then run big
r2, e2 = graph_arm(64, False, "capture@64 (cold scratch)")
if e2:
    fails.append(e2)
print("  now force scratch growth in eager, then replay the OLD graph:")
big = (torch.randn(200000, LATENT, device=dev) * 0.5).to(torch.bfloat16)
quantize_dequantize_reuse(big, G, False)          # reallocates _SCRATCH upward
torch.cuda.synchronize()
if r2 is not None:
    try:
        r2[1].replay()
        torch.cuda.synchronize()
        print("     replay after growth: did not crash "
              "(output validity NOT asserted here -- see note)")
    except Exception as e:
        print(f"     replay after growth RAISED {type(e).__name__}: {e}")
        fails.append("replay after scratch growth raised")
print("  NOTE: sglang captures decode graphs largest-batch-first, which makes the "
      "growth\n        path unlikely in practice, but it is not structurally "
      "prevented.")

print()
print("=" * 74)
print("3. MEMORY -- does the packed path shrink the KV footprint?")
print("=" * 74)
x = (torch.randn(4096, LATENT, device=dev) * 0.5).to(torch.bfloat16)
codes, params, out = quantize_dequantize_reuse(x, G, False)
print(f"  input  c_kv          : {tuple(x.shape)} {x.dtype}   "
      f"{x.numel() * x.element_size() / 1024:.1f} KiB")
print(f"  packed codes         : {tuple(codes.shape)} {codes.dtype}  "
      f"{codes.numel() * codes.element_size() / 1024:.1f} KiB  "
      f"({bytes_per_value(G):.3f} bytes/value)")
print(f"  RETURNED to the pool : {tuple(out.shape)} {out.dtype}  "
      f"{out.numel() * out.element_size() / 1024:.1f} KiB   <-- dequantized")
print("  -> the caller keeps `out` and discards codes/params (`_codes, _params`),")
print("     so the pool still stores full-width floats. This is a WRITE-PATH")
print("     SPEED change, not a memory saving. Matches the docstring: the MLA")
print("     attention path has no INT2 read support.")

print()
print("=" * 74)
print("VERDICT")
print("=" * 74)
if fails:
    for f in fails:
        print(f"  FAIL  {f}")
    print("\nOVERALL: FAIL")
    sys.exit(1)
print("  packed kernel agrees with the simulation, survives capture+replay at a")
print("  fixed batch, and does not change the KV footprint.")
print("\nOVERALL: PASS")
