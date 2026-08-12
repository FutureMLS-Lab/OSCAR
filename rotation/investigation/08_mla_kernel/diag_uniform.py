"""Which groups does the uniform kernel get wrong on real c_kv, and why?"""
import sys, torch
sys.path.insert(0, "/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/oscar/sglang-research/python")
from sglang.QuantKernel.mla_latent_int2 import quantize_pack, dequantize
from sglang.srt.mem_cache.mla_int2_kv_pool import _fake_quant_int2_groupwise

G = 128
c = torch.load(sys.argv[1], map_location="cuda").float().reshape(-1, 512)
ref = _fake_quant_int2_groupwise(c, G, False)
codes, params = quantize_pack(c, G, False)
got = dequantize(codes, params, c.shape, G, torch.float32, False)

d = (ref - got).abs().reshape(-1, G)
bad = (d.max(dim=1).values > 1e-6)
print(f"groups: {d.shape[0]}   bad: {int(bad.sum())} ({100*bad.float().mean():.2f}%)")

x = c.reshape(-1, G)
rng = (x.amax(1) - x.amin(1))
print(f"\nrange over ALL groups : min={rng.min():.3e} median={rng.median():.3e} max={rng.max():.3e}")
if bad.any():
    rb = rng[bad]
    print(f"range over BAD groups : min={rb.min():.3e} median={rb.median():.3e} max={rb.max():.3e}")
    print(f"bad groups with range <= 1e-8: {int((rb <= 1e-8).sum())} / {int(bad.sum())}")
    i = int(torch.nonzero(bad)[0])
    g = x[i]
    print(f"\nexample bad group {i}: min={g.min():.6e} max={g.max():.6e} range={rng[i]:.6e}")
    print(f"  scale used by ref = {max((rng[i]/3).item(), 0):.6e}")
    print(f"  kernel params scale={params[i,0].item():.6e} zero={params[i,1].item():.6e}")
    print(f"  ref[:6] {ref.reshape(-1,G)[i,:6].tolist()}")
    print(f"  got[:6] {got.reshape(-1,G)[i,:6].tolist()}")
    print(f"  raw[:6] {g[:6].tolist()}")
# is it the degenerate-range branch?
deg = rng <= 1e-8
print(f"\ngroups with range <= 1e-8 (degenerate branch): {int(deg.sum())}")
if deg.any():
    print(f"  of those, how many are 'bad': {int((deg & bad).sum())}")
