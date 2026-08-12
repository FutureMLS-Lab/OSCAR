import sys, torch
sys.path.insert(0, "/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/oscar/sglang-research/python")
from sglang.QuantKernel.mla_latent_int2 import quantize_pack, dequantize
from sglang.srt.mem_cache.mla_int2_kv_pool import _fake_quant_int2_groupwise
G = 128
c = torch.load(sys.argv[1], map_location="cuda").float().reshape(-1, 512)
x = c.reshape(-1, G)
ref = _fake_quant_int2_groupwise(c, G, False).reshape(-1, G)
codes, params = quantize_pack(c, G, False)
got = dequantize(codes, params, c.shape, G, torch.float32, False).reshape(-1, G)

# reference codes, recomputed
xmin = x.amin(1, keepdim=True); xmax = x.amax(1, keepdim=True)
scale = torch.where((xmax - xmin).abs() > 1e-8, (xmax - xmin) / 3.0, torch.ones_like(xmin))
qref = ((x - xmin) / scale).round().clamp(0, 3)
# kernel codes, unpacked
b = codes.reshape(-1, G // 4)
lanes = torch.arange(G, device=x.device)
qker = ((b[:, lanes // 4].int() >> (2 * (lanes % 4))) & 3).float()

d = (qref - qker).abs()
print(f"code mismatches: {int((d>0).sum())} / {d.numel()}  ({100*(d>0).float().mean():.3f}%)")
print(f"  by delta: {[(int(k), int((d==k).sum())) for k in d.unique()[:5]]}")
i, j = [t[0].item() for t in torch.nonzero(d > 0, as_tuple=True)]
xv = x[i, j].item(); mn = xmin[i, 0].item(); sc = scale[i, 0].item()
raw = (xv - mn) / sc
print(f"\nfirst mismatch  group={i} lane={j}")
print(f"  x={xv:.10e}  min={mn:.10e}  scale={sc:.10e}")
print(f"  (x-min)/scale = {raw:.17g}")
print(f"  torch.round -> {qref[i,j].item()}   kernel -> {qker[i,j].item()}")
print(f"  frac = {raw - int(raw):.17g}   exact .5 tie? {raw - int(raw) == 0.5}")
# does the kernel's own (x-zero)/scale differ from torch's (x-min)/scale?
print(f"  params: scale={params[i,0].item():.10e} zero={params[i,1].item():.10e}")
print(f"  zero == xmin ? {params[i,1].item() == mn}")
