"""0.024 ms is flat from 1 token to 4096, so it is not compute. Where does it go?

Splits the wrapper cost (3 allocations + reshape + contiguous) from the kernel
launch itself, so the next optimisation targets the right thing.
"""
import sys, time, torch
sys.path.insert(0, "/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/oscar/sglang-research/python")
import triton
from sglang.QuantKernel.mla_latent_int2 import (
    _fused_quant_dequant_kernel, quantize_dequantize_fused,
    _LM_THRESHOLDS, _LM_SPAN, _LM_RATIO, _LM_CENTROIDS)

G, GPB, dev = 128, 4, "cuda"

def timed(fn, iters=200, warmup=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3

for n_tok in (512, 8192):
    x = torch.randn(n_tok, 512, device=dev)
    flat = x.reshape(-1, G).contiguous().float()
    n = flat.shape[0]
    codes = torch.empty((n, G // 4), dtype=torch.uint8, device=dev)
    params = torch.empty((n, 2), dtype=torch.float32, device=dev)
    out = torch.empty_like(flat)
    grid = (triton.cdiv(n, GPB),)

    def kernel_only():
        _fused_quant_dequant_kernel[grid](
            flat, codes, params, out, n, GS=G, GPB=GPB, LLOYD=False,
            T0=_LM_THRESHOLDS[0], T1=_LM_THRESHOLDS[1], T2=_LM_THRESHOLDS[2],
            LM_SPAN3=_LM_SPAN/3.0, LM_RATIO=_LM_RATIO, LM_C0=_LM_CENTROIDS[0])

    def allocs_only():
        torch.empty((n, G // 4), dtype=torch.uint8, device=dev)
        torch.empty((n, 2), dtype=torch.float32, device=dev)
        torch.empty_like(flat)

    t_full = timed(lambda: quantize_dequantize_fused(x, G, False, GPB))
    t_kern = timed(kernel_only)
    t_alloc = timed(allocs_only)
    t_prep = timed(lambda: x.reshape(-1, G).contiguous().float())
    print(f"tokens={n_tok}")
    print(f"  full wrapper      {t_full:.4f} ms")
    print(f"  kernel launch     {t_kern:.4f} ms   ({t_kern/t_full*100:.0f}%)")
    print(f"  3 allocations     {t_alloc:.4f} ms   ({t_alloc/t_full*100:.0f}%)")
    print(f"  reshape+contig+f32 {t_prep:.4f} ms  ({t_prep/t_full*100:.0f}%)")
