"""v1 (one program per group) vs v2 (block-tiled + fused dequant).

v2 must match v1 exactly, not just be faster -- it is the same arithmetic with a
different tiling, so any difference is a bug in the tiling.
"""
import sys, time, torch
sys.path.insert(0, "/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/oscar/sglang-research/python")
from sglang.QuantKernel.mla_latent_int2 import (
    quantize_pack, dequantize, quantize_dequantize_fused)
from sglang.srt.mem_cache.mla_int2_kv_pool import _fake_quant_int2_groupwise

G = 128
dev = "cuda"


def timed(fn, iters=50, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3


print("=== correctness: v2 vs v1 vs the torch reference")
for lloyd in (False, True):
    for n in (7, 1024, 8192):
        x = torch.randn(n, 512, device=dev)
        ref = _fake_quant_int2_groupwise(x, G, lloyd)
        c1, p1 = quantize_pack(x, G, lloyd)
        v1 = dequantize(c1, p1, x.shape, G, torch.float32, lloyd)
        c2, p2, v2 = quantize_dequantize_fused(x, G, lloyd)
        d21 = (v2 - v1).abs().max().item()
        code_mismatch = int((c1 != c2).sum())
        drf = (v2.float() - ref.float()).norm().item() / ref.float().norm().item()
        tag = "lloyd" if lloyd else "uniform"
        print(f"  [{tag}] n={n:<5} v2-vs-v1 max|d|={d21:.2e} codes_differ={code_mismatch} "
              f"v2-vs-ref relL2={drf:.2e}")

print("\n=== speed (uniform)")
print(f"{'tokens':>8} {'v1 pack+deq':>13} {'v2 fused':>11} {'speedup':>9}")
for n in (1, 8, 64, 512, 4096, 8192, 65536):
    x = torch.randn(n, 512, device=dev)
    t1 = timed(lambda: dequantize(*quantize_pack(x, G, False), x.shape, G, torch.float32, False))
    t2 = timed(lambda: quantize_dequantize_fused(x, G, False))
    print(f"{n:>8} {t1:>12.3f}m {t2:>10.3f}m {t1/t2:>8.2f}x")

print("\n=== GROUPS_PER_BLOCK sweep at 8192 tokens")
x = torch.randn(8192, 512, device=dev)
best = None
for gpb in (1, 2, 4, 8, 16, 32, 64):
    t = timed(lambda: quantize_dequantize_fused(x, G, False, gpb))
    print(f"  GPB={gpb:<3} {t:.3f} ms")
    if best is None or t < best[1]: best = (gpb, t)
print(f"  best: GPB={best[0]} at {best[1]:.3f} ms")

print("\n=== full MLA write path, 8192 tokens (rotate -> quant -> unrotate)")
R = torch.linalg.qr(torch.randn(512, 512, device=dev))[0]
t_fake = timed(lambda: _fake_quant_int2_groupwise(x @ R, G, False) @ R.T, iters=20)
t_v1 = timed(lambda: dequantize(*quantize_pack(x @ R, G, False), x.shape, G, torch.float32, False) @ R.T, iters=20)
t_v2 = timed(lambda: quantize_dequantize_fused(x @ R, G, False, best[0])[2] @ R.T, iters=20)
print(f"  torch fake-quant {t_fake:.3f} ms")
print(f"  v1 kernel        {t_v1:.3f} ms  ({t_fake/t_v1:.2f}x vs fake)")
print(f"  v2 kernel        {t_v2:.3f} ms  ({t_fake/t_v2:.2f}x vs fake, {t_v1/t_v2:.2f}x vs v1)")
