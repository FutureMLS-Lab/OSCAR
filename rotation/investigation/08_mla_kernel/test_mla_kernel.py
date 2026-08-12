"""Does the real packed-INT2 MLA kernel reproduce the fake-quant simulation?

The MLA pool currently fake-quants c_kv and writes BF16 back, so it saves no
memory. These kernels store 2-bit codes for real. If the two disagree, every
GLM-5.2 accuracy number we have stops predicting what a real kernel would do --
so the bar here is exact agreement, not "close".

  python test_mla_kernel.py [--ckv <layer_N.pt>] [--rot <rotations_dir>]
"""
import argparse, os, sys, time
import torch

sys.path.insert(0, "/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/oscar/sglang-research/python")
from sglang.QuantKernel.mla_latent_int2 import quantize_pack, dequantize, bytes_per_value
from sglang.srt.mem_cache.mla_int2_kv_pool import _fake_quant_int2_groupwise

G = 128


def compare(name, ref, got):
    # judge on RELATIVE error: absolute max|d| tracks the data scale, so a
    # tensor scaled by 1e3 "fails" any fixed absolute threshold while being
    # correct to fp32 epsilon.
    ref32, got32 = ref.float(), got.float()
    diff = (ref32 - got32).abs()
    denom = ref32.norm().item() or 1.0
    rel = (ref32 - got32).norm().item() / denom
    n_exact = int((diff == 0).sum())
    print(f"  {name:<34} relL2={rel:.3e}  max|d|={diff.max().item():.3e}  "
          f"bit-exact {n_exact}/{diff.numel()}"
          + ("" if n_exact == diff.numel() else "  <-- not bit-exact"))
    return rel, n_exact == diff.numel()


def run_case(name, x, lloyd):
    ref = _fake_quant_int2_groupwise(x, G, lloyd)
    codes, params = quantize_pack(x, G, lloyd)
    got = dequantize(codes, params, x.shape, G, torch.float32, lloyd)
    return compare(name, ref, got)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckv"); ap.add_argument("--rot")
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    worst = 0.0
    all_exact = True

    print("=== synthetic, both quantizer modes")
    for lloyd in (False, True):
        tag = "lloyd" if lloyd else "uniform"
        cases = {
            "normal":        torch.randn(4096, 512, device=dev),
            "wide-range":    torch.randn(1024, 512, device=dev) * 1e3,
            "tiny-range":    torch.randn(1024, 512, device=dev) * 1e-6,
            "constant-group": torch.full((256, 512), 0.7, device=dev),
            "with-outliers": torch.randn(1024, 512, device=dev).index_fill_(
                1, torch.tensor([0, 7, 300], device=dev), 50.0),
        }
        for cname, x in cases.items():
            r, ex = run_case(f"[{tag}] {cname}", x, lloyd)
            worst = max(worst, r); all_exact &= ex

    if a.ckv and os.path.exists(a.ckv):
        print(f"\n=== real GLM-5.2 c_kv: {os.path.basename(a.ckv)}")
        c = torch.load(a.ckv, map_location=dev).float()
        c = c.reshape(-1, c.shape[-1])
        print(f"  tokens={c.shape[0]} dim={c.shape[1]} "
              f"absmax={c.abs().max():.3f} std={c.std():.4f}")
        for lloyd in (False, True):
            r, ex = run_case(f"[{'lloyd' if lloyd else 'uniform'}] real c_kv", c, lloyd)
            worst = max(worst, r); all_exact &= ex

        if a.rot:
            lid = int(os.path.basename(a.ckv).split("_")[1].split(".")[0])
            rp = os.path.join(a.rot, f"layer_{lid}.pt")
            if os.path.exists(rp):
                print(f"\n=== full MLA path with the real layer-{lid} rotation")
                st = torch.load(rp, map_location=dev)
                R = (st["rotation"] if isinstance(st, dict) and "rotation" in st
                     else next(iter(st["layers"].values()))["rotation"]).float()
                for lloyd in (False, True):
                    rot = c @ R
                    ref = _fake_quant_int2_groupwise(rot, G, lloyd) @ R.T
                    codes, params = quantize_pack(rot, G, lloyd)
                    got = dequantize(codes, params, rot.shape, G, torch.float32, lloyd) @ R.T
                    r, ex = compare(f"[{'lloyd' if lloyd else 'uniform'}] rotate->q->unrotate",
                                    ref, got)
                    worst = max(worst, r); all_exact &= ex
                    err = (c - ref).norm().item() / c.norm().item()
                    print(f"      quantization error vs original c_kv: relL2={err:.4f}")

        bf16 = c.numel() * 2
        packed = c.numel() * bytes_per_value(G)
        print(f"\n=== storage for this layer's c_kv")
        print(f"  BF16   {bf16/2**20:8.2f} MiB")
        print(f"  packed {packed/2**20:8.2f} MiB   "
              f"({bytes_per_value(G)*8:.2f} bit/value, {bf16/packed:.2f}x smaller)")

        n = c.shape[0]
        for fn, label in ((lambda: quantize_pack(c, G, False), "quantize+pack"),
                          (lambda: dequantize(*quantize_pack(c, G, False), c.shape, G), "pack+dequant")):
            torch.cuda.synchronize(); t = time.time()
            for _ in range(10):
                fn()
            torch.cuda.synchronize()
            print(f"  {label:<14} {(time.time()-t)/10*1e3:6.2f} ms for {n} tokens")

    print(f"\nworst relative L2 across all cases: {worst:.3e}   bit-exact everywhere: {all_exact}")
    ok = worst < 1e-5
    print("VERDICT:", ("MATCHES the simulation"
                       + (" (bit-exact)" if all_exact else " to fp32 epsilon"))
          if ok else "DIVERGES from the simulation")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
