#!/usr/bin/env python3
"""Prototype: Hadamard + high-precision sensitive subspace (the latent analog of
OSCAR's HP-prefix tokens).

Keep the projection of c onto the top-k Hessian eigenvectors in BF16 (exact);
Hadamard+INT2 the residual. This spends extra precision exactly on the
directions that matter most for the attention score/output — the OSCAR idea,
realized as a *bit-budget* mechanism instead of a rotation (which we showed
cannot beat Hadamard).

Reports functional error vs k and the extra-bit overhead (k fp16 / 512 int2).
"""
from __future__ import annotations
import argparse, math
import torch
from compute_mla_upproj_rotation import KVBProjReader, _resolve_snapshot, latent_hessian
from analyze_latent_quant_error import fake_quant_int2_groupwise

def build_hadamard(n):
    if n == 1: return torch.ones(1,1,dtype=torch.float64)
    h = build_hadamard(n//2)
    return torch.cat([torch.cat([h,h],1),torch.cat([h,-h],1)],0)/math.sqrt(2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", default="/shared/glm51_fp8_c_kv_dump")
    ap.add_argument("--model-path", default="zai-org/GLM-5.1-FP8")
    ap.add_argument("--layers", default="5,20,35,50,65,77")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--ks", default="0,8,16,32,64")
    args = ap.parse_args()
    d = 512
    snap = _resolve_snapshot(args.model_path); reader = KVBProjReader(snap)
    had = build_hadamard(d)
    layers = [int(x) for x in args.layers.split(",")]
    ks = [int(x) for x in args.ks.split(",")]

    methods = ["none", "hadamard"] + [f"had+hp{k}" for k in ks if k>0]
    acc = {m: torch.zeros(2, dtype=torch.float64) for m in methods}

    def quant_hadamard(x):
        y = x @ had.float()
        yq = fake_quant_int2_groupwise(y, args.group_size, lloyd_max=True)
        return yq @ had.T.float()

    for lid in layers:
        c = torch.load(f"{args.dump_dir}/layer_{lid}.pt", map_location="cpu").float().reshape(-1, d)[:args.max_tokens]
        w = reader.get_dequant(lid)
        _, hs, hv = latent_hessian(w, 192, 256, d, args.alpha)
        hs = hs / hs.diagonal().sum(); hv = hv / hv.diagonal().sum()
        H = hs + args.alpha*hv
        eigval, U = torch.linalg.eigh(H)               # ascending; top-k = last cols
        def fe(cq):
            dc = (cq - c).double(); cov = dc.mT @ dc / dc.shape[0]
            return torch.stack([(hs*cov).sum(), (hv*cov).sum()])
        # none
        acc["none"] += fe(fake_quant_int2_groupwise(c, args.group_size, lloyd_max=True))
        # hadamard
        acc["hadamard"] += fe(quant_hadamard(c))
        # had + hp-k : protect top-k sensitive directions (exact), hadamard+int2 the rest
        for k in ks:
            if k == 0: continue
            Uk = U[:, d-k:].float()                    # top-k eigenvectors (most sensitive)
            c_hp = c @ Uk @ Uk.T                        # projection onto sensitive subspace (kept exact)
            c_lp = c - c_hp
            cq = c_hp + quant_hadamard(c_lp)
            acc[f"had+hp{k}"] += fe(cq)

    n = len(layers)
    base = acc["none"]/n; base_total = base[0]+args.alpha*base[1]
    had_total = (acc["hadamard"]/n)[0] + args.alpha*(acc["hadamard"]/n)[1]
    print(f"\n=== {n} layers, α={args.alpha}; lower total=better ===")
    print(f"{'method':12} {'total':>11} {'dB_none':>8} {'dB_had':>8} {'bits/elem':>10}")
    for m in methods:
        v = acc[m]/n; total = v[0]+args.alpha*v[1]
        db = 10*math.log10((total/base_total).clamp(min=1e-30).item())
        dbh = 10*math.log10((total/had_total).clamp(min=1e-30).item())
        if m.startswith("had+hp"):
            k = int(m[6:]); bits = 2.0 + k*16.0/d   # 2 bit base + k fp16 over 512 dims
        else:
            bits = 2.0
        star = " <== beats hadamard" if total < had_total*0.999 else ""
        print(f"{m:12} {total:11.6g} {db:+8.2f} {dbh:+8.2f} {bits:10.3f}{star}")

if __name__ == "__main__":
    main()
