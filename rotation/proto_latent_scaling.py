#!/usr/bin/env python3
"""Prototype: can a sensitivity-aware transform beat Hadamard on the latent?

We allow a *non-orthogonal* forward transform M (c' = c·M) with an explicit
inverse Minv (c_deq = c'_q·Minv), so we can fold a diagonal sensitivity scaling
into the rotation — something a pure orthogonal R can't do. Quantization is the
exact runtime groupwise Lloyd-Max INT2.

Schemes tested (per layer):
  none      : M = I
  hadamard  : M = Had                      (orthogonal; current best)
  had_scale : M = Had · diag(s),  s_i ∝ (diag(Hadᵀ H Had))^p   — scale up the
              directions that matter most in the Hadamard frame
  eig_scale : M = U_h · diag(s),  s_i ∝ λ_i^p                   — Hessian eigvec
              basis with sensitivity scaling

Objective: total = tr(H_score·CovΔc) + α·tr(H_val·CovΔc). Lower is better.
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
import torch
from compute_mla_upproj_rotation import KVBProjReader, _resolve_snapshot, latent_hessian
from analyze_latent_quant_error import fake_quant_int2_groupwise, random_sign_hadamard

def build_hadamard(n):
    if n == 1: return torch.ones(1,1,dtype=torch.float64)
    h = build_hadamard(n//2)
    return torch.cat([torch.cat([h,h],1),torch.cat([h,-h],1)],0)/math.sqrt(2)

def quant_with(c, M, Minv, gs):
    x = c.to(torch.float32) @ M.float()
    xq = fake_quant_int2_groupwise(x, gs, lloyd_max=True)
    return (xq @ Minv.float())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", default="/shared/glm51_fp8_c_kv_dump")
    ap.add_argument("--model-path", default="zai-org/GLM-5.1-FP8")
    ap.add_argument("--layers", default="5,20,35,50,65,77")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--exps", default="0.0,0.25,0.5,0.75,1.0")
    args = ap.parse_args()
    d = 512
    snap = _resolve_snapshot(args.model_path); reader = KVBProjReader(snap)
    had = build_hadamard(d)
    layers = [int(x) for x in args.layers.split(",")]
    exps = [float(x) for x in args.exps.split(",")]

    # method -> accumulated [score_err, val_err]
    methods = ["none", "hadamard"]
    for p in exps:
        methods += [f"had_scale^{p}", f"eig_scale^{p}"]
    acc = {m: torch.zeros(2, dtype=torch.float64) for m in methods}

    for lid in layers:
        c = torch.load(f"{args.dump_dir}/layer_{lid}.pt", map_location="cpu").float().reshape(-1, d)[:args.max_tokens]
        w = reader.get_dequant(lid)
        _, hs, hv = latent_hessian(w, 192, 256, d, args.alpha)
        hs = hs / hs.diagonal().sum(); hv = hv / hv.diagonal().sum()
        H = hs + args.alpha * hv
        eigval, U = torch.linalg.eigh(H)               # ascending
        # sensitivity in Hadamard frame
        Hh = had.T @ H @ had
        had_sens = Hh.diagonal().clamp(min=1e-12)

        def evl(M, Minv):
            cq = quant_with(c, M, Minv, args.group_size)
            dc = (cq - c).double()
            cov = dc.mT @ dc / dc.shape[0]
            return torch.stack([(hs*cov).sum(), (hv*cov).sum()])

        acc["none"] += evl(torch.eye(d, dtype=torch.float64), torch.eye(d, dtype=torch.float64))
        acc["hadamard"] += evl(had, had.T)
        for p in exps:
            # had_scale: M = Had diag(s); Minv = diag(1/s) Hadᵀ
            s = had_sens.pow(p); s = s / s.mean().sqrt() / s.mean().sqrt()  # keep ~unit avg energy
            s = had_sens.pow(p); s = s / s.pow(2).mean().sqrt()
            M = had @ torch.diag(s); Minv = torch.diag(1.0/s) @ had.T
            acc[f"had_scale^{p}"] += evl(M, Minv)
            # eig_scale: M = U diag(se); se_i ∝ λ_i^p
            se = eigval.clamp(min=1e-12).pow(p); se = se / se.pow(2).mean().sqrt()
            M = U @ torch.diag(se); Minv = torch.diag(1.0/se) @ U.T
            acc[f"eig_scale^{p}"] += evl(M, Minv)

    n = len(layers)
    base = (acc["none"]/n); base_total = base[0] + args.alpha*base[1]
    print(f"\n=== {n} layers, α={args.alpha} — total = score+α·val, dB vs none, vs hadamard ===")
    had_total = (acc['hadamard']/n)[0] + args.alpha*(acc['hadamard']/n)[1]
    for m in methods:
        v = acc[m]/n; total = v[0] + args.alpha*v[1]
        db = 10*math.log10((total/base_total).clamp(min=1e-30).item())
        dbh = 10*math.log10((total/had_total).clamp(min=1e-30).item())
        star = "  <-- beats hadamard" if total < had_total*0.999 else ""
        print(f"{m:16} total={total:10.6g}  dB_none{db:+6.2f}  dB_had{dbh:+6.2f}{star}")

if __name__ == "__main__":
    main()
