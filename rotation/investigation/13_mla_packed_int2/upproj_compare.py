#!/usr/bin/env python3
"""Compare latent rotations on the metric that actually predicts accuracy.

Why this script exists
----------------------
The shipped rotation is ``Rcov · P · Hblock``: covariance eigenvectors, a
bit-reversal permutation, and a per-group Hadamard. It minimizes ``‖Δc_kv‖`` --
the reconstruction error of the latent *in isolation*. That is the quantity the
existing sweep (``sweep_latent_bits.py``) reports, and by that measure GLM-5.2
sits at 0.2989 and GLM-5.3 at 0.2906 against the ≤0.10 a 2-bit KV cache wants.

But ``c_kv`` is never consumed directly. It is consumed through the
up-projection:

    score_h = (q_nope_h @ W_UK_h) · c_kv
    out_h   = (Σ_i p_i c_kv_i) @ W_UV_h

so what the model feels is not ``Δc`` but ``Δc @ W_UK^T`` and ``Δc @ W_UV^T``.
An error in a direction that ``kv_b_proj`` attenuates costs nothing; an error in
a direction it amplifies costs a lot. ``Rcov`` cannot know the difference.

**This is the trap the script is built to avoid.** If I evaluated an
up-projection rotation with ``sweep_latent_bits.py``, it would lose by
construction -- ``Rcov`` is near-optimal for latent reconstruction, so anything
that trades latent error for downstream error scores worse on a latent-error
metric. Concluding "upproj is worse" from that would be measuring the wrong
thing. So this script reports **three** errors per configuration:

    latent : ‖Δc‖ / ‖c‖                       (what the old sweep reports)
    score  : ‖Δc @ W_UK^T‖ / ‖c @ W_UK^T‖     (what the attention scores feel)
    value  : ‖Δc @ W_UV^T‖ / ‖c @ W_UV^T‖     (what the output feels)

and expects the shipped rotation to win on the first. Only score/value decide.

Both downstream norms are evaluated through the Hessian identity

    ‖X @ W^T‖_F² = tr(X (WᵀW) Xᵀ) = Σ (X @ H) ⊙ X ,     H = Wᵀ W  [R × R]

which avoids materializing the [N, num_heads·head_dim] projection.

Arms
----
  none    -- no rotation (floor)
  cov     -- Rcov · P · Hblock, the shipped recipe
  upproj  -- R_H · P_H · Hblock, same composition with the covariance
             eigenbasis replaced by the Hessian eigenbasis, and the
             permutation driven by Hessian eigenvalues (sensitivity) rather
             than variance. Same 2 bits, same group size, same 4.00x.

``--alpha`` sweeps the value-path weight in ``H = H_score + α·H_val``. α=1 is
the raw kv_b_proj Gram; the two blocks can differ in scale by orders of
magnitude, so ``--alpha balanced`` (α = tr H_score / tr H_val) is also offered,
which weights the two paths equally instead of letting whichever block has the
larger norm silently own the rotation.

Out-of-sample throughout: the rotation is fitted on the first half of each
layer's dump and the error is measured on the second half.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from compute_kv_rotation import build_hadamard, make_br_perm_matrix  # noqa: E402
from compute_mla_upproj_rotation import (  # noqa: E402
    KVBProjReader,
    _resolve_snapshot,
    latent_hessian,
)


def fake_quant(x: torch.Tensor, group_size: int, bits: int) -> torch.Tensor:
    """Asymmetric per-group uniform quantization -- the pool's storage model.

    Kept byte-identical to sweep_latent_bits.fake_quant so the numbers in this
    script and that one are comparable.
    """
    d = x.shape[-1]
    g = x.reshape(-1, d // group_size, group_size)
    lo = g.amin(-1, keepdim=True)
    hi = g.amax(-1, keepdim=True)
    n = (1 << bits) - 1
    scale = (hi - lo).clamp_min(1e-8) / n
    q = ((g - lo) / scale).round().clamp_(0, n)
    return (q * scale + lo).reshape(x.shape)


def block_hadamard(d: int, group_size: int) -> torch.Tensor:
    h = build_hadamard(group_size)
    hb = torch.zeros(d, d, dtype=torch.float64)
    for g in range(d // group_size):
        hb[g * group_size:(g + 1) * group_size,
           g * group_size:(g + 1) * group_size] = h
    return hb


def compose(basis: torch.Tensor, evals: torch.Tensor, hb: torch.Tensor) -> torch.Tensor:
    """basis · P(evals) · Hblock -- the shipped composition, any basis."""
    return (basis.double() @ make_br_perm_matrix(evals) @ hb).contiguous()


def quad_rel_err(delta: torch.Tensor, ref: torch.Tensor, H: torch.Tensor) -> float:
    """‖Δ @ W^T‖_F / ‖ref @ W^T‖_F, via H = W^T W. No [N, H*dh] materialized."""
    num = ((delta @ H) * delta).sum().clamp_min(0).sqrt()
    den = ((ref @ H) * ref).sum().clamp_min(0).sqrt().clamp_min(1e-12)
    return (num / den).item()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump-path", required=True)
    p.add_argument("--model-path", required=True,
                   help="Snapshot dir or HF repo id -- for kv_b_proj.")
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--bits", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--layers", type=int, default=0,
                   help="Evaluate at most this many dump layers (0 = all).")
    p.add_argument("--alpha", default="balanced",
                   help="'balanced' (tr-matched), or a float, or a "
                        "comma-separated list to sweep.")
    p.add_argument("--variants", default="",
                   help="Comma-separated bits:group configs to evaluate "
                        "alongside, each with its own cov rotation and its "
                        "own block-Hadamard. Lets 'spend 32 more bytes on a "
                        "smaller group' be compared head-to-head against "
                        "'spend them on an HP subspace' at equal ratio.")
    p.add_argument("--hp-k", default="",
                   help="Comma-separated k values for the mixed-precision "
                        "subspace arms (top-k dims held in BF16, residual "
                        "quantized). Costs 2k B/token/layer.")
    args = p.parse_args()

    snapshot = _resolve_snapshot(args.model_path)
    cfg = json.load(open(snapshot / "config.json"))
    # Some checkpoints nest the language config; take whichever level has MLA.
    for c in (cfg, cfg.get("text_config", {}), cfg.get("language_config", {})):
        if isinstance(c, dict) and "kv_lora_rank" in c:
            cfg = c
            break
    R = int(cfg["kv_lora_rank"])
    qk_nope = int(cfg["qk_nope_head_dim"])
    v_head = int(cfg["v_head_dim"])
    print(f"snapshot {snapshot}")
    print(f"  kv_lora_rank={R}  qk_nope_head_dim={qk_nope}  v_head_dim={v_head}")
    print(f"  bits={args.bits}  group_size={args.group_size}  "
          f"(NG={R // args.group_size})")

    reader = KVBProjReader(snapshot)
    hb = block_hadamard(R, args.group_size)

    files = sorted(Path(args.dump_path).glob("layer_*.pt"),
                   key=lambda x: int(x.stem.split("_")[1]))
    if args.layers:
        # Spread the sample across depth rather than taking a prefix -- early
        # layers are not representative of late ones.
        step = max(1, len(files) // args.layers)
        files = files[::step][:args.layers]
    if not files:
        print(f"no layer_*.pt under {args.dump_path}")
        return 1

    alphas: list = []
    for tok in str(args.alpha).split(","):
        tok = tok.strip()
        alphas.append(tok if tok == "balanced" else float(tok))

    # arm -> list of (layer, latent, score, value)
    rows: dict[str, list] = {}

    for f in files:
        lid = int(f.stem.split("_")[1])
        c = torch.load(str(f), map_location="cpu").float().reshape(-1, R)
        c = c[:args.max_tokens].double()
        n = c.shape[0]
        if n < 64:
            print(f"  layer {lid}: only {n} tokens, skipped")
            continue
        half = n // 2
        fit, test = c[:half], c[half:]

        w = reader.get_dequant(lid)
        # α only rescales H_val; get the raw blocks once with α=1.
        _, h_score, h_val = latent_hessian(w, qk_nope, v_head, R, alpha=1.0)

        cov = fit.T @ fit / fit.shape[0]
        cov = (cov + cov.T) / 2
        cov_evals, cov_basis = torch.linalg.eigh(cov)

        arms = {
            "none": torch.eye(R, dtype=torch.float64),
            "cov": compose(cov_basis, cov_evals, hb),
        }

        # Isolate the one variable the upproj arm confounds. `upproj` changes
        # TWO things at once: it drops the whitening (Rcov -> R_H) and it
        # reorders P by sensitivity instead of variance. If the sensitivity
        # information is worth anything, it should show up when we keep the
        # whitening and change ONLY the permutation key.
        #
        # P interleaves the eigen-sorted dims across the NG groups so each
        # group sees a homogeneous dynamic range. Sorting by variance
        # homogenizes range; sorting by sensitivity homogenizes *cost*; the
        # product is the term that actually appears in the error, since a
        # dim hurts in proportion to how much it varies AND how much the
        # up-projection amplifies it.
        h_bal = h_score + (
            h_score.diagonal().sum() / h_val.diagonal().sum().clamp_min(1e-30)
        ) * h_val
        sens = ((cov_basis.T @ h_bal) * cov_basis.T).sum(-1)   # diag in cov basis
        arms["cov+sensP"] = compose(cov_basis, sens, hb)
        arms["cov+var*sensP"] = compose(cov_basis, cov_evals * sens, hb)

        for a in alphas:
            av = (h_score.diagonal().sum() / h_val.diagonal().sum().clamp_min(1e-30)
                  ).item() if a == "balanced" else float(a)
            H = h_score + av * h_val
            H = (H + H.T) / 2
            h_evals, h_basis = torch.linalg.eigh(H)
            tag = f"upproj[a={a if a == 'balanced' else f'{av:g}'}]"
            arms[tag] = compose(h_basis, h_evals, hb)

        # Mixed-precision subspace arms. Different mechanism from everything
        # above: the rotation is unchanged, but the top-k directions are held
        # in BF16 and only the residual is quantized. Costs 2k B/token/layer
        # (k=16 => 288 -> 320, i.e. 4.00x -> 3.60x), so it has to earn its
        # ratio, not merely improve.
        #
        # Two selection rules, same k, so the variable is isolated: the
        # sensitivity basis (top-k eigenvectors of the balanced Hessian) and
        # the variance basis (top-k covariance eigenvectors). The rotation
        # experiment showed sensitivity adds nothing THERE; whether it adds
        # anything HERE is a separate question and this is what answers it.
        hp_arms = []
        for k in [int(x) for x in str(args.hp_k).split(",") if x.strip()]:
            _, u_sens = torch.linalg.eigh(h_bal)
            for name, basis in (("sens", u_sens[:, -k:].T.contiguous()),
                                ("var", cov_basis[:, -k:].T.contiguous())):
                # Re-fit the rotation on the RESIDUAL, not on c: removing the
                # top-k directions changes the statistics the rotation is
                # supposed to whiten, and reusing the full-c rotation would
                # understate the arm.
                res_fit = fit - (fit @ basis.T) @ basis
                cr = res_fit.T @ res_fit / res_fit.shape[0]
                cr = (cr + cr.T) / 2
                rev, rb = torch.linalg.eigh(cr)
                rot_r = compose(rb, rev, hb)

                def _fn(x, _b=basis, _r=rot_r):
                    hp = (x @ _b.T) @ _b          # exact, BF16 in the runtime
                    return hp + fake_quant((x - hp) @ _r, args.group_size,
                                           args.bits) @ _r.T

                cell = 288 + 2 * k
                hp_arms.append(
                    (f"hp{k}-{name} [{1152 / cell:.2f}x]", _fn))

        # Bit/group variants. Each gets its own cov rotation AND its own
        # block-Hadamard, because Hblock is per-group -- reusing the g128
        # rotation for a g32 quantizer would measure a mismatch, not the
        # configuration.
        var_arms = []
        for spec in [s for s in str(args.variants).split(",") if s.strip()]:
            vb, vg = (int(t) for t in spec.split(":"))
            vhb = block_hadamard(R, vg)
            rot_v = compose(cov_basis, cov_evals, vhb)
            # codes + per-group params + the 128-byte k_pe, which is never
            # quantized. Calibrated against the shipped cell: 2-bit g128 gives
            # 128 + 4*8 + 128 = 288 B = 4.00x, and 4-bit g128 gives 416 B =
            # 2.77x -- both match the pool's own reported numbers, so the
            # 8 B/group term is right.
            cell = R * vb // 8 + (R // vg) * 8 + 128

            def _fv(x, _r=rot_v, _g=vg, _b=vb):
                return fake_quant(x @ _r, _g, _b) @ _r.T

            var_arms.append((f"{vb}bit-g{vg} [{1152 / cell:.2f}x]", _fv))

        evals_list = [(t, (lambda x, _r=r: fake_quant(x @ _r, args.group_size,
                                                      args.bits) @ _r.T))
                      for t, r in arms.items()] + var_arms + hp_arms

        for tag, fn in evals_list:
            rec = fn(test)
            delta = rec - test
            lat = (delta.norm() / test.norm().clamp_min(1e-12)).item()
            sc = quad_rel_err(delta, test, h_score)
            va = quad_rel_err(delta, test, h_val)
            rows.setdefault(tag, []).append((lid, lat, sc, va))
        print(f"  layer {lid:2d} done", flush=True)

    def agg(v, i):
        xs = [r[i] for r in v]
        return sum(xs) / len(xs), max(xs)

    print(f"\n{len(rows[next(iter(rows))])} layers, out-of-sample, "
          f"{args.bits}-bit g{args.group_size}\n")
    print(f"  {'arm':<24} {'latent':>16} {'score':>16} {'value':>16}")
    print(f"  {'':<24} {'mean/worst':>16} {'mean/worst':>16} {'mean/worst':>16}")
    for tag, v in rows.items():
        lm, lw = agg(v, 1)
        sm, sw = agg(v, 2)
        vm, vw = agg(v, 3)
        print(f"  {tag:<24} {lm:>7.4f}/{lw:<8.4f} "
              f"{sm:>7.4f}/{sw:<8.4f} {vm:>7.4f}/{vw:<8.4f}")

    # Paired, per layer, against the shipped recipe -- a mean can hide a split.
    base = {r[0]: r for r in rows.get("cov", [])}
    for tag, v in rows.items():
        if tag in ("cov", "none"):
            continue
        print(f"\n  {tag}  vs  cov   (paired per layer, negative = upproj better)")
        for name, i in (("latent", 1), ("score", 2), ("value", 3)):
            d = [(r[i] - base[r[0]][i]) / max(base[r[0]][i], 1e-12)
                 for r in v if r[0] in base]
            wins = sum(x < 0 for x in d)
            print(f"    {name:<7} mean {100 * sum(d) / len(d):+7.2f}%   "
                  f"upproj wins {wins}/{len(d)} layers   "
                  f"range [{100 * min(d):+.1f}%, {100 * max(d):+.1f}%]")

    print("\n  latent is EXPECTED to regress -- cov is fitted for it. "
          "score/value are the ones that decide.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
