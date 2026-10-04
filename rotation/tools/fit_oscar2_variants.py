#!/usr/bin/env python3
"""Fit the OSCAR-2 transform family from startup-calibration moments.

One calibration pass with ``SGLANG_OSCAR_CALIBRATION_SAVE_MOMENTS=1`` leaves
``oscar_moments_rank<r>.pt`` next to the published pair: per (layer, KV head)
the query second moment ``M_q``, the key sum and second moment, and the
energy-weighted value covariance ``S_v``. Every row of the OSCAR-2 component
ablation (PLAN T3) is a closed-form function of those moments (plus ``W_O``
for the output-aware values), so this fits all of them without re-running
the model and writes one K/V checkpoint pair per variant:

  perhead   per-KV-head orthogonal basis           R_k = E_q H P_br
  center    perhead + key centering                k_mean = mu_h
  whiten    centered, query-whitened compact basis R_k = M_q^{1/2} E,          q side M_q^{-1/2} E
  flat      centered, flattened compact basis      R_k = M_q^{1/2} E H P_br,   q side M_q^{-1/2} E H P_br
  stretch   centered, fixed-rate stretch           R_k = X^{1/2} E* H P_br,    q side X^{-1/2} E* H P_br
  outaware  values: post-W_O metric                R_v = G^{1/2} E_v H P_br,  output side G^{-1/2} E_v H P_br
            (G = pooled W_O^T W_O of the query heads reading the KV head;
             keys from --k-base, default flat)

Non-orthogonal K transforms ship their query-side matrix as ``q_rotation``
(so q' . k' == q . k exactly), non-orthogonal V transforms ship ``o_rotation``
(the output is un-rotated as o @ o_rotation^T), and centered variants ship
``k_mean``; ``load_oscar_rotation_field`` in memory_pool.py reads all three.
``--shared`` averages the per-head moments into one basis per layer (the V1
grouping) for any variant.

  python rotation/tools/fit_oscar2_variants.py --moments-dir /scratch/oscar-calib/x \\
      --out /scratch/oscar2/qwen3-8b --variants perhead,center,whiten,flat,stretch,outaware \\
      --model Qwen/Qwen3-8B
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
from sglang.srt.mem_cache.oscar_calibration import (  # noqa: E402
    build_hadamard,
    compose_r_h_pbr,
    make_br_perm_matrix,
)

EIG_FLOOR = 1e-6  # relative to the largest eigenvalue, before roots / inverses


def _sym(m: torch.Tensor) -> torch.Tensor:
    return (m + m.transpose(-1, -2)) / 2


def _eigh_floor(m: torch.Tensor):
    vals, vecs = torch.linalg.eigh(_sym(m))
    vals = torch.clamp(vals, min=float(vals.max()) * EIG_FLOOR)
    return vals, vecs


def sym_pow(m: torch.Tensor, power: float) -> torch.Tensor:
    """Symmetric matrix power through the eigendecomposition (fp64)."""
    vals, vecs = _eigh_floor(m)
    return vecs @ torch.diag(vals**power) @ vecs.T


def unit_mean_eig(m: torch.Tensor) -> torch.Tensor:
    """Scale an SPD matrix to unit mean eigenvalue. A non-orthogonal transform
    stretches the stored rows by the metric; the per-row quantizer scale
    absorbs any global factor, so normalising keeps bf16 ranges sane without
    changing the ratio of the two sides."""
    return m / (torch.trace(m) / m.shape[0])


def orthogonal_basis(cov: torch.Tensor, hadamard: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """OSCAR V1 composition on a covariance: E H P_br, with E the eigenbasis."""
    vals, vecs = torch.linalg.eigh(_sym(cov))
    return compose_r_h_pbr(vecs, vals, hadamard), vals


def metric_basis(metric: torch.Tensor, cov: torch.Tensor, hadamard: torch.Tensor | None):
    """Write-side ``R = A E [H P_br]`` and read-side ``A^{-1} E [H P_br]`` for a
    metric ``A = metric^{1/2}`` and signal covariance ``cov``: ``E`` whitens
    the signal in the metric space, and the optional Hadamard flattens the
    diagonal spectrum for the fixed-rate scalar quantizer."""
    a = sym_pow(metric, 0.5)
    a_inv = sym_pow(metric, -0.5)
    vals, e = torch.linalg.eigh(_sym(a @ cov @ a))
    if hadamard is None:
        tail = e
    else:
        tail = e @ hadamard @ make_br_perm_matrix(vals)
    return a @ tail, a_inv @ tail, vals


def stretch_metric(m_q: torch.Tensor, s_k: torch.Tensor) -> torch.Tensor:
    """Fixed-rate stretch ``X* = S^{-1/2} (S^{1/2} M_q S^{1/2})^{1/2} S^{-1/2}``."""
    s_half = sym_pow(s_k, 0.5)
    s_inv_half = sym_pow(s_k, -0.5)
    inner = sym_pow(s_half @ m_q @ s_half, 0.5)
    return _sym(s_inv_half @ inner @ s_inv_half)


def load_moments(moments_dir: str) -> dict:
    files = sorted(glob.glob(os.path.join(moments_dir, "oscar_moments_rank*.pt")))
    if not files:
        raise SystemExit(f"no oscar_moments_rank*.pt under {moments_dir}")
    parts = [torch.load(f, map_location="cpu") for f in files]
    parts.sort(key=lambda p: p["tp_rank"])
    base = parts[0]
    merged = {k: base[k] for k in ("model_path", "model_revision", "prompt_sha256", "tokens",
                                   "global_kv_heads", "global_q_heads", "head_dim", "v_head_dim")}
    merged["tp_size"] = len(parts)
    layers: dict = {}
    for lid in base["layers"]:
        cat = {}
        for key in ("M_q", "k_sum", "M_k", "S_v"):
            cat[key] = torch.cat([p["layers"][lid][key] for p in parts], dim=0).to(torch.float64)
        cat["count"] = base["layers"][lid]["count"]
        layers[int(lid)] = cat
    merged["layers"] = layers
    heads = next(iter(layers.values()))["M_q"].shape[0]
    if heads != merged["global_kv_heads"]:
        raise SystemExit(f"moments cover {heads} KV heads but the model has {merged['global_kv_heads']}")
    return merged


def load_o_proj(model: str, layer_ids: list[int], revision: str | None = None) -> dict[int, torch.Tensor]:
    """``model.layers.<i>.self_attn.o_proj.weight`` as fp64 ``[hidden, q_heads*head_dim]``."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    local = model if os.path.isdir(model) else snapshot_download(model, revision=revision, allow_patterns=["*.safetensors", "*.json"])
    index = os.path.join(local, "model.safetensors.index.json")
    want = {f"model.layers.{i}.self_attn.o_proj.weight": i for i in layer_ids}
    out: dict[int, torch.Tensor] = {}
    if os.path.exists(index):
        wmap = json.load(open(index))["weight_map"]
        by_file: dict[str, list[str]] = {}
        for name, lid in want.items():
            if name not in wmap:
                raise SystemExit(f"{name} not in {index}")
            by_file.setdefault(wmap[name], []).append(name)
    else:
        by_file = {"model.safetensors": list(want)}
    for fname, names in by_file.items():
        with safe_open(os.path.join(local, fname), framework="pt") as f:
            for name in names:
                out[want[name]] = f.get_tensor(name).to(torch.float64)
    return out


def pooled_output_metric(w_o: torch.Tensor, kv_heads: int, head_dim: int) -> torch.Tensor:
    """``G_h = sum_{j in G_h} W_{O,j}^T W_{O,j}`` over the query heads reading
    KV head ``h`` (``W_{O,j}`` = the ``head_dim`` input columns of ``W_O`` for
    query head ``j``); ``[kv_heads, head_dim, head_dim]``, each at unit mean
    eigenvalue."""
    q_heads = w_o.shape[1] // head_dim
    gqa = q_heads // kv_heads
    g = torch.zeros((kv_heads, head_dim, head_dim), dtype=torch.float64)
    for j in range(q_heads):
        block = w_o[:, j * head_dim : (j + 1) * head_dim]
        g[j // gqa] += block.T @ block
    return torch.stack([unit_mean_eig(_sym(gh)) for gh in g])


def fit_variant(variant: str, mom: dict, *, shared: bool, k_base: str, o_proj: dict | None):
    hd, vd = int(mom["head_dim"]), int(mom["v_head_dim"])
    h_k = build_hadamard(hd)
    h_v = build_hadamard(vd)
    k_layers: dict = {}
    v_layers: dict = {}
    v_metric = "post_wo" if variant == "outaware" else "pre_wo"
    k_variant = k_base if variant == "outaware" else variant
    centered = k_variant in ("center", "whiten", "flat", "stretch")
    for lid, m in sorted(mom["layers"].items()):
        n = float(m["count"])
        m_q, k_sum, m_k, s_v = m["M_q"], m["k_sum"], m["M_k"], m["S_v"]
        mu = k_sum / n
        s_k = _sym(m_k / n - mu[:, :, None] * mu[:, None, :])  # centered key covariance
        if shared:
            m_q = m_q.mean(0, keepdim=True)
            s_k = s_k.mean(0, keepdim=True)
            s_v = s_v.mean(0, keepdim=True)
            mu = mu.mean(0, keepdim=True)
        heads = m_q.shape[0]
        r_k, q_k, ev_k = [], [], []
        for h in range(heads):
            mq = unit_mean_eig(_sym(m_q[h]))
            if k_variant in ("perhead", "center"):
                r, vals = orthogonal_basis(mq, h_k)
                q = None
            elif k_variant == "whiten":
                r, q, vals = metric_basis(mq, s_k[h], None)
            elif k_variant == "flat":
                r, q, vals = metric_basis(mq, s_k[h], h_k)
            elif k_variant == "stretch":
                x = stretch_metric(mq, unit_mean_eig(s_k[h]))
                r, q, vals = metric_basis(x, s_k[h], h_k)
            else:
                raise SystemExit(f"unknown key variant {k_variant}")
            r_k.append(r); q_k.append(q); ev_k.append(vals)
        r_v, o_v, ev_v = [], [], []
        for h in range(heads):
            sv = _sym(s_v[h])
            if v_metric == "pre_wo":
                r, vals = orthogonal_basis(sv, h_v)
                o = None
            else:
                if o_proj is None:
                    raise SystemExit("outaware needs --model for W_O")
                g_all = pooled_output_metric(o_proj[lid], int(mom["global_kv_heads"]), vd)
                g = g_all.mean(0) if shared else g_all[h]
                r, o, vals = metric_basis(g, sv, h_v)
            r_v.append(r); o_v.append(o); ev_v.append(vals)

        def stack(xs):
            if xs[0] is None:
                return None
            t = torch.stack(xs).to(torch.float32)
            return t[0] if shared else t

        entry = {"layer_id": lid, "rotation": stack(r_k), "eigenvalues": stack(ev_k)}
        if q_k[0] is not None:
            entry["q_rotation"] = stack(q_k)
        if centered:
            entry["k_mean"] = (mu[0] if shared else mu).to(torch.float32)
        k_layers[lid] = entry
        ventry = {"layer_id": lid, "rotation": stack(r_v), "eigenvalues": stack(ev_v)}
        if o_v[0] is not None:
            ventry["o_rotation"] = stack(o_v)
        v_layers[lid] = ventry
    common = {
        "format_version": 3,
        "source_grouping": "layer" if shared else "head",
        "calibration": {
            "model_path": mom["model_path"], "model_revision": mom["model_revision"],
            "prompt_sha256": mom["prompt_sha256"], "tokens": mom["tokens"],
            "global_kv_heads": mom["global_kv_heads"], "global_q_heads": mom["global_q_heads"],
            "created_at_unix": time.time(), "fitter": "rotation/tools/fit_oscar2_variants.py",
        },
    }
    k_state = {**common, "objective": f"oscar2_{k_variant}", "transform": {"key": k_variant, "centered": centered,
               "orthogonal": k_variant in ("perhead", "center")}, "layers": k_layers}
    v_state = {**common, "objective": f"oscar2_v_{v_metric}", "transform": {"value": v_metric,
               "orthogonal": v_metric == "pre_wo"}, "layers": v_layers}
    return k_state, v_state


def self_check(k_state: dict, v_state: dict) -> None:
    """q' . k' must equal q . k and the output un-rotation must invert the value transform."""
    gen = torch.Generator().manual_seed(0)
    for lid, e in list(k_state["layers"].items())[:3]:
        r = e["rotation"].to(torch.float64); q = e.get("q_rotation")
        q = r if q is None else q.to(torch.float64)
        if r.dim() == 2:
            r, q = r[None], q[None]
        d = r.shape[-1]
        x = torch.randn(5, d, generator=gen, dtype=torch.float64); y = torch.randn(7, d, generator=gen, dtype=torch.float64)
        for h in range(r.shape[0]):
            err = ((x @ q[h]) @ (y @ r[h]).T - x @ y.T).abs().max().item()
            assert err < 1e-6 * max(1.0, (x @ y.T).abs().max().item()), f"layer {lid} head {h}: q'.k' != q.k ({err})"
    for lid, e in list(v_state["layers"].items())[:3]:
        r = e["rotation"].to(torch.float64); o = e.get("o_rotation")
        o = r if o is None else o.to(torch.float64)
        if r.dim() == 2:
            r, o = r[None], o[None]
        for h in range(r.shape[0]):
            eye = (r[h] @ o[h].T)
            err = (eye - torch.eye(r.shape[-1], dtype=torch.float64)).abs().max().item()
            assert err < 1e-6, f"layer {lid} head {h}: o_rotation does not invert the value transform ({err})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--moments-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="perhead,center,whiten,flat,stretch,outaware")
    ap.add_argument("--k-base", default="flat", help="key transform under the outaware values")
    ap.add_argument("--model", default=None, help="HF id or local dir with W_O (needed for outaware)")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--shared", action="store_true", help="one basis per layer (V1 grouping) instead of per head")
    a = ap.parse_args()
    mom = load_moments(a.moments_dir)
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    o_proj = None
    if "outaware" in variants:
        model = a.model or mom["model_path"]
        o_proj = load_o_proj(model, sorted(mom["layers"]), a.revision or mom.get("model_revision"))
    for v in variants:
        t0 = time.time()
        k_state, v_state = fit_variant(v, mom, shared=a.shared, k_base=a.k_base, o_proj=o_proj)
        self_check(k_state, v_state)
        d = os.path.join(a.out, v + ("-shared" if a.shared else ""))
        os.makedirs(d, exist_ok=True)
        # k_rotation_*.pt / v_rotation_*.pt: the names the run recipes and the
        # smoke harness discover in a rotation directory (one pair per dir).
        torch.save(k_state, os.path.join(d, f"k_rotation_oscar2_{v}.pt"))
        torch.save(v_state, os.path.join(d, f"v_rotation_oscar2_{v}.pt"))
        print(f"[fit] {v:9s} -> {d}  ({len(k_state['layers'])} layers, {time.time() - t0:.1f}s, "
              f"key={k_state['transform']}, value={v_state['transform']})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
