#!/usr/bin/env python3
"""Fit the OSCAR-2 joint latent transform for an MLA model (C2.7, item 3).

The packed pool quantizes the shared latent ``c_kv`` (kv_lora_rank wide), which
is read twice: as the key through ``W_UK`` (logits) and as the value through
``W_UV`` and ``W_O`` (residual output). From a latent dump written by
``SGLANG_OSCAR_MLA_LATENT_DUMP_DIR`` and the checkpoint's ``kv_b_proj`` /
``o_proj`` weights this fits, per layer:

  cov     R_c = E P H_block                      (the current Rcov.P.H_block, plus centering)
  key     R_c = M_K^{1/2} E P H_block            M_K = sum_h E[q_abs,h^T q_abs,h]        (logit error metric)
  value   R_c = M_V^{1/2} E P H_block            M_V = sum_{h,h'} rho_hh' B_h B_h'^T     (post-W_O output error metric,
                                                 B_h = W_UV,h^T W_O,h^T, rho the heads' attention overlap)
  joint   R_c = (M_K/tr + lam * M_V/tr)^{1/2} E P H_block

with E the eigenbasis of the metric-whitened latent covariance, P the
bit-reversal permutation, H_block the per-group Hadamard. Non-orthogonal
variants ship ``q_rotation = R_c^{-T}`` (the query side, and the output side
since key and value share the latent) and every variant ships the latent
``mean``; the packed pool reads both (``_load_latent_companions``).

  python rotation/tools/fit_mla_joint_latent.py --dump-dir /scratch/glm53-latent-dump \\
      --model zai-org/GLM-5.3 --out /scratch/glm53-rotations-joint --variants cov,key,value,joint
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
from sglang.srt.mem_cache.oscar_calibration import build_hadamard, make_br_perm_matrix  # noqa: E402

EIG_FLOOR = 1e-6
# Relative eigenvalue floor of the whitening metric: directions the metric
# rates below this fraction of its largest eigenvalue are treated as that
# fraction, which bounds cond(M^{1/2}) by 10 so the bf16 copies of R_c and
# R_c^{-T} the pool serves with still cancel (a rank-deficient metric would
# otherwise push |R_c^{-T}| into the hundreds and the logit error to 1e-4).
METRIC_FLOOR = 1e-2


def _sym(m: torch.Tensor) -> torch.Tensor:
    return (m + m.transpose(-1, -2)) / 2


def sym_pow(m: torch.Tensor, power: float, floor: float = EIG_FLOOR) -> torch.Tensor:
    vals, vecs = torch.linalg.eigh(_sym(m))
    vals = torch.clamp(vals, min=float(vals.max()) * floor)
    return vecs @ torch.diag(vals**power) @ vecs.T


def unit_mean_eig(m: torch.Tensor) -> torch.Tensor:
    return m / (torch.trace(m) / m.shape[0])


def block_hadamard(d: int, group: int) -> torch.Tensor:
    h = build_hadamard(group)
    out = torch.zeros(d, d, dtype=torch.float64)
    for g in range(d // group):
        out[g * group : (g + 1) * group, g * group : (g + 1) * group] = h
    return out


def load_dump_layer(dump_dir: str, layer_id: int) -> dict:
    """Merge the per-rank files of one layer: latent rows from rank 0 (every
    rank sees the same tokens), query heads concatenated over ranks."""
    files = sorted(glob.glob(os.path.join(dump_dir, f"layer_{layer_id}_rank*.pt")), key=lambda f: int(re.search(r"rank(\d+)", f).group(1)))
    if not files:
        raise SystemExit(f"no layer_{layer_id}_rank*.pt under {dump_dir}")
    parts = [torch.load(f, map_location="cpu") for f in files]
    base = parts[0]
    for p in parts[1:]:
        if int(p["tokens"]) != int(base["tokens"]) or not torch.equal(p["q_rows"], base["q_rows"]):
            raise SystemExit(f"layer {layer_id}: ranks dumped different token sets")
    return {
        "c_kv": base["c_kv"].to(torch.float64),
        "k_pe": base["k_pe"].to(torch.float64),
        "positions": base["positions"].to(torch.int64),
        "q_rows": base["q_rows"].to(torch.int64),
        "q_nope": torch.cat([p["q_nope"] for p in parts], dim=1).to(torch.float64),
        "q_pe": torch.cat([p["q_pe"] for p in parts], dim=1).to(torch.float64),
    }


def dump_layer_ids(dump_dir: str) -> list[int]:
    ids = {int(re.search(r"layer_(\d+)_rank", f).group(1)) for f in glob.glob(os.path.join(dump_dir, "layer_*_rank*.pt"))}
    if not ids:
        raise SystemExit(f"no latent dump under {dump_dir}")
    return sorted(ids)


def latent_statistics(d: dict, *, scaling: float) -> dict:
    """``mean``, centered covariance ``S_c``, the key metric ``M_K`` and the
    heads' attention overlap ``rho`` from one layer's dump."""
    c, k_pe, pos, q_rows, q_nope, q_pe = d["c_kv"], d["k_pe"], d["positions"], d["q_rows"], d["q_nope"], d["q_pe"]
    mean = c.mean(0)
    cc = c - mean
    s_c = _sym(cc.T @ cc / c.shape[0])
    n_heads = q_nope.shape[1]
    m_k = torch.einsum("shr,shq->rq", q_nope, q_nope) / (q_nope.shape[0] * n_heads)
    rho = torch.zeros(n_heads, n_heads, dtype=torch.float64)
    used = 0
    for s in range(q_rows.shape[0]):
        r = int(q_rows[s])
        start = r - int(pos[r])
        if start < 0 or int(pos[start]) != 0:
            continue
        logits = (q_nope[s] @ c[start : r + 1].T + q_pe[s] @ k_pe[start : r + 1].T) * scaling
        a = torch.softmax(logits, dim=-1)
        rho += a @ a.T
        used += 1
    if used == 0:
        raise SystemExit("no sampled query had its whole sequence inside the dump")
    return {"mean": mean, "S_c": s_c, "M_K": _sym(m_k), "rho": rho / used, "rho_samples": used}


def value_metric(w_uv: torch.Tensor, w_o: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """``M_V = sum_{h,h'} rho_hh' B_h B_h'^T`` with ``B_h = W_UV,h^T W_O,h^T``
    (latent -> head value -> residual), through the eigendecomposition of rho
    so the cost is heads x rank x hidden instead of heads^2."""
    heads, v_dim, rank = w_uv.shape
    b = torch.einsum("hvr,hvo->hro", w_uv, w_o)  # [heads, rank, hidden]
    vals, vecs = torch.linalg.eigh(_sym(rho))
    m_v = torch.zeros(rank, rank, dtype=torch.float64)
    for m in range(heads):
        if vals[m] <= 0:
            continue
        c_m = torch.einsum("h,hro->ro", vecs[:, m], b)
        m_v += vals[m] * (c_m @ c_m.T)
    return _sym(m_v)


def fit_layer(stats: dict, *, variant: str, lam: float, group: int, m_v: torch.Tensor | None) -> dict:
    rank = stats["S_c"].shape[0]
    if variant == "cov":
        metric = None
    elif variant == "key":
        metric = unit_mean_eig(stats["M_K"])
    elif variant == "value":
        metric = unit_mean_eig(m_v)
    elif variant == "joint":
        metric = unit_mean_eig(unit_mean_eig(stats["M_K"]) + lam * unit_mean_eig(m_v))
    else:
        raise SystemExit(f"unknown variant {variant}")
    a = torch.eye(rank, dtype=torch.float64) if metric is None else sym_pow(metric, 0.5, METRIC_FLOOR)
    a_inv = torch.eye(rank, dtype=torch.float64) if metric is None else sym_pow(metric, -0.5, METRIC_FLOOR)
    vals, e = torch.linalg.eigh(_sym(a @ stats["S_c"] @ a))
    tail = e @ make_br_perm_matrix(vals) @ block_hadamard(rank, group)
    r_c = a @ tail
    q_c = a_inv @ tail
    out = {"rotation": r_c.to(torch.float32).contiguous(), "mean": stats["mean"].to(torch.float32), "eigenvalues": vals.to(torch.float32),
           "transform": {"latent": variant, "centered": True, "orthogonal": metric is None, "lam": lam, "rho_samples": stats["rho_samples"]}}
    if metric is not None:
        out["q_rotation"] = q_c.to(torch.float32).contiguous()
    out["transform"]["bf16_logit_relerr"] = stored_logit_error(r_c, q_c, torch.bfloat16)
    return out


def stored_logit_error(r_c: torch.Tensor, q_c: torch.Tensor, dtype: torch.dtype) -> float:
    """Relative error of ``(x Q)(y R)^T`` against ``x y^T`` when the pair is
    stored in ``dtype``, the precision the pool serves it in."""
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(64, r_c.shape[0], generator=gen, dtype=torch.float64)
    y = torch.randn(64, r_c.shape[0], generator=gen, dtype=torch.float64)
    r, q = r_c.to(dtype).to(torch.float64), q_c.to(dtype).to(torch.float64)
    exact = x @ y.T
    return float(((x @ q) @ (y @ r).T - exact).norm() / exact.norm())


def self_check(entry: dict) -> None:
    """On the stored fp32 pair: ``q'.k' == q.k`` and the output un-rotation
    inverts the write transform to 1e-4 relative (the metric floor keeps
    cond(R_c) <= 10, so fp32 rounding stays far below that); the bf16 error the
    pool will see is recorded on the entry."""
    r = entry["rotation"].to(torch.float64)
    q = entry.get("q_rotation")
    q = r if q is None else q.to(torch.float64)
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(5, r.shape[0], generator=gen, dtype=torch.float64)
    y = torch.randn(7, r.shape[0], generator=gen, dtype=torch.float64)
    scale = max(1.0, (x @ y.T).abs().max().item())
    err = ((x @ q) @ (y @ r).T - x @ y.T).abs().max().item()
    assert err < 1e-4 * scale, f"q'.k' != q.k ({err / scale:.2e} relative)"
    back = ((y @ r) @ q.T - y).abs().max().item()
    assert back < 1e-4 * max(1.0, y.abs().max().item()), f"output un-rotation does not invert the transform ({back})"
    bf16 = entry["transform"].get("bf16_logit_relerr")
    if bf16 is not None and bf16 > 2e-2:
        raise SystemExit(f"bf16 copies of this transform perturb the logits by {bf16:.2e}; raise METRIC_FLOOR")


def dequantize_block_fp8(weight: torch.Tensor, scale_inv: torch.Tensor, block: tuple[int, int] = (128, 128)) -> torch.Tensor:
    """FP8 weight with per-``block`` ``weight_scale_inv`` back to fp64:
    ``W[i, j] = w[i, j] * s[i // b0, j // b1]``."""
    w = weight.to(torch.float64)
    b0, b1 = block
    rows = torch.arange(w.shape[0]) // b0
    cols = torch.arange(w.shape[1]) // b1
    return w * scale_inv.to(torch.float64)[rows][:, cols]


def mla_dims(cfg) -> dict:
    """``heads / nope / rope / v / rank`` from a config, descending into
    ``text_config`` for the VL wrappers (Kimi-K3)."""
    for c in (cfg, getattr(cfg, "text_config", None)):
        if c is None:
            continue
        get = lambda k: getattr(c, k, None)  # noqa: E731
        if get("kv_lora_rank") is not None:
            return {"heads": int(get("num_attention_heads")), "nope": int(get("qk_nope_head_dim")), "rope": int(get("qk_rope_head_dim")),
                    "v": int(get("v_head_dim")), "rank": int(get("kv_lora_rank"))}
    raise SystemExit("config has no MLA dims (kv_lora_rank)")


def load_mla_weights(model: str, layer_ids: list[int], *, revision: str | None, n_heads: int, nope: int, v_dim: int, prefix: str = "model."):
    """``(w_uv [heads, v, rank], w_o [heads, v, hidden])`` per layer in fp64
    from ``<prefix>layers.<i>.self_attn.kv_b_proj.weight`` ([heads*(nope+v), rank])
    and ``o_proj.weight`` ([hidden, heads*v]); FP8 block-scaled checkpoints are
    dequantized through their ``weight_scale_inv``."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    local = model if os.path.isdir(model) else snapshot_download(model, revision=revision, allow_patterns=["*.safetensors", "*.json"])
    index = os.path.join(local, "model.safetensors.index.json")
    wmap = json.load(open(index))["weight_map"] if os.path.exists(index) else None
    cfg = json.load(open(os.path.join(local, "config.json")))
    block = tuple(cfg.get("quantization_config", {}).get("weight_block_size") or (128, 128))
    want: dict[str, tuple] = {}
    for lid in layer_ids:
        for short in ("kv_b_proj", "o_proj"):
            base = f"{prefix}layers.{lid}.self_attn.{short}.weight"
            want[base] = (lid, short, "w")
            if wmap is None or base + "_scale_inv" in wmap:
                want[base + "_scale_inv"] = (lid, short, "s")
    by_file: dict[str, list[str]] = {}
    for name in want:
        if wmap is not None and name not in wmap:
            raise SystemExit(f"{name} not in {index}")
        by_file.setdefault(wmap[name] if wmap is not None else "model.safetensors", []).append(name)
    raw: dict = {}
    for fname, names in by_file.items():
        with safe_open(os.path.join(local, fname), framework="pt") as f:
            for name in names:
                try:
                    raw[want[name]] = f.get_tensor(name)
                except Exception:
                    if want[name][2] != "s":
                        raise
    out = {}
    for lid in layer_ids:
        mats = {}
        for short in ("kv_b_proj", "o_proj"):
            w = raw[(lid, short, "w")]
            s_inv = raw.get((lid, short, "s"))
            mats[short] = dequantize_block_fp8(w, s_inv, block) if s_inv is not None else w.to(torch.float64)
        kv_b = mats["kv_b_proj"].reshape(n_heads, nope + v_dim, -1)
        w_uv = kv_b[:, nope:, :]  # [heads, v, rank]
        w_o = mats["o_proj"].T.reshape(n_heads, v_dim, -1)  # [heads, v, hidden]
        out[lid] = (w_uv, w_o)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True, help="HF id or local dir (kv_b_proj / o_proj weights and the MLA dims)")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--variants", default="cov,key,value,joint")
    ap.add_argument("--lam", type=float, default=1.0, help="value weight of the joint metric")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--layers", default=None, help="comma-separated subset of layer ids")
    ap.add_argument("--weight-prefix", default="model.", help="tensor name prefix before `layers.<i>` (Kimi-K3: language_model.model.)")
    a = ap.parse_args()
    from transformers import AutoConfig

    dims = mla_dims(AutoConfig.from_pretrained(a.model, revision=a.revision, trust_remote_code=True))
    n_heads, nope, rope, v_dim = dims["heads"], dims["nope"], dims["rope"], dims["v"]
    scaling = 1.0 / math.sqrt(nope + rope)
    layer_ids = [int(x) for x in a.layers.split(",")] if a.layers else dump_layer_ids(a.dump_dir)
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    for v in variants:
        os.makedirs(os.path.join(a.out, v), exist_ok=True)
    need_weights = any(v in ("value", "joint") for v in variants)
    for lid in layer_ids:
        t0 = time.time()
        d = load_dump_layer(a.dump_dir, lid)
        if d["q_nope"].shape[1] != n_heads:
            raise SystemExit(f"layer {lid}: dump has {d['q_nope'].shape[1]} query heads, config says {n_heads} (missing ranks?)")
        stats = latent_statistics(d, scaling=scaling)
        # One layer of weights at a time: o_proj of a 6k-hidden model is 0.8 GB in fp64.
        w_uv, w_o = load_mla_weights(a.model, [lid], revision=a.revision, n_heads=n_heads, nope=nope, v_dim=v_dim, prefix=a.weight_prefix)[lid] if need_weights else (None, None)
        m_v = value_metric(w_uv, w_o, stats["rho"]) if need_weights else None
        for v in variants:
            entry = fit_layer(stats, variant=v, lam=a.lam, group=a.group_size, m_v=m_v)
            self_check(entry)
            torch.save(entry, os.path.join(a.out, v, f"layer_{lid}.pt"))
        print(f"[fit] layer {lid:3d}: rho from {stats['rho_samples']} queries, {time.time() - t0:.1f}s -> {a.out}/{{{','.join(variants)}}}/layer_{lid}.pt", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
