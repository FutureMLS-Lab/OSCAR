#!/usr/bin/env python3
"""Learn per-(layer, KV head) orthogonal K/V rotation corrections through the
INT2 quantizer and attention (the OptR recipe) from the rows the startup
calibration saved with ``SGLANG_OSCAR_CALIBRATION_SAVE_ROWS=1``.

Stage K: ``R_k = R0_k expm(A - A^T)`` with ``A`` trained by Adam, a
straight-through estimator through the write kernel's arithmetic (quantile
clip, per-group min-max or Lloyd-Max levels); objective = squared post-W_O
output error of attention over the quantized keys against exact attention,
summed over the query heads that read the KV head. Stage V: with the
attention weights of the trained quantized keys fixed, the same objective for
the value rotation. ``R0`` comes from a fitted rotation directory (the
closed-form per-head + centering fit), so the trained part is the correction.
The best iterate by training loss is kept, so the result never scores worse
than its init on this objective.

Writes ``k_rotation_oscar2_optr.pt`` / ``v_rotation_oscar2_optr.pt`` in the
per-head checkpoint format the pool loads, plus ``training.json``.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_oscar2_variants import load_o_proj  # noqa: E402

# The write kernel's Lloyd-Max constants (oscar_rotation_clip_int2_kv.py).
LM_THRESHOLD = 0.9810652732849121
LM_EDGE = 1.5095585584640503
LM_RATIO = 1.16


def int2_dequant(x: torch.Tensor, *, clip: float, lloyd_max: bool, group: int) -> torch.Tensor:
    """The write kernel's arithmetic on rows ``[..., hd]``: quantile clip over
    the row, then per-group uniform min-max levels, or Lloyd-Max levels on
    the standardized row with the uniform-equivalent dequant."""
    hd = x.shape[-1]
    if clip > 0.0:
        idx = min(int(clip * hd), hd - 1)
        thr = x.abs().sort(dim=-1).values[..., idx : idx + 1]
        x = torch.maximum(torch.minimum(x, thr), -thr)
    if lloyd_max:
        if group != hd:
            raise ValueError("Lloyd-Max levels are per row; the group must equal head_dim")
        mean = x.mean(-1, keepdim=True)
        diff = x - mean
        std = (diff.pow(2).mean(-1, keepdim=True) + 1e-8).sqrt()
        z = diff / std
        q = (z >= -LM_THRESHOLD).to(x.dtype) + (z >= 0.0).to(x.dtype) + (z >= LM_THRESHOLD).to(x.dtype)
        step = 2.0 * LM_EDGE / 3.0
        scale = step * LM_RATIO * std
        zero = LM_EDGE / step - mean / scale
        return (q - zero) * scale
    if hd % group:
        raise ValueError(f"group {group} does not divide head_dim {hd}")
    g = x.reshape(*x.shape[:-1], hd // group, group)
    mn = g.min(-1, keepdim=True).values
    mx = g.max(-1, keepdim=True).values
    scale = torch.clamp(mx - mn, min=1e-8) / 3.0
    zero = -mn / scale
    q = torch.clamp(torch.floor(g / scale + zero + 0.5), 0.0, 3.0)
    return ((q - zero) * scale).reshape(x.shape)


def quantize_ste(x: torch.Tensor, **kw) -> torch.Tensor:
    """Forward = dequantized value, backward = identity."""
    return x + (int2_dequant(x.detach(), **kw) - x.detach())


def skew_rotation(r0: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    return r0 @ torch.linalg.matrix_exp(a - a.transpose(-1, -2))


def load_rows(rows_dir: str) -> dict:
    """Merge the per-rank row files along the head dimension; the saved query
    heads keep their global indices (``q_head_ids``) for the W_O slices."""
    files = sorted(glob.glob(os.path.join(rows_dir, "oscar_rows_rank*.pt")))
    if not files:
        raise SystemExit(f"no oscar_rows_rank*.pt under {rows_dir}")
    parts = [torch.load(f, map_location="cpu") for f in files]
    parts.sort(key=lambda p: p["tp_rank"])
    base = parts[0]
    merged = {k: base[k] for k in ("model_path", "model_revision", "prompt_sha256", "tokens",
                                   "global_kv_heads", "global_q_heads", "head_dim", "v_head_dim", "q_sample_stride")}
    q_head_ids = []
    for p in parts:
        n = next(iter(p["layers"].values()))["q_samples"].shape[1]
        q_head_ids.append(torch.arange(p["q_head_offset"], p["q_head_offset"] + n))
    merged["q_head_ids"] = torch.cat(q_head_ids)
    layers: dict = {}
    for lid in base["layers"]:
        cat = {key: torch.cat([p["layers"][lid][key] for p in parts], dim=1) for key in ("k", "v", "q_samples")}
        cat["positions"] = base["layers"][lid]["positions"]
        cat["q_scaling"] = base["layers"][lid]["q_scaling"]
        layers[int(lid)] = cat
    merged["layers"] = layers
    heads = next(iter(layers.values()))["k"].shape[1]
    if heads != merged["global_kv_heads"]:
        raise SystemExit(f"rows cover {heads} KV heads but the model has {merged['global_kv_heads']}")
    return merged


def load_init(init_dir: str, kind: str, layer_ids: list[int], heads: int) -> tuple[dict[int, torch.Tensor], dict]:
    """Per-head rotations ``[H, hd, hd]`` (a shared matrix is expanded) of the
    ``k_rotation_*.pt`` / ``v_rotation_*.pt`` in ``init_dir`` plus the raw states."""
    files = sorted(glob.glob(os.path.join(init_dir, f"{kind}_rotation_*.pt")))
    if len(files) != 1:
        raise SystemExit(f"{init_dir} must hold exactly one {kind}_rotation_*.pt, found {len(files)}")
    state = torch.load(files[0], map_location="cpu")
    out: dict[int, torch.Tensor] = {}
    for lid in layer_ids:
        entry = state["layers"].get(lid, state["layers"].get(str(lid)))
        if entry is None:
            raise SystemExit(f"{files[0]} has no layer {lid}")
        r = entry["rotation"].to(torch.float32)
        if entry.get("q_rotation") is not None or entry.get("o_rotation") is not None:
            raise SystemExit("the init must be orthogonal (no q_rotation / o_rotation companions)")
        out[lid] = r.expand(heads, *r.shape) if r.dim() == 2 else r
    return out, state


def attention_groups(q_head_ids: torch.Tensor, global_q_heads: int, kv_heads: int) -> list[torch.Tensor]:
    """For each KV head, the columns of the merged query sample tensor whose
    global query head reads it (standard contiguous GQA grouping)."""
    gqa = global_q_heads // kv_heads
    groups = [torch.nonzero(q_head_ids // gqa == g).flatten() for g in range(kv_heads)]
    if any(len(g) == 0 for g in groups):
        raise SystemExit("a KV head has no saved query heads")
    return groups


def causal_mask(positions: torch.Tensor, query_rows: torch.Tensor) -> torch.Tensor:
    """``[S, T]``: key ``t`` is visible to the query at row ``r`` when both
    belong to the same calibration sequence (positions restart) and ``t <= r``."""
    pos = positions.long()
    new_seq = torch.ones_like(pos, dtype=torch.bool)
    new_seq[1:] = pos[1:] != pos[:-1] + 1
    seq_id = new_seq.cumsum(0) - 1
    same = seq_id[query_rows][:, None] == seq_id[None, :]
    causal = torch.arange(pos.numel())[None, :] <= query_rows[:, None]
    return same & causal


class LayerProblem:
    """One layer's rows on the device plus the W_O slices per KV-head group."""

    def __init__(self, rows: dict, w_o: torch.Tensor, groups: list[torch.Tensor], *, stride: int, device: torch.device):
        t = rows["k"].shape[0]
        query_rows = torch.arange(rows["q_samples"].shape[0]) * stride
        keep = query_rows < t
        self.query_rows = query_rows[keep]
        self.k = rows["k"].to(device=device, dtype=torch.float32)
        self.v = rows["v"].to(device=device, dtype=torch.float32)
        self.q = rows["q_samples"][keep].to(device=device, dtype=torch.float32)
        self.mask = causal_mask(rows["positions"], self.query_rows).to(device)
        scaling = rows["q_scaling"]
        self.scaling = float(scaling) if scaling is not None else 1.0 / math.sqrt(self.k.shape[-1])
        vd = self.v.shape[-1]
        self.groups = groups
        self.w_o = []
        for g in groups:
            # [n, vd, hidden]: W_{O,j}^T for the group's query heads j.
            self.w_o.append(torch.stack([w_o[:, j * vd : (j + 1) * vd].T for j in g.tolist()]).to(device=device, dtype=torch.float32))

    def logits(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        logits = torch.einsum("snd,td->snt", q, k) * self.scaling
        return logits.masked_fill(~self.mask[:, None, :], float("-inf"))

    def exact_attention(self, g: int) -> torch.Tensor:
        return torch.softmax(self.logits(self.q[:, self.groups[g]], self.k[:, g]), dim=-1)

    def quantized_attention(self, g: int, r_k: torch.Tensor, k_mean: torch.Tensor, quant: dict) -> torch.Tensor:
        qg = self.q[:, self.groups[g]]
        k_hat = quantize_ste((self.k[:, g] - k_mean) @ r_k, **quant)
        logits = torch.einsum("snd,td->snt", qg @ r_k, k_hat) + (qg @ k_mean)[..., None]
        logits = (logits * self.scaling).masked_fill(~self.mask[:, None, :], float("-inf"))
        return torch.softmax(logits, dim=-1)

    def output(self, g: int, p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.einsum("snv,nvh->sh", torch.einsum("snt,tv->snv", p, v), self.w_o[g])


def k_stage_losses(problem: LayerProblem, rot, k_mean: torch.Tensor, quant: dict, *, backward: bool) -> tuple[float, float]:
    """Sum over groups of ``|| sum_j (P^q_j - P_j) V W_{O,j} ||^2`` and the
    exact output energy. ``rot(g)`` builds head g's rotation inside the group's
    own graph, so the per-group backward (which bounds the live logits to one
    group) never re-enters a shared subgraph."""
    loss_total = 0.0
    ref_total = 0.0
    for g in range(len(problem.groups)):
        with torch.no_grad():
            p = problem.exact_attention(g)
            out_ref = problem.output(g, p, problem.v[:, g])
            ref_total += out_ref.pow(2).sum().item()
        p_q = problem.quantized_attention(g, rot(g), k_mean[g], quant)
        loss = problem.output(g, p_q - p, problem.v[:, g]).pow(2).sum()
        if backward:
            loss.backward()
        loss_total += loss.item()
    return loss_total, ref_total


def v_stage_losses(problem: LayerProblem, r_k: torch.Tensor, k_mean: torch.Tensor, k_quant: dict,
                   rot_v, v_quant: dict, *, backward: bool) -> tuple[float, float]:
    """Under the attention of the trained quantized keys, ``|| sum_j P^q_j (V_hat - V) W_{O,j} ||^2``."""
    loss_total = 0.0
    ref_total = 0.0
    for g in range(len(problem.groups)):
        with torch.no_grad():
            p_q = problem.quantized_attention(g, r_k[g], k_mean[g], k_quant)
            ref_total += problem.output(g, p_q, problem.v[:, g]).pow(2).sum().item()
        r_v = rot_v(g)
        v_hat = quantize_ste(problem.v[:, g] @ r_v, **v_quant) @ r_v.T
        loss = problem.output(g, p_q, v_hat - problem.v[:, g]).pow(2).sum()
        if backward:
            loss.backward()
        loss_total += loss.item()
    return loss_total, ref_total


def train_rotation(r0: torch.Tensor, loss_fn, *, steps: int, lr: float) -> tuple[torch.Tensor, float, float]:
    """Adam on the skew generator with cosine decay; returns the best iterate
    by training loss with its loss and the init loss."""
    a = torch.zeros_like(r0, requires_grad=True)
    opt = torch.optim.Adam([a], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
    best_loss, best_r = None, r0.detach().clone()
    init_loss = None
    for step in range(steps + 1):
        opt.zero_grad(set_to_none=True)
        loss, _ = loss_fn(lambda g: skew_rotation(r0[g], a[g]), backward=step < steps)
        with torch.no_grad():
            r = skew_rotation(r0, a).detach().clone()
        if init_loss is None:
            init_loss = loss
        if best_loss is None or loss < best_loss:
            best_loss, best_r = loss, r
        if step < steps:
            opt.step()
            sched.step()
    return best_r, float(init_loss), float(best_loss)


def checkpoint_state(kind: str, init_state: dict, rows: dict, layers: dict, args, k_quant: dict, v_quant: dict) -> dict:
    return {
        "format_version": 3,
        "source_grouping": "head",
        "objective": f"oscar2_optr_{kind}",
        "transform": {"key": "optr", "centered": True, "orthogonal": True} if kind == "k"
        else {"value": "optr", "orthogonal": True},
        "calibration": {
            "model_path": rows["model_path"], "model_revision": rows["model_revision"],
            "prompt_sha256": rows["prompt_sha256"], "tokens": rows["tokens"],
            "global_kv_heads": rows["global_kv_heads"], "global_q_heads": rows["global_q_heads"],
            "created_at_unix": time.time(), "fitter": "rotation/tools/train_optr_rotations.py",
            "init": os.path.abspath(args.init_dir), "init_objective": init_state.get("objective"),
            "steps": args.steps, "lr": args.lr, "k_quant": k_quant, "v_quant": v_quant,
        },
        "layers": layers,
    }


def self_check(k_state: dict, v_state: dict) -> None:
    for state in (k_state, v_state):
        for lid, e in state["layers"].items():
            r = e["rotation"].to(torch.float64)
            eye = torch.eye(r.shape[-1], dtype=torch.float64)
            err = (r @ r.transpose(-1, -2) - eye).abs().max().item()
            assert err < 1e-4, f"layer {lid}: rotation is not orthogonal ({err})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows-dir", required=True, help="directory with oscar_rows_rank*.pt")
    ap.add_argument("--init-dir", required=True, help="orthogonal per-head rotation pair to start from (k_mean optional)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None, help="HF id or local dir with W_O (default: the rows' model_path)")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--weight-prefix", default="model.", help="tensor name prefix before `layers.<i>`")
    ap.add_argument("--o-proj-file", default=None, help="torch file {layer_id: W_O [hidden, q_heads*vd]} instead of the HF checkpoint")
    ap.add_argument("--group-size", type=int, default=0, help="INT2 scale group (0 = head_dim)")
    ap.add_argument("--k-clip", type=float, default=0.96)
    ap.add_argument("--v-clip", type=float, default=0.92)
    ap.add_argument("--lloyd-max", type=int, default=0)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--layers", default="", help="comma-separated subset of layer ids (default all)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    rows = load_rows(args.rows_dir)
    layer_ids = sorted(rows["layers"])
    if args.layers:
        layer_ids = [int(x) for x in args.layers.split(",") if x.strip()]
    heads = int(rows["global_kv_heads"])
    hd, vd = int(rows["head_dim"]), int(rows["v_head_dim"])
    group = args.group_size or hd
    k_quant = {"clip": args.k_clip, "lloyd_max": bool(args.lloyd_max), "group": group}
    v_quant = {"clip": args.v_clip, "lloyd_max": bool(args.lloyd_max), "group": group if vd == hd else vd}
    k_init, k_init_state = load_init(args.init_dir, "k", layer_ids, heads)
    v_init, _ = load_init(args.init_dir, "v", layer_ids, heads)
    if args.o_proj_file:
        w_o_all = {int(k): v for k, v in torch.load(args.o_proj_file, map_location="cpu").items()}
    else:
        w_o_all = load_o_proj(args.model or rows["model_path"], layer_ids, args.revision or rows.get("model_revision"), prefix=args.weight_prefix)
    groups = attention_groups(rows["q_head_ids"], int(rows["global_q_heads"]), heads)
    print(f"[optr] {len(layer_ids)} layers x {heads} KV heads, {rows['tokens']} tokens, "
          f"{len(rows['q_head_ids'])} query heads saved, levels={'lloyd_max' if args.lloyd_max else 'uniform'} "
          f"group={group} clip={args.k_clip}/{args.v_clip} steps={args.steps} lr={args.lr} device={device}", flush=True)

    k_layers: dict = {}
    v_layers: dict = {}
    summary: dict = {}
    for lid in layer_ids:
        t0 = time.time()
        problem = LayerProblem(rows["layers"][lid], w_o_all[lid].to(torch.float32), groups, stride=int(rows["q_sample_stride"]), device=device)
        init_entry = k_init_state["layers"].get(lid, k_init_state["layers"].get(str(lid)))
        k_mean = init_entry.get("k_mean")
        k_mean = problem.k.mean(0) if k_mean is None else k_mean.to(device=device, dtype=torch.float32)
        if k_mean.dim() == 1:
            k_mean = k_mean.expand(heads, hd)
        r0_k = k_init[lid].to(device)
        r0_v = v_init[lid].to(device)

        def k_loss(rot, backward):
            return k_stage_losses(problem, rot, k_mean, k_quant, backward=backward)

        r_k, k_init_loss, k_best = train_rotation(r0_k, k_loss, steps=args.steps, lr=args.lr)
        _, k_ref = k_stage_losses(problem, lambda g: r_k[g], k_mean, k_quant, backward=False)

        def v_loss(rot, backward):
            return v_stage_losses(problem, r_k, k_mean, k_quant, rot, v_quant, backward=backward)

        r_v, v_init_loss, v_best = train_rotation(r0_v, v_loss, steps=args.steps, lr=args.lr)
        _, v_ref = v_stage_losses(problem, r_k, k_mean, k_quant, lambda g: r_v[g], v_quant, backward=False)

        k_layers[lid] = {"layer_id": lid, "rotation": r_k.cpu().contiguous(), "eigenvalues": init_entry.get("eigenvalues"),
                         "k_mean": k_mean.cpu().contiguous()}
        v_layers[lid] = {"layer_id": lid, "rotation": r_v.cpu().contiguous(), "eigenvalues": None}
        summary[lid] = {"k_rel_init": math.sqrt(k_init_loss / max(k_ref, 1e-30)), "k_rel_final": math.sqrt(k_best / max(k_ref, 1e-30)),
                        "v_rel_init": math.sqrt(v_init_loss / max(v_ref, 1e-30)), "v_rel_final": math.sqrt(v_best / max(v_ref, 1e-30)),
                        "seconds": time.time() - t0}
        s = summary[lid]
        print(f"[optr] layer {lid:3d}: K rel-err {s['k_rel_init']:.4f} -> {s['k_rel_final']:.4f}   "
              f"V rel-err {s['v_rel_init']:.4f} -> {s['v_rel_final']:.4f}   ({s['seconds']:.1f}s)", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    k_state = checkpoint_state("k", k_init_state, rows, k_layers, args, k_quant, v_quant)
    v_state = checkpoint_state("v", k_init_state, rows, v_layers, args, k_quant, v_quant)
    self_check(k_state, v_state)
    os.makedirs(args.out, exist_ok=True)
    torch.save(k_state, os.path.join(args.out, "k_rotation_oscar2_optr.pt"))
    torch.save(v_state, os.path.join(args.out, "v_rotation_oscar2_optr.pt"))
    mean = lambda key: sum(s[key] for s in summary.values()) / max(len(summary), 1)  # noqa: E731
    totals = {"layers": len(summary), "k_rel_init": mean("k_rel_init"), "k_rel_final": mean("k_rel_final"),
              "v_rel_init": mean("v_rel_init"), "v_rel_final": mean("v_rel_final"), "seconds": sum(s["seconds"] for s in summary.values())}
    with open(os.path.join(args.out, "training.json"), "w") as f:
        json.dump({"args": vars(args), "k_quant": k_quant, "v_quant": v_quant, "layers": summary, "mean": totals}, f, indent=1)
    print(f"[optr] mean rel-err K {totals['k_rel_init']:.4f} -> {totals['k_rel_final']:.4f}, "
          f"V {totals['v_rel_init']:.4f} -> {totals['v_rel_final']:.4f}; {totals['seconds']:.0f}s -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
