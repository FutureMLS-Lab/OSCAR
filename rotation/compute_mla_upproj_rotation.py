#!/usr/bin/env python3
"""Compute OSCAR latent rotations from the MLA up-projection geometry.

Motivation
----------
The first MLA latent rotation (``compute_mla_kv_rotation.py``) aligns the INT2
grid to the eigenbasis of ``c_kv``'s *own* covariance (``c_kv^T c_kv``). That is
the ``ktk``-analog: it minimizes raw reconstruction error of the latent in
isolation and ignores how ``c_kv`` is actually used.

OSCAR's real idea is to align the grid to the *functional sensitivity* of the
downstream computation (``qqt`` for K, ``sst`` for V), not the tensor's own
variance. In MLA the single latent ``c_kv`` feeds **both** the score path and
the value path through the up-projection ``kv_b_proj``:

    score_h = (q_nope_h @ W_UK_h) · c_kv        (absorbed query acts on c_kv)
    out_h   = (Σ_i p_i c_kv_i)   @ W_UV_h       (value read-out of c_kv)

so a quantization error ``Δc`` in the latent propagates as

    score error : (q_nope_h @ W_UK_h) · Δc      → Hessian  W_UK_h^T Σ_q W_UK_h
    output error: Δc @ W_UV_h                   → Hessian  W_UV_h^T W_UV_h

The latent-space Hessian (isotropic-query approximation Σ_q = I) is therefore

    H = Σ_h W_UK_h^T W_UK_h  +  α · Σ_h W_UV_h^T W_UV_h          [512 × 512]

and the OSCAR-for-latent rotation R = eigenvectors(H) (ascending), which the
runtime applies as ``c_rot = c @ R`` / ``c_deq = c_q @ R.T``. With α = 1 this is
exactly the right-singular basis of ``kv_b_proj`` (since [W_UK; W_UV] = kv_b_proj
up to a row reorder).

The up-projection weights come straight from the checkpoint — no calibration
dump is needed. ``kv_b_proj`` is FP8 block-quantized (128×128) in GLM-5.1-FP8,
so we block-dequantize before building the Hessian.

Output: one ``layer_<id>.pt`` (float32 [kv_lora_rank, kv_lora_rank]) per layer,
identical schema to ``compute_mla_kv_rotation.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _resolve_snapshot(model_path: str) -> Path:
    """Accept either a local snapshot dir or a HF cache repo id."""
    p = Path(model_path)
    if (p / "model.safetensors.index.json").exists():
        return p
    # HF cache layout: <HF_HOME>/hub/models--org--name/snapshots/<hash>/
    hub = Path(os.environ.get("HF_HOME", "/shared/huggingface")) / "hub"
    cache = hub / ("models--" + model_path.replace("/", "--"))
    snaps = cache / "snapshots"
    if snaps.is_dir():
        for d in sorted(snaps.iterdir()):
            if (d / "model.safetensors.index.json").exists():
                return d
    raise FileNotFoundError(f"Could not resolve a snapshot dir from {model_path!r}")


def block_dequant(w: torch.Tensor, scale_inv: torch.Tensor, block: int = 128) -> torch.Tensor:
    """Dequantize a block-wise FP8 weight: w_deq[i,j] = w_fp8[i,j] * scale[i//B, j//B]."""
    out_dim, in_dim = w.shape
    wf = w.to(torch.float32)
    s_full = (
        scale_inv.repeat_interleave(block, 0)[:out_dim]
        .repeat_interleave(block, 1)[:, :in_dim]
    )
    return wf * s_full


class KVBProjReader:
    """Lazily reads + dequantizes per-layer kv_b_proj from a sharded checkpoint."""

    def __init__(self, snapshot: Path):
        self.snapshot = snapshot
        idx = json.load(open(snapshot / "model.safetensors.index.json"))
        self.weight_map = idx["weight_map"]

    def layer_ids(self) -> list[int]:
        ids = []
        for k in self.weight_map:
            if k.endswith("self_attn.kv_b_proj.weight"):
                ids.append(int(k.split(".")[2]))
        return sorted(ids)

    def _get(self, key: str) -> torch.Tensor:
        path = self.snapshot / self.weight_map[key]
        with safe_open(str(path), framework="pt") as sf:
            return sf.get_tensor(key)

    def get_dequant(self, layer_id: int) -> torch.Tensor:
        wk = f"model.layers.{layer_id}.self_attn.kv_b_proj.weight"
        w = self._get(wk)
        if w.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            scale = self._get(wk + "_scale_inv")
            return block_dequant(w, scale)
        return w.to(torch.float32)


def latent_hessian(
    w_kv_b: torch.Tensor,
    qk_nope_head_dim: int,
    v_head_dim: int,
    kv_lora_rank: int,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (H, H_score, H_val) — each [kv_lora_rank, kv_lora_rank]."""
    # kv_b_proj.weight: [num_heads*(qk_nope+v), kv_lora_rank]
    w_kc, w_vc = w_kv_b.unflatten(
        0, (-1, qk_nope_head_dim + v_head_dim)
    ).split([qk_nope_head_dim, v_head_dim], dim=1)
    Wk = w_kc.reshape(-1, kv_lora_rank).double()   # [num_heads*qk_nope, R]
    Wv = w_vc.reshape(-1, kv_lora_rank).double()   # [num_heads*v_head,  R]
    h_score = Wk.T @ Wk
    h_val = Wv.T @ Wv
    H = h_score + alpha * h_val
    H = (H + H.T) / 2.0
    return H, h_score, h_val


def compute_rotation(H: torch.Tensor) -> torch.Tensor:
    """Eigenvectors of H, ascending (low-sensitivity dims first)."""
    _, eigvecs = torch.linalg.eigh(H)   # ascending
    return eigvecs.float().contiguous()


def build_hadamard(n: int) -> torch.Tensor:
    if n < 1 or n & (n - 1):
        raise ValueError(f"Hadamard size must be a power of two, got {n}")
    if n == 1:
        return torch.ones(1, 1, dtype=torch.float64)
    h = build_hadamard(n // 2)
    return torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0) / math.sqrt(2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default="zai-org/GLM-5.1-FP8",
                   help="Local snapshot dir or HF cache repo id.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--kv-lora-rank", type=int, default=512)
    p.add_argument("--qk-nope-head-dim", type=int, default=192)
    p.add_argument("--v-head-dim", type=int, default=256)
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Value-path weight in H = H_score + α·H_val. α=1 ⇒ full "
                        "kv_b_proj Gram.")
    p.add_argument("--compose-hadamard", action="store_true",
                   help="Save R·H (eigvecs composed with a Hadamard) instead of R.")
    p.add_argument("--hp-k", type=int, default=0,
                   help="If >0, also emit the top-k most-sensitive eigenvectors of "
                        "the trace-normalized Hessian (H_score/tr + α·H_val/tr) per "
                        "layer as [k, kv_lora_rank] for the HP-subspace runtime path.")
    p.add_argument("--hp-output-dir", type=Path, default=None,
                   help="Output dir for the top-k HP-subspace eigenvectors.")
    p.add_argument("--max-layers", type=int, default=None,
                   help="Limit number of layers (debug).")
    args = p.parse_args()
    if args.hp_k and args.hp_output_dir is None:
        raise ValueError("--hp-k requires --hp-output-dir")

    snapshot = _resolve_snapshot(args.model_path)
    logger.info("Using snapshot %s", snapshot)
    reader = KVBProjReader(snapshot)
    layer_ids = reader.layer_ids()
    if args.max_layers:
        layer_ids = layer_ids[: args.max_layers]
    logger.info("Found %d kv_b_proj layers", len(layer_ids))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    had = build_hadamard(args.kv_lora_rank).double() if args.compose_hadamard else None

    hp_out = None
    if args.hp_k:
        hp_out = Path(args.hp_output_dir)
        hp_out.mkdir(parents=True, exist_ok=True)

    for lid in layer_ids:
        w = reader.get_dequant(lid)
        H, h_score, h_val = latent_hessian(
            w, args.qk_nope_head_dim, args.v_head_dim, args.kv_lora_rank, args.alpha
        )
        R = compute_rotation(H)
        if had is not None:
            R = (R.double() @ had).float().contiguous()
        if hp_out is not None:
            # Top-k eigenvectors of the trace-normalized Hessian (score and value
            # paths weighted equally), most-sensitive first. Saved as [k, R].
            h_norm = h_score / h_score.diagonal().sum() + args.alpha * h_val / h_val.diagonal().sum()
            h_norm = (h_norm + h_norm.T) / 2.0
            _, u = torch.linalg.eigh(h_norm)            # ascending
            uk = u[:, -args.hp_k:].T.float().contiguous()   # [k, kv_lora_rank]
            torch.save(uk, str(hp_out / f"layer_{lid}.pt"))
        # Orthogonality sanity.
        err = (R.T @ R - torch.eye(args.kv_lora_rank)).abs().max().item()
        es = torch.linalg.eigvalsh(h_score)
        ev = torch.linalg.eigvalsh(h_val)
        logger.info(
            "  layer %2d: H_score cond=%.0f  H_val cond=%.0f  ortho_err=%.1e",
            lid, (es[-1] / es[0].clamp(min=1e-9)).item(),
            (ev[-1] / ev[0].clamp(min=1e-9)).item(), err,
        )
        torch.save(R, str(out_dir / f"layer_{lid}.pt"))

    logger.info("Done. %d rotation files → %s", len(layer_ids), out_dir)


if __name__ == "__main__":
    main()
