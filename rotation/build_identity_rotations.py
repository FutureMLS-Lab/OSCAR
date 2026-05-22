#!/usr/bin/env python
"""Generate identity rotation checkpoints for OSCAR INT2 KV cache.

When the SGLang OSCAR rotation is set to identity (`R = I`), the
``rows @ R`` pre-pass becomes a no-op and the int2 quant path runs with only
its fused Hadamard transform — which is the rotation used by
OScaR-KV-Quant's "data-free, calibration-free" Hadamard baseline.

This is the right fallback for models with no offline rotation:
- Qwen3-VL-8B / Qwen3-VL-4B (no calibration dumps exist; multimodal calib
  would require a separate dump pipeline).

Output: a single .pt file per K and V, schema matching ``load_oscar_rotations``:
    {"format_version": "identity-v1",
     "objective": "identity",
     "source_grouping": "identity",
     "layers": {layer_id: {"layer_id": int, "rotation": Tensor[D, D]}, ...}}

Usage:
    python build_identity_rotations.py --num-layers 36 --head-dim 128 \
        --out-dir /path/to/rotations
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def build_identity_state(num_layers: int, head_dim: int, kind: str) -> dict:
    layers = {}
    eye = torch.eye(head_dim, dtype=torch.float32)
    for lid in range(num_layers):
        layers[lid] = {"layer_id": lid, "rotation": eye.clone()}
    return {
        "format_version": "identity-v1",
        "objective": f"identity-{kind}",
        "source_grouping": "identity",
        "layers": layers,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--num-layers", type=int, required=True)
    p.add_argument("--head-dim", type=int, required=True,
                   help="K head dim. V head dim assumed same unless --v-head-dim given.")
    p.add_argument("--v-head-dim", type=int, default=None)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--k-filename", default="k_rotation_qqt_r_h_pbr.pt",
                   help="Matches the name the eval driver expects by default.")
    p.add_argument("--v-filename", default="v_rotation_sst_r_h_pbr.pt")
    args = p.parse_args()

    v_head_dim = args.v_head_dim or args.head_dim
    args.out_dir.mkdir(parents=True, exist_ok=True)

    k_state = build_identity_state(args.num_layers, args.head_dim, "k")
    v_state = build_identity_state(args.num_layers, v_head_dim, "v")

    k_path = args.out_dir / args.k_filename
    v_path = args.out_dir / args.v_filename
    torch.save(k_state, k_path)
    torch.save(v_state, v_path)
    print(f"[identity-rot] wrote {k_path} ({args.num_layers} layers x "
          f"{args.head_dim}x{args.head_dim})")
    print(f"[identity-rot] wrote {v_path} ({args.num_layers} layers x "
          f"{v_head_dim}x{v_head_dim})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
