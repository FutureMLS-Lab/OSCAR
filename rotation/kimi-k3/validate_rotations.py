#!/usr/bin/env python3
"""Validate Kimi-K3 OSCAR rotation checkpoint schema and orthogonality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def validate(path: Path, expected_layers: list[int], head_dim: int) -> dict:
    state = torch.load(path, map_location="cpu")
    if not isinstance(state.get("layers"), dict):
        raise ValueError(f"{path}: missing layers dictionary")
    layers = {int(key): value for key, value in state["layers"].items()}
    missing = sorted(set(expected_layers) - set(layers))
    extras = sorted(set(layers) - set(expected_layers))
    if missing or extras:
        raise ValueError(f"{path}: missing={missing}, extras={extras}")

    max_error = 0.0
    for layer_id in expected_layers:
        rotation = layers[layer_id]["rotation"].float()
        if rotation.shape != (head_dim, head_dim):
            raise ValueError(
                f"{path}: layer {layer_id} shape={tuple(rotation.shape)}, "
                f"expected {(head_dim, head_dim)}"
            )
        if not torch.isfinite(rotation).all():
            raise ValueError(f"{path}: layer {layer_id} contains non-finite values")
        error = (rotation @ rotation.T - torch.eye(head_dim)).abs().max().item()
        max_error = max(max_error, error)
    if max_error > 1e-3:
        raise ValueError(f"{path}: max orthogonality error {max_error:.3e}")
    return {
        "path": str(path),
        "layers": len(layers),
        "head_dim": head_dim,
        "max_orthogonality_error": max_error,
        "objective": state.get("objective"),
        "format_version": state.get("format_version"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--k-rotation", type=Path, required=True)
    parser.add_argument("--v-rotation", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())["text_config"]
    num_layers = config["num_hidden_layers"]
    kda = {layer_id - 1 for layer_id in config["linear_attn_config"]["kda_layers"]}
    full_layers = [layer_id for layer_id in range(num_layers) if layer_id not in kda]
    k_dim = config["qk_nope_head_dim"] + config["qk_rope_head_dim"]
    v_dim = config["v_head_dim"]
    result = {
        "expected_full_attention_layers": full_layers,
        "k": validate(args.k_rotation, full_layers, k_dim),
        "v": validate(args.v_rotation, full_layers, v_dim),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
