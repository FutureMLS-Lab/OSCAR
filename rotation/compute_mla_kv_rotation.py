#!/usr/bin/env python3
"""Compute per-layer OSCAR rotation matrices from dumped c_kv tensors.

Dump layout (produced by mla_int2_kv_pool.py in dump mode):

  <dump_path>/layer_<id>.pt   — float32 tensor [n_tokens, kv_lora_rank]

Output: one ``layer_<id>.pt`` per layer in ``--output-dir``, each file
holding a float32 [kv_lora_rank, kv_lora_rank] orthogonal rotation matrix
(the eigenvectors of the c_kv covariance, sorted by eigenvalue).

The runtime loads these with ``torch.load`` and uses them as R in:
    c_kv_rot = c_kv @ R           (encode)
    c_kv_deq = c_kv_qt @ R.T     (decode after fake-quant)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def compute_rotation(x_flat: torch.Tensor) -> torch.Tensor:
    """Covariance eigenvectors of x (sorted ascending by eigenvalue → R.T packs
    low-variance dims first, which INT2 handles better after clipping)."""
    x = x_flat.double()
    cov = x.T @ x / x.shape[0]
    cov = (cov + cov.T) / 2.0
    _, eigvecs = torch.linalg.eigh(cov)   # ascending order
    return eigvecs.float().contiguous()    # [kv_lora_rank, kv_lora_rank]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump-path", required=True, help="Dir with layer_<id>.pt dump files")
    p.add_argument("--output-dir", required=True, help="Output dir for rotation files")
    p.add_argument("--kv-lora-rank", type=int, default=512)
    args = p.parse_args()

    dump_path = Path(args.dump_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layer_files = sorted(
        dump_path.glob("layer_*.pt"),
        key=lambda p: int(p.stem.split("_", 1)[1]),
    )
    if not layer_files:
        raise FileNotFoundError(f"No layer_*.pt files found in {dump_path}")

    logger.info("Found %d layer dump files in %s", len(layer_files), dump_path)

    for f in layer_files:
        layer_id = int(f.stem.split("_", 1)[1])
        x = torch.load(str(f), map_location="cpu").float()
        logger.info("  layer %d: %s tokens, kv_lora_rank=%d", layer_id, x.shape[0], x.shape[1])
        if x.shape[1] != args.kv_lora_rank:
            logger.warning(
                "  layer %d kv_lora_rank mismatch: got %d, expected %d",
                layer_id, x.shape[1], args.kv_lora_rank,
            )
        R = compute_rotation(x)
        out_path = out_dir / f"layer_{layer_id}.pt"
        torch.save(R, str(out_path))
        logger.info("  → saved %s", out_path)

    logger.info("Done. %d rotation files written to %s", len(layer_files), out_dir)


if __name__ == "__main__":
    main()
