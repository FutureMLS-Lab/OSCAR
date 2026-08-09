#!/usr/bin/env python3
"""Fit per-layer offline Lloyd-Max LUTs for Kimi-K3's 512-d MLA latent."""

from __future__ import annotations

import argparse
import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path

import torch


LUT_SIZE = 8


def layer_dirs(root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir() and path.name.startswith("layer_")
        ),
        key=lambda path: int(path.name.split("_", 1)[1]),
    )


def load_latent(layer_dir: Path) -> torch.Tensor:
    paths = sorted(
        (layer_dir / "latent").glob("*.pt"),
        key=lambda path: int(path.stem),
    )
    if not paths:
        raise FileNotFoundError(f"No latent chunks under {layer_dir}")
    return torch.cat(
        [
            torch.load(path, map_location="cpu", weights_only=True)
            .float()
            .reshape(-1, 512)
            for path in paths
        ],
        dim=0,
    )


def manifest_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.glob("layer_*/latent/*.pt")):
        stat = path.stat()
        digest.update(str(path.relative_to(root)).encode())
        digest.update(str(stat.st_size).encode())
    return digest.hexdigest()


def build_hadamard(size: int, device: torch.device) -> torch.Tensor:
    if size < 1 or size & (size - 1):
        raise ValueError(f"Hadamard size must be a power of two, got {size}")
    value = torch.ones((1, 1), dtype=torch.float32, device=device)
    while value.shape[0] < size:
        value = torch.cat(
            [torch.cat([value, value], 1), torch.cat([value, -value], 1)],
            0,
        ) / math.sqrt(2.0)
    return value


def bit_reversal(size: int, device: torch.device) -> torch.Tensor:
    bits = int(math.log2(size))
    return torch.tensor(
        [int(f"{index:0{bits}b}"[::-1], 2) for index in range(size)],
        dtype=torch.long,
        device=device,
    )


def latent_oscar_rotation(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    covariance = values.T @ values / values.shape[0]
    covariance = (covariance + covariance.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    hadamard = build_hadamard(values.shape[1], values.device)
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    permutation = torch.empty_like(sorted_indices)
    permutation[bit_reversal(values.shape[1], values.device)] = sorted_indices
    rotation = (eigenvectors @ hadamard)[:, permutation]
    error = (
        rotation.T @ rotation
        - torch.eye(rotation.shape[0], device=rotation.device)
    ).abs().max()
    if float(error) > 5e-4:
        raise ValueError(f"Latent rotation is not orthogonal: {float(error):.3e}")
    return rotation, eigenvalues


def clip_rows(values: torch.Tensor, ratio: float) -> torch.Tensor:
    index = min(int(ratio * values.shape[-1]), values.shape[-1] - 1)
    threshold = torch.kthvalue(values.abs(), index + 1, dim=-1).values
    return values.clamp(
        min=-threshold.unsqueeze(-1),
        max=threshold.unsqueeze(-1),
    )


def sample_flat(
    values: torch.Tensor,
    max_samples: int,
    generator: torch.Generator,
) -> torch.Tensor:
    flat = values.flatten()
    if max_samples > 0 and flat.numel() > max_samples:
        indices = torch.randint(
            flat.numel(),
            (max_samples,),
            generator=generator,
            device=flat.device,
        )
        flat = flat[indices]
    return flat


def e4m3_candidates() -> torch.Tensor:
    patterns = torch.arange(256, dtype=torch.uint8)
    values = patterns.view(torch.float8_e4m3fn).float()
    return torch.unique(values[torch.isfinite(values)], sorted=True)


def snap_distinct(centers: torch.Tensor) -> torch.Tensor:
    targets = centers.detach().float().cpu().sort().values
    candidates = e4m3_candidates()
    selected: list[torch.Tensor] = []
    previous = -1
    for index, target in enumerate(targets):
        remaining = targets.numel() - index - 1
        lower = previous + 1
        upper = candidates.numel() - remaining - 1
        insertion = int(torch.searchsorted(candidates, target))
        choices = {
            max(lower, min(upper, insertion)),
            max(lower, min(upper, insertion - 1)),
        }
        best = min(choices, key=lambda item: abs(float(candidates[item] - target)))
        selected.append(candidates[best])
        previous = best
    codebook = torch.stack(selected)
    if not torch.all(codebook[1:] > codebook[:-1]):
        raise ValueError("Snapped E4M3 codebook is not strictly increasing")
    return codebook


def fit_lloyd_max(
    samples: torch.Tensor,
    iterations: int,
) -> torch.Tensor:
    quantiles = (
        torch.arange(LUT_SIZE, device=samples.device, dtype=torch.float32) + 0.5
    ) / LUT_SIZE
    centers = torch.quantile(samples.float(), quantiles)
    for _ in range(iterations):
        boundaries = (centers[:-1] + centers[1:]) * 0.5
        assignment = torch.bucketize(samples, boundaries)
        counts = torch.bincount(assignment, minlength=LUT_SIZE)
        sums = torch.bincount(
            assignment,
            weights=samples,
            minlength=LUT_SIZE,
        )
        updated = centers.clone()
        nonempty = counts > 0
        updated[nonempty] = sums[nonempty] / counts[nonempty]
        if torch.equal(updated, centers):
            break
        centers = updated
    return snap_distinct(centers)


def quantization_mse(values: torch.Tensor, codebook: torch.Tensor) -> float:
    codebook = codebook.to(values.device)
    boundaries = (codebook[:-1] + codebook[1:]) * 0.5
    restored = codebook[torch.bucketize(values, boundaries)]
    return float(torch.mean((values - restored) ** 2))


def state_template(
    *,
    oscar: bool,
    args: argparse.Namespace,
    calibration_hash: str,
) -> dict:
    return {
        "format": "kimi_mla_latent_lutb_int3_e4m3",
        "format_version": 1,
        "method": "lloyd_max",
        "lloyd_max_rule": "float_centroids_round_once_to_distinct_e4m3",
        "oscar": oscar,
        "oscar_objective": "latent_covariance_r_h_pbr" if oscar else None,
        "clip_ratio": args.clip_ratio if oscar else 0.0,
        "latent_dim": 512,
        "lut_granule": {"k": 64, "n": 8},
        "calibration_name": args.calibration_name,
        "calibration_manifest_sha256": calibration_hash,
        "max_samples_per_layer": args.max_samples,
        "seed": args.seed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "layers": {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-path", type=Path, required=True)
    parser.add_argument("--output-no-oscar", type=Path, required=True)
    parser.add_argument("--output-oscar", type=Path, required=True)
    parser.add_argument("--calibration-name", required=True)
    parser.add_argument("--max-samples", type=int, default=2_000_000)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--clip-ratio", type=float, default=0.96)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if not 0.0 < args.clip_ratio <= 1.0:
        parser.error("--clip-ratio must be in (0, 1]")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA fitting requested but no GPU is visible")

    directories = layer_dirs(args.dump_path)
    if not directories:
        raise FileNotFoundError(f"No layer directories under {args.dump_path}")
    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    calibration_hash = manifest_hash(args.dump_path)
    plain_state = state_template(
        oscar=False, args=args, calibration_hash=calibration_hash
    )
    oscar_state = state_template(
        oscar=True, args=args, calibration_hash=calibration_hash
    )

    for directory in directories:
        layer_id = int(directory.name.split("_", 1)[1])
        values = load_latent(directory).to(device)
        if values.shape[-1] != 512:
            raise ValueError(
                f"Layer {layer_id} latent dimension is {values.shape[-1]}, expected 512"
            )

        plain_samples = sample_flat(values, args.max_samples, generator)
        plain_codebook = fit_lloyd_max(plain_samples, args.iterations)
        plain_mse = quantization_mse(plain_samples, plain_codebook)
        plain_state["layers"][layer_id] = {
            "layer_id": layer_id,
            "codebook": plain_codebook,
            "codebook_e4m3_bits": plain_codebook.to(
                torch.float8_e4m3fn
            ).view(torch.uint8),
            "sample_count": int(plain_samples.numel()),
            "calibration_mse": plain_mse,
        }

        rotation, eigenvalues = latent_oscar_rotation(values)
        rotated = clip_rows(values @ rotation, args.clip_ratio)
        oscar_samples = sample_flat(rotated, args.max_samples, generator)
        oscar_codebook = fit_lloyd_max(oscar_samples, args.iterations)
        oscar_mse = quantization_mse(oscar_samples, oscar_codebook)
        oscar_state["layers"][layer_id] = {
            "layer_id": layer_id,
            "codebook": oscar_codebook,
            "codebook_e4m3_bits": oscar_codebook.to(
                torch.float8_e4m3fn
            ).view(torch.uint8),
            "rotation": rotation.float().cpu(),
            "eigenvalues": eigenvalues.float().cpu(),
            "sample_count": int(oscar_samples.numel()),
            "calibration_mse_rotated_domain": oscar_mse,
        }
        print(
            f"layer={layer_id:02d} tokens={values.shape[0]} "
            f"plain_mse={plain_mse:.6g} oscar_mse={oscar_mse:.6g}",
            flush=True,
        )

    for path, state in (
        (args.output_no_oscar, plain_state),
        (args.output_oscar, oscar_state),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)
        print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
