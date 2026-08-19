#!/usr/bin/env python3
"""Hardware-shaped W3A8 LUT-B fake-quantization and WikiText PPL evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.configuration_utils import PretrainedConfig


E4M3_MAX = 448.0
LUT_SIZE = 8
TILE_K = 64
TILE_N = 8
A_SCALE_K = 32
LEGACY_B_SCALE_K = 32
NATIVE_B_SCALE_K = 64
B_SCALE_MODES = {
    "none",
    "global",
    "native_k64",
    "global_native_k64",
    "legacy_k32",
    "global_legacy_k32",
}
RUBIN_NATIVE_B_SCALE_MODES = {
    "none",
    "global",
    "native_k64",
    "global_native_k64",
}
ALGORITHM_ORACLE_B_SCALE_MODES = {
    "legacy_k32",
    "global_legacy_k32",
}
GPTQ_METHODS = {
    "gptq_lm",
    "gptq_sensitivity_lm",
    "gptq_hadamard_lm",
    "gptq_hadamard_sensitivity_lm",
    "inter_lloymax_gptq",
    "inter_lloymax_gptq_sensitivity",
    "inter_lloymax_gptq_hadamard",
    "inter_lloymax_gptq_hadamard_sensitivity",
}
HADAMARD_METHODS = {
    "gptq_hadamard_lm",
    "gptq_hadamard_sensitivity_lm",
    "inter_lloymax_gptq_hadamard",
    "inter_lloymax_gptq_hadamard_sensitivity",
}
INTER_LLOYMAX_GPTQ_METHODS = {
    "inter_lloymax_gptq",
    "inter_lloymax_gptq_sensitivity",
    "inter_lloymax_gptq_hadamard",
    "inter_lloymax_gptq_hadamard_sensitivity",
}
AQLM_METHODS = {"aqlm_reference"}


def pack_int3(indices: torch.Tensor) -> torch.Tensor:
    """Pack logical 3-bit values, eight indices into three bytes."""
    if indices.shape[-1] % 8:
        raise ValueError("INT3 packing requires a multiple of eight values")
    if torch.any((indices < 0) | (indices > 7)):
        raise ValueError("INT3 indices must be in [0, 7]")
    groups = indices.to(torch.int32).reshape(*indices.shape[:-1], -1, 8)
    shifts = torch.arange(8, device=indices.device, dtype=torch.int32) * 3
    words = torch.sum(groups << shifts, dim=-1)
    return torch.stack(
        ((words >> 0) & 0xFF, (words >> 8) & 0xFF, (words >> 16) & 0xFF),
        dim=-1,
    ).to(torch.uint8).flatten(-2)


def unpack_int3(packed: torch.Tensor) -> torch.Tensor:
    """Reverse pack_int3; this is not a Rubin descriptor/swizzle decoder."""
    if packed.shape[-1] % 3:
        raise ValueError("Packed INT3 payload must contain three-byte groups")
    groups = packed.to(torch.int32).reshape(*packed.shape[:-1], -1, 3)
    words = groups[..., 0] | (groups[..., 1] << 8) | (groups[..., 2] << 16)
    shifts = torch.arange(8, device=packed.device, dtype=torch.int32) * 3
    return ((words.unsqueeze(-1) >> shifts) & 7).to(torch.uint8).flatten(-2)


def ue8m0_scale(max_abs: torch.Tensor) -> torch.Tensor:
    """Round max_abs / E4M3_MAX upward to a power-of-two scale."""
    tiny = torch.tensor(2.0**-127, device=max_abs.device, dtype=torch.float32)
    ratio = (max_abs.float() / E4M3_MAX).clamp_min(tiny)
    scale = torch.pow(2.0, torch.ceil(torch.log2(ratio)))
    return torch.where(max_abs > 0, scale, torch.ones_like(scale))


def e4m3_round(values: torch.Tensor) -> torch.Tensor:
    return (
        values.float()
        .clamp(-E4M3_MAX, E4M3_MAX)
        .to(torch.float8_e4m3fn)
        .float()
    )


def weight_global_scale(weight: torch.Tensor) -> torch.Tensor:
    """One FP32 tensor scale; this is folded metadata, not a LUT-B operand."""
    max_abs = weight.float().abs().amax()
    tiny = torch.tensor(2.0**-126, device=weight.device, dtype=torch.float32)
    return (max_abs / E4M3_MAX).clamp_min(tiny)


def b_scale_geometry(
    weight: torch.Tensor,
    tiles: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int | None]:
    """Return groups, UE8M0/local scales, global scale, and local K granule."""
    if mode not in B_SCALE_MODES:
        raise ValueError(f"Unsupported LUT-B scale mode: {mode}")
    use_global = mode in {
        "global",
        "global_native_k64",
        "global_legacy_k32",
    }
    global_scale = (
        weight_global_scale(weight)
        if use_global
        else torch.ones((), device=tiles.device, dtype=torch.float32)
    )
    normalized_tiles = tiles.float() / global_scale
    if mode in {"legacy_k32", "global_legacy_k32"}:
        local_k = LEGACY_B_SCALE_K
    elif mode in {"native_k64", "global_native_k64"}:
        local_k = NATIVE_B_SCALE_K
    else:
        local_k = None
    group_k = local_k or TILE_K
    groups = normalized_tiles.view(
        *normalized_tiles.shape[:-1],
        TILE_K // group_k,
        group_k,
    )
    if local_k is None:
        scales = torch.ones(
            *groups.shape[:-1],
            1,
            device=groups.device,
            dtype=torch.float32,
        )
    else:
        scales = ue8m0_scale(groups.abs().amax(dim=-1, keepdim=True))
    return groups, scales, global_scale, local_k


def initial_centers(values: torch.Tensor, method: str) -> torch.Tensor:
    """Return [tiles, 8] floating-point centers."""
    tile_count = values.shape[0]
    if method == "uniform":
        lo = values.amin(dim=1, keepdim=True)
        hi = values.amax(dim=1, keepdim=True)
        alpha = torch.linspace(
            0.0, 1.0, LUT_SIZE, device=values.device, dtype=torch.float32
        )
        return lo + (hi - lo) * alpha
    if method == "nf3":
        quantiles = (
            torch.arange(LUT_SIZE, device=values.device, dtype=torch.float32)
            + 0.5
        ) / LUT_SIZE
        centers = math.sqrt(2.0) * torch.erfinv(2.0 * quantiles - 1.0)
        # An even-sized symmetric quantile codebook has no exact zero. Replace
        # one of the two central values and mirror per tile based on skew, so
        # sparse/near-zero weights are represented without changing storage.
        centers[LUT_SIZE // 2 - 1] = 0.0
        centers = centers / centers.abs().max() * E4M3_MAX
        positive_skew = centers.expand(tile_count, -1)
        negative_skew = -torch.flip(positive_skew, dims=(1,))
        return torch.where(
            (values.mean(dim=1, keepdim=True) >= 0),
            positive_skew,
            negative_skew,
        )
    if method == "lloyd_max":
        ordered = values.sort(dim=1).values
        positions = (
            (
                torch.arange(
                    LUT_SIZE, device=values.device, dtype=torch.float32
                )
                + 0.5
            )
            / LUT_SIZE
            * values.shape[1]
        ).long()
        return ordered[:, positions]
    raise ValueError(f"Unsupported W3 method: {method}")


def weighted_initial_centers(
    values: torch.Tensor, sample_weights: torch.Tensor
) -> torch.Tensor:
    order = values.argsort(dim=1)
    ordered_values = values.gather(1, order)
    ordered_weights = sample_weights.gather(1, order).clamp_min(0)
    cumulative = ordered_weights.cumsum(dim=1)
    total = cumulative[:, -1:].clamp_min(1e-30)
    quantiles = (
        torch.arange(LUT_SIZE, device=values.device, dtype=torch.float32) + 0.5
    ) / LUT_SIZE
    targets = total * quantiles
    positions = torch.searchsorted(cumulative.contiguous(), targets.contiguous())
    return ordered_values.gather(1, positions.clamp_max(values.shape[1] - 1))


def fit_centers(
    values: torch.Tensor,
    method: str,
    iterations: int,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    base_method = (
        "lloyd_max"
        if method in {"sensitivity_lm", "squeezellm_lm"}
        else method
    )
    if method == "squeezellm_lm" and sample_weights is not None:
        centers = weighted_initial_centers(values, sample_weights)
    else:
        centers = initial_centers(values, base_method)
    if base_method != "lloyd_max":
        return e4m3_round(centers)
    if sample_weights is None:
        sample_weights = torch.ones_like(values)
    for _ in range(iterations):
        assignment = (values.unsqueeze(-1) - centers.unsqueeze(1)).abs().argmin(-1)
        sums = torch.zeros_like(centers)
        counts = torch.zeros_like(centers)
        sums.scatter_add_(1, assignment, values * sample_weights)
        counts.scatter_add_(1, assignment, sample_weights)
        updated = torch.where(
            counts > 0, sums / counts.clamp_min(1e-30), centers
        )
        if torch.equal(updated, centers):
            break
        centers = updated
    return e4m3_round(centers)


def search_ue8m0_scales(
    groups: torch.Tensor,
    scales: torch.Tensor,
    centers: torch.Tensor,
) -> torch.Tensor:
    """Choose a nearby power-of-two scale per [N, K32] subgroup."""
    errors = []
    candidates = []
    expanded_centers = centers[:, None, None, None, :]
    for shift in (-2, -1, 0, 1, 2):
        candidate = scales * (2.0**shift)
        normalized = (groups / candidate).clamp(-E4M3_MAX, E4M3_MAX)
        assignment = (
            normalized.unsqueeze(-1) - expanded_centers
        ).abs().argmin(-1)
        codebook = centers[:, None, None, :].expand(
            -1, groups.shape[-3], groups.shape[-2], -1
        )
        restored = codebook.gather(-1, assignment) * candidate
        errors.append(torch.sum((groups - restored) ** 2, dim=-1, keepdim=True))
        candidates.append(candidate)
    stacked_errors = torch.stack(errors, dim=0)
    choice = stacked_errors.argmin(dim=0)
    stacked_candidates = torch.stack(candidates, dim=0)
    return torch.gather(
        stacked_candidates, 0, choice.unsqueeze(0)
    ).squeeze(0)


def quantize_weight_lutb(
    weight: torch.Tensor,
    *,
    method: str,
    iterations: int,
    chunk_tiles: int,
    b_scale_mode: str = "global_native_k64",
    channel_importance: torch.Tensor | None = None,
    element_importance: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Fake-quantize W[N,K] as physical LUT-B B=W.T[K,N]."""
    if weight.ndim != 2:
        raise ValueError(f"Expected a matrix, got {tuple(weight.shape)}")
    n, k = weight.shape
    padded_n = math.ceil(n / TILE_N) * TILE_N
    padded_k = math.ceil(k / TILE_K) * TILE_K
    padded = F.pad(weight.float(), (0, padded_k - k, 0, padded_n - n))

    n_tiles = padded_n // TILE_N
    k_tiles = padded_k // TILE_K
    tiles = (
        padded.view(n_tiles, TILE_N, k_tiles, TILE_K)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(-1, TILE_N, TILE_K)
    )
    importance_tiles = None
    if element_importance is not None:
        if element_importance.shape != (n, k):
            raise ValueError(
                f"Element importance {tuple(element_importance.shape)} "
                f"does not match W={(n, k)}"
            )
        padded_element_importance = F.pad(
            element_importance.float(),
            (0, padded_k - k, 0, padded_n - n),
        )
        importance_tiles = (
            padded_element_importance.view(
                n_tiles, TILE_N, k_tiles, TILE_K
            )
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(-1, TILE_N, TILE_K)
        )
    elif channel_importance is not None:
        if channel_importance.shape != (k,):
            raise ValueError(
                f"Channel importance {tuple(channel_importance.shape)} "
                f"does not match K={k}"
            )
        padded_importance = F.pad(
            channel_importance.float(), (0, padded_k - k)
        )
        importance_tiles = (
            padded_importance.view(1, k_tiles, 1, TILE_K)
            .expand(n_tiles, -1, TILE_N, -1)
            .contiguous()
            .view(-1, TILE_N, TILE_K)
        )
    restored_tiles = torch.empty_like(tiles)

    for start in range(0, tiles.shape[0], chunk_tiles):
        stop = min(start + chunk_tiles, tiles.shape[0])
        tile = tiles[start:stop]
        groups, scales, global_scale, local_scale_k = b_scale_geometry(
            weight,
            tile,
            b_scale_mode,
        )
        fit_method = "lloyd_max" if method == "scale_lm" else method
        centers = None
        for outer in range(2 if method == "scale_lm" else 1):
            normalized = (groups / scales).clamp(-E4M3_MAX, E4M3_MAX)
            flat = normalized.reshape(normalized.shape[0], -1)
            sample_weights = None
            if importance_tiles is not None:
                # Weighted original-domain MSE:
                # h_k * (scale * normalized_error)^2.
                importance = importance_tiles[start:stop].view_as(groups)
                importance = importance / importance.mean(
                    dim=(-1, -2, -3), keepdim=True
                ).clamp_min(1e-12)
                if method == "squeezellm_lm":
                    importance = importance.clamp_min(1e-8)
                else:
                    importance = importance.clamp(1e-4, 100.0)
                sample_weights = (importance * scales.square()).reshape_as(flat)
            centers = fit_centers(
                flat,
                fit_method,
                iterations,
                sample_weights=sample_weights,
            )
            if (
                method == "scale_lm"
                and outer == 0
                and local_scale_k is not None
            ):
                scales = search_ue8m0_scales(groups, scales, centers)
        assert centers is not None
        assignment = (flat.unsqueeze(-1) - centers.unsqueeze(1)).abs().argmin(-1)
        normalized_restored = centers.gather(1, assignment).view_as(normalized)
        restored = (
            normalized_restored * scales * global_scale
        ).reshape_as(tile)
        restored_tiles[start:stop] = restored

    restored = (
        restored_tiles.view(n_tiles, k_tiles, TILE_N, TILE_K)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(padded_n, padded_k)[:n, :k]
        .to(weight.dtype)
    )
    squared_error = float(
        torch.sum((weight.float() - restored.float()).square())
    )
    squared_signal = float(torch.sum(weight.float().square()))
    return restored, {
        "mse": squared_error / weight.numel(),
        "relative_mse": squared_error / max(squared_signal, 1e-30),
        "num_weights": weight.numel(),
        "b_scale_mode": b_scale_mode,
        "global_scale": (
            float(weight_global_scale(weight))
            if b_scale_mode
            in {"global", "global_native_k64", "global_legacy_k32"}
            else 1.0
        ),
    }


def gptq_inverse_cholesky_blocks(
    hessian: torch.Tensor,
    damp_percent: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    hessian = hessian.float().clone()
    diagonal = torch.diagonal(hessian, dim1=-2, dim2=-1)
    live = diagonal > 1e-12
    mean_diagonal = (
        (diagonal * live).sum(dim=-1)
        / live.sum(dim=-1).clamp_min(1)
    ).clamp_min(1e-8)
    diagonal.add_(damp_percent * mean_diagonal.unsqueeze(-1))
    diagonal.masked_fill_(~live, 1.0)
    identity = torch.eye(
        TILE_K,
        device=hessian.device,
        dtype=hessian.dtype,
    ).unsqueeze(0)
    last_error = None
    for jitter in (0.0, 1e-6, 1e-4, 1e-2):
        try:
            chol = torch.linalg.cholesky(hessian + jitter * identity)
            inverse = torch.cholesky_inverse(chol)
            return torch.linalg.cholesky(inverse, upper=True), ~live
        except RuntimeError as error:
            last_error = error
    raise RuntimeError("GPTQ Hessian Cholesky failed after damping") from last_error


def gptq_quantize_with_centers(
    tiles: torch.Tensor,
    *,
    centers: torch.Tensor,
    scales: torch.Tensor,
    global_scale: torch.Tensor,
    local_scale_k: int | None,
    inverse_cholesky: torch.Tensor,
    dead: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one sequential GPTQ correction pass for a fixed LUT codebook."""
    n_tiles, k_tiles, _, _ = tiles.shape
    work = tiles.float().clone()
    work.masked_fill_(dead.view(1, k_tiles, 1, TILE_K), 0.0)
    quantized = torch.empty_like(work)
    assignments = torch.empty_like(work, dtype=torch.long)
    for column in range(TILE_K):
        values = work[..., column]
        scale_index = column // (local_scale_k or TILE_K)
        column_scale = scales[..., scale_index, 0] * global_scale
        normalized_values = (
            values / column_scale
        ).clamp(-E4M3_MAX, E4M3_MAX)
        assignment = (
            normalized_values.unsqueeze(-1) - centers.unsqueeze(-2)
        ).abs().argmin(dim=-1)
        codebook = centers.unsqueeze(-2).expand(
            n_tiles,
            k_tiles,
            TILE_N,
            LUT_SIZE,
        )
        reconstructed = (
            codebook.gather(-1, assignment.unsqueeze(-1)).squeeze(-1)
            * column_scale
        )
        assignments[..., column] = assignment
        quantized[..., column] = reconstructed
        diagonal = inverse_cholesky[:, column, column].view(
            1, k_tiles, 1
        )
        error = (values - reconstructed) / diagonal
        work[..., column:] -= error.unsqueeze(-1) * inverse_cholesky[
            :, column, column:
        ].view(1, k_tiles, 1, -1)
    return quantized, assignments


def hessian_proxy_objective(
    original: torch.Tensor,
    quantized: torch.Tensor,
    hessian: torch.Tensor,
) -> torch.Tensor:
    """Return tr((W-Q) H (W-Q)^T) over all K64xN8 tiles."""
    error = original.float() - quantized.float()
    return torch.einsum(
        "akni,kij,aknj->",
        error,
        hessian.float(),
        error,
    )


def hessian_lloyd_codebook_update(
    original: torch.Tensor,
    *,
    assignments: torch.Tensor,
    old_centers: torch.Tensor,
    hessian: torch.Tensor,
    scales: torch.Tensor,
    global_scale: torch.Tensor,
    local_scale_k: int | None,
    damping: float,
    chunk_tiles: int,
) -> torch.Tensor:
    """Solve the Hessian-weighted Lloyd M-step for eight scalar centroids.

    For fixed assignments z, each tile minimizes
    sum_n (w_n - S(z_n)c)^T H (w_n - S(z_n)c), where S includes
    the hardware scale applied to each K coordinate.
    """
    n_tiles, k_tiles, _, _ = original.shape
    tile_count = n_tiles * k_tiles
    group_k = local_scale_k or TILE_K
    scale_matrix = (
        scales.squeeze(-1)
        .repeat_interleave(group_k, dim=-1)
        .mul(global_scale)
        .reshape(tile_count, TILE_N, TILE_K)
    )
    flat_original = original.float().reshape(tile_count, TILE_N, TILE_K)
    flat_assignments = assignments.reshape(tile_count, TILE_N, TILE_K)
    flat_old = old_centers.float().reshape(tile_count, LUT_SIZE)
    k_indices = (
        torch.arange(tile_count, device=original.device) % k_tiles
    )
    updated = torch.empty_like(flat_old)
    identity = torch.eye(
        LUT_SIZE, device=original.device, dtype=torch.float32
    )
    for start in range(0, tile_count, chunk_tiles):
        stop = min(start + chunk_tiles, tile_count)
        assignment = flat_assignments[start:stop]
        one_hot = F.one_hot(assignment, LUT_SIZE).float()
        design = one_hot * scale_matrix[start:stop].unsqueeze(-1)
        hessian_chunk = hessian[k_indices[start:stop]].float()
        normal = torch.einsum(
            "cnka,ckl,cnlb->cab",
            design,
            hessian_chunk,
            design,
        )
        target = torch.einsum(
            "cnka,ckl,cnl->ca",
            design,
            hessian_chunk,
            flat_original[start:stop],
        )
        ridge = (
            damping
            * torch.diagonal(normal, dim1=-2, dim2=-1)
            .mean(dim=-1)
            .clamp_min(1e-8)
        )
        regularized = normal + ridge[:, None, None] * identity
        try:
            centers = torch.linalg.solve(
                regularized, target.unsqueeze(-1)
            ).squeeze(-1)
        except RuntimeError:
            centers = torch.matmul(
                torch.linalg.pinv(regularized), target.unsqueeze(-1)
            ).squeeze(-1)
        active = one_hot.sum(dim=(1, 2)) > 0
        centers = torch.where(active, centers, flat_old[start:stop])
        updated[start:stop] = e4m3_round(centers).sort(dim=-1).values
    return updated.view(n_tiles, k_tiles, LUT_SIZE)


def quantize_weight_lutb_gptq(
    weight: torch.Tensor,
    *,
    hessian: torch.Tensor,
    iterations: int,
    damp_percent: float,
    weighted_codebook: bool = False,
    sensitivity_alpha: float = 1.0,
    b_scale_mode: str = "global_native_k64",
    inter_iterations: int = 0,
    inter_tolerance: float = 1e-5,
    inter_codebook_damp: float = 1e-4,
    inter_chunk_tiles: int = 128,
) -> tuple[torch.Tensor, dict[str, object]]:
    if weight.ndim != 2:
        raise ValueError(f"Expected a matrix, got {tuple(weight.shape)}")
    if inter_iterations < 0:
        raise ValueError("inter_iterations must be non-negative")
    if inter_tolerance < 0:
        raise ValueError("inter_tolerance must be non-negative")
    if inter_codebook_damp < 0:
        raise ValueError("inter_codebook_damp must be non-negative")
    if inter_chunk_tiles <= 0:
        raise ValueError("inter_chunk_tiles must be positive")
    n, k = weight.shape
    padded_n = math.ceil(n / TILE_N) * TILE_N
    padded_k = math.ceil(k / TILE_K) * TILE_K
    n_tiles = padded_n // TILE_N
    k_tiles = padded_k // TILE_K
    if hessian.shape != (k_tiles, TILE_K, TILE_K):
        raise ValueError(
            f"GPTQ Hessian {tuple(hessian.shape)} does not match K={k}"
        )

    padded = F.pad(weight.float(), (0, padded_k - k, 0, padded_n - n))
    tiles = (
        padded.view(n_tiles, TILE_N, k_tiles, TILE_K)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    groups, scales, global_scale, local_scale_k = b_scale_geometry(
        weight,
        tiles,
        b_scale_mode,
    )
    normalized = (groups / scales).clamp(-E4M3_MAX, E4M3_MAX)
    flat = normalized.reshape(n_tiles * k_tiles, -1)
    sample_weights = None
    fit_method = "lloyd_max"
    if weighted_codebook:
        hessian_diagonal = torch.diagonal(
            hessian.float(),
            dim1=-2,
            dim2=-1,
        ).view(
            1,
            k_tiles,
            1,
            groups.shape[-2],
            groups.shape[-1],
        )
        importance = hessian_diagonal.expand(
            n_tiles,
            k_tiles,
            TILE_N,
            groups.shape[-2],
            groups.shape[-1],
        )
        importance = importance / importance.mean(
            dim=(-1, -2, -3), keepdim=True
        ).clamp_min(1e-12)
        if sensitivity_alpha != 1.0:
            # h^alpha interpolates between unweighted (alpha=0) and full
            # RMS^2 (alpha=1) codebook weighting; importance is mean-one so
            # the power keeps the weighting scale-free.
            importance = importance.pow(sensitivity_alpha)
        sample_weights = (
            importance.clamp(1e-4, 100.0) * scales.square()
        ).reshape_as(flat)
        fit_method = "sensitivity_lm"
    centers = fit_centers(
        flat,
        fit_method,
        iterations,
        sample_weights=sample_weights,
    ).view(n_tiles, k_tiles, LUT_SIZE)

    inverse_cholesky, dead = gptq_inverse_cholesky_blocks(
        hessian,
        damp_percent,
    )
    quantized, assignments = gptq_quantize_with_centers(
        tiles,
        centers=centers,
        scales=scales,
        global_scale=global_scale,
        local_scale_k=local_scale_k,
        inverse_cholesky=inverse_cholesky,
        dead=dead,
    )
    signal_objective = float(
        hessian_proxy_objective(
            tiles,
            torch.zeros_like(tiles),
            hessian,
        ).clamp_min(1e-30)
    )
    tile_signal_sq = float(torch.sum(tiles.float().square()).clamp_min(1e-30))

    def _tile_relative_mse(candidate: torch.Tensor) -> float:
        return float(
            torch.sum((tiles.float() - candidate.float()).square())
        ) / tile_signal_sq

    current_objective = float(
        hessian_proxy_objective(tiles, quantized, hessian)
    )
    objective_trace = [current_objective / signal_objective]
    mse_trace = [_tile_relative_mse(quantized)]
    codebook_delta_trace: list[float] = []
    accepted_iterations = 0
    for _ in range(inter_iterations):
        proposal = hessian_lloyd_codebook_update(
            tiles,
            assignments=assignments,
            old_centers=centers,
            hessian=hessian,
            scales=scales,
            global_scale=global_scale,
            local_scale_k=local_scale_k,
            damping=inter_codebook_damp,
            chunk_tiles=inter_chunk_tiles,
        )
        best_centers = centers
        best_quantized = quantized
        best_assignments = assignments
        best_objective = current_objective
        for alpha in (1.0, 0.5, 0.25):
            candidate_centers = e4m3_round(
                centers.float() + alpha * (proposal.float() - centers.float())
            ).sort(dim=-1).values
            if torch.equal(candidate_centers, centers):
                continue
            candidate_quantized, candidate_assignments = (
                gptq_quantize_with_centers(
                    tiles,
                    centers=candidate_centers,
                    scales=scales,
                    global_scale=global_scale,
                    local_scale_k=local_scale_k,
                    inverse_cholesky=inverse_cholesky,
                    dead=dead,
                )
            )
            candidate_objective = float(
                hessian_proxy_objective(
                    tiles, candidate_quantized, hessian
                )
            )
            if candidate_objective < best_objective:
                best_centers = candidate_centers
                best_quantized = candidate_quantized
                best_assignments = candidate_assignments
                best_objective = candidate_objective
        improvement = (
            current_objective - best_objective
        ) / max(abs(current_objective), 1e-30)
        if best_centers is centers or improvement <= inter_tolerance:
            break
        codebook_delta_trace.append(
            float(
                torch.mean(
                    (best_centers.float() - centers.float()).square()
                )
            )
        )
        centers = best_centers
        quantized = best_quantized
        assignments = best_assignments
        current_objective = best_objective
        objective_trace.append(current_objective / signal_objective)
        mse_trace.append(_tile_relative_mse(quantized))
        accepted_iterations += 1

    restored = (
        quantized.permute(0, 2, 1, 3)
        .contiguous()
        .view(padded_n, padded_k)[:n, :k]
        .to(weight.dtype)
    )
    squared_error = float(
        torch.sum((weight.float() - restored.float()).square())
    )
    squared_signal = float(torch.sum(weight.float().square()))
    return restored, {
        "mse": squared_error / weight.numel(),
        "relative_mse": squared_error / max(squared_signal, 1e-30),
        "num_weights": weight.numel(),
        "gptq_hessian_blocks": hessian.shape[0],
        "gptq_damp_percent": damp_percent,
        "gptq_weighted_codebook": weighted_codebook,
        "inter_lloymax_gptq": inter_iterations > 0,
        "inter_requested_iterations": inter_iterations,
        "inter_accepted_iterations": accepted_iterations,
        "inter_proxy_objective_trace": objective_trace,
        "inter_mse_trace": mse_trace,
        "inter_codebook_delta_trace": codebook_delta_trace,
        "inter_fixed_point": accepted_iterations < inter_iterations,
        "inter_tolerance": inter_tolerance,
        "inter_codebook_damp": inter_codebook_damp,
        "b_scale_mode": b_scale_mode,
        "global_scale": float(global_scale),
    }


def quantize_weight_aqlm(
    weight: torch.Tensor,
    *,
    hessian: torch.Tensor,
    group_size: int = 8,
    num_codebooks: int = 3,
    code_bits: int = 8,
    iterations: int = 6,
    chunk_rows: int = 256,
    seed: int = 20260818,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Compact, output-aware Additive Quantization (AQLM) reference.

    Each length-`group_size` weight group is encoded as a sum of
    `num_codebooks` learned codes, one per shared codebook of `2**code_bits`
    entries. Codes and codebooks are refined by block-coordinate descent on the
    diagonal-Hessian-weighted objective `sum_k h_k (w_k - w_hat_k)^2`, where
    `h_k` is the calibration activation second moment for input coordinate k.

    This is NOT a Rubin LUT-B contract: additive vector codebooks cannot be
    consumed by the scalar 8-entry E4M3 LUT-B kernel. It is a same-bitrate
    accuracy reference. Approximations relative to full AQLM: diagonal (not
    full) activation Hessian, beam width 1 (coordinate-descent assignment), and
    no end-to-end codebook SGD fine-tuning.
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected a matrix, got {tuple(weight.shape)}")
    n, k = weight.shape
    k_tiles = math.ceil(k / TILE_K)
    padded_k = k_tiles * TILE_K
    if padded_k % group_size:
        raise ValueError("group_size must divide the padded K dimension")
    if hessian.shape != (k_tiles, TILE_K, TILE_K):
        raise ValueError(
            f"AQLM Hessian {tuple(hessian.shape)} does not match K={k}"
        )
    device = weight.device
    num_codes = 1 << code_bits
    g = group_size
    groups_per_row = padded_k // g

    padded = F.pad(weight.float(), (0, padded_k - k))
    w_groups = padded.reshape(n, groups_per_row, g)
    diag = (
        torch.diagonal(hessian.float(), dim1=-2, dim2=-1)
        .reshape(padded_k)
        .clamp_min(0.0)
    )
    diag = diag / diag.mean().clamp_min(1e-12)
    h_groups = diag.reshape(groups_per_row, g)

    generator = torch.Generator(device=device).manual_seed(seed)
    codebooks = torch.zeros(
        num_codebooks, num_codes, g, device=device, dtype=torch.float32
    )
    assignments = torch.zeros(
        n, groups_per_row, num_codebooks, device=device, dtype=torch.long
    )

    flat_groups = w_groups.reshape(-1, g)
    total_groups = flat_groups.shape[0]

    def assign_codebook(target: torch.Tensor, codebook: torch.Tensor,
                        code_norm: torch.Tensor) -> torch.Tensor:
        out = torch.empty(
            target.shape[0], groups_per_row, device=device, dtype=torch.long
        )
        for start in range(0, target.shape[0], chunk_rows):
            stop = min(start + chunk_rows, target.shape[0])
            rh = target[start:stop] * h_groups
            cross = torch.einsum("njg,cg->njc", rh, codebook)
            distance = code_norm.unsqueeze(0) - 2.0 * cross
            out[start:stop] = distance.argmin(dim=-1)
        return out

    def gather_codes(codebook: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        return codebook[codes]

    # Residual-based initialization (sequential k-means-lite per codebook).
    residual = w_groups.clone()
    for m in range(num_codebooks):
        sample = torch.randint(
            0, total_groups, (num_codes,), generator=generator, device=device
        )
        codebooks[m] = residual.reshape(-1, g)[sample].clone()
        code_norm = torch.einsum("jg,cg->jc", h_groups, codebooks[m] ** 2)
        codes = assign_codebook(residual, codebooks[m], code_norm)
        assignments[:, :, m] = codes
        residual = residual - gather_codes(codebooks[m], codes)

    def reconstruct() -> torch.Tensor:
        out = torch.zeros_like(w_groups)
        for m in range(num_codebooks):
            out = out + gather_codes(codebooks[m], assignments[:, :, m])
        return out

    def weighted_objective() -> float:
        error = w_groups - reconstruct()
        return float(torch.sum(h_groups * error.square()))

    objective_trace = [weighted_objective()]
    for _ in range(iterations):
        for m in range(num_codebooks):
            others = torch.zeros_like(w_groups)
            for other in range(num_codebooks):
                if other == m:
                    continue
                others = others + gather_codes(
                    codebooks[other], assignments[:, :, other]
                )
            target = w_groups - others
            code_norm = torch.einsum(
                "jg,cg->jc", h_groups, codebooks[m] ** 2
            )
            codes = assign_codebook(target, codebooks[m], code_norm)
            assignments[:, :, m] = codes
            numerator = torch.zeros(num_codes, g, device=device)
            denominator = torch.zeros(num_codes, g, device=device)
            for start in range(0, n, chunk_rows):
                stop = min(start + chunk_rows, n)
                flat_codes = codes[start:stop].reshape(-1)
                weighted_target = (
                    target[start:stop] * h_groups
                ).reshape(-1, g)
                weights = h_groups.expand(stop - start, -1, -1).reshape(-1, g)
                numerator.index_add_(0, flat_codes, weighted_target)
                denominator.index_add_(0, flat_codes, weights)
            updated = numerator / denominator.clamp_min(1e-12)
            active = denominator.sum(dim=-1) > 0
            codebooks[m] = torch.where(
                active.unsqueeze(-1), updated, codebooks[m]
            )
        objective_trace.append(weighted_objective())

    restored_groups = reconstruct()
    restored = (
        restored_groups.reshape(n, padded_k)[:, :k].to(weight.dtype)
    )
    squared_error = float(
        torch.sum((weight.float() - restored.float()).square())
    )
    squared_signal = float(torch.sum(weight.float().square()))
    index_bits = num_codebooks * code_bits / g
    codebook_bits = (
        num_codebooks * num_codes * g * 16 / max(weight.numel(), 1)
    )
    return restored, {
        "mse": squared_error / weight.numel(),
        "relative_mse": squared_error / max(squared_signal, 1e-30),
        "num_weights": weight.numel(),
        "aqlm_reference": True,
        "aqlm_group_size": g,
        "aqlm_num_codebooks": num_codebooks,
        "aqlm_code_bits": code_bits,
        "aqlm_index_bits_per_weight": index_bits,
        "aqlm_codebook_bits_per_weight": codebook_bits,
        "aqlm_total_bits_per_weight": index_bits + codebook_bits,
        "aqlm_objective_trace": [
            value / max(objective_trace[0], 1e-30)
            for value in objective_trace
        ],
        "lut_b_native": False,
    }


def activation_mxfp8_qdq(
    values: torch.Tensor,
    *,
    block_size: int = A_SCALE_K,
) -> torch.Tensor:
    original_shape = values.shape
    k = original_shape[-1]
    if block_size not in (32, 64):
        raise ValueError(f"Unsupported MXFP8 A scale block: {block_size}")
    padded_k = math.ceil(k / block_size) * block_size
    padded = F.pad(values.float(), (0, padded_k - k))
    groups = padded.reshape(-1, padded_k // block_size, block_size)
    scales = ue8m0_scale(groups.abs().amax(dim=-1, keepdim=True))
    restored = e4m3_round(groups / scales) * scales
    return restored.reshape(*original_shape[:-1], padded_k)[..., :k].to(values.dtype)


def random_hadamard_k64(
    values: torch.Tensor,
    signs: torch.Tensor,
) -> torch.Tensor:
    if values.shape[-1] % TILE_K:
        raise ValueError("Random Hadamard requires a K dimension divisible by 64")
    original_shape = values.shape
    blocks = (
        values.float().reshape(-1, original_shape[-1] // TILE_K, TILE_K)
        * signs.float().view(1, -1, TILE_K)
    )
    transformed = blocks
    step = 1
    while step < TILE_K:
        paired = transformed.view(
            *transformed.shape[:-1],
            TILE_K // (2 * step),
            2,
            step,
        )
        left = paired[..., 0, :]
        right = paired[..., 1, :]
        transformed = torch.stack(
            (left + right, left - right),
            dim=-2,
        ).reshape_as(blocks)
        step *= 2
    transformed = transformed / math.sqrt(TILE_K)
    return transformed.reshape(original_shape).to(values.dtype)


def hadamard_input_hook(module, inputs):
    if not inputs:
        return inputs
    values = random_hadamard_k64(
        inputs[0],
        module._w3_hadamard_signs,
    )
    return (values, *inputs[1:])


def apply_random_hadamard(
    model,
    seed: int,
) -> tuple[list[dict[str, int | str]], list]:
    aliases = linear_hessian_aliases(model)
    signs_by_alias: dict[str, torch.Tensor] = {}
    summaries = []
    handles = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(
            part in name.lower()
            for part in ("vision", "visual", "projector", "audio")
        ):
            continue
        if module.in_features % TILE_K:
            raise ValueError(
                f"Hadamard input dimension for {name} is not K64-aligned"
            )
        alias = aliases.get(name, name)
        signs = signs_by_alias.get(alias)
        if signs is None:
            digest = hashlib.sha256(f"{seed}:{alias}".encode()).digest()
            group_seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
            generator = torch.Generator(device="cpu").manual_seed(group_seed)
            signs = (
                torch.randint(
                    0,
                    2,
                    (module.in_features // TILE_K, TILE_K),
                    generator=generator,
                    dtype=torch.int8,
                )
                .mul_(2)
                .sub_(1)
                .to(module.weight.device)
            )
            signs_by_alias[alias] = signs
        module.weight.data.copy_(
            random_hadamard_k64(module.weight.data, signs)
        )
        module._w3_hadamard_signs = signs
        handles.append(module.register_forward_pre_hook(hadamard_input_hook))
        summaries.append(
            {
                "module": name,
                "alias": alias,
                "blocks": signs.shape[0],
            }
        )
    if not summaries:
        raise ValueError("Random Hadamard found no supported linear layers")
    return summaries, handles


def activation_hook(module, inputs):
    if not inputs:
        return inputs
    values = inputs[0]
    input_scale = getattr(module, "_w3_input_scale", None)
    if input_scale is not None:
        values = values / input_scale.to(device=values.device, dtype=values.dtype)
    if getattr(module, "_w3_activation_mxfp8", False):
        values = activation_mxfp8_qdq(
            values,
            block_size=getattr(module, "_w3_mma_k", A_SCALE_K),
        )
    return (values, *inputs[1:])


def collect_activation_importance(
    model,
    input_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, inputs):
            values = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
            current = values.square().sum(dim=0)
            sums[name] = sums.get(name, torch.zeros_like(current)) + current
            counts[name] = counts.get(name, 0) + values.shape[0]

        return hook

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and not any(
            part in name.lower() for part in ("vision", "visual", "projector")
        ):
            handles.append(module.register_forward_pre_hook(make_hook(name)))
    with torch.inference_mode():
        model(input_ids=input_ids, use_cache=False, return_dict=True)
    for handle in handles:
        handle.remove()
    return {
        name: (value / counts[name]).clamp_min(1e-12)
        for name, value in sums.items()
    }


def linear_hessian_aliases(model) -> dict[str, str]:
    module_names = {id(module): name for name, module in model.named_modules()}
    aliases: dict[str, str] = {}
    for layer_name, layer in model.named_modules():
        if any(
            part in layer_name.lower()
            for part in ("vision", "visual", "projector", "audio")
        ):
            continue
        self_attn = getattr(layer, "self_attn", None)
        mlp = getattr(layer, "mlp", None)
        if self_attn is None or mlp is None:
            continue
        attention = [
            module
            for module in (
                getattr(self_attn, "q_proj", None),
                getattr(self_attn, "k_proj", None),
                getattr(self_attn, "v_proj", None),
            )
            if isinstance(module, torch.nn.Linear)
        ]
        if attention:
            representative = module_names[id(attention[0])]
            for module in attention:
                aliases[module_names[id(module)]] = representative
        mlp_inputs = [
            module
            for module in (
                getattr(mlp, "gate_proj", None),
                getattr(mlp, "up_proj", None),
            )
            if isinstance(module, torch.nn.Linear)
        ]
        if mlp_inputs:
            representative = module_names[id(mlp_inputs[0])]
            for module in mlp_inputs:
                aliases[module_names[id(module)]] = representative

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(
            part in name.lower()
            for part in ("vision", "visual", "projector", "audio")
        ):
            continue
        aliases.setdefault(name, name)
    return aliases


def collect_activation_hessian_blocks(
    model,
    input_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    aliases = linear_hessian_aliases(model)
    representatives = set(aliases.values())
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, inputs):
            values = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
            padded_k = math.ceil(values.shape[-1] / TILE_K) * TILE_K
            padded = F.pad(values, (0, padded_k - values.shape[-1]))
            blocks = padded.view(-1, padded_k // TILE_K, TILE_K)
            current = torch.einsum("tki,tkj->kij", blocks, blocks)
            sums[name] = sums.get(name, torch.zeros_like(current)) + current
            counts[name] = counts.get(name, 0) + values.shape[0]

        return hook

    for name, module in model.named_modules():
        if name in representatives:
            handles.append(module.register_forward_pre_hook(make_hook(name)))
    with torch.inference_mode():
        model(input_ids=input_ids, use_cache=False, return_dict=True)
    for handle in handles:
        handle.remove()

    normalized = {
        name: value / counts[name]
        for name, value in sums.items()
    }
    return {
        name: normalized[representative]
        for name, representative in aliases.items()
        if representative in normalized
    }


def collect_weight_fisher(
    model, calibration_batches: list[torch.Tensor]
) -> list[float]:
    """Accumulate diagonal empirical Fisher from per-sample gradients."""
    parameters = [
        module.weight
        for module in model.modules()
        if isinstance(module, torch.nn.Linear) and module.weight.requires_grad
    ]
    handles = []
    for parameter in parameters:
        parameter._w3_fisher = None

        def accumulate(gradient, target=parameter):
            squared = gradient.detach().float().square()
            if target._w3_fisher is None:
                target._w3_fisher = squared
            else:
                target._w3_fisher.add_(squared)
            return gradient

        handles.append(parameter.register_hook(accumulate))

    losses = []
    try:
        for input_ids in calibration_batches:
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                result = model(
                    input_ids=input_ids,
                    labels=input_ids,
                    use_cache=False,
                    return_dict=True,
                )
                loss = result.loss.float()
                loss.backward()
                losses.append(float(loss.detach()))
            del result, loss
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)

    for parameter in parameters:
        if parameter._w3_fisher is not None:
            parameter._w3_fisher.div_(len(calibration_batches))
    return losses


def detach_tied_output_embedding(model) -> None:
    output = model.get_output_embeddings()
    inputs = model.get_input_embeddings()
    if output is None or inputs is None:
        return
    if output.weight.data_ptr() == inputs.weight.data_ptr():
        output.weight = torch.nn.Parameter(
            output.weight.detach().clone(), requires_grad=False
        )


def force_config_attention_backend(
    config: PretrainedConfig, implementation: str, seen: set[int] | None = None
) -> None:
    if seen is None:
        seen = set()
    if id(config) in seen:
        return
    seen.add(id(config))
    config._attn_implementation = implementation
    config._attn_implementation_internal = implementation
    for value in vars(config).values():
        if isinstance(value, PretrainedConfig):
            force_config_attention_backend(value, implementation, seen)


def force_attention_backend(model, implementation: str) -> None:
    """Propagate the backend into configs held by nested model modules."""
    setter = getattr(model, "set_attn_implementation", None)
    if setter is not None:
        setter(implementation)
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is not None:
            config._attn_implementation = implementation
            config._attn_implementation_internal = implementation
            config.__dict__["_attn_implementation_internal"] = implementation


def awq_group_scale(
    weights: list[torch.Tensor],
    channel_importance: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    x_rms = channel_importance.float().sqrt().clamp_min(1e-8)
    w_max = torch.stack(
        [weight.float().abs().amax(dim=0) for weight in weights]
    ).amax(dim=0).clamp_min(1e-8)
    scale = x_rms.pow(alpha) / w_max.pow(1.0 - alpha)
    scale = scale / torch.exp(torch.mean(torch.log(scale)))
    return scale.clamp(1e-2, 1e2)


def awq_input_scale(
    weight: torch.Tensor,
    channel_importance: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    return awq_group_scale([weight], channel_importance, alpha)


def fold_awq_group(
    norm: torch.nn.Module,
    consumers: list[torch.nn.Linear],
    scale: torch.Tensor,
) -> None:
    norm_weight = getattr(norm, "weight", None)
    if norm_weight is None or norm_weight.numel() != scale.numel():
        raise ValueError("Folded AWQ requires a matching affine RMSNorm")
    scale = scale.to(device=norm_weight.device, dtype=torch.float32)
    norm.weight.data.copy_(
        (norm.weight.data.float() / scale).to(norm.weight.dtype)
    )
    for consumer in consumers:
        if consumer.in_features != scale.numel():
            raise ValueError("Folded AWQ consumer dimension does not match scale")
        consumer.weight.data.copy_(
            (
                consumer.weight.data.float()
                * scale.to(device=consumer.weight.device).unsqueeze(0)
            ).to(consumer.weight.dtype)
        )


def apply_folded_awq(
    model,
    activation_importance: dict[str, torch.Tensor],
    alpha: float,
) -> list[dict[str, float | int | str]]:
    module_names = {id(module): name for name, module in model.named_modules()}
    summaries = []

    def fold_group(
        *,
        group_name: str,
        norm,
        candidates: list[torch.nn.Module | None],
    ) -> None:
        consumers = []
        importance = []
        for candidate in candidates:
            if not isinstance(candidate, torch.nn.Linear):
                continue
            name = module_names.get(id(candidate))
            if name is None or name not in activation_importance:
                continue
            consumers.append(candidate)
            importance.append(activation_importance[name])
        if not consumers:
            return
        shared_importance = torch.stack(importance).mean(dim=0)
        scale = awq_group_scale(
            [consumer.weight.data for consumer in consumers],
            shared_importance,
            alpha,
        )
        fold_awq_group(norm, consumers, scale)
        summaries.append(
            {
                "group": group_name,
                "consumers": len(consumers),
                "scale_min": float(scale.min()),
                "scale_max": float(scale.max()),
            }
        )

    for layer_name, layer in model.named_modules():
        if any(
            part in layer_name.lower()
            for part in ("vision", "visual", "projector", "audio")
        ):
            continue
        self_attn = getattr(layer, "self_attn", None)
        mlp = getattr(layer, "mlp", None)
        input_norm = getattr(layer, "input_layernorm", None)
        if self_attn is None or mlp is None or input_norm is None:
            continue
        fold_group(
            group_name=f"{layer_name}.attention",
            norm=input_norm,
            candidates=[
                getattr(self_attn, "q_proj", None),
                getattr(self_attn, "k_proj", None),
                getattr(self_attn, "v_proj", None),
            ],
        )
        mlp_norm = getattr(layer, "pre_feedforward_layernorm", None)
        if mlp_norm is None:
            mlp_norm = getattr(layer, "post_attention_layernorm", None)
        if mlp_norm is not None:
            fold_group(
                group_name=f"{layer_name}.mlp",
                norm=mlp_norm,
                candidates=[
                    getattr(mlp, "gate_proj", None),
                    getattr(mlp, "up_proj", None),
                ],
            )
    if not summaries:
        raise ValueError("Folded AWQ found no supported RMSNorm fan-out groups")
    return summaries


def apply_fake_quant(
    model,
    *,
    method: str,
    activation: str,
    iterations: int,
    chunk_tiles: int,
    max_linear_layers: int,
    b_scale_mode: str = "global_native_k64",
    mma_k: int = 64,
    activation_importance: dict[str, torch.Tensor] | None = None,
    activation_hessians: dict[str, torch.Tensor] | None = None,
    awq_alpha: float = 0.5,
    gptq_damp_percent: float = 0.01,
    inter_gptq_iterations: int = 3,
    inter_gptq_tolerance: float = 1e-5,
    inter_gptq_codebook_damp: float = 1e-4,
    aqlm_iterations: int = 6,
    sensitivity_alpha: float = 1.0,
) -> dict:
    if mma_k not in (32, 64):
        raise ValueError(f"Unsupported LUT-B MMA K={mma_k}")
    if (
        b_scale_mode in {"legacy_k32", "global_legacy_k32"}
        and mma_k != 32
    ):
        raise ValueError(f"{b_scale_mode} requires --mma-k 32")
    if (
        b_scale_mode in {"native_k64", "global_native_k64"}
        and mma_k != 64
    ):
        raise ValueError(f"{b_scale_mode} requires --mma-k 64")
    if b_scale_mode in ALGORITHM_ORACLE_B_SCALE_MODES:
        print(
            "WARNING: two-K32 B scaling is an algorithm oracle, not a "
            "native PTX 9.4 LUT-B contract",
            flush=True,
        )
    detach_tied_output_embedding(model)
    summaries = []
    handles = []
    linear_count = 0
    started = time.time()
    folded_awq = []
    if method == "awq_folded_lm":
        if activation_importance is None:
            raise ValueError("Folded AWQ requires activation statistics")
        folded_awq = apply_folded_awq(
            model,
            activation_importance,
            awq_alpha,
        )
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(
            part in name.lower()
            for part in ("vision", "visual", "projector", "audio")
        ):
            continue
        if max_linear_layers and linear_count >= max_linear_layers:
            break
        if method != "bf16":
            channel_importance = (
                activation_importance.get(name)
                if activation_importance is not None
                else None
            )
            element_importance = None
            if method == "squeezellm_lm":
                element_importance = getattr(
                    module.weight, "_w3_fisher", None
                )
            weight_to_quantize = module.weight.data
            if method in AQLM_METHODS:
                if activation_hessians is None or name not in activation_hessians:
                    raise ValueError(f"AQLM lacks Hessian statistics for {name}")
                quantized, stats = quantize_weight_aqlm(
                    weight_to_quantize,
                    hessian=activation_hessians[name],
                    iterations=aqlm_iterations,
                )
            elif method in GPTQ_METHODS:
                if activation_hessians is None or name not in activation_hessians:
                    raise ValueError(f"GPTQ lacks Hessian statistics for {name}")
                quantized, stats = quantize_weight_lutb_gptq(
                    weight_to_quantize,
                    hessian=activation_hessians[name],
                    iterations=iterations,
                    damp_percent=gptq_damp_percent,
                    weighted_codebook="sensitivity" in method,
                    sensitivity_alpha=sensitivity_alpha,
                    b_scale_mode=b_scale_mode,
                    inter_iterations=(
                        inter_gptq_iterations
                        if method in INTER_LLOYMAX_GPTQ_METHODS
                        else 0
                    ),
                    inter_tolerance=inter_gptq_tolerance,
                    inter_codebook_damp=inter_gptq_codebook_damp,
                    inter_chunk_tiles=chunk_tiles,
                )
            else:
                quant_method = method
                if method == "awq_lm":
                    if channel_importance is not None:
                        input_scale = awq_input_scale(
                            module.weight.data, channel_importance, awq_alpha
                        )
                        module._w3_input_scale = input_scale.to(
                            device=module.weight.device,
                            dtype=module.weight.dtype,
                        )
                        weight_to_quantize = (
                            module.weight.data.float()
                            * input_scale.unsqueeze(0)
                        ).to(module.weight.dtype)
                    quant_method = "lloyd_max"
                elif method == "awq_folded_lm":
                    quant_method = "lloyd_max"
                quantized, stats = quantize_weight_lutb(
                    weight_to_quantize,
                    method=quant_method,
                    iterations=iterations,
                    chunk_tiles=chunk_tiles,
                    b_scale_mode=b_scale_mode,
                    channel_importance=(
                        channel_importance
                        if method == "sensitivity_lm"
                        else None
                    ),
                    element_importance=element_importance,
                )
            module.weight.data.copy_(quantized)
            module.weight.grad = None
            if hasattr(module.weight, "_w3_fisher"):
                del module.weight._w3_fisher
            summaries.append({"name": name, **stats})
        module._w3_activation_mxfp8 = activation == "mxfp8"
        module._w3_mma_k = mma_k
        if activation == "mxfp8" or method == "awq_lm":
            handles.append(module.register_forward_pre_hook(activation_hook))
        linear_count += 1
        if linear_count % 10 == 0:
            print(f"quantized_linear_layers={linear_count}", flush=True)

    total_weights = sum(item["num_weights"] for item in summaries)
    weighted_relative_mse = (
        sum(item["relative_mse"] * item["num_weights"] for item in summaries)
        / max(total_weights, 1)
    )
    inter_summaries = [
        item for item in summaries if item.get("inter_lloymax_gptq")
    ]
    max_trace_length = max(
        (
            len(item["inter_proxy_objective_trace"])
            for item in inter_summaries
        ),
        default=0,
    )
    inter_proxy_trace = [
        sum(
            item["inter_proxy_objective_trace"][
                min(index, len(item["inter_proxy_objective_trace"]) - 1)
            ]
            * item["num_weights"]
            for item in inter_summaries
        )
        / max(
            sum(item["num_weights"] for item in inter_summaries),
            1,
        )
        for index in range(max_trace_length)
    ]
    max_mse_length = max(
        (len(item.get("inter_mse_trace", [])) for item in inter_summaries),
        default=0,
    )
    inter_mse_trace = [
        sum(
            item["inter_mse_trace"][
                min(index, len(item["inter_mse_trace"]) - 1)
            ]
            * item["num_weights"]
            for item in inter_summaries
        )
        / max(
            sum(item["num_weights"] for item in inter_summaries),
            1,
        )
        for index in range(max_mse_length)
    ]
    inter_layer_records = [
        {
            "name": item["name"],
            "num_weights": item["num_weights"],
            "inter_accepted_iterations": item["inter_accepted_iterations"],
            "inter_mse_trace": item.get("inter_mse_trace", []),
            "inter_proxy_objective_trace": item[
                "inter_proxy_objective_trace"
            ],
        }
        for item in inter_summaries
    ]
    return {
        "linear_layers": linear_count,
        "quantized_weights": total_weights,
        "weighted_relative_mse": weighted_relative_mse,
        "inter_lloymax_gptq_layers": len(inter_summaries),
        "inter_proxy_objective_trace": inter_proxy_trace,
        "inter_mse_trace": inter_mse_trace,
        "inter_layer_records": inter_layer_records,
        "inter_mean_accepted_iterations": (
            sum(
                int(item["inter_accepted_iterations"])
                for item in inter_summaries
            )
            / len(inter_summaries)
            if inter_summaries
            else None
        ),
        "folded_awq_groups": len(folded_awq),
        "folded_awq_consumers": sum(
            int(group["consumers"]) for group in folded_awq
        ),
        "folded_awq_scale_min": (
            min(float(group["scale_min"]) for group in folded_awq)
            if folded_awq
            else None
        ),
        "folded_awq_scale_max": (
            max(float(group["scale_max"]) for group in folded_awq)
            if folded_awq
            else None
        ),
        "quantization_seconds": time.time() - started,
        "hook_count": len(handles),
        "_handles": handles,
    }


def load_blocks(
    *,
    tokenizer,
    cache_path: Path,
    block_size: int,
    split: str = "test",
) -> list[list[int]]:
    if cache_path.is_file():
        state = torch.load(cache_path, map_location="cpu", weights_only=False)
        if state["block_size"] != block_size:
            raise ValueError(
                f"Cached block size {state['block_size']} != {block_size}"
            )
        expected_dataset = f"wikitext-2-raw-v1/{split}"
        if state.get("dataset") != expected_dataset:
            raise ValueError(
                f"Cached dataset {state.get('dataset')} != {expected_dataset}"
            )
        return state["blocks"]
    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split=split
    )
    text = "\n\n".join(dataset["text"])
    tokens = tokenizer(text, add_special_tokens=False).input_ids
    usable = len(tokens) // block_size * block_size
    blocks = [
        tokens[start : start + block_size]
        for start in range(0, usable, block_size)
    ]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "dataset": f"wikitext-2-raw-v1/{split}",
        "block_size": block_size,
        "tokenizer": tokenizer.name_or_path,
        "blocks": blocks,
    }
    temporary_path = cache_path.with_name(
        f".{cache_path.name}.{os.getpid()}.tmp"
    )
    torch.save(state, temporary_path)
    os.replace(temporary_path, cache_path)
    return blocks


def finite_perplexity(mean_nll: float) -> float | None:
    if not math.isfinite(mean_nll) or mean_nll > 80.0:
        return None
    return math.exp(mean_nll)


def fixed_positions(block_size: int, count: int) -> list[int]:
    if block_size < 3:
        raise ValueError("KL blocks require at least three tokens")
    start = min(384, block_size - 2)
    return (
        torch.linspace(start, block_size - 2, steps=count)
        .round()
        .long()
        .unique()
        .tolist()
    )


def prod_truncated_kl(
    teacher_logp: torch.Tensor, student_logp: torch.Tensor
) -> tuple[float, float]:
    """KL(ref~ || cand~) renormalized over the reference top-k support.

    Vendored from kl_probe.metrics.truncated_kl (togethercomputer/kl-probe,
    cloned under CoQuant/kl-probe): both distributions are renormalized over
    the reference top-k ids before the KL sum, and the uncovered reference
    tail mass is returned separately instead of entering the KL. Unlike the
    serving-stack probe, the student logprobs here are exact (gathered from
    full logits), so kl-probe's candidate-floor clamp for ids missing from
    the candidate top-k never triggers.
    """
    p = torch.exp(teacher_logp)
    q = torch.exp(student_logp)
    p_sum = p.sum()
    p_norm = p / p_sum
    q_norm = q / q.sum()
    kl = torch.sum(p_norm * (torch.log(p_norm) - torch.log(q_norm)))
    tail = (1.0 - p_sum).clamp_min(0.0)
    return max(0.0, float(kl)), float(tail)


def cross_entropy_sum(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    rows_per_chunk: int = 256,
) -> float:
    total = 0.0
    for start in range(0, labels.numel(), rows_per_chunk):
        stop = min(start + rows_per_chunk, labels.numel())
        total += float(
            F.cross_entropy(
                logits[start:stop].float(),
                labels[start:stop],
                reduction="sum",
            )
        )
    return total


def evaluate_block(
    model,
    input_ids: torch.Tensor,
    positions: list[int],
    eval_chunk_size: int,
) -> tuple[float, int, torch.Tensor]:
    if eval_chunk_size <= 0 or eval_chunk_size >= input_ids.shape[1]:
        result = model(
            input_ids=input_ids,
            labels=input_ids,
            use_cache=False,
            return_dict=True,
        )
        tokens = input_ids.numel() - 1
        nll = float(result.loss.float()) * tokens
        selected_log_probs = torch.log_softmax(
            result.logits[0, positions].float(), dim=-1
        )
        del result
        return nll, tokens, selected_log_probs

    nll = 0.0
    tokens = 0
    past_key_values = None
    previous_last_logits = None
    selected: dict[int, torch.Tensor] = {}
    sequence_length = input_ids.shape[1]
    for start in range(0, sequence_length, eval_chunk_size):
        stop = min(start + eval_chunk_size, sequence_length)
        chunk_ids = input_ids[:, start:stop]
        result = model(
            input_ids=chunk_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        logits = result.logits[0]
        if previous_last_logits is not None:
            nll += cross_entropy_sum(
                previous_last_logits.unsqueeze(0), chunk_ids[0, :1]
            )
            tokens += 1
        if chunk_ids.shape[1] > 1:
            nll += cross_entropy_sum(logits[:-1], chunk_ids[0, 1:])
            tokens += chunk_ids.shape[1] - 1
        chunk_positions = [
            position for position in positions if start <= position < stop
        ]
        if chunk_positions:
            relative = torch.tensor(
                [position - start for position in chunk_positions],
                device=logits.device,
                dtype=torch.long,
            )
            log_probs = torch.log_softmax(logits[relative].float(), dim=-1)
            selected.update(zip(chunk_positions, log_probs.unbind(0)))
        previous_last_logits = logits[-1].detach().clone()
        past_key_values = result.past_key_values
        del chunk_ids, logits, result
        torch.cuda.empty_cache()

    if tokens != sequence_length - 1:
        raise RuntimeError(f"Scored {tokens} tokens, expected {sequence_length - 1}")
    if set(selected) != set(positions):
        raise RuntimeError("Chunked evaluation did not collect all KL positions")
    selected_log_probs = torch.stack(
        [selected[position] for position in positions], dim=0
    )
    del past_key_values, previous_last_logits, selected
    torch.cuda.empty_cache()
    return nll, tokens, selected_log_probs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--weight-method",
        choices=(
            "bf16",
            "uniform",
            "nf3",
            "lloyd_max",
            "sensitivity_lm",
            "awq_lm",
            "awq_folded_lm",
            "gptq_lm",
            "gptq_sensitivity_lm",
            "gptq_hadamard_lm",
            "gptq_hadamard_sensitivity_lm",
            "inter_lloymax_gptq",
            "inter_lloymax_gptq_sensitivity",
            "inter_lloymax_gptq_hadamard",
            "inter_lloymax_gptq_hadamard_sensitivity",
            "scale_lm",
            "squeezellm_lm",
            "aqlm_reference",
        ),
        required=True,
    )
    parser.add_argument(
        "--activation", choices=("bf16", "mxfp8"), required=True
    )
    parser.add_argument(
        "--b-scale-mode",
        choices=tuple(sorted(B_SCALE_MODES)),
        default="global_native_k64",
        help=(
            "LUT-B scaling contract: none=3.125-bit reference payload; "
            "global=one folded FP32 tensor scale; native_k64=one UE8M0 "
            "factor per B row in each K64xN8 tile. legacy_k32 and "
            "global_legacy_k32 are quality-oracle simulations only: PTX "
            "9.4 LUT-B fixes a K64 compressed block and mxf8f6f4 exposes "
            "only scale_vec::1X/block32."
        ),
    )
    parser.add_argument(
        "--mma-k",
        type=int,
        choices=(32, 64),
        default=64,
        help=(
            "Logical MMA K. K64 is the native PTX 9.4 LUT-B compressed "
            "contract; K32 is retained only for algorithm-oracle runs."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks-path", type=Path, required=True)
    parser.add_argument("--calibration-blocks-path", type=Path)
    parser.add_argument("--block-size", type=int, default=8192)
    parser.add_argument("--max-blocks", type=int, default=16)
    parser.add_argument("--lm-iterations", type=int, default=4)
    parser.add_argument("--chunk-tiles", type=int, default=256)
    parser.add_argument("--max-linear-layers", type=int, default=0)
    parser.add_argument("--calibration-tokens", type=int, default=512)
    parser.add_argument("--fisher-samples", type=int, default=1)
    parser.add_argument("--awq-alpha", type=float, default=0.5)
    parser.add_argument("--gptq-damp-percent", type=float, default=0.01)
    parser.add_argument("--inter-gptq-iterations", type=int, default=3)
    parser.add_argument("--aqlm-iterations", type=int, default=6)
    parser.add_argument("--sensitivity-alpha", type=float, default=1.0)
    parser.add_argument("--inter-gptq-tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--inter-gptq-codebook-damp", type=float, default=1e-4
    )
    parser.add_argument("--hadamard-seed", type=int, default=20260813)
    parser.add_argument("--kl-reference", type=Path)
    parser.add_argument("--write-kl-reference", action="store_true")
    parser.add_argument("--kl-top-k", type=int, default=50)
    parser.add_argument("--kl-positions-per-block", type=int, default=16)
    parser.add_argument("--eval-chunk-size", type=int, default=0)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(20260812)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )
    all_blocks = load_blocks(
        tokenizer=tokenizer,
        cache_path=args.blocks_path,
        block_size=args.block_size,
    )
    blocks = all_blocks[: args.max_blocks or len(all_blocks)]
    config = AutoConfig.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )
    force_config_attention_backend(config, args.attn_implementation)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    ).to("cuda").eval()
    force_attention_backend(model, args.attn_implementation)
    detach_tied_output_embedding(model)
    hadamard_summary = []
    hadamard_handles = []
    if args.weight_method in HADAMARD_METHODS:
        hadamard_summary, hadamard_handles = apply_random_hadamard(
            model,
            args.hadamard_seed,
        )
    activation_importance = None
    activation_hessians = None
    calibration_seconds = 0.0
    calibration_losses = None
    if args.weight_method in {
        "sensitivity_lm",
        "awq_lm",
        "awq_folded_lm",
        "squeezellm_lm",
    } | GPTQ_METHODS | AQLM_METHODS:
        started = time.time()
        calibration_blocks = all_blocks
        if args.calibration_blocks_path is not None:
            calibration_blocks = load_blocks(
                tokenizer=tokenizer,
                cache_path=args.calibration_blocks_path,
                block_size=args.block_size,
                split="train",
            )
        if args.weight_method == "squeezellm_lm":
            if args.calibration_tokens > args.block_size:
                raise ValueError("Calibration sample is longer than a block")
            rng = random.Random(20260812)
            calibration_batches = []
            for _ in range(args.fisher_samples):
                block = calibration_blocks[rng.randrange(len(calibration_blocks))]
                offset = rng.randrange(
                    len(block) - args.calibration_tokens + 1
                )
                calibration_batches.append(
                    torch.tensor(
                        block[offset : offset + args.calibration_tokens],
                        device="cuda",
                        dtype=torch.long,
                    ).unsqueeze(0)
                )
            calibration_losses = collect_weight_fisher(
                model, calibration_batches
            )
            del calibration_batches
        elif args.weight_method in GPTQ_METHODS | AQLM_METHODS:
            calibration_ids = torch.tensor(
                calibration_blocks[0][: args.calibration_tokens],
                device="cuda",
                dtype=torch.long,
            ).unsqueeze(0)
            activation_hessians = collect_activation_hessian_blocks(
                model,
                calibration_ids,
            )
            del calibration_ids
        else:
            calibration_ids = torch.tensor(
                calibration_blocks[0][: args.calibration_tokens],
                device="cuda",
                dtype=torch.long,
            ).unsqueeze(0)
            activation_importance = collect_activation_importance(
                model, calibration_ids
            )
            del calibration_ids
        calibration_seconds = time.time() - started
        torch.cuda.empty_cache()
        print(
            f"calibrated_linear_layers="
            f"{len(activation_importance) if activation_importance else 0} "
            f"hessian_layers="
            f"{len(activation_hessians) if activation_hessians else 0} "
            f"fisher={args.weight_method == 'squeezellm_lm'} "
            f"losses={calibration_losses} seconds={calibration_seconds:.3f}",
            flush=True,
        )
    quant = apply_fake_quant(
        model,
        method=args.weight_method,
        activation=args.activation,
        iterations=args.lm_iterations,
        chunk_tiles=args.chunk_tiles,
        max_linear_layers=args.max_linear_layers,
        b_scale_mode=args.b_scale_mode,
        mma_k=args.mma_k,
        activation_importance=activation_importance,
        activation_hessians=activation_hessians,
        awq_alpha=args.awq_alpha,
        gptq_damp_percent=args.gptq_damp_percent,
        inter_gptq_iterations=args.inter_gptq_iterations,
        inter_gptq_tolerance=args.inter_gptq_tolerance,
        inter_gptq_codebook_damp=args.inter_gptq_codebook_damp,
        aqlm_iterations=args.aqlm_iterations,
        sensitivity_alpha=args.sensitivity_alpha,
    )
    handles = hadamard_handles + quant.pop("_handles")
    quant["hadamard_modules"] = len(hadamard_summary)
    quant["hadamard_aliases"] = len(
        {summary["alias"] for summary in hadamard_summary}
    )
    model.zero_grad(set_to_none=True)
    print(json.dumps({"quantization": quant}, indent=2), flush=True)

    total_nll = 0.0
    scored_tokens = 0
    block_results = []
    teacher_state = None
    if args.kl_reference is not None and args.kl_reference.is_file():
        teacher_state = json.loads(args.kl_reference.read_text())
        if teacher_state["block_size"] != args.block_size:
            raise ValueError("KL teacher and student block sizes differ")
    elif args.kl_reference is not None and not args.write_kl_reference:
        raise FileNotFoundError(args.kl_reference)
    teacher_samples = []
    kl_values = []
    prod_kl_values = []
    prod_kl_tails = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for index, block in enumerate(blocks):
            input_ids = torch.tensor(
                block, device="cuda", dtype=torch.long
            ).unsqueeze(0)
            positions = fixed_positions(
                args.block_size, args.kl_positions_per_block
            )
            nll, tokens, selected_log_probs = evaluate_block(
                model,
                input_ids,
                positions,
                args.eval_chunk_size,
            )
            total_nll += nll
            scored_tokens += tokens
            block_results.append(
                {
                    "block": index,
                    "tokens": tokens,
                    "mean_nll": nll / tokens,
                }
            )
            if args.write_kl_reference:
                top_values, top_indices = torch.topk(
                    selected_log_probs, args.kl_top_k, dim=-1
                )
                tail_probability = (
                    1.0 - torch.exp(top_values).sum(dim=-1)
                ).clamp_min(1e-30)
                for sample_index, position in enumerate(positions):
                    teacher_samples.append(
                        {
                            "block": index,
                            "position": position,
                            "token_ids": top_indices[sample_index].cpu().tolist(),
                            "log_probs": top_values[sample_index].cpu().tolist(),
                            "tail_log_prob": float(
                                torch.log(tail_probability[sample_index])
                            ),
                        }
                    )
            if teacher_state is not None:
                samples = [
                    sample
                    for sample in teacher_state["samples"]
                    if sample["block"] == index
                ]
                if len(samples) != len(positions):
                    raise ValueError(
                        f"Teacher block {index} has {len(samples)} samples, "
                        f"expected {len(positions)}"
                    )
                for sample_index, sample in enumerate(samples):
                    if sample["position"] != positions[sample_index]:
                        raise ValueError("KL teacher positions differ")
                    ids = torch.tensor(
                        sample["token_ids"],
                        device=selected_log_probs.device,
                        dtype=torch.long,
                    )
                    teacher_logp = torch.tensor(
                        sample["log_probs"],
                        device=selected_log_probs.device,
                        dtype=torch.float32,
                    )
                    student_logp = selected_log_probs[sample_index].gather(0, ids)
                    student_tail = (
                        1.0 - torch.exp(student_logp).sum()
                    ).clamp_min(1e-30)
                    teacher_tail_logp = torch.tensor(
                        sample["tail_log_prob"],
                        device=selected_log_probs.device,
                        dtype=torch.float32,
                    )
                    value = torch.sum(
                        torch.exp(teacher_logp)
                        * (teacher_logp - student_logp)
                    )
                    value += torch.exp(teacher_tail_logp) * (
                        teacher_tail_logp - torch.log(student_tail)
                    )
                    kl_values.append(float(value))
                    prod_kl, prod_tail = prod_truncated_kl(
                        teacher_logp, student_logp
                    )
                    prod_kl_values.append(prod_kl)
                    prod_kl_tails.append(prod_tail)
            running_mean_nll = total_nll / scored_tokens
            running_ppl = finite_perplexity(running_mean_nll)
            print(
                f"block={index + 1}/{len(blocks)} "
                f"ppl={running_ppl if running_ppl is not None else 'nonfinite'}",
                flush=True,
            )
            del input_ids, selected_log_probs
            torch.cuda.empty_cache()

    for handle in handles:
        handle.remove()
    mean_nll = total_nll / scored_tokens
    if args.b_scale_mode in {"legacy_k32", "global_legacy_k32"}:
        b_scale_bytes = TILE_N * (TILE_K // LEGACY_B_SCALE_K)
        b_scale_block_k = LEGACY_B_SCALE_K
    elif args.b_scale_mode in {"native_k64", "global_native_k64"}:
        b_scale_bytes = TILE_N * (TILE_K // NATIVE_B_SCALE_K)
        b_scale_block_k = NATIVE_B_SCALE_K
    else:
        b_scale_bytes = 0
        b_scale_block_k = None
    logical_bytes_per_tile = 192 + 8 + b_scale_bytes
    output = {
        "model": args.model_path,
        "weight_method": args.weight_method,
        "activation": args.activation,
        "format": {
            "tile_k": TILE_K,
            "tile_n": TILE_N,
            "mma_k": args.mma_k,
            "indices_bits": 3,
            "lut_entries": LUT_SIZE,
            "lut_dtype": "e4m3",
            "b_scale_mode": args.b_scale_mode,
            "rubin_lutb_isa_compatible": (
                args.b_scale_mode in RUBIN_NATIVE_B_SCALE_MODES
            ),
            "contract_class": (
                "ptx94_native_k64"
                if args.b_scale_mode in RUBIN_NATIVE_B_SCALE_MODES
                else "algorithm_oracle_non_native_two_k32"
            ),
            "contract_warning": (
                None
                if args.b_scale_mode in RUBIN_NATIVE_B_SCALE_MODES
                else (
                    "PTX 9.4 LUT-B uses a K64 compressed block and "
                    "mxf8f6f4 exposes only scale_vec::1X/block32; this "
                    "two-K32 result has no native LUT-B kernel mapping."
                )
            ),
            "b_scale_dtype": "ue8m0" if b_scale_bytes else None,
            "b_scale_block_k": b_scale_block_k,
            "global_scale_dtype": (
                "fp32_folded"
                if args.b_scale_mode
                in {"global", "global_native_k64", "global_legacy_k32"}
                else None
            ),
            "global_scale_rule": (
                "per_weight_tensor_max_abs_div_448"
                if args.b_scale_mode
                in {"global", "global_native_k64", "global_legacy_k32"}
                else None
            ),
            "logical_bytes_per_tile": logical_bytes_per_tile,
            "effective_bits_per_weight": (
                logical_bytes_per_tile * 8 / (TILE_K * TILE_N)
            ),
            "a_dtype": "e4m3" if args.activation == "mxfp8" else "bf16",
            "a_scale_dtype": "ue8m0" if args.activation == "mxfp8" else None,
            "a_scale_block_k": (
                args.mma_k if args.activation == "mxfp8" else None
            ),
        },
        "quantization": quant,
        "calibration_tokens": (
            args.calibration_tokens
            if args.weight_method
            in ({
                "sensitivity_lm",
                "awq_lm",
                "awq_folded_lm",
                "squeezellm_lm",
            } | GPTQ_METHODS)
            else 0
        ),
        "calibration_seconds": calibration_seconds,
        "awq_alpha": (
            args.awq_alpha
            if args.weight_method in {"awq_lm", "awq_folded_lm"}
            else None
        ),
        "awq_folded": args.weight_method == "awq_folded_lm",
        "awq_runtime_rescale": args.weight_method == "awq_lm",
        "gptq_block_diagonal_k": (
            TILE_K
            if args.weight_method in GPTQ_METHODS
            else None
        ),
        "gptq_damp_percent": (
            args.gptq_damp_percent
            if args.weight_method in GPTQ_METHODS
            else None
        ),
        "gptq_weighted_codebook": (
            "sensitivity" in args.weight_method
            if args.weight_method in GPTQ_METHODS
            else None
        ),
        "inter_lloymax_gptq": (
            args.weight_method in INTER_LLOYMAX_GPTQ_METHODS
        ),
        "inter_gptq_iterations": (
            args.inter_gptq_iterations
            if args.weight_method in INTER_LLOYMAX_GPTQ_METHODS
            else 0
        ),
        "aqlm_iterations": (
            args.aqlm_iterations
            if args.weight_method in AQLM_METHODS
            else None
        ),
        "sensitivity_alpha": (
            args.sensitivity_alpha
            if "sensitivity" in args.weight_method
            else None
        ),
        "inter_gptq_tolerance": (
            args.inter_gptq_tolerance
            if args.weight_method in INTER_LLOYMAX_GPTQ_METHODS
            else None
        ),
        "inter_gptq_codebook_damp": (
            args.inter_gptq_codebook_damp
            if args.weight_method in INTER_LLOYMAX_GPTQ_METHODS
            else None
        ),
        "hadamard_seed": (
            args.hadamard_seed
            if args.weight_method in HADAMARD_METHODS
            else None
        ),
        "hadamard_block_k": (
            TILE_K if args.weight_method in HADAMARD_METHODS else None
        ),
        "calibration_dataset": (
            "wikitext-2-raw-v1/train"
            if args.calibration_blocks_path is not None
            else "wikitext-2-raw-v1/test"
        ),
        "fisher_samples": (
            args.fisher_samples
            if args.weight_method == "squeezellm_lm"
            else 0
        ),
        "calibration_total_tokens": (
            args.calibration_tokens * args.fisher_samples
            if args.weight_method == "squeezellm_lm"
            else args.calibration_tokens
            if args.weight_method
            in ({
                "sensitivity_lm",
                "awq_lm",
                "awq_folded_lm",
            } | GPTQ_METHODS)
            else 0
        ),
        "sparse_residual_fraction": 0.0,
        "calibration_losses": calibration_losses,
        "dataset": "wikitext-2-raw-v1/test",
        "block_size": args.block_size,
        "eval_chunk_size": args.eval_chunk_size,
        "num_blocks": len(blocks),
        "scored_tokens": scored_tokens,
        "mean_nll": mean_nll if math.isfinite(mean_nll) else None,
        "ppl": finite_perplexity(mean_nll),
        "blocks": block_results,
        "kl50_mean_nats": (
            sum(kl_values) / len(kl_values) if kl_values else None
        ),
        "kl50_samples": len(kl_values),
        "prod_kl_mean_nats": (
            sum(prod_kl_values) / len(prod_kl_values)
            if prod_kl_values
            else None
        ),
        "prod_kl_ref_tail_mass_mean": (
            sum(prod_kl_tails) / len(prod_kl_tails)
            if prod_kl_tails
            else None
        ),
        "prod_kl_samples": len(prod_kl_values),
    }
    if args.write_kl_reference:
        if args.kl_reference is None:
            raise ValueError("--write-kl-reference requires --kl-reference")
        args.kl_reference.parent.mkdir(parents=True, exist_ok=True)
        args.kl_reference.write_text(
            json.dumps(
                {
                    "kind": "w3a8_topk_teacher_reference",
                    "model": args.model_path,
                    "dataset": "wikitext-2-raw-v1/test",
                    "block_size": args.block_size,
                    "top_k": args.kl_top_k,
                    "positions_per_block": args.kl_positions_per_block,
                    "samples": teacher_samples,
                },
                indent=2,
            )
            + "\n"
        )
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "blocks"}, indent=2))


if __name__ == "__main__":
    main()
