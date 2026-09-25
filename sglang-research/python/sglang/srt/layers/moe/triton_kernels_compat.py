"""Small compatibility boundary for the two Triton-kernels routing APIs."""

from __future__ import annotations

import torch

try:
    from triton_kernels.matmul_ogs import FnSpecs, FusedActivation
except ImportError:
    GatherIndx = RoutingData = ScatterIndx = None
    FnSpecs = FusedActivation = None
    make_ragged_tensor_metadata = None
    routing = None
    IS_AVAILABLE = False
    IS_LEGACY = False
else:
    IS_AVAILABLE = True
    try:
        from triton_kernels.routing import (
            GatherIndx,
            RoutingData,
            ScatterIndx,
            routing,
        )

        IS_LEGACY = False
    except ImportError:
        from triton_kernels.matmul_ogs import GatherIndx, RoutingData, ScatterIndx
        from triton_kernels.tensor import make_ragged_tensor_metadata

        routing = None
        IS_LEGACY = True


def build_routing_from_standard(
    standard_topk,
    *,
    num_experts: int,
    top_k: int,
):
    """Build expert-major routing metadata from SGLang's audited top-k output."""
    if not IS_AVAILABLE:
        raise ImportError("The triton_kernels package is not installed")
    flat_ids = standard_topk.topk_ids.flatten().long()
    combine_idx_long = torch.argsort(flat_ids, stable=True)
    combine_idx = combine_idx_long.to(torch.int32)
    dispatch_idx = torch.empty_like(combine_idx)
    dispatch_idx[combine_idx_long] = torch.arange(
        combine_idx.numel(), device=combine_idx.device, dtype=torch.int32
    )
    expert_hist = torch.bincount(flat_ids, minlength=num_experts).to(torch.int32)
    expert_data = make_ragged_tensor_metadata(expert_hist, dispatch_idx.shape[0])
    gate_scal = standard_topk.topk_weights.flatten()[combine_idx_long]
    routing_data = RoutingData(
        gate_scal,
        expert_data.slice_sizes,
        num_experts,
        top_k,
        expert_data,
    )
    return (
        routing_data,
        GatherIndx(combine_idx, dispatch_idx),
        ScatterIndx(dispatch_idx, combine_idx),
    )


def make_fused_activation(name, fn, arg_names, args, reduction_n):
    if not IS_AVAILABLE:
        raise ImportError("The triton_kernels package is not installed")
    if IS_LEGACY:
        return FusedActivation(
            FnSpecs(name, fn, arg_names, reduction_n=reduction_n), args
        )
    return FusedActivation(FnSpecs(name, fn, arg_names), args, reduction_n)
