"""Patch SGLang to calibrate or fake-quantize MiniMax-M3 routed experts."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F


METHOD = os.environ.get("MINIMAX_ROUTED_W3_METHOD", "").strip()
CALIBRATION_DIR = os.environ.get("MINIMAX_ROUTED_CALIBRATION_DIR", "").strip()
STATS_DIR = os.environ.get("MINIMAX_ROUTED_STATS_DIR", "").strip()
QUANTIZER_PATH = os.environ.get("MINIMAX_ROUTED_W3_QUANTIZER", "").strip()
CHUNK_EXPERTS = int(os.environ.get("MINIMAX_ROUTED_W3_CHUNK_EXPERTS", "1"))
CHUNK_TILES = int(os.environ.get("MINIMAX_ROUTED_W3_CHUNK_TILES", "8192"))
ITERATIONS = int(os.environ.get("MINIMAX_ROUTED_W3_LM_ITERATIONS", "8"))
GPTQ_DAMP = float(os.environ.get("MINIMAX_ROUTED_GPTQ_DAMP", "0.1"))
HADAMARD_SEED = int(os.environ.get("MINIMAX_ROUTED_HADAMARD_SEED", "20260813"))
B_SCALE_MODE = os.environ.get(
    "MINIMAX_ROUTED_B_SCALE_MODE", "global_native_k64"
).strip()
MMA_K = int(os.environ.get("MINIMAX_ROUTED_MMA_K", "64"))
INTER_ITERATIONS = int(
    os.environ.get("MINIMAX_ROUTED_INTER_ITERATIONS", "1")
)
INTER_CHUNK_TILES = int(
    os.environ.get("MINIMAX_ROUTED_INTER_CHUNK_TILES", "1024")
)

SIMPLE_METHODS = {"uniform", "nf3", "lloyd_max", "scale_lm"}
STATS_METHODS = {
    "rms_lm",
    "gptq",
    "hadamard_gate_gptq",
    "hadamard_gate_inter_gptq",
    "awq_gate_lm",
    "squeeze_factorized",
}
HADAMARD_GATE_METHODS = {"hadamard_gate_gptq", "hadamard_gate_inter_gptq"}
A8_METHODS = {"a8_only", "lloyd_max_a8"}


def install_msa_api_compatibility() -> None:
    try:
        import fmha_sm100
    except ImportError:
        return
    aliases = {"plan": "fmha_sm100_plan", "run": "fmha_sm100"}
    installed = []
    for expected, available in aliases.items():
        if not hasattr(fmha_sm100, expected) and hasattr(fmha_sm100, available):
            setattr(fmha_sm100, expected, getattr(fmha_sm100, available))
            installed.append(f"{expected}={available}")
    if installed:
        print("MINIMAX_MSA_API_COMPAT=" + ",".join(installed), flush=True)


def install_cpu_offload_tied_weight_compatibility() -> None:
    if os.environ.get("MINIMAX_CPU_OFFLOAD_TIED_COMPAT") != "1":
        return
    from sglang.srt.utils import offloader

    original = offloader.functional_call

    def functional_call_compat(module, parameter_and_buffer_dicts, *args, **kwargs):
        kwargs.setdefault("tie_weights", False)
        return original(module, parameter_and_buffer_dicts, *args, **kwargs)

    offloader.functional_call = functional_call_compat
    print("MINIMAX_CPU_OFFLOAD_TIED_COMPAT=enabled", flush=True)


def _load_quantizer():
    spec = importlib.util.spec_from_file_location(
        "rubin_w3_quantizer",
        Path(QUANTIZER_PATH),
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _logical_weight(parameter: torch.Tensor, transposed: bool) -> torch.Tensor:
    return (
        parameter.transpose(-1, -2).contiguous()
        if transposed
        else parameter
    )


def _copy_logical(
    destination: torch.Tensor,
    logical: torch.Tensor,
    transposed: bool,
) -> None:
    destination.copy_(
        logical.transpose(-1, -2).contiguous()
        if transposed
        else logical
    )


def _block_hessian(values: torch.Tensor, block: int = 64) -> torch.Tensor:
    padded_k = ((values.shape[-1] + block - 1) // block) * block
    padded = F.pad(values.float(), (0, padded_k - values.shape[-1]))
    shaped = padded.view(-1, padded_k // block, block)
    return torch.einsum("tki,tkj->kij", shaped, shaped) / max(
        values.shape[0],
        1,
    )


def _second_moment(values: torch.Tensor) -> torch.Tensor:
    return values.float().square().mean(dim=0)


def _fallback_by_count(values: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    live = counts > 0
    if live.any():
        fallback = values[live].mean(dim=0)
    else:
        fallback = torch.ones_like(values[0])
    return torch.where(
        live.view(-1, *([1] * (values.ndim - 1))),
        values,
        fallback.unsqueeze(0),
    )


def _gate_activation(layer, gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    limit = layer.moe_runner_config.gemm1_clamp_limit
    alpha = layer.moe_runner_config.gemm1_alpha
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    if alpha is None:
        return F.silu(gate) * up
    return (up + 1.0) * gate * torch.sigmoid(gate * alpha)


def _collect_layer_stats(self, layer, dispatch_output) -> None:
    if getattr(layer, "_minimax_stats_collected", False):
        return
    if not hasattr(dispatch_output, "topk_output"):
        raise ValueError("MiniMax calibration requires StandardDispatchOutput")
    hidden = dispatch_output.hidden_states.detach()
    topk_ids = dispatch_output.topk_output.topk_ids.detach()
    routed = int(layer._num_local_routed)
    transposed = bool(self.use_triton_kernels)
    w13 = _logical_weight(layer.w13_weight[:routed], transposed)

    gate_second = torch.zeros(
        routed,
        hidden.shape[-1],
        dtype=torch.float32,
        device=hidden.device,
    )
    down_second = torch.zeros(
        routed,
        w13.shape[1] // 2,
        dtype=torch.float32,
        device=hidden.device,
    )
    gate_hessian = torch.zeros(
        routed,
        hidden.shape[-1] // 64,
        64,
        64,
        dtype=torch.float32,
        device=hidden.device,
    )
    down_hessian = torch.zeros(
        routed,
        (w13.shape[1] // 2) // 64,
        64,
        64,
        dtype=torch.float32,
        device=hidden.device,
    )
    counts = torch.zeros(routed, dtype=torch.int64, device=hidden.device)

    with torch.inference_mode():
        for expert in range(routed):
            selected = (topk_ids == expert).any(dim=-1)
            if not selected.any():
                continue
            values = hidden[selected]
            intermediate = _gate_activation(
                layer,
                F.linear(values, w13[expert]),
            )
            counts[expert] = values.shape[0]
            gate_second[expert] = _second_moment(values)
            down_second[expert] = _second_moment(intermediate)
            gate_hessian[expert] = _block_hessian(values)
            down_hessian[expert] = _block_hessian(intermediate)

    state = {
        "layer": int(layer.layer_id),
        "ep_rank": int(layer.moe_ep_rank),
        "routed_experts": routed,
        "counts": counts.cpu(),
        "gate_second": gate_second.cpu(),
        "down_second": down_second.cpu(),
        "gate_hessian": gate_hessian.cpu(),
        "down_hessian": down_hessian.cpu(),
    }
    output = (
        Path(CALIBRATION_DIR)
        / f"ep{int(layer.moe_ep_rank):02d}"
        / f"layer{int(layer.layer_id):02d}.pt"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, output)
    layer._minimax_stats_collected = True
    print(
        f"MINIMAX_ROUTED_CALIBRATION={output} tokens={int(counts.sum())}",
        flush=True,
    )


def _load_layer_stats(layer) -> dict:
    path = (
        Path(STATS_DIR)
        / f"ep{int(layer.moe_ep_rank):02d}"
        / f"layer{int(layer.layer_id):02d}.pt"
    )
    state = torch.load(path, map_location="cpu", weights_only=True)
    counts = state["counts"]
    for name in ("gate_second", "down_second", "gate_hessian", "down_hessian"):
        state[name] = _fallback_by_count(state[name], counts)
    return state


def _quantize_simple(
    quantizer,
    parameter,
    routed: int,
    transposed: bool,
    method: str,
) -> dict:
    relative_error = 0.0
    weights = 0
    for start in range(0, routed, CHUNK_EXPERTS):
        stop = min(start + CHUNK_EXPERTS, routed)
        view = parameter.data[start:stop]
        logical = _logical_weight(view, transposed)
        matrix = logical.reshape(-1, logical.shape[-1])
        restored, stats = quantizer.quantize_weight_lutb(
            matrix,
            method=method,
            iterations=ITERATIONS,
            chunk_tiles=CHUNK_TILES,
            b_scale_mode=B_SCALE_MODE,
        )
        _copy_logical(view, restored.view_as(logical), transposed)
        relative_error += stats["relative_mse"] * stats["num_weights"]
        weights += stats["num_weights"]
        del matrix, restored
        torch.cuda.empty_cache()
    return {"weights": weights, "relative_mse": relative_error / max(weights, 1)}


def _quantize_with_stats(
    quantizer,
    parameter,
    routed: int,
    transposed: bool,
    method: str,
    second: torch.Tensor,
    hessian: torch.Tensor,
    inter_iterations: int = 0,
) -> dict:
    relative_error = 0.0
    weights = 0
    for expert in range(routed):
        view = parameter.data[expert]
        logical = _logical_weight(view, transposed)
        importance = second[expert].to(logical.device)
        if method == "rms_lm":
            restored, stats = quantizer.quantize_weight_lutb(
                logical,
                method="sensitivity_lm",
                iterations=ITERATIONS,
                chunk_tiles=CHUNK_TILES,
                b_scale_mode=B_SCALE_MODE,
                channel_importance=importance,
            )
        elif method == "squeeze_factorized":
            row = logical.float().square().mean(dim=1).clamp_min(1e-12)
            element = row.unsqueeze(1) * importance.unsqueeze(0)
            restored, stats = quantizer.quantize_weight_lutb(
                logical,
                method="squeezellm_lm",
                iterations=ITERATIONS,
                chunk_tiles=CHUNK_TILES,
                b_scale_mode=B_SCALE_MODE,
                element_importance=element,
            )
        else:
            restored, stats = quantizer.quantize_weight_lutb_gptq(
                logical,
                hessian=hessian[expert].to(logical.device),
                iterations=ITERATIONS,
                damp_percent=GPTQ_DAMP,
                weighted_codebook=True,
                b_scale_mode=B_SCALE_MODE,
                inter_iterations=inter_iterations,
                inter_tolerance=0.0,
                inter_codebook_damp=1e-4,
                inter_chunk_tiles=INTER_CHUNK_TILES,
            )
        _copy_logical(view, restored, transposed)
        relative_error += stats["relative_mse"] * stats["num_weights"]
        weights += stats["num_weights"]
        del restored
        torch.cuda.empty_cache()
    return {"weights": weights, "relative_mse": relative_error / max(weights, 1)}


def _apply_gate_awq(quantizer, layer, stats, transposed: bool) -> None:
    routed = int(layer._num_local_routed)
    logical = _logical_weight(layer.w13_weight.data[:routed], transposed)
    importance = stats["gate_second"].mean(dim=0).to(logical.device)
    scale = quantizer.awq_group_scale(
        [logical.reshape(-1, logical.shape[-1])],
        importance,
        0.5,
    )
    logical.mul_(scale.view(1, 1, -1).to(logical.dtype))
    _copy_logical(layer.w13_weight.data[:routed], logical, transposed)
    layer._minimax_gate_awq_scale = scale.to(layer.w13_weight.device)


def _apply_gate_hadamard(
    quantizer,
    layer,
    transposed: bool,
) -> torch.Tensor:
    routed = int(layer._num_local_routed)
    logical = _logical_weight(layer.w13_weight.data[:routed], transposed)
    generator = torch.Generator(device="cpu").manual_seed(
        HADAMARD_SEED + int(layer.layer_id) * 131 + int(layer.moe_ep_rank)
    )
    signs = (
        torch.randint(
            0,
            2,
            (logical.shape[-1] // 64, 64),
            generator=generator,
            dtype=torch.int8,
        )
        .mul_(2)
        .sub_(1)
        .to(logical.device)
    )
    rotated = quantizer.random_hadamard_k64(logical, signs)
    _copy_logical(layer.w13_weight.data[:routed], rotated, transposed)
    layer._minimax_gate_hadamard_signs = signs
    return signs


def _rotate_hessian(
    quantizer,
    hessian: torch.Tensor,
    signs: torch.Tensor,
) -> torch.Tensor:
    identity = torch.eye(64, dtype=torch.float32, device=signs.device)
    transforms = []
    for block in range(signs.shape[0]):
        transforms.append(
            quantizer.random_hadamard_k64(
                identity,
                signs[block : block + 1],
            ).float()
        )
    transform = torch.stack(transforms).cpu()
    return torch.einsum(
        "bij,ebjk,bkl->ebil",
        transform.transpose(-1, -2),
        hessian.float(),
        transform,
    )


def install() -> None:
    if not METHOD and not CALIBRATION_DIR:
        return
    if MMA_K not in (32, 64):
        raise ValueError(f"Unsupported MiniMax routed MMA K={MMA_K}")
    if (
        B_SCALE_MODE in {"legacy_k32", "global_legacy_k32"}
        and MMA_K != 32
    ):
        raise ValueError(
            f"{B_SCALE_MODE} requires MINIMAX_ROUTED_MMA_K=32"
        )
    if (
        B_SCALE_MODE in {"native_k64", "global_native_k64"}
        and MMA_K != 64
    ):
        raise ValueError(f"{B_SCALE_MODE} requires MINIMAX_ROUTED_MMA_K=64")
    if B_SCALE_MODE in {"legacy_k32", "global_legacy_k32"}:
        print(
            "MINIMAX_ROUTED_CONTRACT_WARNING="
            "two-K32 is an algorithm oracle, not native PTX 9.4 LUT-B",
            flush=True,
        )
    supported = SIMPLE_METHODS | STATS_METHODS | A8_METHODS
    if METHOD and METHOD not in supported:
        raise ValueError(f"Unsupported MiniMax routed method: {METHOD}")
    quantizer = _load_quantizer()

    from sglang.srt.layers.moe.fused_moe_triton.layer import (
        UnquantizedFusedMoEMethod,
    )

    original_process = UnquantizedFusedMoEMethod.process_weights_after_loading
    original_apply = UnquantizedFusedMoEMethod.apply

    def patched_process(self, layer) -> None:
        original_process(self, layer)
        routed = int(layer._num_local_routed)
        transposed = bool(self.use_triton_kernels)
        if not METHOD or METHOD == "a8_only":
            return
        stats = _load_layer_stats(layer) if METHOD in STATS_METHODS else None
        if METHOD == "awq_gate_lm":
            _apply_gate_awq(quantizer, layer, stats, transposed)
        if METHOD in HADAMARD_GATE_METHODS:
            signs = _apply_gate_hadamard(quantizer, layer, transposed)
            stats["gate_hessian"] = _rotate_hessian(
                quantizer,
                stats["gate_hessian"],
                signs,
            )
            stats["gate_second"] = torch.diagonal(
                stats["gate_hessian"],
                dim1=-2,
                dim2=-1,
            ).reshape(routed, -1)

        if METHOD in SIMPLE_METHODS:
            weight_method = METHOD
            w13 = _quantize_simple(
                quantizer, layer.w13_weight, routed, transposed, weight_method
            )
            w2 = _quantize_simple(
                quantizer, layer.w2_weight, routed, transposed, weight_method
            )
        elif METHOD in {"lloyd_max_a8", "awq_gate_lm"}:
            w13 = _quantize_simple(
                quantizer, layer.w13_weight, routed, transposed, "lloyd_max"
            )
            w2 = _quantize_simple(
                quantizer, layer.w2_weight, routed, transposed, "lloyd_max"
            )
        else:
            effective = (
                "gptq"
                if METHOD in HADAMARD_GATE_METHODS
                else METHOD
            )
            inter_rounds = (
                INTER_ITERATIONS
                if METHOD == "hadamard_gate_inter_gptq"
                else 0
            )
            w13 = _quantize_with_stats(
                quantizer,
                layer.w13_weight,
                routed,
                transposed,
                effective,
                stats["gate_second"],
                stats["gate_hessian"],
                inter_iterations=inter_rounds,
            )
            w2 = _quantize_with_stats(
                quantizer,
                layer.w2_weight,
                routed,
                transposed,
                effective,
                stats["down_second"],
                stats["down_hessian"],
                inter_iterations=inter_rounds,
            )
        print(
            "MINIMAX_ROUTED_W3="
            + json.dumps(
                {
                    "method": METHOD,
                    "layer": int(layer.layer_id),
                    "routed_experts": routed,
                    "w13_relative_mse": w13["relative_mse"],
                    "w2_relative_mse": w2["relative_mse"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def patched_apply(self, layer, dispatch_output):
        if CALIBRATION_DIR:
            _collect_layer_stats(self, layer, dispatch_output)
        hidden = dispatch_output.hidden_states
        if hasattr(layer, "_minimax_gate_awq_scale"):
            hidden = hidden / layer._minimax_gate_awq_scale.to(hidden.dtype)
        if hasattr(layer, "_minimax_gate_hadamard_signs"):
            hidden = quantizer.random_hadamard_k64(
                hidden,
                layer._minimax_gate_hadamard_signs,
            )
        if METHOD in A8_METHODS:
            hidden = quantizer.activation_mxfp8_qdq(
                hidden,
                block_size=MMA_K,
            )
        if hidden is not dispatch_output.hidden_states:
            dispatch_output = dispatch_output._replace(hidden_states=hidden)
        return original_apply(self, layer, dispatch_output)

    UnquantizedFusedMoEMethod.process_weights_after_loading = patched_process
    UnquantizedFusedMoEMethod.apply = patched_apply


install_msa_api_compatibility()
install_cpu_offload_tied_weight_compatibility()
install()
