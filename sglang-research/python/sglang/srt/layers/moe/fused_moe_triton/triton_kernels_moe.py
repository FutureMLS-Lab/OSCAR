# Adapted from https://github.com/vllm-project/vllm/pull/18595/files#diff-f426a6de78c82ffec568eff6811bfbf0043dab5f87f1a8c0cffdbdcb8a81e035

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl
from sgl_kernel import gelu_and_mul, silu_and_mul
from triton_kernels.matmul_ogs import (
    FlexCtx,
    PrecisionConfig,
    matmul_ogs,
)
from triton_kernels.numerics import InFlexData
from triton_kernels.swiglu import swiglu_fn

from sglang.srt.layers.moe.triton_kernels_compat import (
    IS_LEGACY,
    GatherIndx,
    RoutingData,
    ScatterIndx,
    make_fused_activation,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
    from sglang.srt.layers.moe.topk import TopKOutput


@triton.jit(repr=lambda _: "_situ")
def _situ_fn(input, beta, linear_beta):
    gate, up = tl.split(
        tl.reshape(input, (input.shape[0], input.shape[1] // 2, 2))
    )
    gate = gate.to(tl.float32)
    up = up.to(tl.float32)
    if linear_beta is not None:
        up = linear_beta * (2.0 * tl.sigmoid(2.0 * up / linear_beta) - 1.0)
    tanh_gate = 2.0 * tl.sigmoid(2.0 * gate / beta) - 1.0
    return beta * tanh_gate * tl.sigmoid(gate) * up


def quantize(w, dtype, dev, **opt):
    if dtype == "bf16":
        return w.to(torch.bfloat16), InFlexData()


def triton_kernel_moe_forward(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_output: TopKOutput,
    moe_runner_config: MoeRunnerConfig,
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
) -> torch.Tensor:

    from sglang.srt.layers.moe.topk import TopKOutputChecker

    assert TopKOutputChecker.format_is_triton_kernels(topk_output)

    routing_data, gather_idx, scatter_idx = topk_output

    return triton_kernel_fused_experts(
        hidden_states,
        w1,
        w2,
        routing_data,
        gather_idx,
        scatter_idx,
        inplace=False,  # triton kernel doesn't support inplace
        activation=moe_runner_config.activation,
        activation_situ_beta=moe_runner_config.activation_situ_beta,
        activation_situ_linear_beta=moe_runner_config.activation_situ_linear_beta,
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_fp8_w8a8=use_fp8_w8a8,
        per_channel_quant=per_channel_quant,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
    )


# This is a triton implementation of the fused_experts function
def triton_kernel_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    routing_data: RoutingData,
    gather_indx: GatherIndx,
    scatter_indx: ScatterIndx,
    inplace: bool = False,
    activation: str = "silu",
    activation_situ_beta: float = 1.0,
    activation_situ_linear_beta: Optional[float] = None,
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
) -> torch.Tensor:

    assert use_fp8_w8a8 is False, "use_fp8_w8a8 is not supported"
    assert per_channel_quant is False, "per_channel_quant is not supported"
    assert expert_map is None, "expert_map is not supported"
    assert w1_scale is None, "w1_scale is not supported"
    assert w2_scale is None, "w2_scale is not supported"
    assert a1_scale is None, "a1_scale is not supported"
    assert a2_scale is None, "a2_scale is not supported"
    assert block_shape is None, "block_shape is not supported"

    # type check
    assert hidden_states.dtype == torch.bfloat16, "hidden_states must be bfloat16"
    assert w1.dtype == torch.bfloat16, "w1 must be bfloat16"
    assert w2.dtype == torch.bfloat16, "w2 must be bfloat16"

    # Shape check
    assert hidden_states.ndim == 2, "hidden_states must be 2D"
    assert (
        hidden_states.shape[-1] == w1.shape[-2]
    ), f"hidden_states shape[-1] {hidden_states.shape} must be equal to w1 shape[-2] {w1.shape}"
    assert (
        w2.shape[-1] == w1.shape[1]
    ), f"w2 shape[-1] {w2.shape[-1]} must be equal to w1 shape[1] {w1.shape[1]}"

    # feature check
    assert inplace is False, "Inplace is not supported in new triton MoE kernel"

    M, K = hidden_states.shape
    E, _, N = w1.shape
    n_expts_act = routing_data.n_expts_act
    dtype = hidden_states.dtype

    if global_num_experts == -1:
        global_num_experts = E

    # consistent with default implementation
    intermediate_cache2 = torch.empty(
        (M * n_expts_act, N // 2), device="cuda", dtype=dtype
    )

    intermediate_cache1 = matmul_ogs(
        hidden_states,
        w1,
        None,
        routing_data,
        gather_indx=gather_indx,
        gammas=routing_data.gate_scal if apply_router_weight_on_input else None,
    )

    if activation == "silu":
        silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
    elif activation == "gelu":
        gelu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
    elif activation == "situ":
        gate, up = intermediate_cache1.view(-1, N).chunk(2, dim=-1)
        gate = gate.float()
        up = up.float()
        if activation_situ_linear_beta is not None:
            up = activation_situ_linear_beta * torch.tanh(
                up / activation_situ_linear_beta
            )
        intermediate_cache2.copy_(
            (
                activation_situ_beta
                * torch.tanh(gate / activation_situ_beta)
                * torch.sigmoid(gate)
                * up
            ).to(intermediate_cache2.dtype)
        )
    else:
        raise ValueError(f"Unsupported FusedMoe activation: {activation}")

    intermediate_cache3 = matmul_ogs(
        intermediate_cache2,
        w2,
        None,
        routing_data,
        scatter_indx=scatter_indx,
        gammas=None if apply_router_weight_on_input else routing_data.gate_scal,
    )

    return intermediate_cache3


def triton_kernel_moe_with_bias_forward(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w1_pcg,
    b1: torch.Tensor,
    w2: torch.Tensor,
    w2_pcg,
    b2: torch.Tensor,
    topk_output: TopKOutput,
    moe_runner_config: MoeRunnerConfig,
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
) -> torch.Tensor:
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    assert TopKOutputChecker.format_is_triton_kernels(topk_output)

    routing_data, gather_idx, scatter_idx = topk_output

    return triton_kernel_fused_experts_with_bias(
        hidden_states,
        w1=w1,
        w1_pcg=w1_pcg,
        b1=b1,
        w2=w2,
        w2_pcg=w2_pcg,
        b2=b2,
        routing_data=routing_data,
        gather_indx=gather_idx,
        scatter_indx=scatter_idx,
        inplace=False,  # triton kernel doesn't support inplace
        activation=moe_runner_config.activation,
        activation_situ_beta=moe_runner_config.activation_situ_beta,
        activation_situ_linear_beta=moe_runner_config.activation_situ_linear_beta,
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_fp8_w8a8=use_fp8_w8a8,
        per_channel_quant=per_channel_quant,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        gemm1_alpha=moe_runner_config.gemm1_alpha,
        gemm1_clamp_limit=moe_runner_config.gemm1_clamp_limit,
    )


def triton_kernel_fused_experts_with_bias(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w1_pcg,
    b1: torch.Tensor,
    w2: torch.Tensor,
    w2_pcg,
    b2: torch.Tensor,
    routing_data: RoutingData,
    gather_indx: GatherIndx,
    scatter_indx: ScatterIndx,
    inplace: bool = False,
    activation: str = "silu",
    activation_situ_beta: float = 1.0,
    activation_situ_linear_beta: Optional[float] = None,
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    gemm1_alpha: Optional[float] = None,
    gemm1_clamp_limit: Optional[float] = None,
) -> torch.Tensor:
    assert use_fp8_w8a8 is False, "use_fp8_w8a8 is not supported"
    assert per_channel_quant is False, "per_channel_quant is not supported"
    assert expert_map is None, "expert_map is not supported"
    assert w1_scale is None, "w1_scale is not supported"
    assert w2_scale is None, "w2_scale is not supported"
    assert a1_scale is None, "a1_scale is not supported"
    assert a2_scale is None, "a2_scale is not supported"
    assert block_shape is None, "block_shape is not supported"

    # type check
    assert hidden_states.dtype == torch.bfloat16, "hidden_states must be bfloat16"
    for w in (w1, w2):
        # TODO assert bf16 or mxfp4
        # assert (w.dtype == torch.bfloat16) or check-is-mxfp4, f"w must be bfloat16 or mxfp4 {w1.dtype=}"
        pass

    # Shape check
    assert hidden_states.ndim == 2, "hidden_states must be 2D"
    assert (
        hidden_states.shape[-1] == w1.shape[-2]
    ), f"hidden_states shape[-1] {hidden_states.shape} must be equal to w1 shape[-2] {w1.shape}"
    assert (
        w2.shape[-1] == w1.shape[1]
    ), f"w2 shape[-1] {w2.shape[-1]} must be equal to w1 shape[1] {w1.shape[1]}"

    # feature check
    assert inplace is False, "Inplace is not supported in new triton MoE kernel"

    M, K = hidden_states.shape
    E, _, N = w1.shape
    n_expts_act = routing_data.n_expts_act

    if global_num_experts == -1:
        global_num_experts = E

    # TODO maybe completely remove this branch
    if w1.dtype == torch.bfloat16:
        device = "cuda"
        optg = dict()
        w1, w1_flex = quantize(w1, "bf16", device, **optg)
        w1_pcg = PrecisionConfig(flex_ctx=FlexCtx(rhs_data=w1_flex))

        w2, w2_flex = quantize(w2, "bf16", device, **optg)
        w2_pcg = PrecisionConfig(flex_ctx=FlexCtx(rhs_data=w2_flex))

    if activation == "situ":
        act_name = "situ"
        act_fn = _situ_fn
        act_arg_names = ("beta", "linear_beta")
        act_args = (activation_situ_beta, activation_situ_linear_beta)
    elif activation == "silu":
        act_name = "swiglu"
        act_fn = swiglu_fn
        act_arg_names = ("alpha", "limit")
        act_args = (gemm1_alpha, gemm1_clamp_limit)
    else:
        raise ValueError(f"Unsupported FusedMoe activation: {activation}")

    act = make_fused_activation(
        act_name, act_fn, act_arg_names, act_args, reduction_n=2
    )

    leading_shape = () if IS_LEGACY else (1,)
    intermediate_cache = torch.empty(
        (*leading_shape, M * n_expts_act, N // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    output = torch.empty(
        (*leading_shape, M, K),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    matmul_ogs(
        hidden_states,
        w1,
        b1,
        routing_data,
        gather_indx=gather_indx,
        precision_config=w1_pcg,
        gammas=routing_data.gate_scal if apply_router_weight_on_input else None,
        fused_activation=act,
        y=intermediate_cache,
    )

    if IS_LEGACY:
        expert_output = torch.empty(
            (M * n_expts_act, K),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        matmul_ogs(
            intermediate_cache.view(M * n_expts_act, N // 2),
            w2,
            b2,
            routing_data,
            precision_config=w2_pcg,
            y=expert_output,
        )
        if not apply_router_weight_on_input:
            expert_output.mul_(routing_data.gate_scal[:, None])
        token_idx = scatter_indx.dst_indx // n_expts_act
        output_fp32 = torch.zeros(
            (M, K), device=hidden_states.device, dtype=torch.float32
        )
        for start in range(0, token_idx.numel(), 16384):
            end = min(start + 16384, token_idx.numel())
            chunk_token_idx = token_idx[start:end]
            chunk_valid = chunk_token_idx >= 0
            output_fp32.index_add_(
                0,
                chunk_token_idx[chunk_valid].long(),
                expert_output[start:end][chunk_valid].float(),
            )
        output.copy_(output_fp32.to(hidden_states.dtype))
    else:
        matmul_ogs(
            intermediate_cache.view(M * n_expts_act, N // 2),
            w2,
            b2,
            routing_data,
            scatter_indx=scatter_indx,
            precision_config=w2_pcg,
            gammas=None if apply_router_weight_on_input else routing_data.gate_scal,
            y=output,
        )
    return output.view(M, K)
