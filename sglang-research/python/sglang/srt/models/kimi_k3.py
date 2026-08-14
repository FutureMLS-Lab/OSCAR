# Adapted from: https://github.com/vllm-project/vllm/blob/0384aa7150c4c9778efca041ffd1beb3ad2bd694/vllm/model_executor/models/kimi_linear.py

from collections.abc import Iterable
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from sglang.srt.configs.kimi_linear import (
    KimiLinearConfig,
    as_kimi_linear_config,
)
from sglang.srt.distributed import (
    divide,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.attention.fla.fused_norm_gate import FusedRMSNormGated
from sglang.srt.layers.attention.fla.kda import fused_kda_gate
from sglang.srt.layers.communicator import AttentionInputs, get_attn_tp_context
from sglang.srt.layers.dp_attention import (
    get_attention_dp_rank,
    get_attention_tp_group,
    get_attention_tp_rank,
    get_attention_tp_size,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelBatchedLinear,
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    MergedColumnParallelRepeatedLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
    sharded_weight_loader,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
from sglang.srt.models.transformers import maybe_prefix
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import make_layers
from sglang.srt.utils.common import BumpAllocator, add_prefix, set_weight_attrs


class SituAndMul(nn.Module):
    """Kimi's SiTU gated activation."""

    def __init__(self, beta: float = 1.0, linear_beta: Optional[float] = None):
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        gate = gate.float()
        up = up.float()
        activated = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return (activated * up).to(x.dtype)


class KimiMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        activation_situ_beta: float = 1.0,
        activation_situ_linear_beta: Optional[float] = None,
    ):
        super().__init__()
        if hidden_act not in ("silu", "situ"):
            raise ValueError(f"Unsupported Kimi MLP activation: {hidden_act}")
        self.hidden_act = hidden_act
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.situ = SituAndMul(
            beta=activation_situ_beta,
            linear_beta=activation_situ_linear_beta,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        if self.hidden_act == "situ":
            activated = self.situ(gate_up)
        else:
            gate, up = gate_up.chunk(2, dim=-1)
            activated = F.silu(gate) * up
        return self.down_proj(activated)[0]


class OscarQKVCapture:
    """Explicit calibration-only Q/K/V collector for expanded Kimi attention."""

    def __init__(self, output_dir: str, token_limit: int, layer_id: int):
        self.output_dir = Path(output_dir)
        self.token_limit = token_limit
        self.layer_id = layer_id
        self.saved_tokens = 0
        self.chunk_index = 0

    @torch.no_grad()
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> None:
        if (
            self.saved_tokens >= self.token_limit
            or not forward_batch.forward_mode.is_extend()
        ):
            return
        token_count = min(q.shape[0], self.token_limit - self.saved_tokens)
        if token_count <= 0:
            return
        tensors = [x[:token_count].contiguous().detach() for x in (q, k, v)]
        if get_attention_tp_size() > 1:
            group = get_attention_tp_group()
            tensors = [group.all_gather(x, dim=1) for x in tensors]
        if get_attention_tp_rank() == 0 and get_attention_dp_rank() == 0:
            for name, tensor in zip(("q", "k", "v"), tensors):
                directory = self.output_dir / f"layer_{self.layer_id}" / name
                directory.mkdir(parents=True, exist_ok=True)
                torch.save(tensor.cpu(), directory / f"{self.chunk_index}.pt")
        self.saved_tokens += token_count
        self.chunk_index += 1


class KimiMLAAttention(DeepseekV2AttentionMLA):
    """Kimi MLA weights executed through expanded MHA cache storage."""

    def __init__(self, *args, **kwargs):
        prefix = kwargs.get("prefix", "")
        super().__init__(*args, **kwargs)
        self.use_expanded_mha_cache = True
        self.g_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads * self.v_head_dim,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.g_proj",
        )
        try:
            server_args = get_global_server_args()
        except ValueError:
            server_args = None
        dump_path = getattr(server_args, "oscar_qkv_dump_path", None)
        dump_tokens = getattr(server_args, "oscar_qkv_dump_tokens", 0)
        if dump_path and dump_tokens > 0 and getattr(server_args, "dp_size", 1) != 1:
            raise ValueError("OSCAR Q/K/V calibration requires --dp-size 1")
        self.expanded_qkv_capture = (
            OscarQKVCapture(dump_path, dump_tokens, self.layer_id)
            if dump_path and dump_tokens > 0
            else None
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
        *args,
        **kwargs,
    ):
        # DeepSeek decoder layers populate this through LayerCommunicator.
        # Kimi owns its norm/residual path, so provide the same lazy QKV input
        # explicitly before entering the shared MLA implementation.
        attn_tp_context = get_attn_tp_context()
        attn_tp_context.set_attn_inputs(
            AttentionInputs(hidden_states, forward_batch, self.prepare_qkv_latent)
        )
        try:
            return super().forward(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                zero_allocator=zero_allocator,
                *args,
                **kwargs,
            )
        finally:
            attn_tp_context.set_attn_inputs(None)


class KimiMoE(nn.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        layer_idx: int = 0,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        moe_intermediate_size = config.moe_intermediate_size
        num_experts = config.num_experts
        moe_renormalize = config.moe_renormalize
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.num_shared_experts = config.num_shared_experts
        self.layer_idx = layer_idx
        self.alt_stream = alt_stream

        if config.hidden_act not in ("silu", "situ"):
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu and situ are supported."
            )

        routed_hidden_size = getattr(config, "routed_expert_hidden_size", None)
        self.use_latent_moe = routed_hidden_size is not None
        expert_hidden_size = routed_hidden_size or hidden_size
        if self.use_latent_moe:
            self.routed_expert_down_proj = ReplicatedLinear(
                hidden_size,
                expert_hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.routed_expert_down_proj",
            )
            self.routed_expert_up_proj = RowParallelLinear(
                expert_hidden_size,
                hidden_size,
                bias=False,
                input_is_parallel=False,
                quant_config=quant_config,
                prefix=f"{prefix}.routed_expert_up_proj",
            )
            self.routed_expert_norm = (
                RMSNorm(expert_hidden_size, eps=config.rms_norm_eps)
                if getattr(config, "latent_moe_use_norm", False)
                else nn.Identity()
            )

        # Gate always runs at half / full precision for now.
        self.gate = ReplicatedLinear(
            hidden_size,
            num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        self.gate.e_score_correction_bias = nn.Parameter(torch.empty(num_experts))

        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_token,
            hidden_size=expert_hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=self.layer_idx,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            prefix=add_prefix("experts", prefix),
            activation=config.hidden_act,
            activation_situ_beta=getattr(config, "activation_situ_beta", 1.0),
            activation_situ_linear_beta=getattr(
                config, "activation_situ_linear_beta", None
            ),
        )
        # Kimi checkpoints store gate (w1) and up (w3) as separate tensors.
        # The triton_kernels fused activation consumes adjacent gate/up output
        # columns, so the serialized half-by-half rows must be interleaved
        # before the MXFP4 weight layout is swizzled.
        self.experts.interleave_gate_up_for_fused_activation = True

        self.topk = TopK(
            top_k=config.num_experts_per_token,
            renormalize=moe_renormalize,
            use_grouped_topk=True,
            scoring_func=config.moe_router_activation_func,
            num_expert_group=config.num_expert_group,
            topk_group=config.topk_group,
            correction_bias=self.gate.e_score_correction_bias,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=self.experts.should_fuse_routed_scaling_factor_in_topk,
            # Some Fp4 MoE backends require the output format to be bypassed but the MTP layers are unquantized
            # and requires the output format to be standard. We use quant_config to determine the output format.
            output_format=TopKOutputFormat.STANDARD if quant_config is None else None,
        )

        if self.num_shared_experts is not None:
            intermediate_size = moe_intermediate_size * self.num_shared_experts
            self.shared_experts = KimiMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=self.use_latent_moe,
                prefix=f"{prefix}.shared_experts",
                activation_situ_beta=getattr(config, "activation_situ_beta", 1.0),
                activation_situ_linear_beta=getattr(
                    config, "activation_situ_linear_beta", None
                ),
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)

        if self.use_latent_moe:
            shared_output = (
                self.shared_experts(hidden_states)
                if self.num_shared_experts is not None and num_tokens > 0
                else None
            )
            routed_input = self.routed_expert_down_proj(hidden_states)[0]
            router_logits, _ = self.gate(hidden_states)
            topk_output = self.topk(hidden_states, router_logits)
            routed_output = self.experts(routed_input, topk_output)
            if self.tp_size > 1:
                routed_output = tensor_model_parallel_all_reduce(routed_output)
            routed_output = self.routed_expert_norm(routed_output)
            final_hidden_states = self.routed_expert_up_proj(routed_output)[0]
            if shared_output is not None:
                final_hidden_states = final_hidden_states + shared_output
            return final_hidden_states.view(num_tokens, hidden_size)

        shared_output = None

        if (
            self.alt_stream is not None
            and self.num_shared_experts is not None
            and hidden_states.shape[0] > 0
            and get_is_capture_mode()
        ):
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)

            shared_output = self.shared_experts(hidden_states.clone())

            with torch.cuda.stream(self.alt_stream):
                router_logits, _ = self.gate(hidden_states)
                topk_output = self.topk(hidden_states, router_logits)
                final_hidden_states = self.experts(hidden_states, topk_output)

            current_stream.wait_stream(self.alt_stream)
        else:
            if self.num_shared_experts is not None and hidden_states.shape[0] > 0:
                shared_output = self.shared_experts(hidden_states)
            router_logits, _ = self.gate(hidden_states)
            topk_output = self.topk(hidden_states, router_logits)
            final_hidden_states = self.experts(hidden_states, topk_output)

        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output

        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states.view(num_tokens, hidden_size)


class KimiDeltaAttention(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        config: KimiLinearConfig,
        quant_config: Optional[QuantizationConfig] = None,
        rms_norm_eps: float = 1e-5,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.attn_tp_size = get_attention_tp_size()
        self.hidden_size = hidden_size
        self.config = config
        self.head_dim = config.linear_attn_config["head_dim"]
        self.num_heads = config.linear_attn_config["num_heads"]
        self.num_k_heads = config.linear_attn_config["num_heads"]
        self.num_v_heads = config.linear_attn_config["num_heads"]
        self.head_k_dim = config.linear_attn_config["head_dim"]
        self.head_v_dim = config.v_head_dim
        self.layer_idx = layer_idx
        self.prefix = prefix
        assert self.num_heads % self.tp_size == 0
        self.local_num_heads = divide(self.num_heads, self.tp_size)

        projection_size = self.head_dim * self.num_heads
        self.conv_size = config.linear_attn_config["short_conv_kernel_size"]
        self.use_full_rank_gate = config.linear_attn_config.get(
            "use_full_rank_gate", False
        )
        self.gate_lower_bound = config.linear_attn_config.get("gate_lower_bound")

        # TODO: support fusion with quant
        self.do_fuse_qkvbfg = quant_config is None and not self.use_full_rank_gate

        if self.do_fuse_qkvbfg:
            # Fuse: q, k, v, beta (column parallel) + f_a, g_a (replicated)
            self.qkvb_sizes = [
                projection_size,
                projection_size,
                projection_size,
                self.num_heads,
            ]
            self.fg_sizes = [self.head_dim, self.head_dim]

            self.fused_qkvbfg_a_proj = MergedColumnParallelRepeatedLinear(
                self.hidden_size,
                self.qkvb_sizes,  # Column parallel
                self.fg_sizes,  # Replicated: f_a, g_a
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkvbfg_a_proj",
            )
            self.split_sizes = [
                3 * projection_size // self.tp_size,  # qkv
                self.num_heads // self.tp_size,  # beta
                2 * self.head_dim,  # f_a, g_a
            ]
            self.fused_fg_b_proj = ColumnParallelBatchedLinear(
                2, self.head_dim, projection_size, dtype=config.dtype
            )
        else:
            # Unfused path: separate QKVParallelLinear
            attn_tp_rank = get_attention_tp_rank()
            self.qkv_proj = QKVParallelLinear(
                self.hidden_size,
                self.head_dim,
                self.num_heads,
                self.num_k_heads,
                bias=False,
                quant_config=quant_config,
                tp_rank=attn_tp_rank,
                tp_size=self.attn_tp_size,
                v_head_size=self.head_v_dim,
                prefix=f"{prefix}.qkv_proj",
            )

            self.f_a_proj = ReplicatedLinear(
                self.hidden_size,
                self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.f_a_proj",
            )

            self.f_b_proj = ColumnParallelLinear(
                self.head_dim,
                projection_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.f_b_proj",
            )

            self.b_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.b_proj",
            )

            if self.use_full_rank_gate:
                self.g_proj = ColumnParallelLinear(
                    self.hidden_size,
                    projection_size,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.g_proj",
                )
            else:
                self.g_a_proj = ReplicatedLinear(
                    self.hidden_size,
                    self.head_dim,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.g_a_proj",
                )
                self.g_b_proj = ColumnParallelLinear(
                    self.head_dim,
                    projection_size,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.g_b_proj",
                )

        self.dt_bias = nn.Parameter(
            torch.empty(divide(projection_size, self.tp_size), dtype=torch.float32)
        )

        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        self.qkv_conv1d = MergedColumnParallelLinear(
            input_size=self.conv_size,
            output_sizes=[projection_size, projection_size, projection_size],
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.qkv_conv1d",
        )
        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `set_weight_attrs`
        # doesn't allow to override it
        self.qkv_conv1d.weight.data = self.qkv_conv1d.weight.data.unsqueeze(1)

        # Kimi-K3 stores one decay coefficient per key channel, shared by all
        # heads (checkpoint shape: [head_dim]), rather than one per head.
        # Keep it replicated across attention-TP ranks.
        self.A_log = nn.Parameter(torch.empty(self.head_dim, dtype=torch.float32))

        self.o_norm = FusedRMSNormGated(
            self.head_dim, eps=rms_norm_eps, activation="sigmoid"
        )
        self.o_proj = RowParallelLinear(
            projection_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        conv_weights = self.qkv_conv1d.weight.squeeze(1)
        bias = self.qkv_conv1d.bias

        self.attn = RadixLinearAttention(
            layer_id=self.layer_idx,
            num_q_heads=self.num_k_heads // self.attn_tp_size,
            num_k_heads=self.num_k_heads // self.attn_tp_size,
            num_v_heads=self.num_v_heads // self.attn_tp_size,
            head_q_dim=self.head_k_dim,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            conv_weights=conv_weights,
            bias=bias,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
        )
        self.attn.gate_lower_bound = self.gate_lower_bound

    def forward_qkvbfg(self, hidden_states: torch.Tensor):
        qkv, _ = self.qkv_proj(hidden_states)

        # Compute beta, forget_gate, and g_proj_states
        beta = self.b_proj(hidden_states)[0]
        forget_gate = self.f_b_proj(self.f_a_proj(hidden_states)[0])[0]
        g_proj_states = (
            self.g_proj(hidden_states)[0]
            if self.use_full_rank_gate
            else self.g_b_proj(self.g_a_proj(hidden_states)[0])[0]
        )

        return (
            qkv,
            beta,
            forget_gate,
            g_proj_states,
        )

    def forward_qkvbfg_fused(self, hidden_states: torch.Tensor):
        # Single fused projection for all: qkv + beta + f_a + g_a
        fused_states = self.fused_qkvbfg_a_proj(hidden_states)

        qkv, beta, fg_a_states = torch.split(
            fused_states,
            self.split_sizes,
            dim=-1,
        )

        # use batch matmul to calculate forget_gate and g_proj_states
        forget_gate, g_proj_states = self.fused_fg_b_proj(
            fg_a_states.view(-1, 2, self.head_dim).transpose(0, 1)
        )

        return (
            qkv,
            beta,
            forget_gate,
            g_proj_states,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ) -> None:
        if self.do_fuse_qkvbfg:
            mixed_qkv, beta, forget_gate, g_proj_states = self.forward_qkvbfg_fused(
                hidden_states
            )
        else:
            mixed_qkv, beta, forget_gate, g_proj_states = self.forward_qkvbfg(
                hidden_states
            )

        # fused_kda_gate is fused to KimiLinearAttentionBackend with decode
        if not forward_batch.forward_mode.is_decode():
            forget_gate = fused_kda_gate(
                forget_gate,
                self.A_log,
                self.head_dim,
                g_bias=self.dt_bias,
                lower_bound=self.gate_lower_bound,
            )
            beta = beta.float().sigmoid()
            forget_gate = forget_gate.unsqueeze(0)
        beta = beta.unsqueeze(0)

        core_attn_out = self.attn(
            forward_batch,
            mixed_qkv=mixed_qkv,
            a=forget_gate,
            b=beta,
        )

        norm_gate = g_proj_states.unflatten(
            -1, (-1, self.head_dim)
        )  # ... (h d) -> ... h d
        core_attn_out = self.o_norm(core_attn_out, norm_gate)
        core_attn_out = core_attn_out.squeeze(0).flatten(-2)  # 1 n h d -> n (h d)

        return self.o_proj(core_attn_out)[0]


class KimiDecoderLayer(nn.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.alt_stream = alt_stream
        self.layer_idx = layer_idx

        self.is_moe = config.is_moe

        if config.is_kda_layer(layer_idx):
            self.self_attn = KimiDeltaAttention(
                layer_idx=layer_idx,
                hidden_size=config.hidden_size,
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            self.self_attn = KimiMLAAttention(
                layer_id=layer_idx,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                config=config,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                skip_rope=True,
            )

        if (
            self.is_moe
            and config.num_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            self.block_sparse_moe = KimiMoE(
                config=config,
                quant_config=quant_config,
                layer_idx=layer_idx,
                prefix=f"{prefix}.mlp",
                alt_stream=self.alt_stream,
            )
            self.mlp = self.block_sparse_moe
        else:
            self.mlp = KimiMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                activation_situ_beta=getattr(config, "activation_situ_beta", 1.0),
                activation_situ_linear_beta=getattr(
                    config, "activation_situ_linear_beta", None
                ),
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.use_attn_residuals = (
            getattr(config, "attn_res_block_size", None) is not None
        )
        if self.use_attn_residuals:
            self.attn_res_block_size = config.attn_res_block_size
            self.self_attention_res_norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.mlp_res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.self_attention_res_proj = ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attention_res_proj",
            )
            self.mlp_res_proj = ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp_res_proj",
            )

    @staticmethod
    def _apply_attn_res(
        prefix_sum: torch.Tensor,
        block_residual: torch.Tensor,
        proj: ReplicatedLinear,
        norm: RMSNorm,
    ) -> torch.Tensor:
        score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
        output = torch.empty_like(prefix_sum)
        # Residual selection is independent for every token. Chunking the
        # token dimension avoids keeping several full-length FP32 tensors
        # alive at once (about 3.5 GiB each for a 32K Kimi prefill).
        token_chunk_size = 2048
        for start in range(0, prefix_sum.shape[0], token_chunk_size):
            end = min(start + token_chunk_size, prefix_sum.shape[0])
            values = torch.cat(
                (
                    block_residual[start:end],
                    prefix_sum[start:end].unsqueeze(1),
                ),
                dim=1,
            )
            values_float = values.float()
            variance = values_float.square().mean(-1, keepdim=True)
            keys = values_float * torch.rsqrt(variance + norm.variance_epsilon)
            scores = (keys * score_weight).sum(-1)
            probs = scores.softmax(-1).unsqueeze(1)
            output[start:end].copy_(
                torch.matmul(probs, values_float).squeeze(1).to(values.dtype)
            )
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        zero_allocator: BumpAllocator,
        block_residual: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.use_attn_residuals:
            prefix_sum = hidden_states
            if block_residual is None:
                block_residual = hidden_states.new_zeros(
                    hidden_states.shape[0], 0, hidden_states.shape[-1]
                )
            if block_residual.shape[1] > 0:
                hidden_states = self._apply_attn_res(
                    prefix_sum,
                    block_residual,
                    self.self_attention_res_proj,
                    self.self_attention_res_norm,
                )
            if self.layer_idx % self.attn_res_block_size == 0:
                block_residual = torch.cat(
                    (block_residual, prefix_sum.unsqueeze(1)), dim=1
                )
                prefix_sum = None

            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
                forward_batch=forward_batch,
                zero_allocator=zero_allocator,
            )
            prefix_sum = (
                hidden_states if prefix_sum is None else prefix_sum + hidden_states
            )
            hidden_states = self._apply_attn_res(
                prefix_sum,
                block_residual,
                self.mlp_res_proj,
                self.mlp_res_norm,
            )
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            prefix_sum = prefix_sum + hidden_states
            return prefix_sum, None, block_residual

        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual, block_residual


class KimiLinearModel(nn.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        self.config = config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.alt_stream = torch.cuda.Stream()

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: KimiDecoderLayer(
                layer_idx=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=self.alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=f"{prefix}.layers",
        )

        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.use_attn_residuals = (
            getattr(config, "attn_res_block_size", None) is not None
        )
        if self.use_attn_residuals:
            if self.pp_group.is_last_rank:
                self.output_attn_res_norm = RMSNorm(
                    config.hidden_size, eps=config.rms_norm_eps
                )
                self.output_attn_res_proj = ReplicatedLinear(
                    config.hidden_size,
                    1,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.output_attn_res_proj",
                )
            else:
                self.output_attn_res_norm = PPMissingLayer()
                self.output_attn_res_proj = PPMissingLayer()

        world_size = get_tensor_model_parallel_world_size()
        assert config.num_attention_heads % world_size == 0, (
            "num_attention_heads must be divisible by world_size"
        )

    def get_pp_proxy_tensor_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Per-token shapes of the proxy tensors this stage receives.

        With attention residuals the boundary carries `block_residual` instead of
        `residual`: one stacked prefix-sum per residual block, so its width is the
        number of blocks opened before this stage's first layer. Layer indices are
        global and a block opens whenever ``idx % attn_res_block_size == 0``.
        CUDA graph capture allocates these buffers up front, so the shapes have to
        be derivable without running a forward.
        """
        hidden_size = self.config.hidden_size
        if not self.use_attn_residuals:
            return {
                "hidden_states": (hidden_size,),
                "residual": (hidden_size,),
            }
        block_size = self.config.attn_res_block_size
        num_blocks = -(-self.start_layer // block_size)
        return {
            "hidden_states": (hidden_size,),
            "block_residual": (num_blocks, hidden_size),
        }

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: torch.Tensor | None = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
            residual = None
            block_residual = (
                hidden_states.new_zeros(
                    hidden_states.shape[0], 0, hidden_states.shape[-1]
                )
                if self.use_attn_residuals
                else None
            )
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            block_residual = (
                pp_proxy_tensors["block_residual"] if self.use_attn_residuals else None
            )
            residual = None if self.use_attn_residuals else pp_proxy_tensors["residual"]

        total_num_layers = self.end_layer - self.start_layer
        device = hidden_states.device
        zero_allocator = BumpAllocator(
            buffer_size=total_num_layers * 2,
            dtype=torch.float32,
            device=device,
        )
        # TODO: capture aux hidden states
        aux_hidden_states = []
        for i in range(self.start_layer, self.end_layer):
            ctx = get_global_expert_distribution_recorder().with_current_layer(i)
            with ctx:
                layer = self.layers[i]
                hidden_states, residual, block_residual = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    forward_batch=forward_batch,
                    residual=residual,
                    zero_allocator=zero_allocator,
                    block_residual=block_residual,
                )

        if not self.pp_group.is_last_rank:
            tensors = {"hidden_states": hidden_states}
            if self.use_attn_residuals:
                tensors["block_residual"] = block_residual
            else:
                tensors["residual"] = residual
            return PPProxyTensors(tensors)
        else:
            if hidden_states.shape[0] != 0:
                if self.use_attn_residuals:
                    hidden_states = KimiDecoderLayer._apply_attn_res(
                        hidden_states,
                        block_residual,
                        self.output_attn_res_proj,
                        self.output_attn_res_norm,
                    )
                    hidden_states = self.norm(hidden_states)
                elif residual is None:
                    hidden_states = self.norm(hidden_states)
                else:
                    hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states


class KimiLinearForCausalLM(nn.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = KimiLinearModel(
            config, quant_config, prefix=maybe_prefix(prefix, "model")
        )
        self.pp_group = get_pp_group()
        if self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(config=config, logit_scale=logit_scale)

    def get_pp_proxy_tensor_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return self.model.get_pp_proxy_tensor_shapes()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            inputs_embeds,
            pp_proxy_tensors,
        )
        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
        else:
            return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
            # Fused path
            (".fused_qkvbfg_a_proj", ".q_proj", 0),
            (".fused_qkvbfg_a_proj", ".k_proj", 1),
            (".fused_qkvbfg_a_proj", ".v_proj", 2),
            (".fused_qkvbfg_a_proj", ".b_proj", 3),
            (".fused_qkvbfg_a_proj", ".f_a_proj", 4),
            (".fused_qkvbfg_a_proj", ".g_a_proj", 5),
            (".fused_fg_b_proj", ".f_b_proj", 0),
            (".fused_fg_b_proj", ".g_b_proj", 1),
            # Unfused path: separate qkv_proj (when do_fuse_qkvbfg=False)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            # qkv conv fuse
            (".qkv_conv1d", ".q_conv1d", 0),
            (".qkv_conv1d", ".k_conv1d", 1),
            (".qkv_conv1d", ".v_conv1d", 2),
        ]
        params_dict = dict(self.named_parameters())
        cached_a_proj: dict[str, torch.Tensor] = {}
        loaded_params: set[str] = set()
        for args in weights:
            name, loaded_weight = args[:2]
            kwargs = args[2] if len(args) > 2 else {}
            if "rotary_emb.inv_freq" in name:
                continue

            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                if param_name in {
                    ".fused_qkvbfg_a_proj",
                    ".fused_fg_b_proj",
                } or weight_name in {".q_proj", ".k_proj", ".v_proj"}:
                    layer_id = int(name.split(".")[2])
                    if not self.model.start_layer <= layer_id < self.model.end_layer:
                        continue
                    if not self.config.is_kda_layer(layer_id):
                        continue
                # Check if this mapping targets a fused projection (only apply fusion check to fused params)
                if param_name in {".fused_qkvbfg_a_proj", ".fused_fg_b_proj"}:
                    layer = self.model.layers[layer_id].self_attn
                    # Only load to fused projection if fusion is enabled
                    if not getattr(layer, "do_fuse_qkvbfg", False):
                        continue
                name = name.replace(weight_name, param_name)
                if name not in params_dict:
                    continue
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # if is_pp_missing_parameter(name, self):
                #     continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                expert_parts = None
                if ".experts." in name:
                    expert_prefix, expert_tail = name.split(".experts.", 1)
                    parts = expert_tail.split(".", 2)
                    if (
                        len(parts) == 3
                        and parts[0].isdigit()
                        and parts[1] in {"w1", "w2", "w3"}
                    ):
                        expert_parts = (
                            expert_prefix,
                            int(parts[0]),
                            parts[1],
                            parts[2],
                        )
                if expert_parts is not None:
                    expert_prefix, expert_id, shard_id, param_suffix = expert_parts
                    fused_name = "w13_" if shard_id in {"w1", "w3"} else "w2_"
                    name = f"{expert_prefix}.experts.{fused_name}{param_suffix}"
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        expert_id=expert_id,
                        shard_id=shard_id,
                    )
                else:
                    if ".q_a_proj." in name or ".kv_a_proj_with_mqa." in name:
                        fused_name = (
                            name.replace(
                                ".q_a_proj.",
                                ".fused_qkv_a_proj_with_mqa.",
                            )
                            if ".q_a_proj." in name
                            else name.replace(
                                ".kv_a_proj_with_mqa.",
                                ".fused_qkv_a_proj_with_mqa.",
                            )
                        )
                        if fused_name not in params_dict:
                            continue
                        q_name = (
                            name
                            if ".q_a_proj." in name
                            else name.replace(".kv_a_proj_with_mqa.", ".q_a_proj.")
                        )
                        kv_name = (
                            name
                            if ".kv_a_proj_with_mqa." in name
                            else name.replace(".q_a_proj.", ".kv_a_proj_with_mqa.")
                        )
                        cached_a_proj[name] = loaded_weight
                        if q_name in cached_a_proj and kv_name in cached_a_proj:
                            fused_weight = torch.cat(
                                [cached_a_proj.pop(q_name), cached_a_proj.pop(kv_name)],
                                dim=0,
                            )
                            param = params_dict[fused_name]
                            weight_loader = getattr(
                                param, "weight_loader", default_weight_loader
                            )
                            weight_loader(param, fused_weight)
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias")
                        and name not in params_dict
                        and not self.config.is_linear_attn
                    ):  # noqa: E501
                        continue
                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                    if name not in params_dict:
                        continue
                    # if is_pp_missing_parameter(name, self):
                    #     continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, **kwargs)
            loaded_params.add(name)

        if cached_a_proj:
            raise ValueError(
                "Missing paired MLA q_a/kv_a projection weights for "
                f"{sorted(cached_a_proj)}"
            )

        for layer_id in self.config.full_attention_layer_ids:
            if not self.model.start_layer <= layer_id < self.model.end_layer:
                continue
            self_attn = self.model.layers[layer_id].self_attn
            w_kc, w_vc = self_attn.kv_b_proj.weight.unflatten(
                0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
            ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
            self_attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
            self_attn.w_vc = w_vc.contiguous().transpose(1, 2)
            if hasattr(self_attn.kv_b_proj, "weight_scale"):
                self_attn.w_scale = self_attn.kv_b_proj.weight_scale


class KimiK3ForConditionalGeneration(KimiLinearForCausalLM):
    """Text-only Kimi-K3 runtime.

    The official conditional-generation checkpoint also contains vision
    modules. They are intentionally excluded from this language-only runtime;
    every other unexpected top-level tensor is rejected.
    """

    _TEXT_PREFIX = "language_model."
    _IGNORED_MULTIMODAL_PREFIXES = (
        "vision_tower.",
        "multi_modal_projector.",
        "mm_projector.",
    )

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        text_config = as_kimi_linear_config(config.get_text_config())
        super().__init__(text_config, quant_config=quant_config, prefix=prefix)
        self.kimi_k3_config = config

    @classmethod
    def adapt_checkpoint_name(cls, name: str) -> Optional[str]:
        if name.startswith(cls._IGNORED_MULTIMODAL_PREFIXES):
            return None
        if not name.startswith(cls._TEXT_PREFIX):
            raise ValueError(f"Unexpected Kimi-K3 checkpoint tensor: {name}")
        name = name.removeprefix(cls._TEXT_PREFIX)
        if name.startswith("model.layers.93."):
            raise ValueError(
                "MTP layer weights are not supported by the base Kimi-K3 "
                f"runtime: {name}"
            )
        return name.replace(".weight_packed", ".weight")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        def adapted_weights():
            for args in weights:
                name, loaded_weight = args[:2]
                name = self.adapt_checkpoint_name(name)
                if name is None:
                    continue
                yield (name, loaded_weight, *args[2:])

        return super().load_weights(adapted_weights())


EntryClass = KimiK3ForConditionalGeneration
