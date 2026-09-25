# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Inference-only MiniMax-M3 (text-only, dense attention) for SGLang + OSCAR.

Ported from ``minimax_m2.py``. MiniMax-M3 shares the M2 GQA-MoE backbone but adds:
  * Gemma-style RMSNorm everywhere (``x_fp32 * rsqrt(mean(x^2)+eps) * (1 + w)``),
  * per-head QK-norm (over ``head_dim``) before partial RoPE,
  * the ``swigluoai`` gated activation (``(up+1) * gate * sigmoid(alpha*gate)`` with
    clamping) in the dense MLP, the shared expert, and the routed experts,
  * a mixed layout: the first ``first_k_dense`` layers are dense MLP, the rest are
    MoE with 1 always-on shared expert and ``routed_scaling_factor`` on the routed sum.

MiniMax Sparse Attention (MSA) runs DENSE here: the per-layer lightning indexer
weights (``self_attn.index_*``) are skipped, so attention attends the full KV cache.
The vision tower / multimodal projector are skipped (text-only serving). This keeps
the attention a plain dense GQA so OSCAR's INT2 KV pool attaches exactly as for M2.
"""

import logging
from typing import Iterable, Optional, Set, Tuple, Union

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)

# OSCAR calibration QKV dump (opt-in via env). See MiniMaxM3Attention._maybe_dump_qkv.
import os

_M3_DUMP_DIR = os.environ.get("DUMP_M3_QKV_DIR")
_M3_DUMP_CTR: dict = {}


def _get_text_config(config: PretrainedConfig) -> PretrainedConfig:
    """MiniMax-M3 ships a VL wrapper config; the text backbone lives in
    ``config.text_config``. Accept either the wrapper or a bare text config."""
    text_config = getattr(config, "text_config", None)
    return text_config if text_config is not None else config


class MiniMaxM3MLP(nn.Module):
    """SwiGLU-OAI gated MLP, used for the dense layers and the shared expert.

    ``out = down((up + 1) * gate * sigmoid(alpha * gate))`` with
    ``gate <= limit`` and ``|up| <= limit``. Gate/up are contiguous halves.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.swiglu_alpha = config.swiglu_alpha
        self.swiglu_limit = config.swiglu_limit
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        x = (up + 1.0) * gate * torch.sigmoid(gate * self.swiglu_alpha)
        out, _ = self.down_proj(x)
        return out


class MiniMaxM3MoE(nn.Module):
    """Routed experts (FusedMoE, swigluoai) + 1 shared expert.

    ``out = routed_scaling_factor * routed(x) + shared(x)``. The router uses
    independent sigmoid scoring with a per-expert correction bias for selection.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor

        self.e_score_correction_bias = nn.Parameter(
            torch.empty(config.num_local_experts, dtype=torch.float32)
        )
        self.e_score_correction_bias.weight_loader = self.ebias_weight_loader

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )

        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            renormalize=True,
            scoring_func=config.scoring_func,
            correction_bias=self.e_score_correction_bias,
            routed_scaling_factor=1.0,
        )

        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            layer_id=layer_id,
            quant_config=quant_config,
            activation="swigluoai",
            gemm1_alpha=config.swiglu_alpha,
            gemm1_clamp_limit=config.swiglu_limit,
            prefix=add_prefix("experts", prefix),
        )

        # Shared expert returns a per-rank partial sum (reduce_results=False);
        # it is summed with the partial routed output before a single all-reduce.
        self.shared_experts = MiniMaxM3MLP(
            config,
            intermediate_size=config.shared_intermediate_size,
            quant_config=quant_config,
            reduce_results=False,
            prefix=add_prefix("shared_experts", prefix),
        )

    @staticmethod
    def ebias_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight.to(torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Shared expert MUST run before self.experts(): the fused MoE kernel may
        # mutate hidden_states in place, which would corrupt the shared expert's
        # input if it ran afterward.
        shared_output = self.shared_experts(hidden_states)

        router_logits, _ = self.gate(hidden_states.to(torch.float32))
        topk_output = self.topk(hidden_states, router_logits)

        routed = self.experts(hidden_states, topk_output)
        final_hidden_states = routed * self.routed_scaling_factor + shared_output

        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)


class MiniMaxM3Attention(nn.Module):
    """GQA with per-head Gemma QK-norm and partial RoPE (dense; MSA indexer skipped)."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        # head_dim is explicit in the config and differs from hidden/heads.
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // self.total_num_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.rope_theta = getattr(config, "rope_theta", 5000000)
        self.rope_scaling = getattr(config, "rope_scaling", None)
        self.max_position_embeddings = getattr(
            config, "max_position_embeddings", 8192
        )
        self.rotary_dim = getattr(config, "rotary_dim", self.head_dim)

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
            rope_scaling=self.rope_scaling,
        )

        # Per-head Gemma RMSNorm (M3 qk_norm_type == "per_head").
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.layer_id = layer_id
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def _maybe_dump_qkv(self, q, k, v, forward_batch):
        # OSCAR calibration: on prefill, EVERY TP rank saves its post-RoPE q/k/v
        # per layer to layer_<id>/rank_<r>/{q,k,v}/<chunk>.pt (shaped
        # [tokens, local_heads, head_dim]). A merge step later gathers all ranks
        # into the full 64-q-head / 4-distinct-KV-head tensors compute_kv_rotation
        # expects. Enabled by DUMP_M3_QKV_DIR env.
        if _M3_DUMP_DIR is None or not forward_batch.forward_mode.is_extend():
            return
        from sglang.srt.distributed import get_tensor_model_parallel_rank

        r = get_tensor_model_parallel_rank()
        T = q.shape[0]
        out = {
            "q": q.detach().reshape(T, self.num_heads, self.head_dim).to(torch.float16).cpu(),
            "k": k.detach().reshape(T, self.num_kv_heads, self.head_dim).to(torch.float16).cpu(),
            "v": v.detach().reshape(T, self.num_kv_heads, self.head_dim).to(torch.float16).cpu(),
        }
        key = (self.layer_id, r)
        idx = _M3_DUMP_CTR.get(key, 1)
        for name, t in out.items():
            d = os.path.join(_M3_DUMP_DIR, f"layer_{self.layer_id}", f"rank_{r}", name)
            os.makedirs(d, exist_ok=True)
            torch.save(t, os.path.join(d, f"{idx}.pt"))
        _M3_DUMP_CTR[key] = idx + 1

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.reshape(-1, self.head_dim)).view(q.shape)
        k = self.k_norm(k.reshape(-1, self.head_dim)).view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        if _M3_DUMP_DIR is not None:
            self._maybe_dump_qkv(q, k, v, forward_batch)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class MiniMaxM3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = MiniMaxM3Attention(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

        first_k_dense = getattr(config, "first_k_dense_replace", 3)
        self.is_moe = layer_id >= first_k_dense
        if self.is_moe:
            self.block_sparse_moe = MiniMaxM3MoE(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("block_sparse_moe", prefix),
            )
        else:
            self.mlp = MiniMaxM3MLP(
                config,
                intermediate_size=config.dense_intermediate_size,
                quant_config=quant_config,
                reduce_results=True,
                prefix=add_prefix("mlp", prefix),
            )

        self.input_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        if self.is_moe:
            hidden_states = self.block_sparse_moe(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class MiniMaxM3Model(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [
                MiniMaxM3DecoderLayer(
                    config=config,
                    layer_id=i,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class MiniMaxM3ForCausalLM(nn.Module):
    """Text-only causal LM. ``self.model`` + ``self.lm_head`` mirror the
    ``language_model.{model,lm_head}`` checkpoint prefix."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = MiniMaxM3Model(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=None,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(config)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        # ckpt experts: w1=gate, w2=down, w3=up
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_local_experts,
        )

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # Routed experts (w1/w2/w3) are handled below; skip here so
                # gate_proj/up_proj remapping does not corrupt their names.
                if "block_sparse_moe.experts." in name:
                    continue
                new_name = name.replace(weight_name, param_name)
                if new_name.endswith(".bias") and new_name not in params_dict:
                    continue
                if new_name not in params_dict:
                    continue
                param = params_dict[new_name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(new_name)
                break
            else:
                for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    new_name = name.replace(weight_name, param_name)
                    if new_name not in params_dict:
                        continue
                    param = params_dict[new_name]
                    param.weight_loader(
                        param,
                        loaded_weight,
                        new_name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    loaded_params.add(new_name)
                    break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None or name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
        return loaded_params

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation

        text_config = _get_text_config(config)
        return ModelConfigForExpertLocation(
            num_layers=text_config.num_hidden_layers,
            num_logical_experts=text_config.num_local_experts,
            num_groups=None,
        )


class MiniMaxM3SparseForConditionalGeneration(nn.Module):
    """Entry class matching the HF architecture. Text-only: builds just the
    language model from ``config.text_config`` and skips the vision tower / the
    MSA lightning indexer."""

    # Vision / multimodal / sparse-indexer weights skipped for text-only dense.
    _SKIP_PREFIXES = (
        "vision_tower.",
        "multi_modal_projector.",
        "patch_merge_mlp.",
    )

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        text_config = _get_text_config(config)
        self.language_model = MiniMaxM3ForCausalLM(
            text_config,
            quant_config,
            prefix=add_prefix("language_model", prefix),
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.get_input_embeddings(input_ids)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.language_model(input_ids, positions, forward_batch, input_embeds)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        filtered = []
        for name, loaded_weight in weights:
            if name.startswith(self._SKIP_PREFIXES):
                continue
            # MiniMax Sparse Attention indexer (index_q_proj/index_k_proj/
            # index_q_norm/index_k_norm) — unused in dense mode.
            if ".index_" in name:
                continue
            if name.startswith("language_model."):
                name = name[len("language_model.") :]
            filtered.append((name, loaded_weight))
        return self.language_model.load_weights(filtered)


EntryClass = MiniMaxM3SparseForConditionalGeneration
