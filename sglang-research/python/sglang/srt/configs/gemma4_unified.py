# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Vendored config for ``google/gemma-4-*`` (``model_type: gemma4_unified``).

The HF checkpoints declare ``transformers 5.10.0.dev0``; the pinned transformers
in this env (5.3.0) does not know ``gemma4_unified``.  We register a compatible
config so the existing ``gemma4_causal.py`` (``Gemma4ForCausalLM``) can serve the
text backbone with no model-code changes.

Field-name remap (the only real mismatch):  ``gemma4_causal`` and ``ModelConfig``
follow the convention *base = full-attention layers, ``swa_*`` = sliding layers*.
The unified checkpoints instead use ``global_*`` for the full-attention geometry
and the plain names for the sliding geometry.  We translate on the way in:

    sglang ``head_dim``               <- unified ``global_head_dim``        (512, full)
    sglang ``swa_head_dim``           <- unified ``head_dim``               (256, sliding)
    sglang ``num_key_value_heads``    <- unified ``num_global_key_value_heads`` (1, full)
    sglang ``swa_num_key_value_heads``<- unified ``num_key_value_heads``     (8, sliding)
"""

from typing import Any, Optional, Union

from transformers.configuration_utils import PretrainedConfig

from sglang.srt.multimodal.customized_mm_processor_utils import (
    register_customized_processor,
)
from sglang.srt.multimodal.processors.gemma4_unified_processing import (
    Gemma4UnifiedProcessor,
)


class Gemma4UnifiedTextConfig(PretrainedConfig):
    model_type = "gemma4_unified_text"

    def __init__(
        self,
        vocab_size: int = 262144,
        hidden_size: int = 3840,
        intermediate_size: int = 15360,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 16,
        # --- unified names (sliding geometry on the plain names) ---
        num_key_value_heads: int = 8,
        head_dim: int = 256,
        # --- unified names (full-attention geometry on the global names) ---
        num_global_key_value_heads: int = 1,
        global_head_dim: int = 512,
        hidden_activation: str = "gelu_pytorch_tanh",
        max_position_embeddings: int = 262144,
        rms_norm_eps: float = 1e-6,
        sliding_window: int = 1024,
        layer_types: Optional[list[str]] = None,
        rope_parameters: Optional[dict[str, Any]] = None,
        final_logit_softcapping: Optional[float] = 30.0,
        attention_k_eq_v: bool = True,
        num_kv_shared_layers: int = 0,
        use_double_wide_mlp: bool = False,
        use_bidirectional_attention: Optional[str] = "vision",
        hidden_size_per_layer_input: int = 0,
        enable_moe_block: bool = False,
        num_experts: Optional[int] = None,
        moe_intermediate_size: Optional[int] = None,
        top_k_experts: Optional[int] = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # Remap unified -> sglang convention (base = full, swa = sliding).
        # K and V share head_dim within each layer type (gemma uses v_norm on V,
        # no separate v projection dim), so v_head_dim mirrors head_dim.
        self.head_dim = global_head_dim
        self.swa_head_dim = head_dim
        self.v_head_dim = global_head_dim
        self.swa_v_head_dim = head_dim
        self.num_key_value_heads = num_global_key_value_heads
        self.swa_num_key_value_heads = num_key_value_heads

        self.hidden_activation = hidden_activation
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.sliding_window = sliding_window

        if layer_types is None:
            # default Gemma 5:1 sliding:full pattern, last layer full
            layer_types = [
                "sliding_attention" if (i + 1) % 6 else "full_attention"
                for i in range(num_hidden_layers)
            ]
            layer_types[-1] = "full_attention"
        self.layer_types = layer_types

        if rope_parameters is None:
            rope_parameters = {
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "full_attention": {
                    "rope_type": "proportional",
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0,
                },
            }
        self.rope_parameters = rope_parameters

        self.final_logit_softcapping = final_logit_softcapping
        self.attention_k_eq_v = attention_k_eq_v
        self.num_kv_shared_layers = num_kv_shared_layers
        self.use_double_wide_mlp = use_double_wide_mlp
        self.use_bidirectional_attention = use_bidirectional_attention
        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        self.enable_moe_block = enable_moe_block
        self.num_experts = num_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.top_k_experts = top_k_experts
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        super().__init__(**kwargs)


class Gemma4UnifiedVisionConfig(PretrainedConfig):
    """Encoder-free vision config (gemma4_unified_vision).

    Carries the geometry the vision embedder needs: ``model_patch_size`` (48),
    ``mm_embed_dim`` (3840), ``mm_posemb_size`` (1120), ``output_proj_dims`` (3840).
    """

    model_type = "gemma4_unified_vision"

    def __init__(
        self,
        model_patch_size: int = 48,
        patch_size: int = 16,
        pooling_kernel_size: int = 3,
        mm_embed_dim: int = 3840,
        mm_posemb_size: int = 1120,
        output_proj_dims: int = 3840,
        num_soft_tokens: int = 280,
        rms_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        self.model_patch_size = model_patch_size
        self.patch_size = patch_size
        self.pooling_kernel_size = pooling_kernel_size
        self.mm_embed_dim = mm_embed_dim
        self.mm_posemb_size = mm_posemb_size
        self.output_proj_dims = output_proj_dims
        self.num_soft_tokens = num_soft_tokens
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        super().__init__(**kwargs)


@register_customized_processor(Gemma4UnifiedProcessor)
class Gemma4UnifiedConfig(PretrainedConfig):
    model_type = "gemma4_unified"
    sub_configs = {
        "text_config": Gemma4UnifiedTextConfig,
        "vision_config": Gemma4UnifiedVisionConfig,
    }

    def __init__(
        self,
        text_config: Optional[Union[dict, Gemma4UnifiedTextConfig]] = None,
        vision_config: Optional[Union[dict, Gemma4UnifiedVisionConfig]] = None,
        audio_config: Optional[dict] = None,
        boi_token_id: int = 255999,
        eoi_token_id: int = 258882,
        image_token_id: int = 258880,
        video_token_id: int = 258884,
        boa_token_id: int = 256000,
        eoa_token_index: int = 258883,
        audio_token_id: int = 258881,
        **kwargs,
    ):
        if text_config is None:
            text_config = Gemma4UnifiedTextConfig()
        elif isinstance(text_config, dict):
            text_config = Gemma4UnifiedTextConfig(**text_config)
        self.text_config = text_config
        # Expose a structured vision config (encoder-free embedder reads geometry off
        # it); keep audio raw (audio modality not served).
        if isinstance(vision_config, dict):
            vision_config = Gemma4UnifiedVisionConfig(**vision_config)
        self.vision_config = vision_config
        self.audio_config = audio_config

        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.boa_token_id = boa_token_id
        self.eoa_token_index = eoa_token_index
        self.audio_token_id = audio_token_id

        # Convenience mirror used by some sglang sizing paths.
        self.hidden_size = text_config.hidden_size

        super().__init__(**kwargs)


__all__ = [
    "Gemma4UnifiedConfig",
    "Gemma4UnifiedTextConfig",
    "Gemma4UnifiedVisionConfig",
]
