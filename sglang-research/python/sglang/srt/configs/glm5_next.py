"""Config for GLM-5.3-Flash (``glm5_next``).

transformers 5.3 does not know this architecture -- ``AutoConfig`` raises
"Transformers does not recognize this architecture" -- and the checkpoint ships
no remote code. sglang's own ``_CONFIG_REGISTRY`` is the designed way around
that lag, which is why this file exists rather than a transformers upgrade.

Shape, from the checkpoint: 45 layers alternating three KDA linear-attention
layers and one full-attention layer, the full ones being DeepSeek sparse
attention (``index_head_dim`` 128, ``index_topk`` 2048). The latent is
``kv_lora_rank`` 512 with ``qk_rope_head_dim`` **0** -- ``mla_use_nope`` is
True, so there is no positional half of the key at all.

That zero matters to the packed pool rather than only to the model: ``k_pe`` is
never quantized, so it is a fixed floor of 2 * qk_rope_head_dim bytes per token
per layer. At rope 64 that floor is 128 B of a 288 B cell, i.e. 44% of the row;
at rope 0 it vanishes and the same 2-bit packing gives 160 B against BF16's
1024, a **6.40x** pool ratio rather than 4.00x. Only 11 of 45 layers have a KV
cache at all, though, so the end-to-end saving is over a quarter of the model.
"""

from typing import List, Optional

from transformers.configuration_utils import PretrainedConfig


class Glm5NextTextConfig(PretrainedConfig):
    model_type = "glm5_next_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 154880,
        hidden_size: int = 4096,
        intermediate_size: int = 10944,
        num_hidden_layers: int = 45,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 64,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        tie_word_embeddings: bool = False,
        # MLA latent. qk_rope_head_dim is 0 for this model -- see the module
        # docstring; it is not a missing field.
        q_lora_rank: Optional[int] = 1536,
        kv_lora_rank: Optional[int] = 512,
        qk_nope_head_dim: Optional[int] = 256,
        qk_rope_head_dim: Optional[int] = 0,
        qk_head_dim: Optional[int] = 256,
        v_head_dim: Optional[int] = 256,
        mla_use_nope: bool = True,
        # DeepSeek sparse attention (DSA) selector. GLM-5.3-Flash's indexer is
        # NOT GLM-5.2's: it adds K-POOLING, so `index_kpool` and
        # `index_n_heads` are load-bearing rather than cosmetic.
        #
        # index_kpool defaults to 16 in the reference config and is 4 in this
        # checkpoint. Taking the default would size
        # index_kpool_compress_ape as [16, 128] against a stored [4, 128] --
        # that one at least fails loudly, unlike most of the traps here.
        index_head_dim: Optional[int] = 128,
        index_topk: Optional[int] = 2048,
        index_n_heads: int = 32,
        index_kpool: int = 16,
        index_kpool_always_select_tail: bool = True,
        # Layer mix. ``layer_types`` is authoritative -- see is_kda_layer.
        layer_types: Optional[List[str]] = None,
        linear_attn_config: Optional[dict] = None,
        mlp_layer_types: Optional[List[str]] = None,
        # MoE
        n_routed_experts: Optional[int] = 288,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 8,
        moe_intermediate_size: Optional[int] = 2048,
        routed_scaling_factor: float = 2.5,
        n_group: int = 1,
        topk_group: int = 1,
        topk_method: str = "noaux_tc",
        scoring_func: str = "sigmoid",
        norm_topk_prob: bool = True,
        first_k_dense_replace: int = 3,
        num_nextn_predict_layers: int = 1,
        mhc: bool = True,
        swiglu_limit: float = 10.0,
        # Manifold-Constrained Hyper-Connections (mHC, Xie et al. 2026).
        # Names match transformers 5.16.1's Glm5NextTextConfig so the model port
        # can follow the reference implementation field for field.
        hc_mult: int = 4,
        hc_eps: float = 1e-6,
        hc_sinkhorn_iters: int = 20,
        indexer_types: Optional[List[str]] = None,
        # Flat mirrors of linear_attn_config's nested keys. The reference
        # config converts the dict into these; this checkpoint still ships the
        # dict (its transformers_version says 5.16.0, but glm5_next only exists
        # in 5.16.1), so the conversion has to happen HERE or every KDA layer
        # would silently take the default geometry.
        linear_head_dim: int = 128,
        linear_num_heads: int = 64,
        linear_conv_kernel_dim: int = 4,
        linear_lower_bound: Optional[float] = -5.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.mla_use_nope = mla_use_nope
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.index_n_heads = index_n_heads
        self.index_kpool = index_kpool
        self.index_kpool_always_select_tail = index_kpool_always_select_tail

        self.layer_types = layer_types
        self.linear_attn_config = linear_attn_config
        self.mlp_layer_types = mlp_layer_types

        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.topk_method = topk_method
        self.scoring_func = scoring_func
        self.norm_topk_prob = norm_topk_prob
        self.first_k_dense_replace = first_k_dense_replace
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.mhc = mhc
        self.swiglu_limit = swiglu_limit
        self.hc_mult = hc_mult
        self.hc_eps = hc_eps
        self.hc_sinkhorn_iters = hc_sinkhorn_iters

        self.linear_head_dim = linear_head_dim
        self.linear_num_heads = linear_num_heads
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_lower_bound = linear_lower_bound
        if isinstance(linear_attn_config, dict):
            self.linear_head_dim = linear_attn_config.get("head_dim", linear_head_dim)
            self.linear_num_heads = linear_attn_config.get("num_heads", linear_num_heads)
            self.linear_conv_kernel_dim = linear_attn_config.get(
                "short_conv_kernel_size", linear_conv_kernel_dim
            )
            self.linear_lower_bound = linear_attn_config.get(
                "gate_lower_bound", linear_lower_bound
            )
            if linear_attn_config.get("safe_gate", True) and self.linear_lower_bound is None:
                self.linear_lower_bound = -5.0

        # Per-layer indexer mode. The reference derives it from
        # index_topk_pattern, or from index_topk_freq / index_skip_topk_offset.
        self.indexer_types = indexer_types
        if self.indexer_types is None:
            pattern = kwargs.get("index_topk_pattern")
            if pattern is not None:
                self.indexer_types = (
                    [{"F": "full", "S": "shared"}[ch] for ch in pattern]
                    if isinstance(pattern, str)
                    else list(pattern)
                )
            else:
                freq = max(kwargs.get("index_topk_freq", 1), 1)
                offset = kwargs.get("index_skip_topk_offset", 2)
                self.indexer_types = [
                    "full" if (max(i - offset + 1, 0) % freq) == 0 else "shared"
                    for i in range(num_hidden_layers)
                ]

        # The reference normalises "full_attention" to "deepseek_sparse_attention"
        # so downstream code has one spelling to test. Mirror it, or a checkpoint
        # written with the other spelling would take the linear-attention branch.
        if self.layer_types:
            self.layer_types = [
                "deepseek_sparse_attention" if t == "full_attention" else t
                for t in self.layer_types
            ]

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    # ---- layer mix -------------------------------------------------------
    #
    # Derived from ``layer_types``, NOT from ``linear_attn_config["kda_layers"]``,
    # because the two models that share this schema do not share its indexing.
    #
    # KimiLinearConfig.is_kda_layer tests ``(layer_idx + 1) in kda_layers`` --
    # Kimi's list is 1-indexed. GLM-5.3-Flash's is 0-indexed: its kda_layers
    # starts at 0 and agrees with layer_types position for position. Reusing
    # Kimi's predicate here mislabels **23 of 45 layers**, and it does not raise
    # -- it silently routes half the model to the wrong attention type. That is
    # the same failure shape as the K3 rotation-frame bug: two index spaces that
    # overlap enough to look plausible and never throw.
    #
    # ``layer_types`` is positional, so it cannot carry an indexing convention
    # to disagree about. Using it as the source of truth removes the question.

    def _types(self) -> List[str]:
        if self.layer_types:
            return list(self.layer_types)
        cfg = self.linear_attn_config or {}
        kda = set(cfg.get("kda_layers") or [])
        if not kda:
            return ["full_attention"] * self.num_hidden_layers
        # Fallback only. Assume the 0-indexed convention this checkpoint uses,
        # and say so loudly rather than guessing silently.
        return [
            "linear_attention" if i in kda else "deepseek_sparse_attention"
            for i in range(self.num_hidden_layers)
        ]

    def is_kda_layer(self, layer_idx: int) -> bool:
        return self._types()[layer_idx] == "linear_attention"

    @property
    def linear_layer_ids(self) -> List[int]:
        return [i for i in range(self.num_hidden_layers) if self.is_kda_layer(i)]

    @property
    def full_attention_layer_ids(self) -> List[int]:
        return [i for i in range(self.num_hidden_layers) if not self.is_kda_layer(i)]

    @property
    def is_mla(self) -> bool:
        return self.kv_lora_rank is not None

    @property
    def is_linear_attn(self) -> bool:
        return bool(self.linear_layer_ids)

    @property
    def mamba2_cache_params(self):
        from sglang.srt.configs.mamba_utils import (
            KimiLinearCacheParams,
            KimiLinearStateShape,
        )
        from sglang.srt.layers.dp_attention import get_attention_tp_size

        cfg = self.linear_attn_config or {}
        shape = KimiLinearStateShape.create(
            tp_world_size=get_attention_tp_size(),
            num_heads=cfg["num_heads"],
            head_dim=cfg["head_dim"],
            conv_kernel_size=cfg["short_conv_kernel_size"],
        )
        return KimiLinearCacheParams(shape=shape, layers=self.linear_layer_ids)


class Glm5NextVisionConfig(PretrainedConfig):
    model_type = "glm5_next_vision"

    def __init__(
        self,
        depth: int = 24,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        num_heads: int = 16,
        in_channels: int = 3,
        image_size: int = 448,
        patch_size: int = 14,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        out_hidden_size: int = 4096,
        projection_intermediate_size: int = 10240,
        rms_norm_eps: float = 1e-5,
        hidden_act: str = "silu",
        swiglu_limit: float = 10.0,
        attention_bias: bool = True,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        self.depth = depth
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.projection_intermediate_size = projection_intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.swiglu_limit = swiglu_limit
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        super().__init__(**kwargs)


class Glm5NextConfig(PretrainedConfig):
    model_type = "glm5_next"
    sub_configs = {
        "text_config": Glm5NextTextConfig,
        "vision_config": Glm5NextVisionConfig,
    }

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id: int = 154854,
        video_token_id: int = 154855,
        image_start_token_id: int = 154830,
        image_end_token_id: int = 154831,
        video_start_token_id: int = 154832,
        video_end_token_id: int = 154833,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        if isinstance(text_config, dict):
            text_config = Glm5NextTextConfig(**text_config)
        elif text_config is None:
            text_config = Glm5NextTextConfig()
        if isinstance(vision_config, dict):
            vision_config = Glm5NextVisionConfig(**vision_config)
        elif vision_config is None:
            vision_config = Glm5NextVisionConfig()
        self.text_config = text_config
        self.vision_config = vision_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id

        # sglang reads a lot of geometry off the TOP-level config (pool sizing,
        # attention backend selection, the packed-MLA guards). Mirroring the
        # text fields up here keeps those paths working without teaching each
        # of them about text_config -- the same thing other VL configs do.
        for name in (
            "vocab_size", "hidden_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "rms_norm_eps",
            "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim",
            "qk_rope_head_dim", "qk_head_dim", "v_head_dim", "mla_use_nope",
            "index_head_dim", "index_topk", "index_n_heads", "index_kpool",
            "index_kpool_always_select_tail", "indexer_types", "layer_types",
            "hc_mult", "hc_eps", "hc_sinkhorn_iters",
            "linear_head_dim", "linear_num_heads", "linear_conv_kernel_dim",
            "linear_lower_bound",
            "linear_attn_config", "num_nextn_predict_layers",
            "first_k_dense_replace", "n_routed_experts", "rope_theta",
            "rope_scaling", "moe_intermediate_size", "num_experts_per_tok",
        ):
            setattr(self, name, getattr(text_config, name, None))

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def is_kda_layer(self, layer_idx: int) -> bool:
        return self.text_config.is_kda_layer(layer_idx)

    @property
    def linear_layer_ids(self):
        return self.text_config.linear_layer_ids

    @property
    def full_attention_layer_ids(self):
        return self.text_config.full_attention_layer_ids

    @property
    def is_mla(self) -> bool:
        return self.text_config.is_mla

    @property
    def mamba2_cache_params(self):
        return self.text_config.mamba2_cache_params
