# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Serving for ``google/gemma-4-*`` (``Gemma4UnifiedForConditionalGeneration``).

Two serving modes share one entry class:

* **Text-only** (``--enable-multimodal`` off, the arch stays in ``mm_disabled_models``):
  behaves exactly like the original text shim — a thin subclass of
  ``Gemma4ForCausalLM`` over the unwrapped ``text_config``.  This is the path the
  INT2 OSCAR KV-cache GPQA result uses; it is unchanged.

* **Multimodal** (``--enable-multimodal`` on): builds the text backbone via
  ``Gemma4ForCausalLM`` **plus** the encoder-free vision embedder, and merges image
  features into the text embedding stream at ``<|image|>`` positions via
  ``general_mm_embed_routine``.  Vision tokens become ordinary embeddings in the text
  stream, so the INT2 two-group KV pool in the text decoder is unaffected.

Gemma4's vision is **encoder-free**: there is no vision transformer.  Raw merged pixel
patches (48x48x3 = 6912) are projected to LM space by
``vision_embedder`` (LN -> Dense -> LN -> +factorized-posemb -> LN) followed by
``embed_vision`` (RMSNorm[no-scale] -> Linear).  See the HF reference
``Gemma4UnifiedVisionEmbedder`` / ``Gemma4UnifiedMultimodalEmbedder``.
"""

import logging
import os
import re
from typing import Iterable, List, Optional, Set, Tuple

import torch
from torch import nn

from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.layernorm import Gemma4RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import general_mm_embed_routine
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    flatten_nested_list,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.models.gemma4_causal import Gemma4ForCausalLM
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class Gemma4UnifiedVisionEmbedder(nn.Module):
    """Encoder-free embedder: raw merged patches -> LM space.

    Weight names mirror the HF checkpoint exactly (no remap needed):
    ``vision_embedder.{patch_ln1,patch_dense,patch_ln2,pos_embedding,pos_norm}`` and
    ``embed_vision.embedding_projection``.
    """

    def __init__(
        self,
        vision_config,
        text_config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        patch_dim = vision_config.model_patch_size**2 * 3  # 48*48*3 = 6912
        mm_embed_dim = vision_config.mm_embed_dim  # 3840
        out_proj_dim = getattr(vision_config, "output_proj_dims", mm_embed_dim)
        text_hidden = text_config.hidden_size  # 3840
        eps = getattr(vision_config, "rms_norm_eps", 1e-6)

        # vision_embedder.* (checkpoint-native names)
        self.vision_embedder = nn.Module()
        self.vision_embedder.patch_ln1 = nn.LayerNorm(patch_dim)
        self.vision_embedder.patch_dense = nn.Linear(patch_dim, mm_embed_dim)
        self.vision_embedder.patch_ln2 = nn.LayerNorm(mm_embed_dim)
        self.vision_embedder.pos_embedding = nn.Parameter(
            torch.zeros(vision_config.mm_posemb_size, 2, mm_embed_dim)
        )
        self.vision_embedder.pos_norm = nn.LayerNorm(mm_embed_dim)

        # embed_vision.embedding_projection (+ parameter-free pre-projection RMSNorm)
        self.embed_vision = nn.Module()
        self.embed_vision.embedding_pre_projection_norm = Gemma4RMSNorm(
            out_proj_dim, eps=eps, with_scale=False
        )
        self.embed_vision.embedding_projection = ReplicatedLinear(
            out_proj_dim,
            text_hidden,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("embed_vision.embedding_projection", prefix),
        )

    def forward(
        self, pixel_values: torch.Tensor, image_position_ids: torch.Tensor
    ) -> torch.Tensor:
        """Returns (batch, num_patches, text_hidden) including padded positions."""
        ve = self.vision_embedder
        dtype = ve.patch_dense.weight.dtype

        hidden = ve.patch_ln1(pixel_values.to(dtype))
        hidden = ve.patch_dense(hidden)
        hidden = ve.patch_ln2(hidden)

        clamped = image_position_ids.clamp(min=0).long()
        valid = (image_position_ids != -1).to(ve.pos_embedding.dtype).unsqueeze(-1)
        axes = torch.arange(2, device=image_position_ids.device)
        pos_embs = (ve.pos_embedding[clamped, axes] * valid).sum(-2)
        hidden = hidden + pos_embs
        hidden = ve.pos_norm(hidden)

        normed = self.embed_vision.embedding_pre_projection_norm(hidden)
        out, _ = self.embed_vision.embedding_projection(normed)
        return out


class Gemma4UnifiedForConditionalGeneration(Gemma4ForCausalLM):
    """Text + (optional) vision serving for ``gemma4_unified``.

    When ``config.vision_config`` is present *and* ``--enable-multimodal`` is on, the
    vision embedder is built and image features are merged into the text stream.  When
    multimodal is off (the arch is in ``mm_disabled_models``), ``vision_config`` is
    still parsed but the model is used purely as text — identical to the original shim.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        # Composite Gemma4UnifiedConfig carries the text backbone in ``.text_config``;
        # fall back to ``config`` itself if a flat text config is passed.
        self._composite_config = config
        text_config = getattr(config, "text_config", None) or config
        super().__init__(config=text_config, quant_config=quant_config, prefix=prefix)

        # NOTE: do NOT store ``self.language_model = self`` — assigning a module to
        # itself makes it its own submodule (infinite recursion in named_modules).
        # ``general_mm_embed_routine`` takes ``language_model`` as a plain function
        # arg (no module registration), so we just pass ``self`` at the call site.

        # Build the vision embedder ONLY when multimodal is explicitly enabled.
        # ``gemma4_unified`` is in ``mm_disabled_models``, so the server only sets
        # ``enable_multimodal=True`` when the user passes ``--enable-multimodal``.
        # When off (the INT2 OSCAR text/GPQA path), keep the model byte-for-byte
        # identical to the original text-only shim: no vision params, no mm forward.
        from sglang.srt.server_args import get_global_server_args

        try:
            mm_enabled = bool(
                getattr(get_global_server_args(), "enable_multimodal", False)
            )
        except Exception:
            mm_enabled = False
        vision_config = getattr(config, "vision_config", None)
        self.embed_vision = None
        if mm_enabled and vision_config is not None:
            self.embed_vision = Gemma4UnifiedVisionEmbedder(
                vision_config,
                text_config,
                quant_config=quant_config,
                prefix=prefix,
            )

        # Multimodal token ids (read off the composite config).
        self.image_token_id = getattr(config, "image_token_id", None)
        self.video_token_id = getattr(config, "video_token_id", None)
        self.audio_token_id = getattr(config, "audio_token_id", None)

    # ---- multimodal plumbing ------------------------------------------------

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def pad_input_ids(
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
        from sglang.srt.managers.mm_utils import (
            MultiModalityDataPaddingPatternMultimodalTokens,
        )

        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Project raw merged pixel patches to LM space, stripping padding patches.

        Returns ``(total_valid_patches, text_hidden)`` — exactly one row per
        ``<|image|>`` placeholder token in the prompt.
        """
        assert self.embed_vision is not None, "vision embedder not built"
        device = self.embed_vision.vision_embedder.patch_dense.weight.device
        dtype = self.embed_vision.vision_embedder.patch_dense.weight.dtype

        all_embeds = []
        for item in items:
            pvs = flatten_nested_list([item.feature])
            ppos = flatten_nested_list([getattr(item, "image_position_ids", None)])

            for idx, pv in enumerate(pvs):
                # Pre-embedded passthrough (rare): already in text space.
                if pv.dim() in (2, 3) and pv.shape[-1] == self.config.hidden_size:
                    all_embeds.append(pv.to(device))
                    continue

                if idx >= len(ppos) or ppos[idx] is None:
                    raise ValueError(
                        f"pixel_values[{idx}] has no matching image_position_ids."
                    )
                pp = ppos[idx]
                if pv.dim() == 2:
                    pv = pv.unsqueeze(0)
                if pp.dim() == 2:
                    pp = pp.unsqueeze(0)

                pv = pv.to(device=device, dtype=dtype)
                pp = pp.to(device=device)

                feats = self.embed_vision(pv, pp)  # (b, num_patches, hidden)

                # Strip padding patches (position == -1 on both axes).
                padding_mask = (pp == -1).all(dim=-1)  # (b, num_patches)
                feats = feats[~padding_mask]  # (total_valid, hidden)
                all_embeds.append(feats)

        if all_embeds:
            return torch.cat(all_embeds, dim=0)
        return torch.empty(
            0, self.config.hidden_size, device=device, dtype=dtype
        )

    def prepare_attn_masks(
        self,
        forward_batch: ForwardBatch,
        input_ids: torch.Tensor,
        mask_dtype: torch.dtype,
    ):
        """Bidirectional attention within each image's soft-token span during prefill.

        Gemma4 uses bidirectional attention over vision tokens (``use_bidirectional_
        attention == "vision"``).  Only TritonAttnBackend supports a custom mask; for
        other backends we fall back to causal (mirrors gemma4_mm.py / gemma3_mm.py).
        """
        if not isinstance(forward_batch.attn_backend, TritonAttnBackend):
            logger.warning_once(
                "Bidirectional attention for image tokens requires TritonAttnBackend; "
                "falling back to causal attention."
            )
            return
        assert forward_batch.forward_mode == ForwardMode.EXTEND

        masks_list = []
        mask_indptr = torch.zeros(
            forward_batch.batch_size + 1, dtype=torch.int32, device=input_ids.device
        )
        for i in range(forward_batch.batch_size):
            extend_seq_len = forward_batch.extend_seq_lens[i]
            prefix_len = forward_batch.extend_prefix_lens[i]
            m = torch.zeros(
                extend_seq_len,
                extend_seq_len + prefix_len,
                dtype=mask_dtype,
                device=input_ids.device,
            )
            m.fill_(1)
            m = m.tril(diagonal=prefix_len)

            mm_inputs = forward_batch.mm_inputs[i]
            if mm_inputs is not None:
                for mm_item in mm_inputs.mm_items:
                    if mm_item.is_image():
                        for im_begin, im_end in mm_item.offsets:
                            if (
                                im_begin >= prefix_len
                                and im_end < prefix_len + extend_seq_len
                            ):
                                m[
                                    im_begin - prefix_len : im_end + 1 - prefix_len,
                                    im_begin : im_end + 1,
                                ] = 1
            masks_list.append(m.flatten())
            mask_indptr[i + 1] = mask_indptr[i] + m.nelement()

        if masks_list:
            forward_batch.attn_backend.forward_metadata.mask_indptr = mask_indptr
            forward_batch.attn_backend.forward_metadata.custom_mask = torch.cat(
                masks_list, dim=0
            )

    @torch.no_grad()
    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: Optional[torch.Tensor] = None,
        forward_batch: ForwardBatch = None,
        input_embeds: torch.Tensor = None,
        **kwargs,
    ):
        # ``general_mm_embed_routine`` re-enters as ``language_model(input_ids=None,
        # forward_batch=..., input_embeds=..., positions=...)``.  Detect that inner
        # call (input_ids is None / embeds supplied) and run the text backbone, which
        # returns *hidden states* (logits_processor is applied by the outer call).
        if positions is None:
            positions = kwargs.pop("positions", None)
        else:
            kwargs.pop("positions", None)

        # The inner re-entry from general_mm_embed_routine passes input_ids=None and a
        # pre-built input_embeds.  Run the text backbone (returns hidden states).
        if input_ids is None and input_embeds is not None:
            return self.model(
                input_ids, positions, forward_batch, input_embeds, **kwargs
            )

        # Text-only path (vision not built): identical to the original shim.
        if self.embed_vision is None:
            return super().forward(
                input_ids, positions, forward_batch, input_embeds, **kwargs
            )

        # Bidirectional attention over image soft tokens is a quality optimization but
        # is INCOMPATIBLE with the INT2 KV cache: setting forward_metadata.custom_mask
        # forces the generic ``extend_attention_fwd`` triton kernel (which does
        # ``tl.dot(q, k)`` on the raw K buffer) instead of the INT2 quantized-dense
        # prefill path — and that kernel rejects the INT2-packed (int8-storage) buffer
        # ("only int8 supported!").  So default to CAUSAL attention for image tokens
        # (a documented gemma3/gemma4 fallback) and only enable the bidirectional mask
        # when explicitly requested AND the KV cache is not INT2.
        kv_pool = getattr(forward_batch, "token_to_kv_pool", None)
        kv_is_int2 = getattr(kv_pool, "dtype", None) == "int2"
        want_bidir = os.environ.get("SGLANG_GEMMA4U_BIDIRECTIONAL_IMAGE", "0") == "1"
        if (
            want_bidir
            and not kv_is_int2
            and forward_batch.forward_mode == ForwardMode.EXTEND
            and forward_batch.contains_image_inputs()
        ):
            self.prepare_attn_masks(forward_batch, input_ids, mask_dtype=torch.bool)

        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self,
            data_embedding_funcs={Modality.IMAGE: self.get_image_feature},
            positions=positions,
            **kwargs,
        )
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    # ---- weight loading -----------------------------------------------------

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Route vision weights to the embedder; text weights to Gemma4ForCausalLM."""
        if self.embed_vision is None:
            return super().load_weights(weights)

        vision_params = dict(self.embed_vision.named_parameters())

        def _vision_local_name(name: str) -> Optional[str]:
            # Checkpoint: model.vision_embedder.* / model.embed_vision.embedding_projection.*
            n = name
            if n.startswith("model."):
                n = n[len("model.") :]
            if n.startswith("vision_embedder.") or n.startswith("embed_vision."):
                return n
            return None

        text_weights: List[Tuple[str, torch.Tensor]] = []
        loaded: Set[str] = set()
        for name, w in weights:
            # Drop audio modality entirely.
            if "embed_audio." in name or "audio_tower." in name or "audio" in name.split(".")[1:2]:
                continue
            local = _vision_local_name(name)
            if local is not None and local in vision_params:
                p = vision_params[local]
                wl = getattr(p, "weight_loader", None)
                if wl is not None:
                    wl(p, w)
                else:
                    assert p.shape == w.shape, f"{local}: {p.shape} vs {w.shape}"
                    p.data.copy_(w)
                loaded.add("embed_vision." + local)
                continue
            if local is not None:
                # vision-namespaced but unknown (e.g. audio embed) -> skip.
                continue
            text_weights.append((name, w))

        loaded |= super().load_weights(text_weights)
        return loaded


EntryClass = Gemma4UnifiedForConditionalGeneration
