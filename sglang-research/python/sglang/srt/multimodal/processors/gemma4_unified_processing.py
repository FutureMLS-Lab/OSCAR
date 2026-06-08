# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Vendored ``Gemma4UnifiedProcessor`` (transformers 5.9-compatible).

Ties the GemmaTokenizer + the vendored ``Gemma4UnifiedImageProcessor`` and performs
``<|image|>`` placeholder expansion + tokenization, mirroring the transformers 5.10.1
``Gemma4UnifiedProcessor.__call__`` (which is unavailable in the pinned 5.9 env).

Registered for ``model_type == "gemma4_unified"`` so sglang's ``get_processor`` builds
this instead of the AutoProcessor (which would fall back to a bare tokenizer in 5.9).
"""

import json
import os
from typing import List, Optional, Union

import torch
from transformers import AutoTokenizer

from sglang.srt.multimodal.processors.gemma4_unified_image_processing import (
    Gemma4UnifiedImageProcessor,
)


class Gemma4UnifiedProcessor:
    """Minimal HF-style processor: ``.tokenizer``, ``.image_processor``, ``__call__``.

    Not derived from ``ProcessorMixin`` (its call/loading machinery diverges across the
    5.9/5.10 boundary); sglang only needs ``.tokenizer``, an optional ``.image_processor``,
    and a ``__call__(text=..., images=...)`` returning ``input_ids`` + image tensors.
    """

    def __init__(
        self,
        tokenizer,
        image_processor: Gemma4UnifiedImageProcessor,
        image_seq_length: int = 280,
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_seq_length = image_seq_length

        self.image_token = getattr(tokenizer, "image_token", "<|image|>")
        self.image_token_id = getattr(tokenizer, "image_token_id", None)
        if self.image_token_id is None:
            self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.boi_token = getattr(tokenizer, "boi_token", "<|image>")
        self.eoi_token = getattr(tokenizer, "eoi_token", "<image|>")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *args,
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
        **kwargs,
    ) -> "Gemma4UnifiedProcessor":
        # Strip kwargs the bare image processor / tokenizer do not accept.
        kwargs.pop("use_fast", None)

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )

        # Load image_processor config from processor_config.json if present.
        img_cfg: dict = {}
        image_seq_length = 280
        path = pretrained_model_name_or_path
        proc_cfg_path = None
        if os.path.isdir(path):
            cand = os.path.join(path, "processor_config.json")
            if os.path.exists(cand):
                proc_cfg_path = cand
        else:
            try:
                from huggingface_hub import hf_hub_download

                proc_cfg_path = hf_hub_download(
                    path, "processor_config.json", revision=revision
                )
            except Exception:
                proc_cfg_path = None
        if proc_cfg_path is not None:
            with open(proc_cfg_path) as f:
                pc = json.load(f)
            img_cfg = pc.get("image_processor", {}) or {}
            image_seq_length = pc.get("image_seq_length", image_seq_length)

        image_processor = Gemma4UnifiedImageProcessor.from_dict(img_cfg)
        return cls(
            tokenizer=tokenizer,
            image_processor=image_processor,
            image_seq_length=image_seq_length,
        )

    def _expand_placeholders(self, text: str, num_soft_tokens: List[int]) -> str:
        """Replace each ``<|image|>`` with ``<boi> <image>*N <eoi>`` in order."""
        idx = 0
        out_parts = []
        cursor = 0
        token = self.image_token
        while True:
            pos = text.find(token, cursor)
            if pos == -1:
                out_parts.append(text[cursor:])
                break
            out_parts.append(text[cursor:pos])
            n = num_soft_tokens[idx]
            out_parts.append(
                f"{self.boi_token}{self.image_token * n}{self.eoi_token}"
            )
            idx += 1
            cursor = pos + len(token)
        if idx != len(num_soft_tokens):
            raise ValueError(
                f"Found {idx} `{token}` placeholders in text but {len(num_soft_tokens)} "
                "images were processed."
            )
        return "".join(out_parts)

    def __call__(
        self,
        text: Union[str, List[str]],
        images=None,
        videos=None,
        audios=None,
        audio=None,
        return_tensors: Optional[str] = "pt",
        padding: bool = True,
        **kwargs,
    ) -> dict:
        if isinstance(text, str):
            text = [text]
        if len(text) != 1:
            raise ValueError(
                "Gemma4UnifiedProcessor (sglang) handles one prompt at a time."
            )
        prompt = text[0]

        out: dict = {}
        if images:
            if not isinstance(images, (list, tuple)):
                images = [images]
            img_out = self.image_processor(images, return_tensors="pt")
            out["pixel_values"] = img_out["pixel_values"]
            out["image_position_ids"] = img_out["image_position_ids"]
            prompt = self._expand_placeholders(
                prompt, img_out["num_soft_tokens_per_image"]
            )

        enc = self.tokenizer(prompt, return_tensors=return_tensors, padding=padding)
        out["input_ids"] = enc["input_ids"]
        if "attention_mask" in enc:
            out["attention_mask"] = enc["attention_mask"]
        return out


__all__ = ["Gemma4UnifiedProcessor"]
