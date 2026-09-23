# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""sglang multimodal processor for ``Gemma4UnifiedForConditionalGeneration``.

Image-only (the unified audio modality is not served).  Mirrors the gemma4.py
processor but binds to the encoder-free unified model and the vendored
``Gemma4UnifiedProcessor`` (which produces ``pixel_values`` = raw merged patches and
``image_position_ids``).
"""

from typing import Dict, List, Optional, Union

from sglang.srt.managers.multimodal_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalProcessorOutput
from sglang.srt.models.gemma4_unified import Gemma4UnifiedForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens


class Gemma4UnifiedSGLangProcessor(SGLangBaseProcessor):
    """Image multimodal processor for the encoder-free gemma4_unified model."""

    models = [Gemma4UnifiedForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        self.IM_START_TOKEN_ID = hf_config.boi_token_id
        self.IM_END_TOKEN_ID = hf_config.eoi_token_id

        self.mm_tokens = MultimodalSpecialTokens(
            image_token_id=hf_config.image_token_id,
        ).build(_processor)

        # Persist the image processor's position-id output onto MultimodalDataItem.
        self.ATTR_NAME_TO_MODALITY["image_position_ids"] = Modality.IMAGE

    async def process_mm_data_async(
        self,
        image_data: Optional[List[Union[str, bytes, Dict]]] = None,
        audio_data: Optional[List[Union[str, bytes, Dict]]] = None,
        input_text: str = "",
        request_obj=None,
        *args,
        **kwargs,
    ):
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
        )

        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_token_id=self.mm_tokens.image_token_id,
        )
