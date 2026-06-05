# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Text serving for ``google/gemma-4-*`` (``Gemma4UnifiedForConditionalGeneration``).

The unified checkpoints share the Gemma-4 text backbone already implemented by
``Gemma4ForCausalLM`` (gemma4_causal.py): sandwich norms, ``layer_scalar``,
per-head qk/v RMSNorm, ``attention_k_eq_v`` (full layers reuse k_proj as v), the
``model.language_model.`` weight prefix, and the sliding/full hybrid-SWA cache.

The only thing this shim does is unwrap the composite multimodal config down to
its text sub-config (the vision/audio towers are not built for text serving) and
register the unified architecture name so the model registry resolves it.
"""

from typing import Optional

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.models.gemma4_causal import Gemma4ForCausalLM


class Gemma4UnifiedForConditionalGeneration(Gemma4ForCausalLM):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        # Composite Gemma4UnifiedConfig carries the text backbone in `.text_config`;
        # fall back to `config` itself if a flat text config is passed.
        text_config = getattr(config, "text_config", None) or config
        super().__init__(
            config=text_config, quant_config=quant_config, prefix=prefix
        )


EntryClass = Gemma4UnifiedForConditionalGeneration
