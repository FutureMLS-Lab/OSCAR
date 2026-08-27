"""GLM-5.3-Flash's KDA linear-attention layer, reusing Kimi's implementation.

GLM-5.3-Flash and Kimi Linear both interleave Kimi Delta Attention with a full
attention layer, and their KDA weight names line up one for one (``q/k/v_proj``,
``q/k/v_conv1d``, ``b_proj``, ``f_a_proj``/``f_b_proj``/``dt_bias``/``A_log``,
``g_a_proj``/``g_b_proj``, ``o_norm``, ``o_proj``). So the 240-line
``KimiDeltaAttention`` is reused rather than copied: a copy would drift from
upstream and the difference between the two models would stop being visible.

**There is exactly one incompatibility, and it is silent.**
``KimiDeltaAttention`` sets ``self.head_v_dim = config.v_head_dim``. For Kimi
that field IS the KDA value head width. For GLM-5.3-Flash it is the **MLA**
value head width, 256, while the KDA head is **128** -- the checkpoint's
``v_proj`` is [8192, 4096] = 64 heads x 128. Handing the real config straight
through builds every KDA layer at double the value width: shapes that are
self-consistent inside the module, a load that fails only if someone checks,
and no exception.

A shim rather than a subclass because the incompatibility is in a value read
during ``__init__``, not in a method that could be overridden. The class was
audited for exactly which config fields it touches -- ``dtype``,
``linear_attn_config`` and ``v_head_dim``, plus the dict keys ``head_dim``,
``num_heads``, ``short_conv_kernel_size`` -- so the shim's surface is known and
small rather than assumed. ``_KdaConfigView`` forwards everything else to the
real config, so a future upstream change that reads a new field keeps working;
only the audited three are intercepted.
"""

from __future__ import annotations

from typing import Any, Optional

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.models.kimi_linear import KimiDeltaAttention

# The exact set audited in kimi_linear.py's KimiDeltaAttention. If upstream
# starts reading another field, the view forwards it untouched -- but these are
# the ones whose meaning differs between the two models.
_AUDITED_FIELDS = ("dtype", "linear_attn_config", "v_head_dim")


class _KdaConfigView:
    """Presents a Glm5NextTextConfig the way KimiDeltaAttention expects one."""

    def __init__(self, config: Any):
        self._config = config

    @property
    def v_head_dim(self) -> int:
        # THE override. `linear_head_dim`, not the MLA `v_head_dim`.
        return self._config.linear_head_dim

    @property
    def linear_attn_config(self) -> dict:
        # Built from the flattened fields rather than passed through, because
        # this checkpoint's nested dict is the pre-5.16.1 spelling and a future
        # one may drop it. The flat fields are normalised in the config class,
        # so they are the single source of truth for KDA geometry.
        return {
            "head_dim": self._config.linear_head_dim,
            "num_heads": self._config.linear_num_heads,
            "short_conv_kernel_size": self._config.linear_conv_kernel_dim,
            "gate_lower_bound": self._config.linear_lower_bound,
        }

    def __getattr__(self, name: str) -> Any:
        # Only reached for names not defined above, so the two overrides always
        # win and everything else is the real config.
        return getattr(self._config, name)


class Glm5NextKdaAttention(KimiDeltaAttention):
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
        rms_norm_eps: float = 1e-5,
        prefix: str = "",
        **kwargs,
    ) -> None:
        view = _KdaConfigView(config)
        super().__init__(
            layer_idx=layer_idx,
            hidden_size=hidden_size,
            config=view,
            quant_config=quant_config,
            rms_norm_eps=rms_norm_eps,
            prefix=prefix,
            **kwargs,
        )
        # Assert the thing the shim exists to guarantee, at build time, rather
        # than discovering it as a shape error 45 layers into a 306 GB load.
        expected = config.linear_head_dim
        if self.head_v_dim != expected:
            raise ValueError(
                f"KDA value head width is {self.head_v_dim}, expected "
                f"{expected} (config.linear_head_dim). The config view did not "
                f"take effect; config.v_head_dim={getattr(config, 'v_head_dim', None)} "
                f"is the MLA width and must NOT be used here."
            )
