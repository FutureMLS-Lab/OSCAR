"""Shim for upstream's model_executor.runner accessors.

Upstream moved a handful of runner-scoped helpers here. Only the one Kimi-K3
imports is provided, forwarding to where this fork still keeps it.
"""

from sglang.srt.model_executor.cuda_graph_runner import (  # noqa: F401
    get_is_capture_mode,
)

__all__ = ["get_is_capture_mode"]
