"""Availability probe for upstream's ``sglang.kernels`` package.

Upstream ships a compiled/JIT kernel package that this fork does not vendor.
Model code imports from it at roughly thirty sites, and every one of those
sites already has a slow path beside it -- a triton kernel, ``F.linear``,
``x * sigmoid(gate)``, the unfused router. The imports are written as bare
statements because upstream knows the package is there, so in this fork they
raise ModuleNotFoundError from inside the first real forward pass: after the
weights load, tens of minutes into a multi-GPU run, reading like a model
defect rather than a missing optional dependency.

The capability gates that dispatch to those kernels ask whether the *GPU*
supports them and not whether the kernel exists. This closes that gap in one
place so the answer is uniform, cached, and logged once instead of being
rediscovered one crash at a time.
"""

from __future__ import annotations

import importlib
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def kernels_module_available(dotted: str) -> bool:
    """Whether ``dotted`` (under ``sglang.kernels``) imports in this build.

    Cached per module name: the answer cannot change within a process, and
    these gates sit on per-token paths.
    """
    try:
        importlib.import_module(dotted)
        return True
    except Exception as exc:  # ImportError, and anything the module raises
        _log_once(dotted, exc)
        return False


def _log_once(dotted: str, exc: BaseException) -> None:
    if isinstance(exc, ModuleNotFoundError):
        logger.info(
            "%s is not available in this build; using the fallback path "
            "(same result, lower throughput).",
            dotted,
        )
    else:
        # A kernel package that exists but fails to import is a real problem
        # worth seeing, even though the fallback keeps the run alive.
        logger.warning("%s failed to import (%s); using the fallback path.", dotted, exc)


def have_kernels() -> bool:
    """Whether the ``sglang.kernels`` package exists at all."""
    return kernels_module_available("sglang.kernels")
