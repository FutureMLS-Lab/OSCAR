"""sgl_kernel compatibility shim for OSCAR dump path.

The vendored `sglang-dump-qkv` (older sglang fork) was built against an older
`sgl_kernel` that exports a handful of AWQ/GPTQ/Machete/Cutlass/FP4 legacy
symbols. The `oscar` conda env's newer `sgl_kernel` no longer exports
some of them.

OSCAR only ever dumps from BF16 or FP8 models, so the missing symbols are
never actually CALLED — they only fail at module import time. This shim
installs a fallback `__getattr__` on the `sgl_kernel` module so any missing
attribute returns a stub that raises NotImplementedError if it's ever
actually invoked. That lets all the import-time references succeed under
the oscar env while preserving a clear error if anything in our dump path
unexpectedly does need a removed kernel.

Activated by adding this directory to PYTHONPATH; Python auto-imports
`sitecustomize` at interpreter startup before any application module runs.
"""

import types


def _pytorch_sampling_fallback(probs, *args, **kwargs):
    """Pure-PyTorch fallback for top_k_top_p / min_p sampling.

    Dump path only needs ONE token to come out per request (max_new_tokens=1)
    so we can ignore top_k/top_p/min_p constraints and just sample from probs.
    Returns argmax (deterministic) so the dump is reproducible.
    """
    import torch
    return torch.argmax(probs, dim=-1)


def _install_sgl_kernel_compat():
    try:
        import sgl_kernel
    except ImportError:
        return  # no sgl_kernel at all — nothing to patch

    # Runtime sampling stubs — return argmax(probs). Only used by the dump
    # path where max_new_tokens=1 and the token value doesn't matter.
    _RUNTIME_SAMPLING_FALLBACKS = (
        "top_k_top_p_sampling_from_probs",
        "top_p_sampling_from_probs",
        "min_p_sampling_from_probs",
        "top_k_top_p_sampling_from_logits",
        "top_k_mask_logits",
    )

    class _SglKernelProxy(types.ModuleType):
        def __getattr__(self, name):
            # Called only when `name` is not in the module's normal __dict__.
            if name in _RUNTIME_SAMPLING_FALLBACKS:
                return _pytorch_sampling_fallback
            # Otherwise, return a stub that raises only if actually called.
            def _stub(*args, **kwargs):
                raise NotImplementedError(
                    f"sgl_kernel.{name} is unavailable in this build (likely "
                    "the oscar env). OSCAR dump path should not exercise it."
                )
            return _stub

    # Re-class the existing module so __getattr__ becomes effective for
    # subsequent `from sgl_kernel import <missing>` calls.
    sgl_kernel.__class__ = _SglKernelProxy


def _install_sgl_kernel_version_shim():
    """Override `importlib.metadata.version('sgl-kernel')` to satisfy sglang's
    `assert_pkg_version` check. The oscar env builds sgl_kernel from source
    without a pip-installed .dist-info, so the metadata lookup raises
    PackageNotFoundError even though `import sgl_kernel` works fine.
    """
    import importlib.metadata as _md

    _orig_version = _md.version

    def _patched_version(name):
        try:
            return _orig_version(name)
        except _md.PackageNotFoundError:
            if name in ("sgl-kernel", "sgl_kernel"):
                return "99.0.0"
            raise

    _md.version = _patched_version


_install_sgl_kernel_compat()
_install_sgl_kernel_version_shim()
