"""Per-rank Triton cache redirection.

Prepended to PYTHONPATH by eval_oscar_gpqa.sh so that every Python interpreter
(including TP workers spawned by sglang) reads LOCAL_RANK from the environment
and routes Triton's on-disk cache into a rank-specific subdirectory.

This breaks the multi-process race where TP ranks within the same job compile
identical kernel hashes and clobber each other's launcher .so / metadata files.

Activate by setting OSCAR_TRITON_PER_RANK_BASE in the parent environment; if
unset, this module is a no-op and Triton uses its default TRITON_CACHE_DIR.
"""
import os


def _apply():
    base = os.environ.get("OSCAR_TRITON_PER_RANK_BASE")
    if not base:
        return
    rank = os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))
    target = os.path.join(base, f"rank{rank}")
    try:
        os.makedirs(target, exist_ok=True)
    except OSError:
        return
    os.environ["TRITON_CACHE_DIR"] = target


_apply()
