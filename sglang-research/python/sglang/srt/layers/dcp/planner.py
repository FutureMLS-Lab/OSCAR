"""Decode-context-parallel planner: not implemented in this fork.

Upstream models import this unconditionally and call it only when --dcp-size is
set. This fork has no DCP, so the import must succeed and the call must fail
loudly -- returning empty metadata would let a DCP run proceed with silently
wrong attention partitioning.
"""


def prepare_decode_context_parallel_metadata(*args, **kwargs):
    raise NotImplementedError(
        "Decode context parallel (--dcp-size) is not implemented in this fork; "
        "run without it."
    )
