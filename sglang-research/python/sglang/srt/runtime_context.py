"""Compatibility shim for upstream's runtime-context accessors.

Upstream reaches parallelism, execution and platform facts through
``get_parallel()`` / ``get_exec()`` / ``get_platform()``. This fork still has
the older free functions, so porting a model file from upstream otherwise means
rewriting 30-odd call sites by hand and getting one of them subtly wrong.

Only the surface upstream's Kimi-K3 actually touches is implemented, and each
field forwards to this fork's existing source of truth rather than caching a
copy -- the values change with the process's parallel groups, so a snapshot
taken at import time would be wrong.
"""

from __future__ import annotations


class _Parallel:
    @property
    def tp_size(self) -> int:
        from sglang.srt.distributed import get_tensor_model_parallel_world_size

        return get_tensor_model_parallel_world_size()

    @property
    def tp_rank(self) -> int:
        from sglang.srt.distributed import get_tensor_model_parallel_rank

        return get_tensor_model_parallel_rank()

    @property
    def attn_tp_size(self) -> int:
        from sglang.srt.layers.dp_attention import get_attention_tp_size

        return get_attention_tp_size()

    @property
    def attn_tp_rank(self) -> int:
        from sglang.srt.layers.dp_attention import get_attention_tp_rank

        return get_attention_tp_rank()

    @property
    def attn_tp_group(self):
        from sglang.srt.layers.dp_attention import get_attention_tp_group

        return get_attention_tp_group()

    @property
    def enable_dp_lm_head(self) -> bool:
        return bool(getattr(_server_args(), "enable_dp_lm_head", False))

    @property
    def enable_shared_experts_attn_tp(self) -> bool:
        # Upstream's name for "the shared experts run under attention TP".
        # This fork spells the inverse as disable_shared_experts_fusion.
        sa = _server_args()
        if hasattr(sa, "enable_shared_experts_attn_tp"):
            return bool(sa.enable_shared_experts_attn_tp)
        return not bool(getattr(sa, "disable_shared_experts_fusion", False))

    @property
    def enable_dense_mlp_attn_tp(self) -> bool:
        return bool(getattr(_server_args(), "enable_dense_mlp_attn_tp", False))


class _ExecComm:
    @property
    def enable_symm_mem(self) -> bool:
        return bool(getattr(_server_args(), "enable_torch_symm_mem", False))


class _ExecMoe:
    @property
    def moe_a2a_backend(self):
        from sglang.srt.layers.moe.utils import get_moe_a2a_backend

        return get_moe_a2a_backend()

    @property
    def moe_runner_backend(self):
        from sglang.srt.layers.moe.utils import get_moe_runner_backend

        return get_moe_runner_backend()


class _ExecDeterministic:
    @property
    def enable_deterministic_inference(self) -> bool:
        return bool(getattr(_server_args(), "enable_deterministic_inference", False))


class _Exec:
    # Upstream groups these into sub-bags (`get_exec().moe.moe_a2a_backend`,
    # `get_exec().deterministic.enable_deterministic_inference`). Mirror the
    # shape, not just the values: a flat bool here would make
    # `.deterministic.enable_deterministic_inference` an AttributeError
    # mid-load, which is exactly the class of drift this shim exists to avoid.
    comm = _ExecComm()
    moe = _ExecMoe()
    deterministic = _ExecDeterministic()


class _Mm:
    @property
    def mm_attention_backend(self):
        return getattr(_server_args(), "mm_attention_backend", None)


class _Platform:
    @property
    def is_blackwell(self) -> bool:
        from sglang.srt.utils import is_sm100_supported

        return bool(is_sm100_supported())


class _NoServerArgs:
    """Every flag reads False before the server is up.

    These fields are only meaningful inside a running server, but the module
    is also imported by weight-only and config-only tooling, where raising
    "Global server args is not set yet!" would be a crash rather than an
    answer.
    """

    def __getattr__(self, _name):
        return False


_NO_SERVER_ARGS = _NoServerArgs()


def _server_args():
    from sglang.srt.server_args import get_global_server_args

    try:
        return get_global_server_args()
    except Exception:
        return _NO_SERVER_ARGS


_PARALLEL = _Parallel()
_MM = _Mm()
_EXEC = _Exec()
_PLATFORM = _Platform()


def get_parallel() -> _Parallel:
    return _PARALLEL


def get_exec() -> _Exec:
    return _EXEC


def get_mm() -> _Mm:
    return _MM


def get_platform() -> _Platform:
    return _PLATFORM
