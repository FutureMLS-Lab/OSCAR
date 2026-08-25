"""Memory pool configurators for profiling and sizing KV cache pools.

Each model architecture has its own configurator that computes pool sizes
from available GPU memory using a unified coeff+bias model:

    available_bytes = max_tokens * coeff + bias
    max_tokens = (available_bytes - bias) / coeff

Two entry points, same core computation:
- calculate_pool_sizes(available_bytes, page_size): profiling path
- calculate_pool_sizes_from_max_tokens(max_tokens, page_size): constraint path
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.configs.model_config import get_nsa_index_head_dim, is_deepseek_nsa
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool
from sglang.srt.mem_cache.unified_kv_pool import (
    compute_page_geometry,
    resolve_hp_dtype,
    resolve_scale_dtype,
)
from sglang.srt.utils.common import is_float4_e2m1fn_x2


def _attention_supports_mixed_kv(server_args) -> bool:
    """Return True when the active attention-backend configuration can run
    the unified HP+int2 KV pool end-to-end.

    The pool's read-path (dequant + rotation + int2 decode) lives in the
    Triton backend today. The FA3 prefill backend gained a rotation-aware
    int2 prefill path in
    ``python/sglang/srt/layers/attention/quantized_kv_prefill.py``, so the
    hybrid ``prefill=fa3 + decode=triton`` configuration is supported too.
    Any other combination stays on the plain ``MHATokenToKVPool`` int2 path.

    ``prefill_attention_backend`` and ``decode_attention_backend`` can both
    be unset (``None``), in which case the model runner falls back to
    ``attention_backend`` for both roles (see
    ``ServerArgs.get_attention_backends``). Apply the same resolution here
    so partially-specified configurations (e.g. only ``--attention-backend
    triton``, or only ``--prefill-attention-backend fa3`` with
    ``--attention-backend triton``) are recognized correctly.
    """
    ab = getattr(server_args, "attention_backend", None)
    pab = getattr(server_args, "prefill_attention_backend", None) or ab
    dab = getattr(server_args, "decode_attention_backend", None) or ab
    if pab == "triton" and dab == "triton":
        return True
    if pab == "fa3" and dab == "triton":
        return True
    return False


@dataclass
class MemoryPoolConfig:
    """Resolved memory pool config, shared between target and draft workers."""

    max_total_num_tokens: int
    max_running_requests: Optional[int] = None
    full_max_total_num_tokens: Optional[int] = None
    swa_max_total_num_tokens: Optional[int] = None

    mem_fraction_static: Optional[float] = None

    def __post_init__(self):
        if self.max_total_num_tokens <= 0:
            msg = "Not enough memory. Please try to increase --mem-fraction-static."
            if self.mem_fraction_static is not None:
                msg += f" Current value: mem_fraction_static={self.mem_fraction_static}"
            raise RuntimeError(msg)


if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def _resolve_quant_group_count(head_dim: int, group_size: Optional[int]) -> int:
    effective_group_size = head_dim if group_size is None else group_size
    if effective_group_size <= 0:
        raise ValueError(
            f"kv_cache_quant_group_size must be positive, got {effective_group_size}"
        )
    if head_dim % effective_group_size != 0:
        raise ValueError(
            f"head_dim ({head_dim}) must be divisible by "
            f"kv_cache_quant_group_size ({effective_group_size})"
        )
    return head_dim // effective_group_size


def _get_int_kv_bytes_per_head_pair(
    k_head_dim: int,
    v_head_dim: int,
    kv_cache_dtype: str,
    group_size: Optional[int],
    scale_dtype_bytes: int = 4,
) -> int:
    """Bytes per *quant-token* per head-pair for the int2 KV cache.

    Used by both the non-mixed path and the mixed path: the scheduler's
    ``max_total_num_tokens`` is denominated in quant tokens (= slot ids on the
    int2 tier), and the mixed allocator's ``size`` = ``(num_pages - 1) * N_Q``
    is also in quant tokens, so the leak check
    ``size - available - evictable - protected`` closes in a single unit.
    """
    assert kv_cache_dtype == "int2", (
        f"Only int2 quant KV is supported, got {kv_cache_dtype}"
    )
    pack_factor = 4
    k_groups = _resolve_quant_group_count(k_head_dim, group_size)
    v_groups = _resolve_quant_group_count(v_head_dim, group_size)
    packed_k_bytes = k_head_dim // pack_factor
    packed_v_bytes = v_head_dim // pack_factor
    # Interleaved (scale, zero) per group in ``scale_dtype``.
    scales_zeros_bytes = 2 * scale_dtype_bytes * (k_groups + v_groups)
    return packed_k_bytes + packed_v_bytes + scales_zeros_bytes


def _get_unified_mixed_kv_bytes_per_quant_token(
    k_head_dim: int,
    v_head_dim: int,
    hp_dtype_bytes: int,
    scale_dtype_bytes: int,
    k_groups: int,
    v_groups: int,
    n_q: int,
) -> int:
    """Shared-arena bytes per *quant token* in the unified pool.

    A page of the shared K/V arena is sized to host either 1 HP token or N_Q
    quant tokens; its byte size is therefore ``(head_dim + v_head_dim) *
    hp_dtype_bytes``. Per quant token that is this over ``N_Q``. Scales/zeros
    live in a parallel arena (one entry per quant slot) and are included
    here so the scheduler's cell-size matches the actual allocator size.
    """
    arena_bytes_per_page = (k_head_dim + v_head_dim) * hp_dtype_bytes
    arena_bytes_per_quant_token = arena_bytes_per_page // n_q
    scales_zeros_bytes_per_quant_token = (
        2 * scale_dtype_bytes * (k_groups + v_groups)
    )
    return arena_bytes_per_quant_token + scales_zeros_bytes_per_quant_token


class MemoryPoolConfigurator:
    """Base class for memory pool configurators.

    Subclasses compute pool sizes for their architecture via coeff+bias model.
    Both entry points return MemoryPoolConfig (with max_running_requests=None,
    to be filled by the consumer).
    """

    def calculate_pool_sizes(
        self, available_bytes: int, page_size: int
    ) -> MemoryPoolConfig:
        """Profiling path: compute pool sizes from available bytes."""
        raise NotImplementedError

    def calculate_pool_sizes_from_max_tokens(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        """Constraint path: recalculate pool sizes from a constrained max_tokens."""
        raise NotImplementedError


def _packed_mla_latent_enabled(mr: ModelRunner) -> bool:
    """Whether this run stores the MLA latent as packed INT2 codes.

    Must agree exactly with the routing in ``model_runner_kv_cache_mixin``: a
    configurator that disagrees with the pool it is sizing either wastes the
    difference or runs the allocator off the end of the buffers.
    """
    from sglang.srt.environ import envs

    base = (
        mr.use_mla_backend
        and envs.SGLANG_OSCAR_MLA_KV_PACKED.get()
        and bool(envs.SGLANG_OSCAR_MLA_KV_ROTATION_PATH.get())
    )
    if not base:
        return False
    # The mambaish branch in model_runner_kv_cache_mixin additionally refuses
    # speculative decoding and disaggregation, and this predicate did not. On
    # an MLA + mambaish model with either enabled, the configurator sized the
    # pool at 288 B/token while the router built a BF16 pool at 1152 -- a 4x
    # under-size, which is the "runs the allocator off the end" case this
    # docstring warns about rather than the harmless wasteful one.
    #
    # Non-mambaish MLA has no such conditions, so it keeps the plain predicate.
    if getattr(mr, "mambaish_config", None):
        sa = mr.server_args
        return (
            sa.disaggregation_mode in (None, "null")
            and sa.speculative_algorithm is None
        )
    return True


class DefaultPoolConfigurator(MemoryPoolConfigurator):
    """Configurator for standard models: MHA, MLA, NSA, FP4.

    coeff = cell_size (bytes per token across all layers)
    bias = 0
    """

    def __init__(self, mr: ModelRunner):
        # Determine effective number of layers for KV cache
        if mambaish := mr.mambaish_config:
            effective_layer_ids = [
                i
                for i in mambaish.full_attention_layer_ids
                if mr.start_layer <= i < mr.end_layer
            ]
            num_layers = len(effective_layer_ids)
        else:
            num_layers = mr.num_effective_layers

        self._cell_size = self._compute_cell_size(mr, num_layers)

        # DFLASH: scale cell_size to account for draft model KV cache
        if mr.spec_algorithm.is_dflash() and not mr.is_draft_worker:
            from sglang.srt.speculative.dflash_utils import (
                scale_kv_cell_size_per_token_for_dflash,
            )

            draft_num_layers = getattr(mr, "dflash_draft_num_layers", None)
            if (
                draft_num_layers is not None
                and int(draft_num_layers) > 0
                and int(num_layers) > 0
            ):
                self._cell_size = scale_kv_cell_size_per_token_for_dflash(
                    target_cell_size_per_token=self._cell_size,
                    target_num_layers=int(num_layers),
                    draft_num_layers=int(draft_num_layers),
                )

    def _compute_cell_size(self, mr: ModelRunner, num_layers: int) -> int:
        """Compute per-token KV cache cost in bytes. Subclasses can override."""
        # args to config cell size
        from sglang.srt.environ import envs

        model_config = mr.model_config
        kv_cache_dtype = mr.kv_cache_dtype
        kv_quant_group_size = getattr(
            mr.server_args, "kv_cache_quant_group_size", None
        )
        tp_size = get_attention_tp_size()

        # Mixed HP+quant pool: the shared arena bytes per logical token are
        # ``(head_dim + v_head_dim) * hp_dtype_bytes`` (== 1 HP token worth).
        # Plus the scales/zeros arena sized for every page's quant view.
        enable_mixed_kv = (
            kv_cache_dtype == "int2"
            and envs.SGLANG_ENABLE_MIXED_KV_WINDOWS.get()
            and _attention_supports_mixed_kv(mr.server_args)
            and not mr.is_hybrid_swa
            and mr.server_args.disaggregation_mode in (None, "null")
            and mr.server_args.speculative_algorithm is None
        )

        if kv_cache_dtype == "int2":
            scale_dtype = resolve_scale_dtype(envs.SGLANG_MIXED_KV_SCALE_DTYPE.get())
            scale_bytes = torch.empty(0, dtype=scale_dtype).element_size()
        else:
            scale_bytes = None

        if enable_mixed_kv:
            hp_dtype = resolve_hp_dtype(envs.SGLANG_MIXED_KV_HP_DTYPE.get())
            hp_dtype_bytes = torch.empty(0, dtype=hp_dtype).element_size()
            _, n_q = compute_page_geometry(hp_dtype)
            k_groups = _resolve_quant_group_count(
                model_config.head_dim, kv_quant_group_size
            )
            v_groups = _resolve_quant_group_count(
                model_config.v_head_dim, kv_quant_group_size
            )
            # max_total_num_tokens is denominated in *quant tokens* (slot ids
            # on the int2 tier). This matches the unified allocator's
            # scheduler-facing ``size = (num_pages - 1) * N_Q``.
            bytes_per_head = _get_unified_mixed_kv_bytes_per_quant_token(
                model_config.head_dim,
                model_config.v_head_dim,
                hp_dtype_bytes,
                scale_bytes,
                k_groups,
                v_groups,
                n_q,
            )
            kv_size = None
        elif kv_cache_dtype == "int2":
            bytes_per_head = _get_int_kv_bytes_per_head_pair(
                model_config.head_dim,
                model_config.v_head_dim,
                kv_cache_dtype,
                kv_quant_group_size,
                scale_bytes,
            )
            kv_size = None
        else:
            kv_size = torch._utils._element_size(kv_cache_dtype)
            bytes_per_head = None

        if mr.use_mla_backend:
            if _packed_mla_latent_enabled(mr):
                # Packed INT2 latent: codes + group params + BF16 k_pe. Without
                # this the pool would be sized as if it still stored 576 BF16
                # values per token and ``max_total_num_tokens`` would not move
                # off the BF16 arm's number -- which is exactly the symptom that
                # says the storage change did not land.
                from sglang.srt.mem_cache.mla_packed_kv_pool import (
                    packed_latent_bytes_per_token,
                )

                cell_size = (
                    packed_latent_bytes_per_token(
                        model_config.kv_lora_rank,
                        model_config.qk_rope_head_dim,
                        envs.SGLANG_OSCAR_MLA_KV_GROUP_SIZE.get(),
                    )
                    * num_layers
                )
                # The BF16 window arena is a fixed cost, not a per-token one:
                # (sink + recent) rows per request slot, every layer. Charging
                # it here keeps the pool inside the mem-fraction budget instead
                # of overshooting it by a couple of GB at the end of startup.
                p = (
                    envs.SGLANG_MIXED_KV_PREFIX_TOKENS.get()
                    if envs.SGLANG_MIXED_KV_PREFIX_TOKENS.is_set()
                    else 64
                )
                r = (
                    envs.SGLANG_MIXED_KV_RECENT_TOKENS.get()
                    if envs.SGLANG_MIXED_KV_RECENT_TOKENS.is_set()
                    else 256
                )
                # ``mr.max_running_requests`` does not exist yet -- it is
                # resolved *from* the pool size this call is computing, and its
                # default estimate climbs with the pool, so a 3x bigger pool
                # would silently ask for a 3x bigger window arena. Read the flag
                # instead, and require it: an unpinned arena is either 60x too
                # big or quietly leaves requests without windows.
                max_reqs = mr.server_args.max_running_requests
                if max_reqs is None:
                    raise ValueError(
                        "SGLANG_OSCAR_MLA_KV_PACKED needs --max-running-requests "
                        "pinned: the BF16 window arena is sized per request slot "
                        "and the default is derived from the pool size, which "
                        "this branch changes."
                    )
                self._fixed_overhead_bytes = (
                    (1 + max_reqs * (p + r))
                    * model_config.kv_lora_rank
                    * 2
                    * num_layers
                )
            else:
                cell_size = (
                    (model_config.kv_lora_rank + model_config.qk_rope_head_dim)
                    * num_layers
                    * kv_size
                )
            if is_float4_e2m1fn_x2(kv_cache_dtype):
                # kv_scale_buffer
                scale_block_size = 16
                cell_size = (cell_size // 2) + (
                    (
                        (model_config.kv_lora_rank + model_config.qk_rope_head_dim)
                        // scale_block_size
                    )
                    * num_layers
                    * kv_size
                )

            # Add indexer KV cache overhead for NSA models (DeepSeek V3.2)
            if is_deepseek_nsa(model_config.hf_config):
                index_head_dim = get_nsa_index_head_dim(model_config.hf_config)
                indexer_size_per_token = (
                    index_head_dim
                    + index_head_dim // NSATokenToKVPool.quant_block_size * 4
                )
                element_size = torch._utils._element_size(
                    NSATokenToKVPool.index_k_with_scale_buffer_dtype
                )
                cell_size += indexer_size_per_token * num_layers * element_size
        else:
            if bytes_per_head is not None:
                cell_size = (
                    model_config.get_num_kv_heads(tp_size) * bytes_per_head * num_layers
                )
            else:
                cell_size = (
                    model_config.get_num_kv_heads(tp_size)
                    * (model_config.head_dim + model_config.v_head_dim)
                    * num_layers
                    * kv_size
                )

            if is_float4_e2m1fn_x2(kv_cache_dtype):
                # kv_scale_buffer
                scale_block_size = 16
                n = model_config.get_num_kv_heads(tp_size)
                k = model_config.head_dim
                cell_size = (cell_size // 2) + (
                    (n * k * num_layers * 2 * kv_size) // scale_block_size
                )

        return cell_size

    def calculate_pool_sizes(
        self, available_bytes: int, page_size: int
    ) -> MemoryPoolConfig:
        available_bytes -= getattr(self, "_fixed_overhead_bytes", 0)
        max_total_num_tokens = available_bytes // self._cell_size
        max_total_num_tokens = max_total_num_tokens // page_size * page_size
        return MemoryPoolConfig(max_total_num_tokens=max_total_num_tokens)

    def calculate_pool_sizes_from_max_tokens(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        max_total_num_tokens = max_total_num_tokens // page_size * page_size
        return MemoryPoolConfig(max_total_num_tokens=max_total_num_tokens)


class HybridSWAPoolConfigurator(MemoryPoolConfigurator):
    """Configurator for hybrid sliding window attention models (Gemma2, Command-R, MiMo).

    Splits available memory between full attention and SWA pools.
    Does NOT inherit DefaultPoolConfigurator — different coeff model.
    """

    def __init__(self, mr: ModelRunner):
        model_config = mr.model_config
        kv_cache_dtype = mr.kv_cache_dtype
        kv_quant_group_size = getattr(
            mr.server_args, "kv_cache_quant_group_size", None
        )
        tp_size = get_attention_tp_size()

        if (
            kv_cache_dtype == "int2"
            and kv_quant_group_size is not None
        ):
            raise ValueError(
                "--kv-cache-quant-group-size is only supported for the "
                "full-attention Triton int2 KV cache path and is not supported "
                "with hybrid SWA models"
            )

        if kv_cache_dtype == "int2":
            full_per_token = model_config.get_num_kv_heads(tp_size) * (
                _get_int_kv_bytes_per_head_pair(
                    model_config.head_dim,
                    model_config.v_head_dim,
                    kv_cache_dtype,
                    kv_quant_group_size,
                )
            )
            swa_per_token = model_config.get_swa_num_kv_heads(tp_size) * (
                _get_int_kv_bytes_per_head_pair(
                    model_config.swa_head_dim,
                    model_config.swa_v_head_dim,
                    kv_cache_dtype,
                    kv_quant_group_size,
                )
            )
            kv_size = None
        else:
            kv_size = torch._utils._element_size(kv_cache_dtype)
            full_per_token = (
                model_config.get_num_kv_heads(tp_size)
                * (model_config.head_dim + model_config.v_head_dim)
                * kv_size
            )
            swa_per_token = (
                model_config.get_swa_num_kv_heads(tp_size)
                * (model_config.swa_head_dim + model_config.swa_v_head_dim)
                * kv_size
            )

        self._full_layers_num = len(model_config.full_attention_layer_ids)
        self._swa_layers_num = len(model_config.swa_attention_layer_ids)
        assert (
            self._swa_layers_num > 0
        ), "Hybrid SWA model must have at least one SWA layer"

        self._swa_full_tokens_ratio = mr.server_args.swa_full_tokens_ratio

        # Full layer per-token memory (bytes)
        self._full_per_token = full_per_token

        # SWA layer per-token memory (bytes)
        self._swa_per_token = swa_per_token

        # Bytes per token of max_total_num_tokens.
        #
        # Hybrid (full_layers > 0): max_total = full_tokens, so cell_size accounts
        # for both pools: F*nf + r*S*ns (where swa_tokens = full_tokens * r).
        #
        # All-SWA (full_layers == 0): max_total = swa_tokens directly. The ratio
        # is meaningless here -- there is no full pool to relate to, and every
        # token beyond the sliding window can be evicted. So cell_size = S*ns,
        # with no ratio factor applied.
        if self._full_layers_num == 0:
            self._cell_size = self._swa_per_token * self._swa_layers_num
        else:
            self._cell_size = (
                self._full_per_token * self._full_layers_num
                + self._swa_full_tokens_ratio
                * self._swa_per_token
                * self._swa_layers_num
            )

    def _solve_pool_sizes(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        """Core computation: split max_total_num_tokens into full/swa pool sizes."""

        def align_page_size(x: int) -> int:
            return (x // page_size) * page_size

        if self._full_layers_num == 0:
            # All-SWA: no full pool, max_total = actual SWA pool size.
            # Ratio is not applied -- see __init__ comment.
            swa_tokens = align_page_size(max_total_num_tokens)
            logger.info(
                f"Use sliding window memory pool (all SWA). "
                f"swa_layer_tokens={swa_tokens}"
            )
            return MemoryPoolConfig(
                max_total_num_tokens=swa_tokens,
                full_max_total_num_tokens=0,
                swa_max_total_num_tokens=swa_tokens,
            )

        # Hybrid: full_tokens = max_total_num_tokens, swa_tokens = full_tokens * ratio
        full_tokens = align_page_size(max_total_num_tokens)
        swa_tokens = align_page_size(int(full_tokens * self._swa_full_tokens_ratio))

        logger.info(
            f"Use sliding window memory pool. "
            f"full_layer_tokens={full_tokens}, swa_layer_tokens={swa_tokens}"
        )

        return MemoryPoolConfig(
            max_total_num_tokens=full_tokens,
            full_max_total_num_tokens=full_tokens,
            swa_max_total_num_tokens=swa_tokens,
        )

    def calculate_pool_sizes(
        self, available_bytes: int, page_size: int
    ) -> MemoryPoolConfig:
        max_total_num_tokens = int(available_bytes // self._cell_size)
        return self._solve_pool_sizes(max_total_num_tokens, page_size)

    def calculate_pool_sizes_from_max_tokens(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        return self._solve_pool_sizes(max_total_num_tokens, page_size)


def create_memory_pool_configurator(
    mr: ModelRunner,
) -> MemoryPoolConfigurator:
    """Factory: select the right configurator for the model architecture."""
    if mr.is_hybrid_swa:
        return HybridSWAPoolConfigurator(mr)
    # Future: MambaPoolConfigurator
    return DefaultPoolConfigurator(mr)
