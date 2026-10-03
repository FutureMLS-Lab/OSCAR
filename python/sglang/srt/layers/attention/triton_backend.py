from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func
from sglang.kernels.ops.attention.flash_attention_v3 import _is_fa3_supported
from sglang.kernels.ops.attention.metadata import get_num_kv_splits_triton
from sglang.kernels.ops.attention.mla_kv_pack_quantize_fp8 import (
    mla_kv_pack_quantize_fp8,
)
from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.configs.model_config import (
    AttentionArch,
    is_dspark_draft,
    is_kimi_k3,
    is_qwen3_5,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.quantized_kv_prefill import (
    _apply_oscar_rotation,
    _pool_uses_oscar_rotation,
    apply_inverse_v_rotation,
    apply_segmented_hadamard_transform,
    dequantize_prefix_kv,
    prepare_quantized_extend_qkv,
)
from sglang.srt.layers.attention.verify_mask import VerifyMask, maybe_create_verify_mask
from sglang.srt.layers.dcp import (
    cp_lse_ag_out_rs_mha,
    create_triton_kv_indices_for_dcp_triton,
    get_dcp_lens,
)
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
    cuda_graph_fully_disabled,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_exec, get_parallel, get_schedule, get_spec
from sglang.srt.speculative.spec_utils import (
    draft_kv_indices_buffer_width,
    draft_kv_indices_used_len,
    generate_draft_decode_kv_indices,
    resolve_draft_decode_window,
)
from sglang.srt.utils import (
    get_bool_env_var,
    get_device_core_count,
    get_int_env_var,
    is_cuda,
    is_gfx95_supported,
    is_gfx942_supported,
    is_hip,
    is_xpu,
    next_power_of_2,
)

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_gfx942 = is_gfx942_supported()
_is_xpu = is_xpu()

if _is_cuda:
    from sgl_kernel.utils import is_arch_support_pdl

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput


def _is_packed_mla_pool(pool) -> bool:
    """True for the packed-INT2 latent pool (``mla_packed_kv_pool``).

    Duck-typed rather than an isinstance import so this module keeps no
    dependency on the OSCAR pool, and so a pool that only *partly* implements
    the contract cannot half-qualify.
    """
    return hasattr(pool, "packed_read_operands") and hasattr(pool, "materialize_rows")


def _is_int2_pool(pool) -> bool:
    """True for the int2 KV pools (``UnifiedInt2HPKVPool`` and the pure-int2
    MHA pool), which report the string ``"int2"`` as their dtype.

    Duck-typed like ``_is_packed_mla_pool``: the probe is the pool contract
    (``get_raw_key_buffer`` / ``get_key_scales_zeros`` / ``set_kv_buffer(...,
    already_hadamard_transformed, is_decode)``), not a class.
    """
    return getattr(pool, "dtype", None) == "int2"


def _pool_mixed_kv_active(pool) -> bool:
    """True when the pool keeps HP prefix/recent windows beside the int2 tier.

    ``mixed_kv_enabled()`` is True only for ``UnifiedInt2HPKVPool`` (SWAKVPool /
    MHA pools lack the method), so this probe already excludes the plain
    hybrid-SWA path.
    """
    probe = getattr(pool, "mixed_kv_enabled", None)
    return probe is not None and probe() is True


@triton.jit
def _count_mixed_hp_lens_kernel(
    req_to_token_ptr,       # int32 [num_req_slots, max_ctx]
    req_pool_indices_ptr,   # int64 [bs]
    seq_lens_ptr,           # int32 [bs]
    hp_lens_ptr,            # int32 [bs]
    start_pos_ptr,          # int32 [bs] or None -- per-req scan start position
    rtt_stride_row,
    HP_OFFSET: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Count per-request HP lengths without dense mask materialization.

    This keeps the req-pool indirection fused with the tier classification so
    ``_build_mixed_kv_indices`` never has to materialize a gathered ``rows``
    tensor or per-token boolean masks. The quant tier length is derived from
    ``(seq_len - start) - hp_len`` on the Python side.

    ``start_pos_ptr`` (optional) restricts the scan to token positions
    ``[start, seq_len)``. It is used by the sliding-window mixed-decode variant
    to drop out-of-window tokens (both the prefix-sink HP tokens and the
    out-of-window quant bulk); ``None`` => start at 0 (full context, unchanged
    behavior for every existing caller / full-attention layer).
    """
    req = tl.program_id(0)
    req_pool_idx = tl.load(req_pool_indices_ptr + req).to(tl.int64)
    seq_len = tl.load(seq_lens_ptr + req).to(tl.int32)
    start = tl.zeros((), dtype=tl.int32)
    if start_pos_ptr:
        start = tl.load(start_pos_ptr + req).to(tl.int32)

    hp_count = tl.zeros((), dtype=tl.int32)
    num_loops = tl.cdiv(seq_len - start, BLOCK_SIZE)
    for i in range(num_loops):
        offs = start + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = offs < seq_len
        slot = tl.load(
            req_to_token_ptr + req_pool_idx * rtt_stride_row + offs.to(tl.int64),
            mask=valid,
            other=0,
        ).to(tl.int64)
        hp_count += tl.sum((valid & (slot >= HP_OFFSET)).to(tl.int32), axis=0)

    tl.store(hp_lens_ptr + req, hp_count)


@triton.jit
def _scatter_mixed_kv_indices_kernel(
    req_to_token_ptr,       # int32 [num_req_slots, max_ctx]
    req_pool_indices_ptr,   # int64 [bs]
    seq_lens_ptr,           # int32 or int64 [bs] -- cast inside
    hp_kv_indptr_ptr,       # int32 [bs + 1]   already cumsum'd
    quant_kv_indptr_ptr,    # int32 [bs + 1]   already cumsum'd
    hp_kv_indices_ptr,      # int64 [*] destination, pre-sized
    quant_kv_indices_ptr,   # int64 [*] destination, pre-sized
    start_pos_ptr,          # int32 [bs] or None -- per-req scan start position
    rtt_stride_row,
    HP_OFFSET: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Slot-id-classified scatter into hp/quant index buffers, one block per req.

    For each request i we walk ``req_to_token[req_pool_indices[i], start..seq_len)``
    in ``BLOCK_SIZE`` chunks. Each lane decides whether its slot id is HP
    (``slot >= HP_OFFSET``) or quant, then contributes to within-block exclusive
    prefix sums that act as scatter offsets into the pre-cumsum'd
    ``hp_kv_indptr`` / ``quant_kv_indptr`` tier-local layout. No masked-select,
    no Python bs-loop, and no D2H sync: stride and offset arithmetic is all on
    device with shapes known statically.

    ``start_pos_ptr`` (optional) restricts the scan to ``[start, seq_len)``;
    ``None`` => start at 0 (full context, unchanged for every existing caller /
    full-attention layer). The windowed sliding-decode variant passes
    ``start = max(0, seq_len - window)`` so out-of-window positions (the prefix
    sink + the out-of-window quant bulk) are never emitted. Because the scan is
    monotone in position and quant entries are stored in scan order, the emitted
    quant indices for a sliding layer are exactly the in-window quant bulk in
    ascending position order.
    """
    req = tl.program_id(0)
    req_pool_idx = tl.load(req_pool_indices_ptr + req).to(tl.int64)
    seq_len = tl.load(seq_lens_ptr + req).to(tl.int32)
    hp_base = tl.load(hp_kv_indptr_ptr + req).to(tl.int64)
    quant_base = tl.load(quant_kv_indptr_ptr + req).to(tl.int64)
    start = tl.zeros((), dtype=tl.int32)
    if start_pos_ptr:
        start = tl.load(start_pos_ptr + req).to(tl.int32)

    # Running counters for the chunked scatter. Triton tracks these as scalar
    # SSA values that accumulate across the Python-side for loop below.
    hp_running = tl.zeros((), dtype=tl.int32)
    quant_running = tl.zeros((), dtype=tl.int32)

    num_loops = tl.cdiv(seq_len - start, BLOCK_SIZE)
    for i in range(num_loops):
        offs = start + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = offs < seq_len
        slot = tl.load(
            req_to_token_ptr + req_pool_idx * rtt_stride_row + offs.to(tl.int64),
            mask=valid,
            other=0,
        ).to(tl.int64)
        # HP slot ids start at exactly ``HP_OFFSET`` (page 0 is a valid HP
        # page), so the boundary is ``>=`` not ``>``. The unified pool /
        # allocator (``unified_kv_pool._split_global_locs``,
        # ``unified_kv_allocator.free``) and the GPU flush kernel
        # (``gpu_flush_int2``) all classify by ``>=``; using ``>`` here would
        # misclassify HP slot id ``HP_OFFSET`` as quant and read OOB from
        # the quant buffer.
        is_hp = valid & (slot >= HP_OFFSET)
        is_quant = valid & (slot < HP_OFFSET)  # == valid & ~is_hp; explicit to avoid ~bool dtype quirks

        hp_inc = is_hp.to(tl.int32)
        quant_inc = is_quant.to(tl.int32)

        # tl.cumsum gives an inclusive prefix; subtract the lane value to get
        # the exclusive prefix (= rank of this lane among HP/quant entries
        # within this block).
        hp_rank = tl.cumsum(hp_inc, axis=0) - hp_inc
        quant_rank = tl.cumsum(quant_inc, axis=0) - quant_inc

        tl.store(
            hp_kv_indices_ptr + hp_base + (hp_running + hp_rank).to(tl.int64),
            slot - HP_OFFSET,
            mask=is_hp,
        )
        tl.store(
            quant_kv_indices_ptr + quant_base + (quant_running + quant_rank).to(tl.int64),
            slot,
            mask=is_quant,
        )

        hp_running += tl.sum(hp_inc, axis=0)
        quant_running += tl.sum(quant_inc, axis=0)


_MLA_DECODE_MIN_BLOCK_KV = 32


def _mla_decode_kv_splits_cap(
    base_max_kv_splits: int, sm_count: int, max_context_len: int
) -> int:
    if sm_count <= 0:
        return base_max_kv_splits
    sm_cap = next_power_of_2(sm_count)
    ctx_cap = next_power_of_2(triton.cdiv(max_context_len, _MLA_DECODE_MIN_BLOCK_KV))
    return max(base_max_kv_splits, min(sm_cap, ctx_cap))


def _should_use_verify_shared_kv(model_config, topk, use_mla, use_verify_splitkv):
    if not is_gfx95_supported() or topk != 1:
        return False
    if use_mla:
        return is_kimi_k3(model_config.hf_config)
    if is_dspark_draft(model_config.hf_config):
        return use_verify_splitkv
    return (
        use_verify_splitkv
        and is_qwen3_5(model_config.hf_config)
        and model_config.get_num_kv_heads(
            get_parallel().attn_tp_size, get_parallel().attn_dcp_size
        )
        == 1
    )


def logit_capping_mod(logit_capping_method, logit_cap):
    # positive logit_cap -> tanh cap
    if logit_capping_method == "tanh":
        return logit_cap
    else:
        raise ValueError()


@dataclass
class ForwardMetadata:
    attn_logits: torch.Tensor
    attn_lse: torch.Tensor
    max_extend_len: int
    num_kv_splits: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    qo_indptr: torch.Tensor
    custom_mask: torch.Tensor
    mask_indptr: torch.Tensor
    # Sliding window
    window_kv_indptr: torch.Tensor
    window_kv_indices: torch.Tensor
    window_num_kv_splits: torch.Tensor
    window_kv_offsets: torch.Tensor
    # Separate attn_logits for SWA layers when v_head_dim differs
    swa_attn_logits: Optional[torch.Tensor] = None
    # full->SWA translated out_cache_loc (SWA KV-store write target)
    swa_out_cache_loc: Optional[torch.Tensor] = None
    # PHYSICAL full-attn write target for the unified pool (eager: translated tensor;
    # cuda-graph: capture-stable buffer view). None for non-unified pools.
    out_cache_loc_full_physical: Optional[torch.Tensor] = None
    # Lean decode (persistent-grid partial-result buffers)
    lean_Mp: Optional[torch.Tensor] = None
    lean_Lp: Optional[torch.Tensor] = None
    lean_Op: Optional[torch.Tensor] = None
    lean_locks: Optional[torch.Tensor] = None
    # Per-tier indptr/indices for the unified single-launch mixed int2 path.
    mixed_hp_kv_indptr: Optional[torch.Tensor] = None
    mixed_hp_kv_indices: Optional[torch.Tensor] = None
    mixed_quant_kv_indptr: Optional[torch.Tensor] = None
    mixed_quant_kv_indices: Optional[torch.Tensor] = None
    # Single combined stage-1 scratch: HP splits in the first hp_max slots,
    # quant splits in the next quant_max slots. Stage-2 reduces both in one
    # launch.
    mixed_attn_logits: Optional[torch.Tensor] = None
    mixed_attn_lse: Optional[torch.Tensor] = None
    # SWA-geometry mixed scratch (gemma4_unified two-group). The mixed-decode
    # stage-2 derives the LSE stride via ``// Lv`` from the logits buffer, so
    # the scratch width must equal the layer's v_head_dim. Sliding layers
    # (v_head_dim 256) need their own scratch separate from the full-layer
    # scratch (v_head_dim 512); selected per-layer in forward_decode.
    mixed_swa_attn_logits: Optional[torch.Tensor] = None
    mixed_swa_attn_lse: Optional[torch.Tensor] = None
    # Per-tier split counts populated by get_num_kv_splits_triton.
    mixed_hp_num_kv_splits: Optional[torch.Tensor] = None
    mixed_quant_num_kv_splits: Optional[torch.Tensor] = None
    # Sliding-window mixed-decode indices (gemma4_unified two-group). For
    # SLIDING layers the quant bulk and HP tier are capped to the last
    # ``sliding_window`` tokens: the prefix-sink HP tokens and the out-of-window
    # quant bulk are dropped. Built only when the backend has a sliding window
    # AND the mixed pool is active; full-attention layers keep the unwindowed
    # ``mixed_{hp,quant}_kv_*`` above. ``forward_decode`` selects per layer on
    # ``layer.sliding_window_size``.
    mixed_swa_hp_kv_indptr: Optional[torch.Tensor] = None
    mixed_swa_hp_kv_indices: Optional[torch.Tensor] = None
    mixed_swa_quant_kv_indptr: Optional[torch.Tensor] = None
    mixed_swa_quant_kv_indices: Optional[torch.Tensor] = None


class TritonAttnBackend(AttentionBackend):
    # CUDA-graph replay rebuilds metadata from preallocated kv_indptr/kv_indices
    # buffers; it never reads seq_lens_cpu / seq_lens_sum.
    needs_cpu_seq_lens: bool = False

    # kv_indptr/qo_indptr are preallocated at (req pool + 1); an extend batch
    # can never carry more seqs than the pool.
    extend_dummy_seqs_capped_by_req_pool: bool = True

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
    ):
        # Lazy import to avoid the initialization of cuda context
        from sglang.kernels.ops.attention.decode_attention import (
            _LEAN_BLOCK_M,
            _lean_decode_launch_params,
            decode_attention_fwd,
            lean_capture_policy,
            lean_decode_seqlen_gate,
        )
        from sglang.kernels.ops.attention.extend_attention import (
            build_unified_kv_indices,
            can_use_dense_prefill_fp8,
            dense_prefill_attention_fwd,
            extend_attention_fwd,
            extend_attention_fwd_unified,
        )
        from sglang.kernels.ops.attention.verify_mla import verify_shared_kv_fwd
        from sglang.kernels.ops.attention.verify_splitkv import verify_splitkv_fwd
        from sglang.srt.layers.attention.triton_ops.decode_attention import (
            decode_attention_fwd_int2_unified,
            decode_attention_fwd_quantized,
        )
        from sglang.srt.layers.attention.triton_ops.decode_attention_pq import (
            decode_attention_fwd_pq_unified,
        )

        super().__init__()

        self.decode_attention_fwd = torch.compiler.disable(decode_attention_fwd)
        self.decode_attention_fwd_quantized = torch.compiler.disable(
            decode_attention_fwd_quantized
        )
        self.decode_attention_fwd_int2_unified = torch.compiler.disable(
            decode_attention_fwd_int2_unified
        )
        self.decode_attention_fwd_pq_unified = torch.compiler.disable(
            decode_attention_fwd_pq_unified
        )
        # Work-Centric (Lean) Attention activation. None => auto-gate from host-side
        # seqlen metadata in forward_decode; True/False => explicit override.
        self.enable_lean_attention = get_exec().kernel.enable_lean_attention
        self._lean_decode_seqlen_gate = lean_decode_seqlen_gate
        self._lean_capture_policy = lean_capture_policy
        self.extend_attention_fwd = torch.compiler.disable(extend_attention_fwd)
        self.extend_attention_fwd_unified = torch.compiler.disable(
            extend_attention_fwd_unified
        )
        self.build_unified_kv_indices = torch.compiler.disable(build_unified_kv_indices)
        # Dense (non-absorbed) MLA prefill over a materialized prefix; see
        # handle_attention_triton for when the dispatcher selects it.
        self.dense_prefill_attention_fwd = torch.compiler.disable(
            dense_prefill_attention_fwd
        )
        self.can_use_dense_prefill_fp8 = can_use_dense_prefill_fp8
        # Cumulative full sequence lengths addressing the one-shot K/V; built
        # on first use per forward and reset by init_forward_metadata.
        self._dense_one_shot_kv_indptr = None
        # Split-KV EAGLE-verify kernel; enabled below once topk is known (valid only at topk == 1).
        self.verify_splitkv_fwd = torch.compiler.disable(verify_splitkv_fwd)
        # Grouped-head split-KV verify kernel for MLA or one shared local KV head.
        self.verify_shared_kv_fwd = torch.compiler.disable(verify_shared_kv_fwd)

        # Parse args
        self.skip_prefill = skip_prefill
        max_bs = model_runner.req_to_token_pool.size
        self.sliding_window_size = model_runner.sliding_window_size
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool_allocator = model_runner.token_to_kv_pool_allocator
        self.use_sliding_window_kv_pool = isinstance(self.token_to_kv_pool, SWAKVPool)
        # Lets the Triton wrappers specialize on PAGE_SIZE; page_size=1 is
        # byte-identical to the slot-based envelope.
        self.page_size = getattr(model_runner, "page_size", 1) or 1
        self.kv_index_translator = model_runner.kv_index_translator
        self.num_draft_tokens = get_spec().speculative_num_draft_tokens
        self.dllm_block_size = (
            model_runner.decode_num_tokens_per_req()
            if get_exec().dllm.dllm_algorithm is not None
            else None
        )
        self.target_verify_num_tokens_per_req = model_runner.decode_num_tokens_per_req()
        self.speculative_num_steps = get_spec().speculative_num_steps
        self.topk = get_spec().speculative_eagle_topk or 0
        # Split-KV verify is bit-equivalent only for a pure-causal chain (topk==1)
        # and is gfx95-only; else fall back to extend_attention_fwd.
        self.use_verify_splitkv = (
            is_gfx95_supported()
            and envs.SGLANG_ENABLE_SPLITKV_VERIFY.get()
            and self.topk == 1
        )
        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA
        # The grouped-head verify kernel is tuned for Kimi-K3 MLA and Qwen3.5
        # GQA with exactly one TP-local KV head.
        self.use_verify_shared_kv = _should_use_verify_shared_kv(
            model_runner.model_config,
            self.topk,
            self.use_mla,
            self.use_verify_splitkv,
        )
        # TODO: this logic should be fixed in non-hip platform
        self.is_hip_dspark_draft = (
            _is_hip
            and model_runner.is_draft_worker
            and model_runner.spec_algorithm.is_dspark()
        )
        if self.is_hip_dspark_draft:
            # Drafts never join the dcp group so we ignore it
            self.dcp_size = 1
            self.dcp_rank = 0
        else:
            self.dcp_size = get_parallel().attn_dcp_size
            self.dcp_rank = get_parallel().attn_dcp_rank
        self.num_head = (
            model_runner.model_config.get_max_num_attention_heads()
            // get_parallel().attn_tp_size
        ) * self.dcp_size
        self.num_kv_head = model_runner.model_config.get_num_kv_heads(
            get_parallel().attn_tp_size, get_parallel().attn_dcp_size
        )
        # Ported from the dump fork: per-layer state for the env-driven
        # DUMP_KVCACHE Q/K/V hook in ``forward_extend``. Inert unless
        # ``DUMP_KVCACHE=true`` so it's safe in production.
        self._dump_kvcache_enabled = get_bool_env_var("DUMP_KVCACHE", "false")
        self._dump_kv_done_layers = set()
        self._dump_saved_tokens = {}
        self._dump_chunk_idx = {}
        mla_config = model_runner.model_config
        self.use_dense_fp8_chunked_prefill = (
            self.use_mla
            and is_gfx95_supported()
            and envs.SGLANG_TRITON_DENSE_PREFILL_ATTN.get()
            and envs.SGLANG_TRITON_FP8_PREFILL_ATTN.get()
            and model_runner.kv_cache_dtype == torch.float8_e4m3fn
            and self.num_head == 12
            and mla_config.qk_nope_head_dim + mla_config.qk_rope_head_dim == 192
            and mla_config.v_head_dim == 128
            and mla_config.kv_lora_rank == 512
        )
        # forward_mha discovers these hooks dynamically. Hiding them when the
        # exact Kimi-K3 FP8 configuration is absent keeps all other models on
        # their existing code paths.
        if not self.use_dense_fp8_chunked_prefill:
            self.prepare_chunked_prefill_qkv = None
            self.pack_prefix_chunk_kv = None
        # The decode kernel's "// Lv" stride trick requires attn_logits.shape[-1]
        # to exactly match the layer's v_head_dim, so hybrid SWA models with
        # differing SWA/full v_head_dim need a second buffer for SWA layers.
        full_v_head_dim = model_runner.model_config.v_head_dim
        swa_v_head_dim = model_runner.model_config.swa_v_head_dim
        if self.sliding_window_size is not None and swa_v_head_dim != full_v_head_dim:
            self.v_head_dim = full_v_head_dim
            self.swa_v_head_dim = swa_v_head_dim
        elif mambaish_config(model_runner.model_config) is not None:
            # For hybrid linear models, layer_id = 0 may not be full attention
            # (e.g. NemotronH's full-attn layers are [5,12,19,...]). mambaish_config
            # unions mamba2 (NemotronH/FalconH1/...), hybrid-GDN, kimi-linear, and
            # linear-attn specs, so we ask get_v_head_dim() instead of indexing
            # layer 0, which is not guaranteed to be a full-attention layer.
            self.v_head_dim = model_runner.token_to_kv_pool.get_v_head_dim()
            self.swa_v_head_dim = None
        else:
            _pool = model_runner.token_to_kv_pool
            if _is_packed_mla_pool(_pool):
                # The packed MLA pool has no BF16 value buffer to measure -- reading
                # one is exactly the mistake it refuses to serve -- so ask it for the
                # width instead. Everything else keeps the buffer probe.
                self.v_head_dim = _pool.kv_lora_rank
            elif _is_int2_pool(_pool):
                # The int2 pools store V packed four values per byte, so the raw
                # buffer's last dim is head_dim // 4; the pool records the width.
                self.v_head_dim = _pool.v_head_dim
            else:
                # Use start_layer instead of 0 to handle pipeline parallelism.
                # In PP, start_layer may be > 0, so layer 0 isn't in this stage's buffer.
                self.v_head_dim = _pool.get_value_buffer(_pool.start_layer).shape[-1]
            self.swa_v_head_dim = None
        self.packed_mla_pool = (
            _is_packed_mla_pool(model_runner.token_to_kv_pool) and self.use_mla
        )
        if self.dcp_size > 1 and (
            self.packed_mla_pool or _is_int2_pool(model_runner.token_to_kv_pool)
        ):
            # The int2 / packed-latent decode and extend paths dispatch before
            # the DCP branches and read the whole KV, not this rank's shard.
            raise NotImplementedError(
                "OSCAR int2 / packed-latent KV pools do not support DCP "
                "(attn_dcp_size > 1) on the Triton backend."
            )
        self.max_context_len = model_runner.model_config.context_len
        # Group-factored decode: ON by default above a context threshold.
        #
        # Measured end to end on GLM-5.2-FP8 (tp8, B200, packed 2-bit + OSCAR),
        # gf / production decode tok/s, two back-to-back server launches with
        # this as the only variable:
        #
        #     ctx      conc=1   conc=8   conc=32
        #     1000      0.877    0.967     0.922     <-- gf LOSES
        #     2000      0.985    1.011     1.044
        #     4000      1.068    1.037     1.092
        #     16000     1.295    1.285     1.660
        #     32000     1.369    1.353     1.720
        #
        # The crossover sits between 2k and 4k, and the loss at 1k reproduces on
        # a different model (0.862x on DeepSeek-V2-Lite), so it is the kernel,
        # not node noise. gf's window pass is a fixed per-STEP cost: at 1k it is
        # most of the work, at 32k it is a rounding error, which is also why the
        # ratio IMPROVES with concurrency once the context is long.
        #
        # Why this is decided ONCE, from the server's context length, and not
        # per batch from the actual sequence lengths: a CUDA graph's capture key
        # is the batch size, NOT the sequence length, so one graph serves
        # seq=100 and seq=32000 alike. A per-batch branch would simply bake in
        # whichever side ran at capture time and never execute again at replay.
        # Per-server is the only adaptivity that survives graph capture.
        #
        # The threshold is 8192 rather than the 3000-ish crossover because the
        # errors are asymmetric: guessing wrong costs at most ~12% when prompts
        # turn out short, and costs a 1.3-1.7x speedup when they turn out long.
        # A server configured for long context is one where long decodes are
        # worth optimizing for.
        self._gf_enabled = (
            envs.SGLANG_OSCAR_MLA_PACKED_GF.get()
            if envs.SGLANG_OSCAR_MLA_PACKED_GF.is_set()
            else self.max_context_len >= 8192
        )
        # Persistent split-count scratch for the gf path; sized on first use and
        # kept alive for the lifetime of any CUDA graph that captured it.
        self._gf_split_bufs = None
        # ``mixed_kv_enabled()`` is True only for ``UnifiedInt2HPKVPool``
        # (SWAKVPool / MHA pools lack the method), so this gate already
        # excludes the plain hybrid-SWA path. The unified pool can now span
        # heterogeneous SWA geometry (gemma4_unified two-group), in which case
        # ``sliding_window_size`` / ``swa_v_head_dim`` are set on the backend
        # but the pool is still the unified mixed pool. The mixed-KV decode
        # reads the full context (there is no windowed mixed-decode variant),
        # so for the two-group case sliding layers attend over the full
        # sequence in *decode*; the sliding-window mask still applies in
        # prefill (extend uses layer.sliding_window_size).
        self.enable_mixed_kv = (
            _pool_mixed_kv_active(model_runner.token_to_kv_pool) and not self.use_mla
        )
        self.mixed_hp_prefix_tokens = (
            model_runner.token_to_kv_pool.hp_prefix_tokens
            if self.enable_mixed_kv
            else 0
        )
        self.mixed_hp_recent_tokens = (
            model_runner.token_to_kv_pool.hp_recent_tokens
            if self.enable_mixed_kv
            else 0
        )
        self.mixed_hp_global_offset = (
            model_runner.token_to_kv_pool.hp_global_offset
            if self.enable_mixed_kv
            else 0
        )
        # Mixed-KV decode uses a fixed HP split count because the HP window is
        # bounded by ``hp_prefix + hp_recent + flush_interval - 1`` tokens.
        # ``SGLANG_MIXED_KV_HP_MAX_SPLITS`` is therefore the direct per-request
        # HP cap for the unified int2 decode path.
        self.max_hp_kv_splits = (
            envs.SGLANG_MIXED_KV_HP_MAX_SPLITS.get()
            if self.enable_mixed_kv
            else 0
        )
        # Output dtype for per-tier intermediate buffers in the mixed-KV path.
        self.model_dtype = model_runner.dtype
        self.device = model_runner.device
        self.device_core_count = get_device_core_count(model_runner.gpu_id)
        # Lean decode persistent-grid size (depends only on head architecture).
        kv_group_num = self.num_head // self.num_kv_head
        self.lean_total_programs, _, _ = _lean_decode_launch_params(
            self.num_kv_head, kv_group_num
        )
        # BLOCK_M for Lean partial-result buffers; kept as an attribute so the
        # cuda-graph / eager buffer allocators (separate methods) can size them.
        self.lean_block_m = _LEAN_BLOCK_M
        self.static_kv_splits = get_bool_env_var(
            "SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS", "false"
        )
        self.max_kv_splits = get_exec().kernel.triton_attention_num_kv_splits
        if self.use_mla and not _is_xpu:
            self.max_kv_splits = _mla_decode_kv_splits_cap(
                self.max_kv_splits,
                self.device_core_count,
                self.max_context_len,
            )
            if _is_gfx942:
                # gfx942's 304 CUs round up to 512 splits, doubling the persistent
                # fp32 attn_logits buffer to ~4 GiB on Kimi-K2.6 and faulting in
                # ROCm graph replay; pin to 256 to match validated gfx950 behavior.
                self.max_kv_splits = min(self.max_kv_splits, 256)
        if _is_cuda:
            self.use_pdl = is_arch_support_pdl()
        else:
            self.use_pdl = False

        self.allow_bidirectional_attention_in_extend = (
            # BCG captures one complete prefill forward. It is therefore safe
            # for encoder-style attention, unlike the other CUDA graph modes
            # that can split or pad requests. Eager prefill remains supported
            # as before.
            (
                cuda_graph_fully_disabled()
                or check_cuda_graph_backend(Phase.PREFILL, Backend.BREAKABLE)
            )
            and get_schedule().chunked_prefill_size == -1
        )

        self.enable_deterministic = (
            get_exec().deterministic.enable_deterministic_inference
        )

        if self.enable_deterministic:
            # Fixed split tile size for batch invariance
            self.split_tile_size = get_int_env_var(
                "SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE", 256
            )
            self.static_kv_splits = False
        else:
            self.split_tile_size = get_exec().kernel.triton_attention_split_tile_size

        if self.split_tile_size is not None:
            self.max_kv_splits = (
                self.max_context_len + self.split_tile_size - 1
            ) // self.split_tile_size

        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        # TODO(Jianan Ji): verify behavior when kv_indptr_buf is provided and sliding window is enabled
        if kv_indptr_buf is None:
            self.kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )
        else:
            self.kv_indptr = kv_indptr_buf

        # Sliding window may need a second buffer for interleaved attention types
        self.window_kv_indptr = None
        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            if kv_indptr_buf is None:
                self.window_kv_indptr = torch.zeros(
                    (max_bs + 1,), dtype=torch.int32, device=model_runner.device
                )
            else:
                self.window_kv_indptr = torch.zeros_like(kv_indptr_buf)

        if not self.skip_prefill:
            self.qo_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int64, device=model_runner.device
            )

            self.mask_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int64, device=model_runner.device
            )

        self.forward_metadata: ForwardMetadata = None
        self._verify_mask = None
        # Tree-mask scratch is fetched from the target backend only.
        self.is_draft_runner = model_runner.is_draft_worker

        # Auto-detect BLOCK_M that extend_attention kernel will use for this model.
        # This is used by the scheduler's tile-budget admission logic to match
        # the kernel's actual tile size.
        head_dim = model_runner.model_config.head_dim
        from sglang.kernels.ops.attention.extend_attention import (
            _get_block_sizes_for_extend_attention,
        )

        _, _, _, block_m, _, _ = _get_block_sizes_for_extend_attention(
            Lq=head_dim, Lv=head_dim
        )
        self.extend_attention_block_m = block_m

    def get_num_kv_splits(
        self,
        num_kv_splits: torch.Tensor,
        seq_lens: torch.Tensor,
        max_kv_splits: Optional[int] = None,
    ):
        """Fill ``num_kv_splits`` with a per-sequence split count.

        ``max_kv_splits`` overrides the per-call upper bound (defaults to
        ``self.max_kv_splits``). The mixed-KV path uses the override to cap
        the HP-side split count independently of the quant/primary side.
        """
        if max_kv_splits is None:
            max_kv_splits = self.max_kv_splits
        num_token, num_seq = num_kv_splits.shape[0], seq_lens.shape[0]
        # NOTE(alcanderian): Considering speculative_decodeing,
        # num_kv_splits.shape[0] will be topk * real_num_token.
        # And the real_num_token is num_seq in decoding phase.
        num_group = num_token // num_seq

        assert num_group * num_seq == num_token, (
            f"num_seq({num_seq}), num_token({num_token}), something goes wrong!"
        )

        if (
            self.static_kv_splits or self.device_core_count <= 0
        ) and not self.enable_deterministic:
            num_kv_splits.fill_(max_kv_splits)
            return

        if self.split_tile_size is not None and self.enable_deterministic:
            if num_group > 1:
                expanded_seq_lens = seq_lens.repeat_interleave(num_group)
            else:
                expanded_seq_lens = seq_lens

            num_kv_splits[:] = torch.clamp(
                (expanded_seq_lens + self.split_tile_size - 1)
                // self.split_tile_size,
                max=max_kv_splits,
            )
            return

        if num_seq < 256:
            SCHEDULE_SEQ = 256
        else:
            SCHEDULE_SEQ = triton.next_power_of_2(num_seq)

        get_num_kv_splits_triton[(1,)](
            num_kv_splits,
            seq_lens,
            num_seq,
            num_group,
            self.num_head,
            self.num_kv_head,
            max_kv_splits,
            self.device_core_count,
            MAX_NUM_SEQ=SCHEDULE_SEQ,
        )

    def _build_mixed_kv_indices(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        hp_kv_indptr: torch.Tensor,
        hp_kv_indices: torch.Tensor,
        quant_kv_indptr: torch.Tensor,
        quant_kv_indices: torch.Tensor,
        bs: int,
        start_pos: Optional[torch.Tensor] = None,
    ):
        """Classify each token's slot id as HP vs quant and scatter into the
        caller-provided per-tier index buffers.

        ``start_pos`` (optional, int32 ``[bs]``): per-request scan start
        position. When ``None`` the scan covers the full ``[0, seq_len)`` range
        (every existing caller / full-attention layer -- unchanged). When given
        (sliding-window mixed-decode variant) the scan covers
        ``[start_pos, seq_len)`` so out-of-window positions (the prefix-sink HP
        tokens and the out-of-window quant bulk) are dropped before they reach
        the per-tier kernels.

        Sync-free on the decode hot path. Previously this routine ran a
        ``for i in range(bs)`` Python loop with ``rows[hp_mask[i]]``
        masked-selects whose output shape is data-dependent -- each
        masked-select forces a cudaStreamSynchronize so PyTorch can learn the
        size. That was the single biggest CPU-critical-path blocker in
        mixed-KV decode after the flush pipeline was fused.

        The replacement:
          * ``hp_kv_indptr`` is built from a Triton HP-length counting kernel
            that streams ``req_to_token`` through the req-pool indirection --
            no dense gather/mask materialization, no sync.
          * ``quant_kv_indptr`` is derived from the full sequence lengths minus
            the HP prefix sum, so there is no separate quant-length pass.
          * The per-(req, pos) scatter into ``hp_kv_indices`` /
            ``quant_kv_indices`` happens inside a single triton kernel
            (``_scatter_mixed_kv_indices_kernel``) that walks each request's
            ``[0, seq_len)`` range in ``BLOCK_SIZE`` chunks and uses
            ``tl.cumsum`` for within-block ranks. No Python bs-loop, no
            masked-select, no sync.
        """
        seq_lens = seq_lens[:bs]
        req_pool_indices = req_pool_indices[:bs].to(torch.int64)
        # Cast seq_lens to int32 once; both mixed-KV Triton kernels want
        # int32. Keeps the conversion off the hot path's per-step alloc trail.
        seq_lens_i32 = seq_lens.to(torch.int32)
        if start_pos is not None:
            start_pos = start_pos[:bs].to(torch.int32)
            # Per-request scanned length = seq_len - start (windowed). Clamp at 0
            # for safety though start is always <= seq_len by construction.
            scanned_lens_i32 = torch.clamp(seq_lens_i32 - start_pos, min=0)
        else:
            scanned_lens_i32 = seq_lens_i32
        hp_lens = torch.empty_like(seq_lens_i32)
        # Count directly from ``req_to_token`` so the hot path no longer
        # materializes a dense gathered ``rows`` tensor or boolean masks.
        _count_mixed_hp_lens_kernel[(bs,)](
            self.req_to_token,
            req_pool_indices,
            seq_lens_i32,
            hp_lens,
            start_pos,
            self.req_to_token.stride(0),
            HP_OFFSET=int(self.mixed_hp_global_offset),
            BLOCK_SIZE=512,
            num_warps=2,
            num_stages=1,
        )

        # indptr = exclusive prefix sum of per-req lengths. ``cumsum`` + slice
        # assignment are shape-static so no D2H read is forced. The leading
        # ``[0]`` element stays at zero from the buffer's ``torch.zeros``
        # allocation; assigning a Python scalar there would force a sync H2D
        # copy that blocks the CPU on prior decode work, recreating the
        # ~1.5 ms inter-step bubble. The quant length is derived from the
        # *scanned* (windowed) length minus the HP prefix sum.
        hp_kv_indptr[1 : bs + 1] = torch.cumsum(hp_lens, dim=0)
        quant_kv_indptr[1 : bs + 1] = torch.cumsum(scanned_lens_i32, dim=0)
        quant_kv_indptr[1 : bs + 1] -= hp_kv_indptr[1 : bs + 1]

        # Single triton launch scatters the tier-classified slot ids directly
        # into the pre-sized destination buffers. BLOCK_SIZE here is the
        # per-request chunk size; picking 512 matches
        # ``create_flashinfer_kv_indices_triton`` and balances occupancy
        # against the ``tl.cumsum`` reduction depth.
        _scatter_mixed_kv_indices_kernel[(bs,)](
            self.req_to_token,
            req_pool_indices,
            seq_lens_i32,
            hp_kv_indptr,
            quant_kv_indptr,
            hp_kv_indices,
            quant_kv_indices,
            start_pos,
            self.req_to_token.stride(0),
            HP_OFFSET=int(self.mixed_hp_global_offset),
            BLOCK_SIZE=512,
            num_warps=2,
            num_stages=1,
        )

    def _mixed_swa_start_pos(self, seq_lens: torch.Tensor) -> torch.Tensor:
        """Per-request first in-window position for the sliding mixed-decode scan.

        ``window`` = sliding_window_size + 1 tokens to match the validated
        prefill mask (key >= q_abs - (sliding_window_size) in
        ``_sdpa_varlen_prefill`` / flash ``window_size=(w-1, 0)``).
        """
        window_tokens = self.sliding_window_size + 1
        return torch.clamp(seq_lens.to(torch.int32) - window_tokens, min=0)

    def _alloc_eager_mixed_kv_metadata(
        self, forward_batch: ForwardBatch, bs: int
    ) -> dict:
        """Per-tier indices, split counts and the combined stage-1 scratch for
        the eager unified int2 decode; returns the ``ForwardMetadata`` mixed_*
        fields."""
        # This is the eager path, so ``bs`` is the real batch size and
        # ``seq_lens_sum`` bounds both tiers exactly: every position is
        # classified as either HP or quant, so ``hp_total + quant_total ==
        # seq_lens_sum``. Padded replays never come through here -- they go
        # through ``_fill_cuda_graph_mixed_kv_buffers``, which writes into the
        # ``max_bs * max_context_len`` graph buffers.
        seq_lens_sum = forward_batch.seq_lens_sum
        if seq_lens_sum is None:
            # gpu_only: seq_lens_sum may be None; over-allocate is safe (ragged write).
            seq_lens_sum = bs * self.max_context_len
        dev = self.device
        fields = {
            "mixed_hp_kv_indptr": torch.zeros((bs + 1,), dtype=torch.int32, device=dev),
            "mixed_quant_kv_indptr": torch.zeros(
                (bs + 1,), dtype=torch.int32, device=dev
            ),
            "mixed_hp_kv_indices": torch.empty(
                seq_lens_sum, dtype=torch.int64, device=dev
            ),
            "mixed_quant_kv_indices": torch.empty(
                seq_lens_sum, dtype=torch.int64, device=dev
            ),
        }
        total_splits = self.max_kv_splits + self.max_hp_kv_splits
        # Single combined stage-1 scratch. LSE is pre-filled with -inf so the
        # tier-agnostic stage-2 can skip unused splits.
        fields["mixed_attn_logits"] = torch.empty(
            (bs, self.num_head, total_splits, self.v_head_dim),
            dtype=torch.float32,
            device=dev,
        )
        fields["mixed_attn_lse"] = torch.full(
            (bs, self.num_head, total_splits),
            float("-inf"),
            dtype=torch.float32,
            device=dev,
        )
        # Separate SWA-geometry mixed scratch (sliding layers).
        if self.swa_v_head_dim is not None:
            fields["mixed_swa_attn_logits"] = torch.empty(
                (bs, self.num_head, total_splits, self.swa_v_head_dim),
                dtype=torch.float32,
                device=dev,
            )
            fields["mixed_swa_attn_lse"] = torch.full(
                (bs, self.num_head, total_splits),
                float("-inf"),
                dtype=torch.float32,
                device=dev,
            )
        fields["mixed_hp_num_kv_splits"] = torch.full(
            (bs,), self.max_hp_kv_splits, dtype=torch.int32, device=dev
        )
        # HP uses the fixed cap above; only the quant tier is right-sized, and
        # it uses the full sequence length as a cheap planning proxy instead of
        # per-tier mixed-KV counts.
        quant_num_kv_splits = torch.empty((bs,), dtype=torch.int32, device=dev)
        self.get_num_kv_splits(quant_num_kv_splits, forward_batch.seq_lens)
        fields["mixed_quant_num_kv_splits"] = quant_num_kv_splits
        self._build_mixed_kv_indices(
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            fields["mixed_hp_kv_indptr"],
            fields["mixed_hp_kv_indices"],
            fields["mixed_quant_kv_indptr"],
            fields["mixed_quant_kv_indices"],
            bs,
        )
        # Sliding-window mixed-decode indices (gemma4_unified two-group). For
        # SLIDING layers the HP+quant tiers must be capped to the last
        # ``sliding_window`` tokens; otherwise a sliding layer would (wrongly)
        # attend to the full prior context in decode for seq_len >
        # sliding_window. We scan only positions [seq_len - window, seq_len):
        # this drops the out-of-window quant bulk AND the prefix-sink HP tokens
        # (which fall below the window once seq_len-window > sink).
        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            fields["mixed_swa_hp_kv_indptr"] = torch.zeros(
                (bs + 1,), dtype=torch.int32, device=dev
            )
            fields["mixed_swa_quant_kv_indptr"] = torch.zeros(
                (bs + 1,), dtype=torch.int32, device=dev
            )
            # Windowed scan emits at most ``window_tokens`` indices per
            # request; the full-context buffers are an upper bound, so reuse
            # that size to avoid a sync on the windowed total.
            fields["mixed_swa_hp_kv_indices"] = torch.empty(
                seq_lens_sum, dtype=torch.int64, device=dev
            )
            fields["mixed_swa_quant_kv_indices"] = torch.empty(
                seq_lens_sum, dtype=torch.int64, device=dev
            )
            self._build_mixed_kv_indices(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                fields["mixed_swa_hp_kv_indptr"],
                fields["mixed_swa_hp_kv_indices"],
                fields["mixed_swa_quant_kv_indptr"],
                fields["mixed_swa_quant_kv_indices"],
                bs,
                start_pos=self._mixed_swa_start_pos(forward_batch.seq_lens),
            )
        return fields

    def _init_cuda_graph_mixed_kv_state(self, max_bs: int, max_num_tokens: int):
        """Capture-stable per-tier index buffers and stage-1 scratch for the
        unified int2 decode."""
        dev = self.device
        self.cuda_graph_mixed_hp_kv_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=dev
        )
        self.cuda_graph_mixed_quant_kv_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=dev
        )
        self.cuda_graph_mixed_hp_kv_indices = torch.zeros(
            (max_num_tokens * self.max_context_len),
            dtype=torch.int64,
            device=dev,
        )
        self.cuda_graph_mixed_quant_kv_indices = torch.zeros(
            (max_num_tokens * self.max_context_len),
            dtype=torch.int64,
            device=dev,
        )
        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            # Sliding layers need their own windowed HP/quant indices. Without
            # them the decode path falls back to the full-context indices and
            # reads KV from outside the window -- which shows up as digit soup
            # at small cuda-graph bs and an illegal access at larger bs.
            self.cuda_graph_mixed_swa_hp_kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=dev
            )
            self.cuda_graph_mixed_swa_quant_kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=dev
            )
            self.cuda_graph_mixed_swa_hp_kv_indices = torch.zeros(
                (max_num_tokens * self.max_context_len),
                dtype=torch.int64,
                device=dev,
            )
            self.cuda_graph_mixed_swa_quant_kv_indices = torch.zeros(
                (max_num_tokens * self.max_context_len),
                dtype=torch.int64,
                device=dev,
            )
        else:
            self.cuda_graph_mixed_swa_hp_kv_indptr = None
            self.cuda_graph_mixed_swa_quant_kv_indptr = None
            self.cuda_graph_mixed_swa_hp_kv_indices = None
            self.cuda_graph_mixed_swa_quant_kv_indices = None
        total_splits = self.max_kv_splits + self.max_hp_kv_splits
        # Sliding layers have their own head geometry (gemma4: 256 vs 512 on
        # full layers), so they need their own stage-1 scratch. Sharing the
        # full-geometry buffer writes at the wrong stride.
        if self.swa_v_head_dim is not None:
            self.cuda_graph_mixed_swa_attn_logits = torch.zeros(
                (max_num_tokens, self.num_head, total_splits, self.swa_v_head_dim),
                dtype=torch.float32,
                device=dev,
            )
            self.cuda_graph_mixed_swa_attn_lse = torch.full(
                (max_num_tokens, self.num_head, total_splits),
                float("-inf"),
                dtype=torch.float32,
                device=dev,
            )
        else:
            self.cuda_graph_mixed_swa_attn_logits = None
            self.cuda_graph_mixed_swa_attn_lse = None
        # Single combined stage-1 scratch. LSE pre-filled to -inf so the
        # tier-agnostic stage-2 skips unused splits.
        self.cuda_graph_mixed_attn_logits = torch.zeros(
            (max_num_tokens, self.num_head, total_splits, self.v_head_dim),
            dtype=torch.float32,
            device=dev,
        )
        self.cuda_graph_mixed_attn_lse = torch.full(
            (max_num_tokens, self.num_head, total_splits),
            float("-inf"),
            dtype=torch.float32,
            device=dev,
        )
        self.cuda_graph_mixed_hp_num_kv_splits = torch.full(
            (max_num_tokens,), self.max_hp_kv_splits, dtype=torch.int32, device=dev
        )
        self.cuda_graph_mixed_quant_num_kv_splits = torch.zeros(
            (max_num_tokens,), dtype=torch.int32, device=dev
        )

    def _cuda_graph_mixed_metadata_fields(self) -> dict:
        """The mixed_* ``ForwardMetadata`` fields for a captured decode: views
        of the capture-stable buffers, so capture and replay read one address."""
        if not self.enable_mixed_kv:
            return {}
        return dict(
            mixed_hp_kv_indptr=self.cuda_graph_mixed_hp_kv_indptr,
            mixed_hp_kv_indices=self.cuda_graph_mixed_hp_kv_indices,
            mixed_quant_kv_indptr=self.cuda_graph_mixed_quant_kv_indptr,
            mixed_quant_kv_indices=self.cuda_graph_mixed_quant_kv_indices,
            mixed_attn_logits=self.cuda_graph_mixed_attn_logits,
            mixed_attn_lse=self.cuda_graph_mixed_attn_lse,
            mixed_swa_attn_logits=self.cuda_graph_mixed_swa_attn_logits,
            mixed_swa_attn_lse=self.cuda_graph_mixed_swa_attn_lse,
            mixed_swa_hp_kv_indptr=self.cuda_graph_mixed_swa_hp_kv_indptr,
            mixed_swa_hp_kv_indices=self.cuda_graph_mixed_swa_hp_kv_indices,
            mixed_swa_quant_kv_indptr=self.cuda_graph_mixed_swa_quant_kv_indptr,
            mixed_swa_quant_kv_indices=self.cuda_graph_mixed_swa_quant_kv_indices,
            mixed_hp_num_kv_splits=self.cuda_graph_mixed_hp_num_kv_splits,
            mixed_quant_num_kv_splits=self.cuda_graph_mixed_quant_num_kv_splits,
        )

    def _fill_cuda_graph_mixed_kv_buffers(
        self, bs: int, req_pool_indices: torch.Tensor, seq_lens: torch.Tensor
    ):
        """Refill the capture-stable per-tier buffers for a decode capture or
        replay (runs before ``graph.replay()``, outside the captured region)."""
        self._build_mixed_kv_indices(
            req_pool_indices,
            seq_lens,
            self.cuda_graph_mixed_hp_kv_indptr,
            self.cuda_graph_mixed_hp_kv_indices,
            self.cuda_graph_mixed_quant_kv_indptr,
            self.cuda_graph_mixed_quant_kv_indices,
            bs,
        )
        if self.cuda_graph_mixed_swa_quant_kv_indptr is not None:
            # Same windowed indices the non-graph path builds; the decode path
            # only takes the sliding branch when these are non-None, so
            # skipping them silently served out-of-window KV on every sliding
            # layer.
            self._build_mixed_kv_indices(
                req_pool_indices,
                seq_lens,
                self.cuda_graph_mixed_swa_hp_kv_indptr,
                self.cuda_graph_mixed_swa_hp_kv_indices,
                self.cuda_graph_mixed_swa_quant_kv_indptr,
                self.cuda_graph_mixed_swa_quant_kv_indices,
                bs,
                start_pos=self._mixed_swa_start_pos(seq_lens[:bs]),
            )
        self.cuda_graph_mixed_hp_num_kv_splits[:bs] = self.max_hp_kv_splits
        self.get_num_kv_splits(
            self.cuda_graph_mixed_quant_num_kv_splits[:bs], seq_lens[:bs]
        )
        # The unified attention wrapper fills LSE with -inf every call, so the
        # shared scratch is always in a known state entering stage-2. No extra
        # reset needed here.

    def _forward_extend_quantized_dense(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        causal: bool,
        pre_rotated_q: Optional[torch.Tensor] = None,
        pre_rotated_k: Optional[torch.Tensor] = None,
        pre_rotated_v: Optional[torch.Tensor] = None,
        need_v_inverse_override: Optional[bool] = None,
    ):
        kv_pool = self.token_to_kv_pool
        q3 = (
            pre_rotated_q
            if pre_rotated_q is not None
            else q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        )
        k3 = pre_rotated_k if pre_rotated_k is not None else k.contiguous()
        v3 = pre_rotated_v if pre_rotated_v is not None else v.contiguous()
        if need_v_inverse_override is None:
            q3, k3, v3, need_v_inverse = prepare_quantized_extend_qkv(
                kv_pool,
                layer,
                q3,
                k3,
                v3,
                q_already_hadamard_transformed=pre_rotated_q is not None,
                kv_already_hadamard_transformed=(
                    pre_rotated_k is not None and pre_rotated_v is not None
                ),
            )
        else:
            need_v_inverse = need_v_inverse_override

        prefix_k, prefix_v = dequantize_prefix_kv(
            kv_pool,
            layer.layer_id,
            self.forward_metadata.kv_indices,
            q3.dtype,
        )

        unified_k_parts = []
        unified_v_parts = []
        unified_k_lens = []
        prefix_indptr = self.forward_metadata.kv_indptr
        extend_start_loc = forward_batch.extend_start_loc
        for i, extend_len in enumerate(forward_batch.extend_seq_lens_cpu):
            prefix_start = int(prefix_indptr[i].item())
            prefix_end = int(prefix_indptr[i + 1].item())
            extend_start = int(extend_start_loc[i].item())
            extend_end = extend_start + int(extend_len)
            req_k = torch.cat(
                [prefix_k[prefix_start:prefix_end], k3[extend_start:extend_end]], dim=0
            )
            req_v = torch.cat(
                [prefix_v[prefix_start:prefix_end], v3[extend_start:extend_end]], dim=0
            )
            unified_k_parts.append(req_k)
            unified_v_parts.append(req_v)
            unified_k_lens.append(req_k.shape[0])

        unified_k = torch.cat(unified_k_parts, dim=0) if unified_k_parts else k3[:0]
        unified_v = torch.cat(unified_v_parts, dim=0) if unified_v_parts else v3[:0]
        cu_seqlens_q = self.forward_metadata.qo_indptr.to(torch.int32)
        cu_seqlens_k = torch.empty(
            (len(unified_k_lens) + 1,), dtype=torch.int32, device=self.device
        )
        cu_seqlens_k[0] = 0
        cu_seqlens_k[1:] = torch.cumsum(
            torch.tensor(unified_k_lens, dtype=torch.int32, device=self.device), dim=0
        )

        # Sliding-window layers (gemma4_unified two-group): the prefix here is
        # the *full* dequantized context, so we let flash_attn apply the
        # sliding-window mask via window_size=(w-1, 0). Full-attention layers
        # keep the unbounded (-1, -1) window. This makes int2 prefill match the
        # model's per-layer attention span.
        if layer.sliding_window_size is not None and layer.sliding_window_size > 0:
            window_size = (layer.sliding_window_size - 1, 0)
        else:
            window_size = (-1, -1)

        head_dim = q3.shape[-1]
        softcap = logit_capping_mod(layer.logit_capping_method, layer.logit_cap)
        # Two reasons to leave FlashAttention: it caps head_dim at 256 (gemma4's
        # full-attention layers are 512), and sgl-kernel only builds it for
        # sm8x/sm90, so it raises on Blackwell. The SDPA pass handles arbitrary
        # head_dim, causal + sliding window via an additive mask, and MQA/GQA via
        # enable_gqa -- but it has no softcap, so a capping layer must not
        # silently take it.
        use_sdpa = head_dim > 256 or not _is_fa3_supported()
        if use_sdpa and softcap:
            raise NotImplementedError(
                f"int2 prefill needs a softcap ({softcap}) that the SDPA fallback "
                f"cannot apply, and FlashAttention is unavailable here "
                f"(head_dim={head_dim}, fa3_supported={_is_fa3_supported()})."
            )
        if use_sdpa:
            result = self._sdpa_varlen_prefill(
                q3,
                unified_k_parts,
                unified_v_parts,
                cu_seqlens_q,
                forward_batch.extend_seq_lens_cpu,
                unified_k_lens,
                layer.scaling,
                causal,
                window_size[0] if window_size[0] >= 0 else -1,
            )
        else:
            result = flash_attn_varlen_func(
                q=q3,
                k=unified_k,
                v=unified_v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max(forward_batch.extend_seq_lens_cpu),
                max_seqlen_k=max(unified_k_lens) if unified_k_lens else 0,
                softmax_scale=layer.scaling,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
            )
        result = apply_inverse_v_rotation(result, kv_pool, layer, need_v_inverse)
        o.copy_(result.view_as(o))
        return o

    # Query rows per SDPA call in the fallback path. 1024 x 32k x 8 heads x 4B
    # is ~1 GB, which fits alongside weights on every SKU we serve.
    _SDPA_Q_CHUNK_DEFAULT = 1024

    def _sdpa_varlen_prefill(
        self,
        q3: torch.Tensor,                  # [total_q, num_q_heads, head_dim]
        k_parts: list,                     # per-req [k_len_i, num_kv_heads, head_dim]
        v_parts: list,                     # per-req [k_len_i, num_kv_heads, v_head_dim]
        cu_seqlens_q: torch.Tensor,        # int32 [bs+1]
        extend_seq_lens_cpu,               # list[int] per req (query lengths)
        k_lens: list,                      # list[int] per req (full kv lengths)
        sm_scale: float,
        causal: bool,
        sliding_window: int,               # >=0 window size (w-1 left); -1 disabled
    ) -> torch.Tensor:
        """Varlen prefill via per-request SDPA, for head_dim > 256 (FA caps at
        256). Operates on the already-dequantized dense K/V. Builds an additive
        mask combining causality + (optional) sliding window. MQA/GQA handled by
        ``enable_gqa=True``. Returns ``[total_q, num_q_heads, v_head_dim]``.
        """
        _SDPA_Q_CHUNK = int(
            os.environ.get("SGLANG_SDPA_Q_CHUNK", self._SDPA_Q_CHUNK_DEFAULT)
        )
        num_q_heads = q3.shape[1]
        v_head_dim = v_parts[0].shape[-1] if v_parts else q3.shape[-1]
        out = q3.new_empty((q3.shape[0], num_q_heads, v_head_dim))
        q_starts = cu_seqlens_q.tolist()
        for i, q_len in enumerate(extend_seq_lens_cpu):
            q_len = int(q_len)
            if q_len == 0:
                continue
            k_len = int(k_lens[i])
            qs = int(q_starts[i])
            # [1, H, q_len, hd] / [1, Hkv, k_len, hd]
            ki = k_parts[i].transpose(0, 1).unsqueeze(0)
            vi = v_parts[i].transpose(0, 1).unsqueeze(0)
            k_abs = torch.arange(k_len, device=q3.device).unsqueeze(0)  # [1, k_len]
            gqa = num_q_heads != ki.shape[1]
            # Cap the score matrix at ~chunk * k_len entries. Each query row is
            # independent, so chunking changes nothing but peak memory.
            chunk = max(1, min(q_len, _SDPA_Q_CHUNK))
            for c0 in range(0, q_len, chunk):
                c1 = min(c0 + chunk, q_len)
                qi = q3[qs + c0 : qs + c1].transpose(0, 1).unsqueeze(0)
                # Query position p (0-based within request) corresponds to
                # absolute key index (k_len - q_len + p): the last q_len keys are
                # the extend tokens, the leading (k_len - q_len) are the prefix.
                q_abs = torch.arange(
                    k_len - q_len + c0, k_len - q_len + c1, device=q3.device
                ).unsqueeze(1)
                allowed = torch.ones(
                    (c1 - c0, k_len), dtype=torch.bool, device=q3.device
                )
                if causal:
                    allowed &= k_abs <= q_abs
                if sliding_window >= 0:
                    # window covers keys [q_abs - sliding_window, q_abs]
                    allowed &= k_abs >= (q_abs - sliding_window)
                attn_mask = torch.zeros(
                    (c1 - c0, k_len), dtype=qi.dtype, device=q3.device
                )
                attn_mask.masked_fill_(~allowed, float("-inf"))
                oi = torch.nn.functional.scaled_dot_product_attention(
                    qi, ki, vi, attn_mask=attn_mask, scale=sm_scale, enable_gqa=gqa
                )
                out[qs + c0 : qs + c1] = oi.squeeze(0).transpose(0, 1)
                del qi, attn_mask, allowed, q_abs, oi
        return out

    def _dcp_lens(self, lens: torch.Tensor, start: Optional[torch.Tensor] = None):
        return get_dcp_lens(lens, self.dcp_size, self.dcp_rank, start)

    def _dcp_kv_indices(
        self,
        req_pool_indices: torch.Tensor,
        lens: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: Optional[torch.Tensor] = None,
        kv_start_idx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Build per-DCP-rank sharded KV indptr/indices. eager passes kv_indices=None
        # (fresh tensor); cuda-graph passes an address-stable buffer to fill in place.
        dcp_lens = self._dcp_lens(lens, kv_start_idx)
        kv_indptr[1 : len(req_pool_indices) + 1] = torch.cumsum(dcp_lens, dim=0)
        kv_indptr = kv_indptr[: len(req_pool_indices) + 1]
        if kv_indices is None:
            kv_indices = torch.empty(
                int(dcp_lens.sum().item()), dtype=torch.int64, device=self.device
            )
        create_triton_kv_indices_for_dcp_triton[(len(req_pool_indices),)](
            self.req_to_token,
            req_pool_indices,
            dcp_lens,
            kv_indptr,
            kv_start_idx,
            kv_indices,
            self.req_to_token.stride(0),
            self.dcp_size,
            self.dcp_rank,
        )
        return kv_indptr, kv_indices, dcp_lens

    def _fill_kv_indptr_and_indices(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        kv_indices: torch.Tensor,
    ) -> torch.Tensor:
        kv_indptr = self.kv_indptr[: bs + 1]
        kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)
        self.kv_index_translator.fill_packed_read_stream(
            req_pool_indices=req_pool_indices[:bs],
            seq_lens=seq_lens[:bs],
            indptr=kv_indptr,
            total_tokens=kv_indices.numel(),
            out=kv_indices,
        )
        return kv_indptr

    def _update_decode_kv_buffers(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ):
        """Fill KV (and SWA) cuda-graph buffers for decode/idle mode.

        Returns ``(kv_indptr, window_kv_indptr, window_kv_lens, num_kv_splits_lens)``
        where ``window_kv_lens`` is ``None`` when sliding-window is disabled and
        ``num_kv_splits_lens`` is the per-request length used to size kv splits
        (per-DCP-rank length clamped to >=1 when DCP is enabled, full seq_lens
        otherwise).
        """
        seq_lens = seq_lens[:bs]
        req_pool_indices = req_pool_indices[:bs]
        if self.dcp_size > 1:
            # DCP: per-rank sharded; write into the same cuda-graph buffers
            # _build_cuda_graph_forward_metadata reads back.
            _, _, dcp_seq_lens = self._dcp_kv_indices(
                req_pool_indices,
                seq_lens,
                self.kv_indptr,
                self.cuda_graph_kv_indices,
                None,
            )
            kv_indptr = self.kv_indptr[: bs + 1]
            num_kv_splits_lens = dcp_seq_lens.clamp_min(1)
        else:
            kv_indptr = self._fill_kv_indptr_and_indices(
                bs, seq_lens, req_pool_indices, self.cuda_graph_kv_indices
            )
            num_kv_splits_lens = seq_lens
        window_kv_indptr = self.window_kv_indptr
        window_kv_lens = None
        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            window_kv_indptr, _, window_kv_lens, _ = update_sliding_window_buffer(
                self.window_kv_indptr,
                self.kv_index_translator,
                req_pool_indices,
                self.sliding_window_size,
                seq_lens,
                bs,
                token_to_kv_pool=self.token_to_kv_pool,
                window_kv_indices=self.cuda_graph_window_kv_indices,
            )
        return kv_indptr, window_kv_indptr, window_kv_lens, num_kv_splits_lens

    def _target_verify_num_tokens_per_req(self, spec_info: Optional[SpecInput]) -> int:
        # Runtime metadata may vary by step; nonpositive means use capture width.
        if spec_info is None or spec_info.num_tokens_per_req <= 0:
            return self.target_verify_num_tokens_per_req
        return spec_info.num_tokens_per_req

    def _update_target_verify_buffers(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        spec_info,
        req_pool_indices: torch.Tensor,
    ):
        """Fill all cuda-graph buffers for target_verify mode."""
        num_tokens_per_req = self._target_verify_num_tokens_per_req(spec_info)
        qo_indptr = self.qo_indptr[: bs + 1]
        qo_indptr[: bs + 1] = torch.arange(
            0,
            (1 + bs) * num_tokens_per_req,
            step=num_tokens_per_req,
            dtype=torch.int32,
            device=self.device,
        )
        kv_indptr = self._fill_kv_indptr_and_indices(
            bs, seq_lens, req_pool_indices, self.cuda_graph_kv_indices
        )
        window_kv_indptr = self.window_kv_indptr
        window_kv_indices = None
        window_num_kv_splits = None
        window_kv_offsets = None
        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            window_kv_indices = self.cuda_graph_window_kv_indices
            window_num_kv_splits = self.cuda_graph_window_num_kv_splits
            window_kv_offsets = self.cuda_graph_window_kv_offsets
            window_kv_indptr, window_kv_indices, _, window_kv_offsets[:bs] = (
                update_sliding_window_buffer(
                    self.window_kv_indptr,
                    self.kv_index_translator,
                    req_pool_indices,
                    self.sliding_window_size,
                    seq_lens[:bs],
                    bs,
                    token_to_kv_pool=self.token_to_kv_pool,
                    window_kv_indices=window_kv_indices,
                )
            )
        custom_mask = (
            self._verify_mask.buffer if self._verify_mask is not None else None
        )
        if spec_info is not None and spec_info.custom_mask is not None:
            custom_mask[: spec_info.custom_mask.shape[0]] = spec_info.custom_mask
        else:
            custom_mask = None
        seq_mask_len = num_tokens_per_req * (seq_lens + num_tokens_per_req)
        mask_indptr = self.mask_indptr[: bs + 1]
        mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len, dim=0)
        return (
            qo_indptr,
            kv_indptr,
            custom_mask,
            mask_indptr,
            window_kv_indptr,
            window_kv_indices,
            window_num_kv_splits,
            window_kv_offsets,
        )

    def _update_dllm_buffers(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ):
        # The current canvas is passed as dense K/V to extend attention. The
        # paged read stream must contain only the already encoded context.
        block_size = self.dllm_block_size
        prefix_lens = (seq_lens[:bs] - block_size).clamp_min(0)
        self.qo_indptr[: bs + 1] = torch.arange(
            0,
            (bs + 1) * block_size,
            block_size,
            dtype=torch.int32,
            device=self.device,
        )
        self._fill_kv_indptr_and_indices(
            bs, prefix_lens, req_pool_indices, self.cuda_graph_kv_indices
        )
        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            update_sliding_window_buffer(
                self.window_kv_indptr,
                self.kv_index_translator,
                req_pool_indices,
                self.sliding_window_size,
                prefix_lens,
                bs,
                token_to_kv_pool=self.token_to_kv_pool,
                window_kv_indices=self.cuda_graph_window_kv_indices,
            )

    def _update_draft_extend_buffers(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        req_pool_indices: torch.Tensor,
    ):
        """Fill QO + KV cuda-graph buffers for draft_extend mode."""
        seq_lens = seq_lens[:bs]
        # V2 draft-extend fills num_draft_tokens per req; num_steps+1 only equals
        # that when topk == 1.
        num_tokens_per_req = (
            self.num_draft_tokens
            if forward_mode.is_draft_extend_v2()
            else self.speculative_num_steps + 1
        )
        qo_indptr = self.qo_indptr[: bs + 1]
        qo_indptr[: bs + 1] = torch.arange(
            0,
            bs * num_tokens_per_req + 1,
            step=num_tokens_per_req,
            dtype=torch.int32,
            device=self.device,
        )
        # DRAFT_EXTEND_V2: kv_indptr/kv_indices cover only the prefix (extend K/V go
        # separately). Capture warmup lacks extend_seq_lens_tensor -> fall back to
        # zeros; clamp at 0 so padded rows (seq_lens==fill 1) don't go negative.
        if (
            spec_info is not None
            and getattr(spec_info, "extend_seq_lens_tensor", None) is not None
        ):
            extend_seq_lens = spec_info.extend_seq_lens_tensor[:bs].to(torch.int32)
        else:
            extend_seq_lens = torch.zeros(bs, dtype=torch.int32, device=seq_lens.device)
        kv_lens = torch.clamp(seq_lens - extend_seq_lens, min=0).to(torch.int32)
        kv_indptr = self._fill_kv_indptr_and_indices(
            bs, kv_lens, req_pool_indices, self.cuda_graph_kv_indices
        )
        return qo_indptr, kv_indptr, num_tokens_per_req

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        bs = forward_batch.batch_size
        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens
        forward_mode = forward_batch.forward_mode
        spec_info = forward_batch.spec_info

        if in_capture:
            assert forward_batch.encoder_lens is None, "Not supported"
            # Multi-step spec decode: kv buffers come from spec_info, not the
            # cuda-graph pool, so replay is not involved.
            if forward_mode.is_decode_or_idle() and spec_info is not None:
                self.forward_metadata = ForwardMetadata(
                    attn_logits=self.cuda_graph_attn_logits,
                    attn_lse=self.cuda_graph_attn_lse,
                    max_extend_len=None,
                    num_kv_splits=self.cuda_graph_num_kv_splits,
                    kv_indptr=spec_info.kv_indptr,
                    kv_indices=spec_info.kv_indices,
                    qo_indptr=None,
                    custom_mask=None,
                    mask_indptr=None,
                    window_kv_indptr=self.window_kv_indptr,
                    window_kv_indices=None,
                    window_num_kv_splits=None,
                    window_kv_offsets=None,
                    swa_attn_logits=self.cuda_graph_swa_attn_logits,
                    lean_Mp=self.cuda_graph_lean_Mp,
                    lean_Lp=self.cuda_graph_lean_Lp,
                    lean_Op=self.cuda_graph_lean_Op,
                    lean_locks=self.cuda_graph_lean_locks,
                )
                return

            self._apply_cuda_graph_metadata(
                bs=bs,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                forward_mode=forward_mode,
                spec_info=spec_info,
            )
            out_cache_loc_full_physical = self._fill_cuda_graph_write_locs(
                forward_batch, bs
            )
            swa_out_cache_loc = self._fill_cuda_graph_swa_out_cache_loc(
                forward_batch, in_capture=True
            )
            self.forward_metadata = self._build_cuda_graph_forward_metadata(
                bs,
                forward_mode,
                spec_info,
                swa_out_cache_loc,
                out_cache_loc_full_physical,
            )
        else:
            self._apply_cuda_graph_metadata(
                bs=bs,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                forward_mode=forward_mode,
                spec_info=spec_info,
            )
            # Metadata view is reused from capture; just refill the buffers.
            self._fill_cuda_graph_write_locs(forward_batch, bs)
            self._fill_cuda_graph_swa_out_cache_loc(forward_batch)

    def _fill_cuda_graph_swa_out_cache_loc(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ) -> Optional[torch.Tensor]:
        """Refill the SWA write-target buffer from the batch's derived
        sliding-window write loc, returning the [:n] view (None for non-SWA /
        multi-step draft) so the captured store reads fresh slots on replay.
        """
        if not self.use_sliding_window_kv_pool:
            return None
        out_cache_loc = forward_batch.out_cache_loc
        if (
            out_cache_loc is None
            or out_cache_loc.shape[0] > self.cuda_graph_swa_out_cache_loc.shape[0]
        ):
            return None
        n = out_cache_loc.shape[0]
        self.cuda_graph_swa_out_cache_loc[n:].zero_()
        if in_capture:
            self.cuda_graph_swa_out_cache_loc[:n].zero_()
        else:
            self.cuda_graph_swa_out_cache_loc[:n].copy_(
                self.kv_index_translator.sliding_window_write_loc_for(out_cache_loc)
            )
        return self.cuda_graph_swa_out_cache_loc[:n]

    def _fill_cuda_graph_write_locs(
        self, forward_batch: ForwardBatch, bs: int
    ) -> Optional[torch.Tensor]:
        """Runs BEFORE graph.replay(), so it reads the live post-compaction
        v2p; no-op for non-unified pools."""
        # The buffer exists only for a translating pool; return before naming it.
        if not self.kv_index_translator.is_translating:
            return None
        return self.kv_index_translator.fill_capture_write_loc(
            out=self.cuda_graph_out_cache_loc_full_physical,
            forward_batch=forward_batch,
            width=self.cuda_graph_out_cache_loc_full_physical.numel(),
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init auxiliary variables for triton attention backend."""

        self._dense_one_shot_kv_indptr = None
        bs = forward_batch.batch_size
        window_kv_indptr = self.window_kv_indptr
        window_kv_indices = None
        window_num_kv_splits = None
        window_kv_offsets = None
        swa_attn_logits = None
        spec_info = forward_batch.spec_info
        # Lean decode buffers are only allocated on the decode path below; default
        # to None so the shared ForwardMetadata constructor works for extend/verify.
        lean_Mp = lean_Lp = lean_Op = lean_locks = None
        # Mixed-KV (HP + int2) per-tier metadata exists only for a decode batch
        # on the unified pool; every other mode leaves the fields at None.
        mixed_fields = {}

        if forward_batch.forward_mode.is_decode_or_idle():
            if spec_info is None or spec_info.kv_indptr is None:
                # kv_indptr is None for draft-extend's idle batch; build from seq_lens.
                if self.dcp_size > 1:
                    # DCP: per-rank sharded KV indices, else each rank reads the
                    # whole KV instead of its owner shard.
                    kv_indptr, kv_indices, _ = self._dcp_kv_indices(
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        self.kv_indptr,
                    )
                else:
                    # gpu_only: seq_lens_sum may be None; over-allocate is safe (ragged write).
                    seq_lens_sum = forward_batch.seq_lens_sum
                    if seq_lens_sum is None:
                        seq_lens_sum = bs * self.max_context_len
                    kv_indices = torch.empty(
                        seq_lens_sum, dtype=torch.int64, device=self.device
                    )
                    kv_indptr = self._fill_kv_indptr_and_indices(
                        bs,
                        forward_batch.seq_lens,
                        forward_batch.req_pool_indices,
                        kv_indices,
                    )
                if (
                    self.sliding_window_size is not None
                    and self.sliding_window_size > 0
                ):
                    window_kv_indptr, window_kv_indices, window_kv_lens, _ = (
                        update_sliding_window_buffer(
                            self.window_kv_indptr,
                            self.kv_index_translator,
                            forward_batch.req_pool_indices,
                            self.sliding_window_size,
                            forward_batch.seq_lens,
                            bs,
                            self.device,
                            self.token_to_kv_pool,
                        )
                    )
                    window_num_kv_splits = torch.empty(
                        (bs,), dtype=torch.int32, device=self.device
                    )
                    self.get_num_kv_splits(window_num_kv_splits, window_kv_lens)
                if self.enable_mixed_kv:
                    mixed_fields = self._alloc_eager_mixed_kv_metadata(
                        forward_batch, bs
                    )
            else:
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices
                bs = kv_indptr.shape[0] - 1

            attn_logits = torch.empty(
                (bs, self.num_head, self.max_kv_splits, self.v_head_dim),
                dtype=torch.float32,
                device=self.device,
            )
            if self.swa_v_head_dim is not None:
                swa_attn_logits = torch.empty(
                    (bs, self.num_head, self.max_kv_splits, self.swa_v_head_dim),
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                swa_attn_logits = None
            attn_lse = torch.empty(
                (bs, self.num_head, self.max_kv_splits),
                dtype=torch.float32,
                device=self.device,
            )
            num_kv_splits = torch.empty((bs,), dtype=torch.int32, device=self.device)
            self.get_num_kv_splits(
                num_kv_splits,
                (
                    self._dcp_lens(forward_batch.seq_lens).clamp_min(1)
                    if self.dcp_size > 1
                    else forward_batch.seq_lens
                ),
            )

            # Lean decode persistent-grid partial-result buffers.
            lean_Mp = torch.empty(
                (self.lean_total_programs, self.lean_block_m),
                dtype=torch.float32,
                device=self.device,
            )
            lean_Lp = torch.empty(
                (self.lean_total_programs, self.lean_block_m),
                dtype=torch.float32,
                device=self.device,
            )
            lean_Op = torch.empty(
                (self.lean_total_programs, self.lean_block_m, self.v_head_dim),
                dtype=torch.float32,
                device=self.device,
            )
            lean_locks = torch.zeros(
                (self.lean_total_programs,), dtype=torch.int32, device=self.device
            )

            qo_indptr = None
            custom_mask = None
            mask_indptr = None
            max_extend_len = None
        elif forward_batch.forward_mode.is_target_verify():
            bs = len(forward_batch.req_pool_indices)
            num_tokens_per_req = self._target_verify_num_tokens_per_req(spec_info)
            qo_indptr = torch.arange(
                0,
                (1 + bs) * num_tokens_per_req,
                step=num_tokens_per_req,
                dtype=torch.int32,
                device=self.device,
            )
            # gpu_only: seq_lens_sum may be None; over-allocate is safe (ragged write).
            seq_lens_sum = forward_batch.seq_lens_sum
            if seq_lens_sum is None:
                seq_lens_sum = bs * self.max_context_len
            kv_indices = torch.empty(
                seq_lens_sum, dtype=torch.int64, device=self.device
            )
            kv_indptr = self._fill_kv_indptr_and_indices(
                bs,
                forward_batch.seq_lens,
                forward_batch.req_pool_indices,
                kv_indices,
            )

            if self.sliding_window_size is not None and self.sliding_window_size > 0:
                # window_kv_offsets gives the start position in custom mask
                (
                    window_kv_indptr,
                    window_kv_indices,
                    window_kv_lens,
                    window_kv_offsets,
                ) = update_sliding_window_buffer(
                    self.window_kv_indptr,
                    self.kv_index_translator,
                    forward_batch.req_pool_indices,
                    self.sliding_window_size,
                    forward_batch.seq_lens,
                    bs,
                    self.device,
                    self.token_to_kv_pool,
                )

            custom_mask = spec_info.custom_mask
            seq_mask_len = num_tokens_per_req * (
                forward_batch.seq_lens + num_tokens_per_req
            )
            mask_indptr = self.mask_indptr
            mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len[:bs], dim=0)
            mask_indptr = mask_indptr[: bs + 1]
            max_extend_len = num_tokens_per_req
            num_kv_splits = None
            attn_logits = None
            attn_lse = None

        else:
            if self.dcp_size > 1:
                kv_indptr, kv_indices, _ = self._dcp_kv_indices(
                    forward_batch.req_pool_indices,
                    forward_batch.extend_prefix_lens,
                    self.kv_indptr,
                )
            else:
                # gpu_only leaves _cpu unset; over-allocate is safe (ragged write).
                if forward_batch.extend_prefix_lens_cpu is not None:
                    kv_indices_len = sum(forward_batch.extend_prefix_lens_cpu)
                else:
                    kv_indices_len = bs * self.max_context_len
                kv_indices = torch.empty(
                    kv_indices_len,
                    dtype=torch.int64,
                    device=self.device,
                )
                kv_indptr = self._fill_kv_indptr_and_indices(
                    bs,
                    forward_batch.extend_prefix_lens,
                    forward_batch.req_pool_indices,
                    kv_indices,
                )
            if self.sliding_window_size is not None and self.sliding_window_size > 0:
                (
                    window_kv_indptr,
                    window_kv_indices,
                    window_kv_lens,
                    window_kv_offsets,
                ) = update_sliding_window_buffer(
                    self.window_kv_indptr,
                    self.kv_index_translator,
                    forward_batch.req_pool_indices,
                    self.sliding_window_size,
                    forward_batch.extend_prefix_lens,
                    bs,
                    self.device,
                    self.token_to_kv_pool,
                )

            qo_indptr = self.qo_indptr
            qo_indptr[1 : bs + 1] = torch.cumsum(forward_batch.extend_seq_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]
            custom_mask = None
            mask_indptr = None
            attn_logits = None
            attn_lse = None
            # Defensive GPU-max fallback when extend_seq_lens_cpu is absent.
            if forward_batch.extend_seq_lens_cpu is not None:
                max_extend_len = max(forward_batch.extend_seq_lens_cpu)
            else:
                max_extend_len = int(forward_batch.extend_seq_lens.max())
            num_kv_splits = None

        swa_out_cache_loc = None
        if self.use_sliding_window_kv_pool and forward_batch.out_cache_loc is not None:
            swa_out_cache_loc = self.kv_index_translator.sliding_window_write_loc_for(
                forward_batch.out_cache_loc
            )

        self.forward_metadata = ForwardMetadata(
            attn_logits,
            attn_lse,
            max_extend_len,
            num_kv_splits,
            kv_indptr,
            kv_indices,
            qo_indptr,
            custom_mask,
            mask_indptr,
            window_kv_indptr,
            window_kv_indices,
            window_num_kv_splits,
            window_kv_offsets,
            swa_attn_logits=swa_attn_logits,
            swa_out_cache_loc=swa_out_cache_loc,
            out_cache_loc_full_physical=(
                forward_batch.out_cache_loc
                if self.kv_index_translator.is_translating
                else None
            ),
            lean_Mp=lean_Mp,
            lean_Lp=lean_Lp,
            lean_Op=lean_Op,
            lean_locks=lean_locks,
            **mixed_fields,
        )

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
        cuda_graph_num_kv_splits_buf: Optional[torch.Tensor] = None,
    ):
        self.cuda_graph_attn_logits = torch.zeros(
            (max_num_tokens, self.num_head, self.max_kv_splits, self.v_head_dim),
            dtype=torch.float32,
            device=self.device,
        )
        if self.swa_v_head_dim is not None:
            self.cuda_graph_swa_attn_logits = torch.zeros(
                (
                    max_num_tokens,
                    self.num_head,
                    self.max_kv_splits,
                    self.swa_v_head_dim,
                ),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            self.cuda_graph_swa_attn_logits = None
        self.cuda_graph_attn_lse = torch.zeros(
            (max_num_tokens, self.num_head, self.max_kv_splits),
            dtype=torch.float32,
            device=self.device,
        )

        # Lean decode persistent-grid partial-result buffers (shared across all layers).
        self.cuda_graph_lean_Mp = torch.zeros(
            (self.lean_total_programs, self.lean_block_m),
            dtype=torch.float32,
            device=self.device,
        )
        self.cuda_graph_lean_Lp = torch.zeros(
            (self.lean_total_programs, self.lean_block_m),
            dtype=torch.float32,
            device=self.device,
        )
        self.cuda_graph_lean_Op = torch.zeros(
            (self.lean_total_programs, self.lean_block_m, self.v_head_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.cuda_graph_lean_locks = torch.zeros(
            (self.lean_total_programs,), dtype=torch.int32, device=self.device
        )

        if cuda_graph_num_kv_splits_buf is None:
            self.cuda_graph_num_kv_splits = torch.full(
                (max_num_tokens,),
                self.max_kv_splits,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            self.cuda_graph_num_kv_splits = cuda_graph_num_kv_splits_buf

        if kv_indices_buf is None:
            self.cuda_graph_kv_indices = torch.zeros(
                (max_num_tokens * self.max_context_len),
                dtype=torch.int64,
                device=self.device,
            )
        else:
            self.cuda_graph_kv_indices = kv_indices_buf

        # Layout is draft * (seq_len + draft) per request (seq_mask_len cumsum
        # below) -- the same bound the shared sizing covers. Read as uint8.
        self._verify_mask = maybe_create_verify_mask(
            is_draft_runner=self.is_draft_runner,
            skip_prefill=self.skip_prefill,
            max_bs=max_bs,
            max_context_len=self.max_context_len,
            num_draft_tokens=self.num_draft_tokens,
            device=self.device,
            is_read=True,
            dtype=torch.uint8,
        )

        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            if kv_indices_buf is None:
                self.cuda_graph_window_kv_indices = torch.zeros(
                    (max_num_tokens * self.sliding_window_size),
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                self.cuda_graph_window_kv_indices = torch.zeros_like(kv_indices_buf)

            self.cuda_graph_window_num_kv_splits = torch.full(
                (max_num_tokens,),
                self.max_kv_splits,
                dtype=torch.int32,
                device=self.device,
            )

            self.cuda_graph_window_kv_offsets = torch.zeros(
                (max_bs,),
                dtype=torch.int32,
                device=self.device,
            )

        if self.use_sliding_window_kv_pool:
            # SWA write-target buffer; refilled at replay from out_cache_loc.
            self.cuda_graph_swa_out_cache_loc = torch.zeros(
                (max_num_tokens,),
                dtype=torch.int64,
                device=self.device,
            )

        if self.kv_index_translator.is_translating:
            # Unified pool full-attention write-target buffer, refilled at replay
            # (-> KVWriteLoc.full_loc). Capture-stable, mirrors cuda_graph_swa_out_cache_loc.
            self.cuda_graph_out_cache_loc_full_physical = torch.zeros(
                (max_num_tokens,),
                dtype=torch.int64,
                device=self.device,
            )

        if self.enable_mixed_kv:
            self._init_cuda_graph_mixed_kv_state(max_bs, max_num_tokens)

    def _build_cuda_graph_forward_metadata(
        self,
        bs: int,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        swa_out_cache_loc: Optional[torch.Tensor] = None,
        out_cache_loc_full_physical: Optional[torch.Tensor] = None,
    ) -> ForwardMetadata:
        """Construct ForwardMetadata from the current cuda-graph buffer state.

        Called by capture after the buffer-update helpers have already run
        (either via replay or directly).  All fields reference the same
        self.cuda_graph_* tensors that the captured graph kernels will
        read — the Python object is rebuilt each capture, but the underlying
        GPU memory addresses are stable. ``swa_out_cache_loc`` is the
        pre-allocated SWA write-target buffer view (or None for non-SWA).
        """
        swa = self.sliding_window_size is not None and self.sliding_window_size > 0
        if forward_mode.is_decode_or_idle():
            return ForwardMetadata(
                attn_logits=self.cuda_graph_attn_logits,
                attn_lse=self.cuda_graph_attn_lse,
                max_extend_len=None,
                num_kv_splits=self.cuda_graph_num_kv_splits,
                kv_indptr=self.kv_indptr[: bs + 1],
                kv_indices=self.cuda_graph_kv_indices,
                qo_indptr=None,
                custom_mask=None,
                mask_indptr=None,
                window_kv_indptr=self.window_kv_indptr[: bs + 1] if swa else None,
                window_kv_indices=self.cuda_graph_window_kv_indices if swa else None,
                window_num_kv_splits=(
                    self.cuda_graph_window_num_kv_splits if swa else None
                ),
                window_kv_offsets=None,
                swa_attn_logits=self.cuda_graph_swa_attn_logits,
                swa_out_cache_loc=swa_out_cache_loc,
                out_cache_loc_full_physical=out_cache_loc_full_physical,
                lean_Mp=self.cuda_graph_lean_Mp,
                lean_Lp=self.cuda_graph_lean_Lp,
                lean_Op=self.cuda_graph_lean_Op,
                lean_locks=self.cuda_graph_lean_locks,
                **self._cuda_graph_mixed_metadata_fields(),
            )
        elif forward_mode.is_target_verify():
            custom_mask = (
                self._verify_mask.buffer
                if self._verify_mask is not None
                and spec_info is not None
                and spec_info.custom_mask is not None
                else None
            )
            max_extend_len = self._target_verify_num_tokens_per_req(spec_info)
            return ForwardMetadata(
                attn_logits=None,
                attn_lse=None,
                max_extend_len=max_extend_len,
                num_kv_splits=None,
                kv_indptr=self.kv_indptr[: bs + 1],
                kv_indices=self.cuda_graph_kv_indices,
                qo_indptr=self.qo_indptr[: bs + 1],
                custom_mask=custom_mask,
                mask_indptr=self.mask_indptr[: bs + 1],
                window_kv_indptr=self.window_kv_indptr[: bs + 1] if swa else None,
                window_kv_indices=self.cuda_graph_window_kv_indices if swa else None,
                window_num_kv_splits=(
                    self.cuda_graph_window_num_kv_splits if swa else None
                ),
                window_kv_offsets=self.cuda_graph_window_kv_offsets if swa else None,
                swa_out_cache_loc=swa_out_cache_loc,
                out_cache_loc_full_physical=out_cache_loc_full_physical,
            )
        elif forward_mode.is_dllm_extend():
            return ForwardMetadata(
                attn_logits=None,
                attn_lse=None,
                max_extend_len=self.dllm_block_size,
                num_kv_splits=None,
                kv_indptr=self.kv_indptr[: bs + 1],
                kv_indices=self.cuda_graph_kv_indices,
                qo_indptr=self.qo_indptr[: bs + 1],
                custom_mask=None,
                mask_indptr=None,
                window_kv_indptr=self.window_kv_indptr[: bs + 1] if swa else None,
                window_kv_indices=self.cuda_graph_window_kv_indices if swa else None,
                window_num_kv_splits=None,
                window_kv_offsets=None,
                swa_out_cache_loc=swa_out_cache_loc,
                out_cache_loc_full_physical=out_cache_loc_full_physical,
            )
        elif forward_mode.is_draft_extend_v2():
            return ForwardMetadata(
                attn_logits=None,
                attn_lse=None,
                # Must match the per-req query count (num_tokens_per_req) used to
                # build qo_indptr above, else the extend kernel grid is too small
                # for topk > 1 (num_draft_tokens > num_steps+1) and drops query
                # blocks.
                max_extend_len=(
                    self.num_draft_tokens
                    if forward_mode.is_draft_extend_v2()
                    else self.speculative_num_steps + 1
                ),
                num_kv_splits=None,
                kv_indptr=self.kv_indptr[: bs + 1],
                kv_indices=self.cuda_graph_kv_indices,
                qo_indptr=self.qo_indptr[: bs + 1],
                custom_mask=None,
                mask_indptr=None,
                window_kv_indptr=self.window_kv_indptr,
                window_kv_indices=None,
                window_num_kv_splits=None,
                window_kv_offsets=None,
                swa_out_cache_loc=swa_out_cache_loc,
                out_cache_loc_full_physical=out_cache_loc_full_physical,
            )
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=} for CUDA Graph.")

    def _apply_cuda_graph_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ) -> None:
        """Shared capture+replay body for the cuda-graph init path.

        Public entry: :py:meth:`init_forward_metadata_out_graph`.
        """
        # NOTE: encoder_lens expected to be zeros or None
        if forward_mode.is_decode_or_idle():
            assert spec_info is None, "Multi-step cuda graph init is not done here."
            _, _, window_kv_lens, num_kv_splits_lens = self._update_decode_kv_buffers(
                bs, seq_lens, req_pool_indices
            )
            self.get_num_kv_splits(
                self.cuda_graph_num_kv_splits[:bs], num_kv_splits_lens[:bs]
            )
            if window_kv_lens is not None:
                self.get_num_kv_splits(
                    self.cuda_graph_window_num_kv_splits[:bs], window_kv_lens[:bs]
                )
            if self.enable_mixed_kv:
                self._fill_cuda_graph_mixed_kv_buffers(bs, req_pool_indices, seq_lens)
        elif forward_mode.is_target_verify():
            bs = len(req_pool_indices)
            self._update_target_verify_buffers(
                bs, seq_lens, spec_info, req_pool_indices
            )
        elif forward_mode.is_dllm_extend():
            self._update_dllm_buffers(bs, seq_lens, req_pool_indices)
        elif forward_mode.is_draft_extend_v2():
            self._update_draft_extend_buffers(
                bs, seq_lens, forward_mode, spec_info, req_pool_indices
            )
        else:
            raise ValueError(
                f"Invalid forward mode: {forward_mode=} for CUDA Graph replay."
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    @property
    def verify_mask(self) -> Optional[VerifyMask]:
        return self._verify_mask

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        pass

    @property
    def pack_all_prefix_chunks(self) -> bool:
        """Pack every prefix chunk into one FP8 buffer when capacity allows."""
        return self.use_dense_fp8_chunked_prefill

    @property
    def fuse_prefix_into_extend(self) -> bool:
        """Attend the packed prefix and current chunk in one launch."""
        return self.use_dense_fp8_chunked_prefill

    def prepare_chunked_prefill_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert the current dense MHA chunk once and reuse Q for prefix passes."""
        fp8_dtype = torch.float8_e4m3fn
        output_dtype = q.dtype
        if output_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            output_dtype = torch.bfloat16
        forward_batch._triton_dense_fp8_output_dtype = output_dtype

        if q.dtype != fp8_dtype:
            q = q.to(fp8_dtype)
        if k.dtype != fp8_dtype:
            k = k.to(fp8_dtype)
        if v.dtype != fp8_dtype:
            v = v.to(fp8_dtype)
        return q.contiguous(), k.contiguous(), v.contiguous()

    def pack_prefix_chunk_kv(
        self,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack a materialized dense prefix directly into unit-scale FP8 K/V."""
        return mla_kv_pack_quantize_fp8(
            k_nope,
            k_pe,
            v,
            fp8_dtype=torch.float8_e4m3fn,
            enable_pdl=False,
        )

    def _can_run_dense_fp8_chunked_mha(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ) -> bool:
        return (
            self.use_dense_fp8_chunked_prefill
            and forward_batch.attn_attend_prefix_cache is not None
            and self.forward_metadata.custom_mask is None
            and q.dtype == torch.float8_e4m3fn
            and k.dtype == torch.float8_e4m3fn
            and v.dtype == torch.float8_e4m3fn
            and layer.tp_q_head_num == 12
            and layer.tp_k_head_num == 12
            and layer.qk_head_dim == 192
            and layer.v_head_dim == 128
            and (layer.sliding_window_size is None or layer.sliding_window_size <= -1)
            and layer.logit_cap <= 0
        )

    def _forward_dense_fp8_chunked_mha(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ):
        """Run current or cached dense FP8 K/V through Triton extend attention."""
        output_dtype = getattr(
            forward_batch, "_triton_dense_fp8_output_dtype", torch.bfloat16
        )
        output = torch.empty(
            (q.shape[0], layer.tp_q_head_num, layer.v_head_dim),
            dtype=output_dtype,
            device=q.device,
        )

        prefix_k = getattr(forward_batch, "fused_prefix_k", None)
        if prefix_k is not None:
            # Prefix is non-causal while the current chunk is causal. The
            # extend kernel already implements precisely that two-stage mask.
            prefix_v = forward_batch.fused_prefix_v
            self.extend_attention_fwd(
                q,
                k,
                v,
                output,
                prefix_k,
                prefix_v,
                self.forward_metadata.qo_indptr,
                forward_batch.prefix_chunk_cu_seq_lens[0],
                forward_batch.prefix_dense_kv_indices[: prefix_k.shape[0]],
                None,
                True,
                None,
                self.forward_metadata.max_extend_len,
                1.0,
                1.0,
                sm_scale=layer.scaling,
                page_size=1,
                extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
                identity_kv_indices=True,
            )
            return output

        lse = torch.empty(
            (q.shape[0], layer.tp_q_head_num),
            dtype=torch.float32,
            device=q.device,
        )

        if forward_batch.attn_attend_prefix_cache:
            chunk_idx = forward_batch.prefix_chunk_idx
            assert chunk_idx is not None and chunk_idx >= 0
            kv_indptr = forward_batch.prefix_chunk_cu_seq_lens[chunk_idx]
            kv_indices = forward_batch.prefix_dense_kv_indices[: k.shape[0]]
            self.extend_attention_fwd(
                q,
                k[:0],
                v[:0],
                output,
                k,
                v,
                self.forward_metadata.qo_indptr,
                kv_indptr,
                kv_indices,
                None,
                False,
                None,
                self.forward_metadata.max_extend_len,
                1.0,
                1.0,
                sm_scale=layer.scaling,
                lse_extend=lse,
                skip_extend=True,
                page_size=1,
                extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
                identity_kv_indices=True,
            )
            # Empty ragged rows are returned as output=0, LSE=-inf, so the
            # portable merge_state operation ignores them exactly.
        else:
            self.extend_attention_fwd(
                q,
                k,
                v,
                output,
                k[:0],
                v[:0],
                self.forward_metadata.qo_indptr,
                forward_batch.mha_empty_kv_indptr,
                self.forward_metadata.kv_indices[:0],
                None,
                True,
                None,
                self.forward_metadata.max_extend_len,
                1.0,
                1.0,
                sm_scale=layer.scaling,
                lse_extend=lse,
                skip_prefix=True,
                page_size=1,
                extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            )

        if forward_batch.mha_return_lse:
            return output, lse
        return output

    def _set_kv_buffer(
        self,
        forward_batch: ForwardBatch,
        layer: RadixAttention,
        loc_info,
        k: torch.Tensor,
        v: torch.Tensor,
        k_scale=None,
        v_scale=None,
    ) -> None:
        # DCP writes to the local physical shard (loc = out_cache_loc //
        # dcp_size) through the masked path so each rank only stores the tokens
        # it owns. Non-DCP keeps the original write loc and plain set_kv_buffer.
        if self.dcp_size > 1:
            # The rank-local slot of a physical loc is physical.
            loc = KVWriteLoc(
                forward_batch.out_cache_loc // self.dcp_size,
                physical=forward_batch.out_cache_loc_is_physical,
            )
            if (
                forward_batch.positions is not None
                and forward_batch.positions.numel() == loc.loc.numel()
            ):
                dcp_kv_mask = forward_batch.positions % self.dcp_size == self.dcp_rank
            else:
                dcp_kv_mask = forward_batch.dcp_kv_mask
            kwargs = {"dcp_kv_mask": dcp_kv_mask}
        else:
            loc = loc_info
            kwargs = {}
        if k_scale is None and v_scale is None:
            self.token_to_kv_pool.set_kv_buffer(layer, loc, k, v, **kwargs)
        else:
            self.token_to_kv_pool.set_kv_buffer(
                layer, loc, k, v, k_scale, v_scale, **kwargs
            )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
        score_mod=None,
        aux_tensors=None,
    ):
        if (
            k is not None
            and v is not None
            and sinks is None
            and score_mod is None
            and aux_tensors is None
            and self._can_run_dense_fp8_chunked_mha(q, k, v, layer, forward_batch)
        ):
            return self._forward_dense_fp8_chunked_mha(q, k, v, layer, forward_batch)

        # TODO: reuse the buffer across layers
        attn_out = getattr(forward_batch, "_attn_output", None)
        if attn_out is not None:
            o = attn_out
        elif layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        kv_pool = self.token_to_kv_pool
        # Mixed two-group pool (gemma4_unified): sliding-window layers also
        # store int2 KV, so they must take the int2 dense prefill path too.
        # The sliding-window mask is then applied by flash_attn's window_size
        # (see ``_forward_extend_quantized_dense``); the full prefix is
        # dequantized and flash masks out-of-window keys. For non-mixed int2
        # pools (uniform full-attention models) the original ``sliding_window
        # < 0`` gate is unchanged (those never have sliding layers).
        mixed_pool_active = _pool_mixed_kv_active(kv_pool)
        layer_is_sliding = (
            layer.sliding_window_size is not None and layer.sliding_window_size > -1
        )
        use_quantized_dense_prefill = (
            _is_int2_pool(kv_pool)
            and (not layer_is_sliding or mixed_pool_active)
            and self.forward_metadata.custom_mask is None
        )
        if _is_int2_pool(kv_pool) and not use_quantized_dense_prefill:
            # The generic extend kernel cannot read the int2-packed buffers; it
            # would die inside Triton with "only int8 supported!", far from the
            # cause. Name the gate that failed instead.
            raise RuntimeError(
                "int2 KV pool reached the generic extend kernel: "
                f"layer_id={layer.layer_id} sliding_window_size={layer.sliding_window_size} "
                f"mixed_pool_active={mixed_pool_active} "
                f"custom_mask={'set' if self.forward_metadata.custom_mask is not None else 'none'} "
                f"pool={type(kv_pool).__name__}"
            )
        kv_from_pool = False
        if k is None and v is None and _is_int2_pool(kv_pool):
            # A KV-shared layer (Gemma-4's last layers) attends with the K/V its
            # source layer wrote for this forward's tokens. In an int2 pool those
            # rows exist only as stored, so dequantize them -- they come back in
            # the pool's rotated frame -- and hand them to the int2 prefill as
            # already-rotated K/V. The source layer owns the cache write.
            k, v = dequantize_prefix_kv(
                kv_pool, layer.layer_id, forward_batch.out_cache_loc, q.dtype
            )
            kv_from_pool = True
            save_kv_cache = False
        pre_rotated_q = None
        pre_rotated_k = None
        pre_rotated_v = None
        need_v_inverse = None
        if (
            not self.enable_deterministic
            and use_quantized_dense_prefill
            and k is not None
            and v is not None
        ):
            # Int2 prefill used to rotate K/V once for attention and again when
            # writing the KV cache. Pre-rotate them here so both consumers can
            # share the same tensors.
            pre_rotated_q, pre_rotated_k, pre_rotated_v, need_v_inverse = (
                prepare_quantized_extend_qkv(
                    kv_pool,
                    layer,
                    q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                    k.contiguous(),
                    v.contiguous(),
                    kv_already_hadamard_transformed=kv_from_pool,
                )
            )

        self._maybe_dump_qkv(q, k, v, layer, forward_batch)

        if k is None and v is None:
            pool = self.token_to_kv_pool
            cache_loc = forward_batch.out_cache_loc
            if isinstance(pool, SWAKVPool) and pool.layers_mapping[layer.layer_id][1]:
                assert self.forward_metadata.swa_out_cache_loc is not None, (
                    "window-layer read-back before the metadata carried a "
                    "sliding-window write loc"
                )
                cache_loc = self.forward_metadata.swa_out_cache_loc
            k_buffer, v_buffer = pool.get_kv_buffer(layer.layer_id)
            k = k_buffer[cache_loc]
            v = v_buffer[cache_loc]
        elif k is None or v is None:
            raise ValueError("Both k and v should be None or not None")
        else:
            # Save KV cache first (must do this before unified kernel)
            if save_kv_cache:
                loc_info = KVWriteLoc.for_batch(
                    forward_batch,
                    swa_loc=self.forward_metadata.swa_out_cache_loc,
                    full_loc=self.forward_metadata.out_cache_loc_full_physical,
                )
                # The OSCAR pools take a bare loc tensor, not a KVWriteLoc: they
                # are static pools (no v2p translation, no SWA sub-pool), so
                # ``loc`` is already the kernel-facing slot id.
                if pre_rotated_k is not None and pre_rotated_v is not None:
                    # Deferred int2 save: write the pre-rotated K/V so the pool
                    # keeps the rotated-domain representation.
                    # ``already_hadamard_transformed=True`` tells the pool to
                    # skip its own rotation.
                    kv_pool.set_kv_buffer(
                        layer,
                        forward_batch.out_cache_loc,
                        pre_rotated_k,
                        pre_rotated_v,
                        layer.k_scale,
                        layer.v_scale,
                        already_hadamard_transformed=True,
                        is_decode=False,
                    )
                elif _is_int2_pool(kv_pool):
                    # int2 pool outside the dense int2 prefill (deterministic
                    # mode / custom mask): the pool rotates and quantizes itself.
                    kv_pool.set_kv_buffer(
                        layer,
                        forward_batch.out_cache_loc,
                        k,
                        v,
                        layer.k_scale,
                        layer.v_scale,
                    )
                elif self.packed_mla_pool:
                    # ``[c_kv | k_pe]`` rows; the packed pool rotates and packs
                    # on the write and takes no scale parameters.
                    kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)
                elif layer.k_scale is None:
                    self._set_kv_buffer(forward_batch, layer, loc_info, k, v)
                elif self.use_mla:
                    # For MLA, scale K manually before storing since MLATokenToKVPool
                    # doesn't accept scale parameters. Clone to protect k from mutation
                    # since it's used later in the attention kernel.
                    k_scaled = k.clone().div_(layer.k_scale)
                    self.token_to_kv_pool.set_kv_buffer(
                        layer,
                        loc_info,
                        k_scaled,
                        v,
                    )
                else:
                    self._set_kv_buffer(
                        forward_batch,
                        layer,
                        loc_info,
                        k.clone(),  # cloned to protect k,v from in-place mutation in set_kv_buffer
                        v.clone(),
                        layer.k_scale,
                        layer.v_scale,
                    )

        logits_soft_cap = logit_capping_mod(layer.logit_capping_method, layer.logit_cap)

        causal = True
        if (
            layer.is_cross_attention
            or layer.attn_type == AttentionType.ENCODER_ONLY
            or (
                layer.attn_type == AttentionType.DECODER_BIDIRECTIONAL
                and (
                    self.allow_bidirectional_attention_in_extend
                    # A DLLM graph contains complete, fixed-width canvases;
                    # padding adds requests, never tokens inside a canvas.
                    or forward_batch.forward_mode.is_dllm_extend()
                )
            )
        ):
            causal = False

        # Dense one-shot MLA prefill (AttnForwardMethod.MHA_ONE_SHOT): k/v were
        # up-projected out of the latent cache and span prefix + current chunk,
        # so they no longer line up row-for-row with q the way
        # extend_attention_fwd requires. Route to the single-loop dense kernel.
        # A prefix-chunk phase (attn_attend_prefix_cache) also carries a longer
        # k/v, but the dispatcher never hands Triton MHA_CHUNKED_KV.
        if (
            forward_batch.mha_one_shot
            and not forward_batch.attn_attend_prefix_cache
            and k is not None
            and k.shape[0] != q.shape[0]
        ):
            return self._forward_extend_dense_one_shot(
                q,
                k,
                v,
                o,
                layer,
                forward_batch,
                causal,
                logits_soft_cap,
                sinks=sinks,
                score_mod=score_mod,
            )

        if self.dcp_size > 1:
            if score_mod is not None:
                raise NotImplementedError(
                    "DCP Triton extend does not support score_mod"
                )
            return self._forward_extend_dcp(
                q, k, v, layer, forward_batch, causal, logits_soft_cap, sinks
            )

        # Deterministic mode: use unified 1-stage kernel
        if self.enable_deterministic:
            return self._forward_extend_unified(
                q,
                o,
                layer,
                forward_batch,
                causal,
                logits_soft_cap,
                sinks,
                score_mod=score_mod,
                aux_tensors=aux_tensors,
            )

        # Normal mode: use original 2-stage kernel
        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            bidirectional_extend = (
                layer.attn_type == AttentionType.DECODER_BIDIRECTIONAL
            )
            sliding_window_size = (
                -1 if bidirectional_extend else layer.sliding_window_size
            )
            kv_indptr = self.forward_metadata.window_kv_indptr
            kv_indices = self.forward_metadata.window_kv_indices
            window_kv_offsets = self.forward_metadata.window_kv_offsets
        else:
            sliding_window_size = -1
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices
            window_kv_offsets = None

        if layer.k_scale is not None and layer.v_scale is not None:
            k_descale = layer.k_scale_float
            v_descale = layer.v_scale_float
        else:
            k_descale = 1.0
            v_descale = 1.0

        if use_quantized_dense_prefill:
            return self._forward_extend_quantized_dense(
                q,
                k,
                v,
                o,
                layer,
                forward_batch,
                causal,
                pre_rotated_q=pre_rotated_q,
                pre_rotated_k=pre_rotated_k,
                pre_rotated_v=pre_rotated_v,
                need_v_inverse_override=need_v_inverse,
            )

        if self.packed_mla_pool:
            # Extend reads only the *reused prefix* rows (the tokens of this
            # forward are passed in as k/v), so the row set is small and bounded
            # by the radix hit, not by the context. Staging them dense costs one
            # dequant pass and keeps a second specialised kernel -- and a second
            # place to get head tiling wrong -- out of the tree. If a workload
            # ever makes this the hot path, it is the same dequant the decode
            # kernel already fuses.
            pool = self.token_to_kv_pool
            n_prefix = int(kv_indices.numel())
            if n_prefix > 0:
                staged = pool.materialize_rows(layer.layer_id, kv_indices)
                k_buffer = staged
                v_buffer = staged[..., : pool.kv_lora_rank]
                kv_indices = torch.arange(
                    n_prefix, dtype=kv_indices.dtype, device=kv_indices.device
                )
            else:
                d = pool.latent_row_dim()
                k_buffer = torch.zeros((1, 1, d), dtype=q.dtype, device=q.device)
                v_buffer = k_buffer[..., : pool.kv_lora_rank]
        else:
            k_buffer = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
            v_buffer = self.token_to_kv_pool.get_value_buffer(layer.layer_id)

        # Split-KV EAGLE-verify fast path (ROCm/Triton). On target-verify
        # (topk=1 causal chain), run the bandwidth-efficient split-KV kernel
        # instead of the serial-prefix extend kernel. verify_splitkv_fwd()
        # returns True if it ran (o written), or False for any case it cannot
        # serve bit-equivalently (its can_handle() gates on non-causal / sinks /
        # sliding-window / ragged / topk>1), so we fall through to
        # extend_attention_fwd below. Correctness is never at risk.
        # Route target-verify to the grouped-head kernel when eligible, else the
        # per-head split-KV kernel.
        if self.use_verify_shared_kv:
            verify_fwd = self.verify_shared_kv_fwd
        elif self.use_verify_splitkv:
            verify_fwd = self.verify_splitkv_fwd
        else:
            verify_fwd = None
        if (
            verify_fwd is not None
            and score_mod is None
            and forward_batch.forward_mode.is_target_verify()
            and verify_fwd(
                q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                k.contiguous(),
                v.contiguous(),
                o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                k_buffer,
                v_buffer,
                self.forward_metadata.qo_indptr,
                kv_indptr,
                kv_indices,
                self.forward_metadata.custom_mask,
                causal,
                self.forward_metadata.mask_indptr,
                self.forward_metadata.max_extend_len,
                k_descale,
                v_descale,
                layer.scaling,
                logit_cap=logits_soft_cap,
                sliding_window_size=sliding_window_size,
                sinks=sinks,
                window_kv_offsets=window_kv_offsets,
                xai_temperature_len=layer.xai_temperature_len,
                max_bs=self.req_to_token_pool.size,
            )
        ):
            return o

        self.extend_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k.contiguous(),
            v.contiguous(),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            k_buffer,
            v_buffer,
            self.forward_metadata.qo_indptr,
            kv_indptr,
            kv_indices,
            self.forward_metadata.custom_mask,
            causal,
            self.forward_metadata.mask_indptr,
            self.forward_metadata.max_extend_len,
            k_descale,
            v_descale,
            layer.scaling,
            logit_cap=logits_soft_cap,
            sliding_window_size=sliding_window_size,
            sinks=sinks,
            window_kv_offsets=window_kv_offsets,
            xai_temperature_len=layer.xai_temperature_len,
            page_size=self.page_size,
            score_mod=score_mod,
            aux_tensors=aux_tensors,
            extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        )
        return o

    def _dense_one_shot_kv_indptr_for(self, forward_batch: ForwardBatch):
        """Cumulative full sequence lengths addressing the one-shot K/V rows.

        The MHA one-shot K/V is gathered with fetch_mha_one_shot_kv_indices(),
        which lays sequences out back to back at their full seq_len -- so the
        row offsets are cumsum(seq_lens), not the prefix-only kv_indptr that
        forward_metadata carries for the paged extend path.
        """
        if self._dense_one_shot_kv_indptr is None:
            bs = forward_batch.batch_size
            kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=self.device)
            kv_indptr[1:] = torch.cumsum(forward_batch.seq_lens[:bs], dim=0)
            self._dense_one_shot_kv_indptr = kv_indptr
        return self._dense_one_shot_kv_indptr

    def _forward_extend_dense_one_shot(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        causal: bool,
        logits_soft_cap: float,
        sinks: Optional[torch.Tensor] = None,
        score_mod=None,
    ):
        # Guarded rather than silently fallen back on: dropping to
        # extend_attention_fwd with a longer-than-q k/v would read the wrong
        # rows and quietly return wrong numbers.
        if sinks is not None or score_mod is not None:
            raise NotImplementedError(
                "Triton dense one-shot prefill does not support sinks/score_mod"
            )
        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            raise NotImplementedError(
                "Triton dense one-shot prefill does not support sliding windows"
            )
        if layer.k_scale is not None or layer.v_scale is not None:
            raise NotImplementedError(
                "Triton dense one-shot prefill does not support KV descales"
            )
        if layer.xai_temperature_len is not None and layer.xai_temperature_len > 0:
            raise NotImplementedError(
                "Triton dense one-shot prefill does not support xai temperature"
            )

        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        k = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
        v = v.view(-1, layer.tp_k_head_num, layer.v_head_dim)

        if self.can_use_dense_prefill_fp8(
            q, k, v, is_causal=causal, logit_cap=logits_soft_cap
        ):
            # Cast Q, K and V separately, matching the zero-prefix FP8 gate in
            # extend_attention_fwd (and Aiter's opt-in behavior).
            q = q.to(torch.float8_e4m3fn)
            k = k.to(torch.float8_e4m3fn)
            v = v.to(torch.float8_e4m3fn)

        self.dense_prefill_attention_fwd(
            q,
            k.contiguous(),
            v.contiguous(),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            self.forward_metadata.qo_indptr,
            self._dense_one_shot_kv_indptr_for(forward_batch),
            self.forward_metadata.max_extend_len,
            sm_scale=layer.scaling,
            logit_cap=logits_soft_cap,
            is_causal=causal,
        )
        return o

    def _forward_extend_dcp(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        causal: bool,
        logits_soft_cap: float,
        sinks: Optional[torch.Tensor],
    ):
        if sinks is not None:
            raise NotImplementedError("DCP Triton extend does not support sinks")
        if self.forward_metadata.custom_mask is not None:
            raise NotImplementedError("DCP Triton extend does not support custom masks")
        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            raise NotImplementedError(
                "DCP Triton extend does not support sliding window"
            )

        group = get_parallel().dcp_group
        q_local = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim).contiguous()
        total_tokens, local_heads, _ = q_local.shape

        kv_indptr = self.forward_metadata.kv_indptr
        kv_indices = self.forward_metadata.kv_indices
        max_extend_len = self.forward_metadata.max_extend_len

        if layer.k_scale is not None and layer.v_scale is not None:
            k_descale = layer.k_scale_float
            v_descale = layer.v_scale_float
        else:
            k_descale = 1.0
            v_descale = 1.0

        k_buffer = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_buffer = self.token_to_kv_pool.get_value_buffer(layer.layer_id)

        current_out = torch.zeros(
            (total_tokens, local_heads, layer.v_head_dim),
            device=q.device,
            dtype=torch.float32,
        )
        current_lse = torch.full(
            (total_tokens, local_heads),
            -float("inf"),
            device=q.device,
            dtype=torch.float32,
        )

        # Select the replicated K/V heads matching this rank's Q shard.
        if k.numel() > 0:
            if layer.tp_k_head_num > 1:
                kv_head_start = (
                    group.rank_in_group * layer.tp_k_head_num // group.world_size
                )
                kv_head_end = max(
                    (group.rank_in_group + 1) * layer.tp_k_head_num // group.world_size,
                    kv_head_start + 1,
                )
                k = k[:, kv_head_start:kv_head_end]
                v = v[:, kv_head_start:kv_head_end]

            empty_kv_indptr = torch.zeros_like(kv_indptr)
            self.extend_attention_fwd(
                q_local,
                k.contiguous(),
                v.contiguous(),
                current_out,
                k_buffer,
                v_buffer,
                self.forward_metadata.qo_indptr,
                empty_kv_indptr,
                kv_indices[:0],
                None,
                causal,
                None,
                max_extend_len,
                1.0,
                1.0,
                sm_scale=layer.scaling,
                logit_cap=logits_soft_cap,
                xai_temperature_len=layer.xai_temperature_len,
                lse_extend=current_lse,
                skip_prefix=True,
            )

        if kv_indices.numel() == 0:
            return current_out.reshape(-1, layer.tp_q_head_num * layer.v_head_dim).to(
                q.dtype
            )

        # Prefix KV is sharded across DCP ranks, so compute each rank's
        # partial attention with all gathered query heads and merge by LSE.
        q_all = group.all_gather(q_local, dim=1).contiguous()
        total_heads = q_all.shape[1]
        prefix_out = torch.zeros(
            (total_tokens, total_heads, layer.v_head_dim),
            device=q.device,
            dtype=torch.float32,
        )
        prefix_lse = torch.full(
            (total_tokens, total_heads),
            -float("inf"),
            device=q.device,
            dtype=torch.float32,
        )
        empty_k = k[:0].contiguous()
        empty_v = v[:0].contiguous()
        self.extend_attention_fwd(
            q_all,
            empty_k,
            empty_v,
            prefix_out,
            k_buffer,
            v_buffer,
            self.forward_metadata.qo_indptr,
            kv_indptr,
            kv_indices,
            None,
            False,
            None,
            max_extend_len,
            k_descale,
            v_descale,
            sm_scale=layer.scaling,
            logit_cap=logits_soft_cap,
            xai_temperature_len=layer.xai_temperature_len,
            lse_extend=prefix_lse,
            skip_extend=True,
        )

        prefix_out, prefix_lse = cp_lse_ag_out_rs_mha(
            prefix_out, prefix_lse, group, return_lse=True
        )
        final_lse = torch.logaddexp(prefix_lse, current_lse)
        prefix_scale = torch.exp(prefix_lse - final_lse).unsqueeze(-1)
        current_scale = torch.exp(current_lse - final_lse).unsqueeze(-1)
        prefix_scale = torch.nan_to_num(prefix_scale, nan=0.0, posinf=0.0, neginf=0.0)
        current_scale = torch.nan_to_num(current_scale, nan=0.0, posinf=0.0, neginf=0.0)
        out = prefix_out * prefix_scale + current_out * current_scale
        return out.reshape(-1, layer.tp_q_head_num * layer.v_head_dim).to(q.dtype)

    def _forward_extend_unified(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        causal: bool,
        logits_soft_cap: float,
        sinks: Optional[torch.Tensor],
        score_mod=None,
        aux_tensors=None,
    ):
        """
        Unified 1-stage extend attention for deterministic inference.
        Both prefix and extend KV are accessed through unified kv_indices.
        """
        bs = forward_batch.batch_size

        # Determine sliding window settings
        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            bidirectional_extend = (
                layer.attn_type == AttentionType.DECODER_BIDIRECTIONAL
            )
            sliding_window_size = (
                -1 if bidirectional_extend else layer.sliding_window_size
            )
            # Note: for unified kernel, we use full kv_indptr (not window)
            prefix_kv_indptr = self.forward_metadata.window_kv_indptr
            prefix_kv_indices = self.forward_metadata.window_kv_indices
            # Compute window start positions (absolute position of first key in window)
            # window_start_pos = seq_len - window_len
            window_kv_lens = prefix_kv_indptr[1 : bs + 1] - prefix_kv_indptr[:bs]
            if forward_batch.extend_prefix_lens is not None:
                window_start_pos = (
                    forward_batch.extend_prefix_lens[:bs] - window_kv_lens
                )
            elif forward_batch.forward_mode.is_target_verify():
                window_start_pos = forward_batch.seq_lens[:bs] - window_kv_lens
            else:
                window_start_pos = None
        else:
            sliding_window_size = -1
            prefix_kv_indptr = self.forward_metadata.kv_indptr
            prefix_kv_indices = self.forward_metadata.kv_indices
            window_start_pos = None

        extend_kv_indices = forward_batch.out_cache_loc
        pool = self.token_to_kv_pool
        if (
            layer.sliding_window_size is not None
            and layer.sliding_window_size > -1
            and isinstance(pool, SWAKVPool)
            and pool.layers_mapping[layer.layer_id][1]
        ):
            extend_kv_indices = self.forward_metadata.swa_out_cache_loc
            assert extend_kv_indices is not None, (
                "window-layer extend before the metadata carried a "
                "sliding-window write loc"
            )
        elif self.forward_metadata.out_cache_loc_full_physical is not None:
            extend_kv_indices = self.forward_metadata.out_cache_loc_full_physical

        # Capture batches may not have a spec_info, so use the attention
        # metadata's resolved uniform verify width when extend lengths are absent.
        if forward_batch.extend_seq_lens is None:
            if not forward_batch.forward_mode.is_target_verify():
                raise RuntimeError(
                    "extend_seq_lens is None outside TARGET_VERIFY mode."
                )
            extend_seq_lens = torch.full(
                (bs,),
                self.forward_metadata.max_extend_len,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            extend_seq_lens = forward_batch.extend_seq_lens

        if forward_batch.extend_start_loc is None:
            extend_start_loc = torch.cat(
                [
                    torch.zeros(1, dtype=torch.int32, device=self.device),
                    torch.cumsum(extend_seq_lens[:-1], dim=0),
                ]
            )
        else:
            extend_start_loc = forward_batch.extend_start_loc

        unified_kv_indptr, unified_kv_indices, prefix_lens = (
            self.build_unified_kv_indices(
                prefix_kv_indptr,
                prefix_kv_indices,
                extend_start_loc,
                extend_seq_lens,
                extend_kv_indices,
                bs,
            )
        )

        # Convert prefix_lens to int32 for the kernel
        prefix_lens = prefix_lens.to(torch.int32)

        if layer.k_scale is not None and layer.v_scale is not None:
            k_descale = layer.k_scale_float
            v_descale = layer.v_scale_float
        else:
            k_descale = 1.0
            v_descale = 1.0

        # Call unified kernel
        self.extend_attention_fwd_unified(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),
            self.token_to_kv_pool.get_value_buffer(layer.layer_id),
            k_descale,
            v_descale,
            self.forward_metadata.qo_indptr,
            unified_kv_indptr,
            unified_kv_indices,
            prefix_lens,
            self.forward_metadata.max_extend_len,
            custom_mask=self.forward_metadata.custom_mask,
            mask_indptr=self.forward_metadata.mask_indptr,
            sm_scale=layer.scaling,
            logit_cap=logits_soft_cap,
            is_causal=causal,
            sliding_window_size=sliding_window_size,
            sinks=sinks,
            window_start_pos=window_start_pos,
            xai_temperature_len=layer.xai_temperature_len,
            page_size=self.page_size,
            score_mod=score_mod,
            aux_tensors=aux_tensors,
        )

        return o

    def _maybe_dump_qkv(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ) -> None:
        """Env-driven post-RoPE Q/K/V dump for OSCAR calibration. Ported from the
        sglang-dump-qkv fork. Inert unless DUMP_KVCACHE=true. Saves up to
        DUMP_KVCACHE_TOKENS tokens per layer to DUMP_KVCACHE_DIR/layer_<id>/{q,k,v}/<chunk>.pt
        plus a parallel seq_lens dir so the calibration script can split chunks
        back into per-request samples. For hybrid models (Qwen3.5, etc.) this
        only fires for layers that actually go through the triton attention
        backend -- full-attention layers -- so the dump naturally skips
        linear/mamba layers without any extra filtering. Per-layer shapes are
        preserved, so heterogeneous head_dim (gemma4_unified sliding 256 / full
        512) is handled naturally.
        """
        if (
            not self._dump_kvcache_enabled
            or k is None
            or v is None
            or layer.layer_id in self._dump_kv_done_layers
        ):
            return
        dump_tokens = get_int_env_var("DUMP_KVCACHE_TOKENS", 100)
        layer_id = layer.layer_id
        saved_so_far = self._dump_saved_tokens.get(layer_id, 0)
        chunk_idx = self._dump_chunk_idx.get(layer_id, 0)
        remaining = dump_tokens - saved_so_far
        if remaining <= 0:
            return
        num_tokens = q.shape[0]
        tokens_to_save = min(num_tokens, remaining)
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        q_dump = (
            q[:tokens_to_save]
            .view(-1, layer.tp_q_head_num, layer.qk_head_dim)
            .contiguous()
            .detach()
        )
        k_dump = k[:tokens_to_save].contiguous().detach()
        v_dump = v[:tokens_to_save].contiguous().detach()
        chunk_seq_lens = []
        if forward_batch.extend_seq_lens is not None:
            remain = tokens_to_save
            for slen in forward_batch.extend_seq_lens.tolist():
                if remain <= 0:
                    break
                take = min(slen, remain)
                chunk_seq_lens.append(take)
                remain -= take
        else:
            chunk_seq_lens = [tokens_to_save]
        chunk_seq_lens_t = torch.tensor(chunk_seq_lens, dtype=torch.int32)
        tp_size = get_parallel().attn_tp_size
        tp_rank = get_parallel().attn_tp_rank
        if tp_size > 1:
            attn_tp_group = get_parallel().attn_tp_group
            q_dump = attn_tp_group.all_gather(q_dump, dim=1)
            k_dump = attn_tp_group.all_gather(k_dump, dim=1)
            v_dump = attn_tp_group.all_gather(v_dump, dim=1)
        if tp_rank == 0:
            save_dir = os.environ.get("DUMP_KVCACHE_DIR", ".")
            for name, tensor in (("q", q_dump), ("k", k_dump), ("v", v_dump)):
                chunk_dir = os.path.join(save_dir, f"layer_{layer_id}", name)
                os.makedirs(chunk_dir, exist_ok=True)
                torch.save(tensor.cpu(), os.path.join(chunk_dir, f"{chunk_idx}.pt"))
            seq_dir = os.path.join(save_dir, f"layer_{layer_id}", "seq_lens")
            os.makedirs(seq_dir, exist_ok=True)
            torch.save(chunk_seq_lens_t, os.path.join(seq_dir, f"{chunk_idx}.pt"))
        self._dump_saved_tokens[layer_id] = saved_so_far + tokens_to_save
        self._dump_chunk_idx[layer_id] = chunk_idx + 1
        if saved_so_far + tokens_to_save >= dump_tokens:
            self._dump_kv_done_layers.add(layer_id)

    def _forward_decode_int2(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        layer: RadixAttention,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        attn_logits: torch.Tensor,
        logits_soft_cap: float,
        sinks: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Decode over an int2 pool: rotate q into the KV frame, run the
        quantized (or two-tier HP + quant) split-KV kernels, undo the V
        rotation on the output."""
        kv_pool = self.token_to_kv_pool
        uses_oscar = _pool_uses_oscar_rotation(kv_pool)

        q_for_decode = q.contiguous().view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        mixed_decode_metadata_available = (
            self.forward_metadata.mixed_hp_kv_indptr is not None
        )
        mixed_decode_enabled = (
            self.enable_mixed_kv and sinks is None and mixed_decode_metadata_available
        )
        if self.enable_mixed_kv and mixed_decode_metadata_available and sinks is not None:
            raise NotImplementedError(
                "Mixed KV windows do not support sink tokens in Triton decode."
            )

        # Hard guarantee that the upstream gating actually held: if mixed
        # KV is enabled with an int2 pool, ``init_forward_metadata`` must
        # have built the per-tier indices. Falling through to the
        # non-mixed ``decode_attention_fwd_quantized`` path would treat
        # HP slot ids (>= HP_OFFSET) as quant slot ids and read OOB
        # garbage from the quant buffer. The known offenders are the
        # ``spec_info != None`` decode-or-idle paths (currently gated out
        # at server-args / model-runner level); this assertion makes the
        # gating load-bearing at the kernel boundary so any future
        # widening of those upstream gates surfaces here loudly instead
        # of silently corrupting attention output.
        if self.enable_mixed_kv:
            assert mixed_decode_metadata_available, (
                "Mixed-KV pool active but mixed decode metadata not built. "
                "spec_info / non-decode-or-idle paths must not reach the "
                "mixed-KV decode dispatch -- check upstream gating in "
                "ServerArgs._unified_mixed_kv_active and "
                "model_runner_kv_cache_mixin._init_pools."
            )

        oscar_layer_idx = layer.layer_id - kv_pool.start_layer

        if uses_oscar:
            # q is [bs, q_heads, hd]; a per-head rotation is indexed by KV
            # head, so under GQA each KV head's matrix serves
            # ``kv_group_num`` consecutive query heads.
            R_k_dec = kv_pool._R_k[oscar_layer_idx]
            q_kv_group = (
                q_for_decode.shape[1] // R_k_dec.shape[0] if R_k_dec.dim() == 3 else 1
            )
            q_for_decode = _apply_oscar_rotation(q_for_decode, R_k_dec, q_kv_group)
        else:
            q_for_decode = apply_segmented_hadamard_transform(q_for_decode)
        if mixed_decode_enabled:
            bs = q_for_decode.shape[0]
            # Select the mixed scratch whose width matches this layer's
            # v_head_dim. The unified stage-2 derives the LSE stride via
            # ``// Lv`` from the logits buffer, so the scratch width MUST
            # equal v_head_dim. Sliding layers (gemma4_unified two-group)
            # use the SWA-sized scratch; full layers use the default one.
            is_sliding_layer = (
                layer.sliding_window_size is not None and layer.sliding_window_size > 0
            )
            if (
                self.forward_metadata.mixed_swa_attn_logits is not None
                and self.swa_v_head_dim is not None
                and layer.v_head_dim == self.swa_v_head_dim
            ):
                mixed_logits = self.forward_metadata.mixed_swa_attn_logits[:bs]
                mixed_lse = self.forward_metadata.mixed_swa_attn_lse[:bs]
            else:
                mixed_logits = self.forward_metadata.mixed_attn_logits[:bs]
                mixed_lse = self.forward_metadata.mixed_attn_lse[:bs]
            # Sliding layers attend only to the last ``sliding_window``
            # tokens in decode -- use the windowed HP+quant indices built in
            # init_forward_metadata (drops the prefix-sink HP tokens and the
            # out-of-window quant bulk). Full-attention layers (and any model
            # without a sliding window) keep the full-context indices. The
            # quant split count (sized from full seq_len) is a safe upper
            # bound for the smaller windowed quant length: the int2 stage-1
            # early-exits on empty splits.
            if (
                is_sliding_layer
                and self.forward_metadata.mixed_swa_quant_kv_indptr is not None
            ):
                decode_hp_kv_indptr = self.forward_metadata.mixed_swa_hp_kv_indptr
                decode_hp_kv_indices = self.forward_metadata.mixed_swa_hp_kv_indices
                decode_quant_kv_indptr = self.forward_metadata.mixed_swa_quant_kv_indptr
                decode_quant_kv_indices = (
                    self.forward_metadata.mixed_swa_quant_kv_indices
                )
            else:
                decode_hp_kv_indptr = self.forward_metadata.mixed_hp_kv_indptr
                decode_hp_kv_indices = self.forward_metadata.mixed_hp_kv_indices
                decode_quant_kv_indptr = self.forward_metadata.mixed_quant_kv_indptr
                decode_quant_kv_indices = self.forward_metadata.mixed_quant_kv_indices
            if kv_pool.pq_k_set is not None:
                self.decode_attention_fwd_pq_unified(
                    q_for_decode,
                    kv_pool.get_hp_key_buffer(layer.layer_id),
                    kv_pool.get_hp_value_buffer(layer.layer_id),
                    kv_pool.get_raw_key_buffer(layer.layer_id),
                    kv_pool.get_raw_value_buffer(layer.layer_id),
                    kv_pool.get_value_scales_zeros(layer.layer_id),
                    o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                    decode_hp_kv_indptr,
                    decode_hp_kv_indices,
                    decode_quant_kv_indptr,
                    decode_quant_kv_indices,
                    mixed_logits,
                    mixed_lse,
                    self.forward_metadata.mixed_hp_num_kv_splits[:bs],
                    self.forward_metadata.mixed_quant_num_kv_splits[:bs],
                    self.max_hp_kv_splits,
                    self.max_kv_splits,
                    layer.scaling,
                    k_codebook=kv_pool.pq_k_codebook(layer.layer_id),
                    k_codes2=kv_pool.get_raw_key_buffer2(layer.layer_id),
                    k_codebook2=kv_pool.pq_k_codebook2(layer.layer_id),
                    v_codebook=kv_pool.pq_v_codebook(layer.layer_id),
                    logit_cap=logits_soft_cap,
                    sinks=sinks,
                    xai_temperature_len=layer.xai_temperature_len,
                )
            else:
                self.decode_attention_fwd_int2_unified(
                    q_for_decode,
                    kv_pool.get_hp_key_buffer(layer.layer_id),
                    kv_pool.get_hp_value_buffer(layer.layer_id),
                    kv_pool.get_raw_key_buffer(layer.layer_id),
                    kv_pool.get_raw_value_buffer(layer.layer_id),
                    kv_pool.get_key_scales_zeros(layer.layer_id),
                    kv_pool.get_value_scales_zeros(layer.layer_id),
                    o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                    decode_hp_kv_indptr,
                    decode_hp_kv_indices,
                    decode_quant_kv_indptr,
                    decode_quant_kv_indices,
                    mixed_logits,
                    mixed_lse,
                    self.forward_metadata.mixed_hp_num_kv_splits[:bs],
                    self.forward_metadata.mixed_quant_num_kv_splits[:bs],
                    self.max_hp_kv_splits,
                    self.max_kv_splits,
                    layer.scaling,
                    logit_cap=logits_soft_cap,
                    sinks=sinks,
                    xai_temperature_len=layer.xai_temperature_len,
                )
        else:
            # Use optimized quantized attention kernel
            self.decode_attention_fwd_quantized(
                q_for_decode,
                kv_pool.get_raw_key_buffer(layer.layer_id),
                kv_pool.get_raw_value_buffer(layer.layer_id),
                kv_pool.get_key_scales_zeros(layer.layer_id),
                kv_pool.get_value_scales_zeros(layer.layer_id),
                o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                kv_indptr,
                kv_indices,
                attn_logits,
                self.forward_metadata.attn_lse,
                self.forward_metadata.num_kv_splits,
                self.max_kv_splits,
                layer.scaling,
                kv_pool.dtype,
                logit_cap=logits_soft_cap,
                sinks=sinks,
                xai_temperature_len=layer.xai_temperature_len,
            )
        # int2: V is always rotated, so apply the inverse rotation to the
        # output. Oscar mode uses ``o @ R_v.T``; Hadamard mode re-applies
        # the segmented FWHT (self-inverse with 1/sqrt(N)).
        if uses_oscar:
            R_v = kv_pool._R_v[oscar_layer_idx]
            o3 = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            if R_v.dim() == 2:
                o3.copy_((o3.to(R_v.dtype) @ R_v.T).to(o3.dtype))
            else:
                Rv_h = R_v.repeat_interleave(max(1, o3.shape[1] // R_v.shape[0]), dim=0)
                o3.copy_(
                    torch.einsum("thd,hed->the", o3.to(R_v.dtype), Rv_h).to(o3.dtype)
                )
        else:
            o = apply_segmented_hadamard_transform(o)
        return o

    def _forward_decode_packed_mla(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        layer: RadixAttention,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        attn_logits: torch.Tensor,
        k_descale: float,
        logits_soft_cap: float,
    ) -> torch.Tensor:
        """Packed-INT2 latent: dequantize inside the KV loop instead of
        loading a BF16 row that does not exist. 288 B/token read instead
        of 1152."""
        from sglang.srt.layers.attention.triton_ops.mla_packed_decode import (
            packed_mla_decode_fwd,
            packed_mla_decode_gf_fwd,
        )

        pool = self.token_to_kv_pool
        if self._gf_enabled:
            # Group-factored two-pass path. Validated against the override
            # kernel after stage 2 (rel 3.65e-03, inside bf16 rounding,
            # with the arena confirmed load-bearing at 1.77e-01) and 4.75x
            # faster in the microbenchmark.
            #
            # It needs one split slot for the BF16 window partial, and it
            # BORROWS rather than allocates: the packed pass runs on
            # num_kv_splits - 1 and the window writes the freed last slot,
            # so stage 2 still reads num_kv_splits and no buffer, and in
            # particular no CUDA-graph capture buffer, changes shape.
            # Enlarging the split axis would have touched every model on
            # this backend to speed up one pool.
            ns = self.forward_metadata.num_kv_splits
            # Persistent buffers, filled IN PLACE.
            #
            # These are kernel arguments, and a CUDA graph captures the
            # pointer it was given. Allocating them per call inside the
            # captured region means replay reads whatever now lives at an
            # address the allocator has since recycled -- which is
            # consistent with a kernel that passes its equivalence gate
            # eagerly at three shapes and still garbles in a captured
            # server. Sizing from ns and reusing keeps one address alive
            # for the graph's lifetime.
            buf = self._gf_split_bufs
            if buf is None or buf[0].shape[0] < ns.shape[0]:
                buf = (
                    torch.empty_like(ns),
                    torch.empty_like(ns),
                )
                self._gf_split_bufs = buf
            ns_quant, ns_merge = buf[0][: ns.shape[0]], buf[1][: ns.shape[0]]
            # inference_mode for the same reason the launcher needs it:
            # these buffers derive from `ns`, which under CUDA-graph capture
            # is an INFERENCE TENSOR, and `out=` is an in-place write just
            # like fill_(). I fixed only the explicit fill_/zero_ first and
            # this one killed the very next capture -- `out=` does not look
            # like mutation at a glance, which is exactly why it was missed.
            with torch.inference_mode():
                torch.clamp(ns - 1, min=1, out=ns_quant)
                torch.add(ns_quant, 1, out=ns_merge)
            # ns_merge is ns_quant + 1, NOT the original ns.
            #
            # They agree whenever ns >= 2, but at ns == 1 the clamp keeps
            # ns_quant at 1, so the packed pass writes split 0 and the
            # window writes slot 1 -- while ns says to read one split. The
            # window partial is then dropped, and the packed pass has
            # already excluded those tokens, so they are lost outright. On a
            # short sequence the window IS most of the sequence, which is
            # why the live probe returned '!!!!!!' on 55-token prompts while
            # the microbenchmark, run at 20000 tokens where ns is never 1,
            # passed its equivalence gate.
            packed_mla_decode_gf_fwd(
                q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                pool,
                layer.layer_id,
                o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                attn_logits,
                self.forward_metadata.attn_lse,
                kv_indptr,
                kv_indices,
                ns_quant,
                self.max_kv_splits - 1,
                layer.scaling * k_descale,
                logit_cap=logits_soft_cap,
                num_kv_splits_plus1=ns_merge,
            )
            return o
        packed_mla_decode_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            pool,
            layer.layer_id,
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            attn_logits,
            self.forward_metadata.attn_lse,
            kv_indptr,
            kv_indices,
            self.forward_metadata.num_kv_splits,
            self.max_kv_splits,
            layer.scaling * k_descale,
            logit_cap=logits_soft_cap,
        )
        return o

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
        score_mod=None,
        aux_tensors=None,
    ):
        # During torch.compile, there is a bug in rotary_emb that causes the
        # output value to have a 3D tensor shape. This reshapes the output correctly.
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        # TODO: reuse the buffer across layers
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        logits_soft_cap = logit_capping_mod(layer.logit_capping_method, layer.logit_cap)

        if save_kv_cache:
            # The OSCAR pools take a bare loc tensor, not a KVWriteLoc (static
            # pools: ``out_cache_loc`` is already the kernel-facing slot id).
            if _is_int2_pool(self.token_to_kv_pool):
                # ``is_decode=True``: the unified pool routes a single-token
                # write to the HP-recent ring (no boolean masking, safe under
                # CUDA-graph capture) and must not take the quant+HP mixed path.
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    forward_batch.out_cache_loc,
                    k,
                    v,
                    layer.k_scale,
                    layer.v_scale,
                    is_decode=True,
                )
            elif self.packed_mla_pool:
                if layer.k_scale is not None:
                    # The packed pool takes no scale parameters; k is unused
                    # after this point in decode, so scale in place.
                    k.div_(layer.k_scale)
                self.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v
                )
            elif self.use_mla:
                if layer.k_scale is not None:
                    # MLATokenToKVPool doesn't accept scale parameters; k is unused
                    # after this point in decode, so scale in place.
                    k.div_(layer.k_scale)
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    # `full_loc` carries the pre-translated loc under the unified
                    # pool, refreshed into a capture-stable buffer before replay —
                    # translating inside set_kv_buffer would be captured and replay
                    # a stale v2p. None (-> raw loc) for static pools.
                    KVWriteLoc.for_batch(
                        forward_batch,
                        swa_loc=self.forward_metadata.swa_out_cache_loc,
                        full_loc=self.forward_metadata.out_cache_loc_full_physical,
                    ),
                    k,
                    v,
                )
            else:
                self._set_kv_buffer(
                    forward_batch,
                    layer,
                    KVWriteLoc.for_batch(
                        forward_batch,
                        swa_loc=self.forward_metadata.swa_out_cache_loc,
                        full_loc=self.forward_metadata.out_cache_loc_full_physical,
                    ),
                    k,
                    v,
                    layer.k_scale,
                    layer.v_scale,
                )

        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            kv_indptr = self.forward_metadata.window_kv_indptr
            kv_indices = self.forward_metadata.window_kv_indices
        else:
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices

        if layer.k_scale is not None and layer.v_scale is not None:
            k_descale = layer.k_scale_float
            v_descale = layer.v_scale_float
        else:
            k_descale = 1.0
            v_descale = 1.0

        # Select the correctly-sized attn_logits buffer for this layer.
        # The triton kernel's // Lv stride trick requires attn_logits.shape[-1]
        # to exactly match the layer's v_head_dim.
        attn_logits = self.forward_metadata.attn_logits
        if (
            self.forward_metadata.swa_attn_logits is not None
            and layer.v_head_dim == self.swa_v_head_dim
        ):
            attn_logits = self.forward_metadata.swa_attn_logits

        # Resolve Work-Centric (Lean) Attention activation. In auto mode (None) the decision
        # depends on whether this forward is a CUDA-graph capture: during capture seq_lens are
        # the fill value (1), so the seq-len gate would always bake the standard kernel and Lean
        # would never activate on the default path. There we key the bake on capture-time-known
        # signals (batch, head-tiles, is_mla) via lean_capture_policy -- Lean's fixed persistent
        # grid still adapts to raggedness on-device at replay. In eager decode, real seq_lens
        # are known, so lean_decode_seqlen_gate uses them. Deterministic inference requires
        # the batch-invariant standard path; the SGLANG_DISABLE_LEAN_ATTENTION kill-switch
        # also forces that path. Otherwise, an explicit True/False override is respected.
        from sglang.srt.environ import envs
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            get_is_capture_mode,
        )

        if self.enable_deterministic or envs.SGLANG_DISABLE_LEAN_ATTENTION.get():
            enable_lean = False
        else:
            enable_lean = self.enable_lean_attention
            if enable_lean is None:
                kv_group_num = layer.tp_q_head_num // layer.tp_k_head_num
                is_mla = layer.qk_head_dim != layer.v_head_dim
                if get_is_capture_mode():
                    enable_lean = self._lean_capture_policy(
                        layer.tp_q_head_num,
                        kv_group_num,
                        forward_batch.batch_size,
                        is_mla,
                    )
                else:
                    enable_lean = self._lean_decode_seqlen_gate(
                        layer.tp_q_head_num,
                        kv_group_num,
                        forward_batch.batch_size,
                        forward_batch.seq_lens_sum,
                        is_mla,
                    )

        # Int2 quantized KV cache path (the only supported quant tier).
        if _is_int2_pool(self.token_to_kv_pool):
            return self._forward_decode_int2(
                q,
                o,
                layer,
                kv_indptr,
                kv_indices,
                attn_logits,
                logits_soft_cap,
                sinks,
            )
        if self.packed_mla_pool:
            return self._forward_decode_packed_mla(
                q,
                o,
                layer,
                kv_indptr,
                kv_indices,
                attn_logits,
                k_descale,
                logits_soft_cap,
            )

        if self.dcp_size > 1:
            if score_mod is not None:
                raise NotImplementedError(
                    "DCP Triton decode does not support score_mod"
                )
            group = get_parallel().dcp_group
            with use_symmetric_memory(group):
                q_for_decode = q.view(
                    -1, layer.tp_q_head_num, layer.qk_head_dim
                ).contiguous()
            q_for_decode = group.all_gather(q_for_decode, dim=1).contiguous()
            o_for_decode = torch.empty(
                (q_for_decode.shape[0], q_for_decode.shape[1], layer.v_head_dim),
                dtype=torch.float32,
                device=q.device,
            )
            self.forward_metadata.attn_lse.fill_(-float("inf"))
            self.decode_attention_fwd(
                q_for_decode,
                self.token_to_kv_pool.get_key_buffer(layer.layer_id),
                self.token_to_kv_pool.get_value_buffer(layer.layer_id),
                o_for_decode,
                kv_indptr,
                kv_indices,
                attn_logits,
                self.forward_metadata.attn_lse,
                self.forward_metadata.num_kv_splits,
                self.max_kv_splits,
                layer.scaling,
                k_descale,
                v_descale,
                logit_cap=logits_soft_cap,
                sinks=sinks,
                xai_temperature_len=layer.xai_temperature_len,
                enable_lean=enable_lean,
                lean_Mp=self.forward_metadata.lean_Mp,
                lean_Lp=self.forward_metadata.lean_Lp,
                lean_Op=self.forward_metadata.lean_Op,
                lean_locks=self.forward_metadata.lean_locks,
            )
            local_lse = torch.logsumexp(
                self.forward_metadata.attn_lse[
                    : q_for_decode.shape[0], : q_for_decode.shape[1], :
                ],
                dim=-1,
            )
            o = cp_lse_ag_out_rs_mha(o_for_decode, local_lse, group)
            return o.reshape(-1, layer.tp_q_head_num * layer.v_head_dim).to(q.dtype)

        self.decode_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),
            self.token_to_kv_pool.get_value_buffer(layer.layer_id),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            kv_indptr,
            kv_indices,
            attn_logits,
            self.forward_metadata.attn_lse,
            self.forward_metadata.num_kv_splits,
            self.max_kv_splits,
            layer.scaling,
            k_descale,
            v_descale,
            logit_cap=logits_soft_cap,
            sinks=sinks,
            xai_temperature_len=layer.xai_temperature_len,
            has_mla=self.use_mla,
            use_pdl=self.use_pdl,
            page_size=self.page_size,
            score_mod=score_mod,
            aux_tensors=aux_tensors,
            enable_lean=enable_lean,
            lean_Mp=self.forward_metadata.lean_Mp,
            lean_Lp=self.forward_metadata.lean_Lp,
            lean_Op=self.forward_metadata.lean_Op,
            lean_locks=self.forward_metadata.lean_locks,
        )
        return o


class TritonMultiStepDraftBackend:
    """
    Wrap multiple triton attention backends as one for multiple consecutive
    draft decoding steps.
    """

    needs_cpu_seq_lens: bool = False

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        max_bs = model_runner.req_to_token_pool.size * self.topk
        self.kv_indptr = torch.zeros(
            (
                self.speculative_num_steps,
                max_bs + 1,
            ),
            dtype=torch.int32,
            device=model_runner.device,
        )
        self.attn_backends: List[TritonAttnBackend] = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                TritonAttnBackend(
                    model_runner,
                    skip_prefill=True,
                    kv_indptr_buf=self.kv_indptr[i],
                )
            )
        self.max_context_len = self.attn_backends[0].max_context_len
        self.num_head = (
            model_runner.model_config.get_max_num_attention_heads()
            // get_parallel().attn_tp_size
        )
        self.device = model_runner.device
        # Cached variables for generate_draft_decode_kv_indices
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]
        self.page_size = get_schedule().page_size
        self.draft_window_size, self.draft_sink_size = resolve_draft_decode_window(
            model_runner
        )

    def common_template(
        self,
        forward_batch: ForwardBatch,
        kv_indices_buffer: Optional[torch.Tensor],
        call_fn: int,
    ):
        if kv_indices_buffer is None:
            kv_indices_buffer = self.cuda_graph_kv_indices

        num_seqs = forward_batch.batch_size
        bs = self.topk * num_seqs
        seq_lens_sum = forward_batch.seq_lens_sum
        if seq_lens_sum is None:
            # seq_lens_sum here only slice-clamps a preallocated kv_indices buffer;
            # over-estimate is safe. Use a static UB to skip the per-iter .sum().item() D2H.
            seq_lens_sum = num_seqs * self.max_context_len

        generate_draft_decode_kv_indices[
            (self.speculative_num_steps, num_seqs, self.topk)
        ](
            forward_batch.req_pool_indices,
            self.req_to_token_pool.req_to_token,
            forward_batch.seq_lens,
            kv_indices_buffer,
            self.kv_indptr,
            forward_batch.positions,
            self.pool_len,
            kv_indices_buffer.shape[1],
            self.kv_indptr.shape[1],
            next_power_of_2(num_seqs),
            next_power_of_2(self.speculative_num_steps),
            next_power_of_2(bs),
            self.page_size,
            self.draft_window_size,
            self.draft_sink_size,
        )

        if call_fn is None:
            return

        for i in range(self.speculative_num_steps - 1):
            forward_batch.spec_info.kv_indptr = self.kv_indptr[i, : bs + 1]
            forward_batch.spec_info.kv_indices = kv_indices_buffer[i][
                : draft_kv_indices_used_len(seq_lens_sum, self.topk, bs, i + 1)
            ]
            call_fn(i, forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        kv_indices_width = draft_kv_indices_buffer_width(
            forward_batch.batch_size, self.topk, self.max_context_len
        )
        kv_indices = torch.empty(
            (self.speculative_num_steps, kv_indices_width),
            dtype=torch.int64,
            device=self.device,
        )

        def call_fn(i, forward_batch):
            forward_batch.spec_info.kv_indptr = (
                forward_batch.spec_info.kv_indptr.clone()
            )
            forward_batch.spec_info.kv_indices = (
                forward_batch.spec_info.kv_indices.clone()
            )
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, kv_indices, call_fn)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        kv_indices_width = draft_kv_indices_buffer_width(
            max_bs, self.topk, self.max_context_len
        )
        self.cuda_graph_kv_indices = torch.zeros(
            (self.speculative_num_steps, kv_indices_width),
            dtype=torch.int64,
            device=self.device,
        )
        self.cuda_graph_num_kv_splits = torch.full(
            (max_num_tokens,),
            self.attn_backends[0].max_kv_splits,
            dtype=torch.int32,
            device=self.device,
        )

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(
                max_bs,
                max_num_tokens,
                kv_indices_buf=self.cuda_graph_kv_indices[i],
                cuda_graph_num_kv_splits_buf=self.cuda_graph_num_kv_splits,
            )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        from sglang.srt.model_executor.forward_batch_info import build_inner_fb_view

        if in_capture:
            inner_fb = build_inner_fb_view(
                forward_batch,
                bs=forward_batch.batch_size,
                forward_mode=ForwardMode.DECODE,
            )

            def call_fn(i, _forward_batch):
                self.attn_backends[i].init_forward_metadata_out_graph(
                    inner_fb, in_capture=True
                )

            self.common_template(forward_batch, None, call_fn)
        else:
            bs = forward_batch.batch_size
            self.common_template(forward_batch, None, None)

            # NOTE: Multi-step's attention backends use the slice of
            # - kv_indptr buffer (cuda graph and non-cuda graph)
            # - kv_indices buffer (cuda graph only)
            # So we don't need to assign the KV indices inside the attention backend.

            # Compute num_kv_splits only once
            num_token = bs * self.topk
            self.attn_backends[-1].get_num_kv_splits(
                self.attn_backends[-1].cuda_graph_num_kv_splits[:num_token],
                forward_batch.seq_lens[:bs],
            )

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch) -> None:
        for attn_backend in self.attn_backends:
            attn_backend.init_forward_metadata_in_graph(forward_batch)


def update_sliding_window_buffer(
    window_kv_indptr,
    translator,
    req_pool_indices,
    sliding_window_size,
    seq_lens,
    bs,
    device=None,
    token_to_kv_pool=None,
    window_kv_indices=None,
):
    """Fill window KV buffers for sliding-window attention.

    Pass ``window_kv_indices`` to write into a pre-allocated buffer (CUDA-graph
    path); omit it (or pass ``None``) to allocate a fresh tensor (eager path,
    requires ``device``).

    Unified pool: the gather reads the swa sub-pool's own id space (built
    directly from virtual ids through the swa side's own v2p), so the window
    indices come out already swa-side ids -- no translate here, eager or
    captured. Static SWA pools gather full-token ids from req_to_token and keep
    the legacy full->swa translate below.
    """
    window_kv_lens = torch.minimum(
        seq_lens,
        torch.tensor(sliding_window_size),
    )
    window_kv_indptr[1 : bs + 1] = torch.cumsum(window_kv_lens, dim=0)
    window_kv_indptr = window_kv_indptr[: bs + 1]
    if window_kv_indices is None:
        window_kv_indices = torch.empty(
            window_kv_indptr[-1], dtype=torch.int64, device=device
        )
    window_kv_start_idx = seq_lens - window_kv_lens
    translated = translator.fill_packed_read_stream(
        req_pool_indices=req_pool_indices[:bs],
        seq_lens=window_kv_lens,
        indptr=window_kv_indptr,
        total_tokens=window_kv_indices.numel(),
        out=window_kv_indices,
        kv_start_idx=window_kv_start_idx,
        sliding_window=translator.reads_are_translated,
    )
    if not translated and isinstance(token_to_kv_pool, BaseSWAKVPool):
        kv_last_index = window_kv_indptr[-1]
        window_kv_indices[:kv_last_index] = (
            token_to_kv_pool.translate_loc_from_full_to_swa(
                window_kv_indices[:kv_last_index]
            )
        )
    return window_kv_indptr, window_kv_indices, window_kv_lens, window_kv_start_idx
