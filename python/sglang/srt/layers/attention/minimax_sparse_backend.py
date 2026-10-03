from __future__ import annotations

import copy
import logging
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional, Tuple

import msgspec
import torch

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.configs.model_config import (
    get_minimax_sparse_attention_config,
    get_minimax_sparse_disable_value_layer_ids,
    get_minimax_sparse_layer_ids,
    get_minimax_sparse_score_type,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import (
    AttentionBackend,
    SharedReadEnds,
)
from sglang.srt.layers.attention.minimax_sparse_staging import (
    build_slot_to_ragged,
    decode_block_rows,
    fill_decode_fake_table,
    prefill_fake_table,
    stage_decode_blocks_fused,
)
from sglang.srt.layers.attention.quantized_kv_prefill import (
    apply_inverse_v_rotation,
    build_prefix_indices_from_req_to_token,
    dequantize_prefix_kv,
    prepare_quantized_extend_qkv,
)
from sglang.srt.layers.moe.utils import is_tbo_enabled
from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool
from sglang.srt.mem_cache.minimax_int2_kv_pool import MiniMaxInt2SparseKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import (
    get_parallel,
    get_spec,
)
from sglang.srt.server_args import m3_fp8_attn_gemm_enabled
from sglang.srt.utils import is_gfx95_supported, is_hip, is_npu

if is_npu():
    from sglang.kernels.ops.attention.minimax_sparse.common.index import (
        topk_index_reduce,
    )

# Adaptive block_size_q thresholds (cut K-cache traffic; affects only the serial loop).
_BSQ_THRESHOLD_64 = 4096  # max_seqlen_k >= 4K  -> BSQ=64
_BSQ_THRESHOLD_32 = 1024  # max_seqlen_k >= 1K  -> BSQ=32
_BSQ_THRESHOLD_16 = 512  # max_seqlen_k >= 512 -> BSQ=16
# BSQ<=64 is UB-safe for the prefill indexer (Q tile up to 8KB at BSQ=64).


def _native_indexer_enabled() -> bool:
    # Native AscendC packed indexer switch (default off).
    return envs.SGLANG_MINIMAX_NPU_NATIVE_INDEXER.get()


def _native_attn_enabled() -> bool:
    # Native AscendC sparse MAIN-attention switch (default off).
    return envs.SGLANG_MINIMAX_NPU_NATIVE_ATTN.get()


if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def _kv_cache_to_bnsd(
    k_cache: torch.Tensor, v_cache: torch.Tensor, page_size: int
) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
    """Reshape NHD slot-major KV caches to BNSD [pages, page_size, heads, dim].

    Already-paged 4D inputs pass through unchanged.
    """
    if k_cache.dim() == 4:
        num_pages, _, num_kv_heads, head_dim = k_cache.shape
        return k_cache, v_cache, num_pages, num_kv_heads, head_dim
    num_pages = k_cache.shape[0] // page_size
    num_kv_heads = k_cache.shape[1]
    head_dim = k_cache.shape[2]
    return (
        k_cache.view(num_pages, page_size, num_kv_heads, head_dim),
        v_cache.view(num_pages, page_size, num_kv_heads, head_dim),
        num_pages,
        num_kv_heads,
        head_dim,
    )


def _idx_cache_to_bnsd(
    idx_k_cache: torch.Tensor,
    idx_v_cache: Optional[torch.Tensor],
    page_size: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, int]:
    """Reshape NHD slot-major index caches to BNSD; already-paged 4D passes through."""
    if idx_k_cache.dim() == 4:
        return idx_k_cache, idx_v_cache, idx_k_cache.shape[2], idx_k_cache.shape[3]
    num_pages = idx_k_cache.shape[0] // page_size
    idx_kv_heads = idx_k_cache.shape[1]
    idx_dim = idx_k_cache.shape[2]
    idx_v_bnsd = (
        None
        if idx_v_cache is None
        else idx_v_cache.view(num_pages, page_size, idx_kv_heads, idx_dim)
    )
    return (
        idx_k_cache.view(num_pages, page_size, idx_kv_heads, idx_dim),
        idx_v_bnsd,
        idx_kv_heads,
        idx_dim,
    )


def _quant_q_fp8(q: torch.Tensor, q_scale: Optional[float]) -> torch.Tensor:
    # Same convention as the KV pools: the fp8 tensor stores value/scale and
    # the attention kernels multiply the logits back by the scale (None = unit).
    if q_scale is not None:
        q = q / q_scale
    return q.to(torch.float8_e4m3fn)


def _is_int2_mixed_pool(pool) -> bool:
    # The OSCAR per-head INT2 pool with BF16 windows, probed on the pool
    # contract the way the triton backend selects its int2 paths.
    return pool.dtype == "int2" and pool.mixed_kv_enabled() is True



class _Int2DecodeBuffers(msgspec.Struct, frozen=True):
    """Persistent decode staging for the int2 pool, sized for ``max_bs``."""

    max_bs: int
    fake: torch.Tensor  # [max_bs, req_to_token cols + 1] int32; last col = dump
    rows: torch.Tensor  # [max_bs, n_blocks, block_size] int32 staged row ids
    slot_ids: torch.Tensor  # [max_bs] int32 = arange
    k: torch.Tensor  # [max_bs * n_blocks * block_size, kv_heads, head_dim]
    v: torch.Tensor  # [max_bs * n_blocks * block_size, kv_heads, v_head_dim]


class _Int2PrefillStaging(msgspec.Struct, frozen=True):
    """Per-ForwardBatch prefill staging tables for the int2 pool."""

    owner: object  # the ForwardBatch these were built for
    flat: torch.Tensor  # [sum(seq_lens)] int64 slots in ragged request order
    fake: torch.Tensor  # [bs, max_seqlen_k] int32 = cu_seqlens_k[b] + pos
    slot_ids: torch.Tensor  # [bs] int32 = arange

class MiniMaxSparseAttnBackend(AttentionBackend):
    def __init__(self, runner: ModelRunner):
        # The sparse kernels gather K/V rows by slot id out of a BF16 (or fp8)
        # cache. The OSCAR int2 pool has no such cache, so on it every
        # sparse layer dequantizes the rows a forward attends into a staging
        # buffer and the kernels read that through a fake page table; see the
        # "OSCAR INT2 pool" section below.
        self.int2 = _is_int2_mixed_pool(runner.token_to_kv_pool)
        if not self.int2:
            assert isinstance(runner.token_to_kv_pool, MiniMaxSparseKVPool)
        self.is_npu = is_npu()
        self.is_hip = is_hip()
        self.kv_pool = runner.token_to_kv_pool
        self.hisparse_coordinator = runner.hisparse_coordinator
        self.token_to_kv_pool = runner.token_to_kv_pool  # alias for TboAttnBackend
        self.req_to_token_pool = runner.req_to_token_pool  # pool obj for TboAttnBackend
        self.req_to_token = runner.req_to_token_pool.req_to_token
        self.max_context_len = int(runner.model_config.context_len)
        # Per-forward cache for the native decode block table (rebuilt each forward).
        self._native_decode_bt: dict = {}
        self.fp8_attn_gemm = m3_fp8_attn_gemm_enabled(
            resolving_view(runner.server_args)
        )
        if self.fp8_attn_gemm:
            assert self.kv_pool.main_pool.dtype == torch.float8_e4m3fn, (
                "fp8 attn-GEMM mode requires an fp8_e4m3fn main KV pool, got "
                f"{self.kv_pool.main_pool.dtype}"
            )

        hf_config = runner.model_config.hf_config
        sparse_cfg = get_minimax_sparse_attention_config(hf_config)
        self.idx_head_dim = sparse_cfg["sparse_index_dim"]
        self.dense_layer_ids, self.sparse_layer_ids = get_minimax_sparse_layer_ids(
            sparse_cfg
        )
        self.disable_value_layer_ids: set[int] = set(
            get_minimax_sparse_disable_value_layer_ids(sparse_cfg)
        )
        self.score_type: str = get_minimax_sparse_score_type(sparse_cfg)

        # Plain Python int so it is safe inside CUDA graphs (no .item() at graph time).
        self._max_seqlen_q: int = 1
        self._max_seqlen_k: int = 1

        # NPU: per-forward cached metadata for the triton paths (rebuilt each forward).
        self._prefill_meta: Optional[SimpleNamespace] = None
        # (owning ForwardBatch, cu_seqlens, seq_lens, prefix_lens, cu_seqblocks_q,
        # max_seqblock_q, all_seqblock_q). The owner is part of the key because one
        # metadata init can be followed by more than one ForwardBatch reaching the
        # layers (two-batch overlap splits into two children with different
        # extend_seq_lens); a hit requires the SAME object, not just a live cache.
        self._prefill_seqblock_meta: Optional[tuple] = None
        self._extend_meta: Optional[SimpleNamespace] = None
        self._extend_meta_key: Optional[int] = None
        self._decode_seq_lens_i32_cg: dict[int, torch.Tensor] = {}
        self._verify_meta_cg: dict[tuple, SimpleNamespace] = {}
        self._linear_verify_meta: Optional[SimpleNamespace] = None

        self.block_size_q = 1
        self.block_size_k = sparse_cfg["sparse_block_size"]
        if "sparse_init_block" in sparse_cfg:
            self.init_blocks = sparse_cfg["sparse_init_block"]
        else:
            init_tokens = sparse_cfg["sparse_init_tokens"]
            self.init_blocks = (
                init_tokens + self.block_size_k - 1
            ) // self.block_size_k
        if "sparse_local_block" in sparse_cfg:
            self.local_blocks = sparse_cfg["sparse_local_block"]
        else:
            local_tokens = sparse_cfg["sparse_local_tokens"]
            self.local_blocks = (
                local_tokens + self.block_size_k - 1
            ) // self.block_size_k + 1
        self.topk_blocks = sparse_cfg["sparse_topk_blocks"]
        if self.int2:
            self._init_int2(
                model_dtype=runner.dtype, device=runner.device, sparse_cfg=sparse_cfg
            )
        if self.hisparse_coordinator is not None:
            selected_tokens = self.topk_blocks * self.block_size_k
            assert selected_tokens <= self.hisparse_coordinator.device_buffer_size, (
                f"MiniMax M3 selects {selected_tokens} sparse-attention tokens, "
                "but the HiSparse device buffer holds only "
                f"{self.hisparse_coordinator.device_buffer_size}."
            )
            self._loc_mapping = (
                self.kv_pool.main_pool.full_to_hisparse_device_index_mapping
            )
        else:
            self._loc_mapping = None

        # MSA (fmha_sm100) is SM100-only; fall back to the Triton sparse path when
        # the kernel is unavailable or its constraints don't hold.
        if self.is_npu:
            self.use_msa = False
            # Prime the native sparse op probe before cuda-graph capture.
            from sgl_kernel_npu.attention.gqa_share_sparse_attention import (
                _get_native_sparse_op,
            )

            self._native_sparse_ok = _get_native_sparse_op() is not None
        else:
            self._native_sparse_ok = False
            from sglang.srt.layers.attention.minimax_sparse_ops.msa import (
                msa_available,
            )

            # MSA (fmha_sm100) runs bf16, or uniform fp8_e4m3 under fp8 attn-GEMM mode
            # (which also casts q to fp8). An fp8 main KV cache WITHOUT the flag
            # would pair a bf16 q with fp8 K/V — unsupported by fmha_sm100's
            # uniform-dtype kernels — so it stays on the Triton sparse path (which
            # dequants fp8 on load). e5m2 is never allowed into MSA (fmha_sm100's
            # variant lookup would silently dispatch the e4m3 kernel).
            _main_kv_is_fp8 = self.kv_pool.main_pool.dtype in (
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            )
            _msa_fp8_ok = (
                self.fp8_attn_gemm
                and self.kv_pool.main_pool.dtype == torch.float8_e4m3fn
            )
            # The int2 pool is staged into per-token rows (page_size 1 in the
            # kernels' terms), which the 128-token-page MSA kernel cannot read.
            self.use_msa = (
                not self.int2
                and not envs.SGLANG_DISABLE_MSA.get()
                and self.hisparse_coordinator is None
                and msa_available()
                and self.block_size_k == 128
                and self.kv_pool.page_size == self.block_size_k
                and self.topk_blocks in (4, 8, 16, 32)
                and (not _main_kv_is_fp8 or _msa_fp8_ok)
            )
            if (
                not self.use_msa
                and not self.int2
                and not envs.SGLANG_DISABLE_MSA.get()
                and msa_available()
                and self.block_size_k == 128
                and self.kv_pool.page_size != self.block_size_k
            ):
                logger.warning(
                    "MiniMax-M3 MSA decode disabled: page_size=%d != sparse block size "
                    "%d. Pass --page-size 128 (with an attention backend that allows it, "
                    "e.g. fa4 or trtllm_mha) to enable the faster MSA kernel; falling "
                    "back to the Triton sparse path.",
                    self.kv_pool.page_size,
                    self.block_size_k,
                )

        self._msa_dec_meta = None
        if self.use_msa:
            self.num_q_heads = (
                runner.model_config.num_attention_heads // get_parallel().attn_tp_size
            )
            self.num_kv_heads = self.kv_pool.main_pool.head_num
            self._msa_nb_max = (
                self.max_context_len + self.block_size_k - 1
            ) // self.block_size_k
            self._msa_cg: dict[int, tuple] = {}

        self.page_size = self.kv_pool.page_size
        # The dense-main decode runs trtllm over the real paged cache, which
        # the int2 pool does not have.
        self.use_dense_sparse_decode = (
            (not self.is_npu)
            and not self.int2
            and self.hisparse_coordinator is None
            and envs.SGLANG_OPT_USE_MINIMAX_DENSE_SPARSE_DECODE.get()
            and self.block_size_k % self.page_size == 0
            # _dense_sparse_main_decode calls trtllm decode with a bf16 q and
            # unit bmm scales — no fp8 handling yet (follow-up).
            and not self.fp8_attn_gemm
        )
        from sglang.srt.model_executor.cuda_graph_config import (
            Backend,
            Phase,
            check_cuda_graph_backend,
        )

        spec = get_spec()
        self.speculative_num_draft_tokens = spec.speculative_num_draft_tokens
        if self.is_hip and spec.speculative_algorithm is not None:
            if spec.speculative_eagle_topk != 1:
                raise NotImplementedError(
                    "MiniMax-M3 ROCm speculative attention requires a linear "
                    "draft chain (--speculative-eagle-topk 1)."
                )
        _decode_cuda_graph = not check_cuda_graph_backend(
            Phase.DECODE, Backend.DISABLED
        )
        self._use_msa_decode = self.use_msa and (
            not _decode_cuda_graph or envs.SGLANG_OPT_USE_MSA_DECODE_UNDER_GRAPH.get()
        )

        # MSA + spec decode + cuda graph crashes mid-capture: TARGET_VERIFY batches
        # route to forward_extend, dereferencing absent extend metadata. Fail at startup.
        if (
            self.use_msa
            and _decode_cuda_graph
            and spec.speculative_algorithm is not None
        ):
            raise NotImplementedError(
                "MiniMax-M3 MSA attention does not support speculative decoding under "
                "CUDA graph. Use --disable-cuda-graph, set SGLANG_DISABLE_MSA=1, or "
                "disable speculative decoding."
            )
        self._msa_owns_decode = self._use_msa_decode and not (
            self.use_dense_sparse_decode and self.kv_pool.main_pool.head_num == 1
        )
        self.dense_backend: Optional[AttentionBackend] = None

        # Top-k reuse across layers is not wired into the int2 staging path.
        self.index_topk_freq = (
            max(int(envs.SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ.get()), 1)
            if is_hip() and is_gfx95_supported() and not is_tbo_enabled()
            and not self.int2
            else 1
        )
        self.index_cache_enabled = self.index_topk_freq > 1
        # topk_index_reduce widens the last dim to idx_group_size * topk_blocks
        # (union of the group's selections), so the shared decode buffer must be
        # that wide. Head split mirrors MiniMaxM3 sparse attention's.
        self._idx_group_size = 1
        if self.index_cache_enabled:
            _num_idx_heads = max(
                sparse_cfg["sparse_num_index_heads"] // get_parallel().attn_tp_size, 1
            )
            self._idx_group_size = max(
                _num_idx_heads // self.kv_pool.main_pool.head_num, 1
            )
        # Persistent per-bs device buffer for decode top-k reuse. Allocated eagerly
        # outside CUDA-graph capture; the captured graph only copies into and reads
        # from a fixed address.
        self._decode_topk_buf: dict = {}
        self._topk_group_of_layer: dict[int, int] = {}
        self._topk_is_source: dict[int, bool] = {}
        for ordinal, lid in enumerate(
            lid for lid in self.sparse_layer_ids if lid in self.disable_value_layer_ids
        ):
            group = ordinal // self.index_topk_freq
            self._topk_group_of_layer[lid] = group
            self._topk_is_source[lid] = (ordinal % self.index_topk_freq) == 0
        self._topk_cache: dict = {}
        self._topk_cache_owner: Optional[ForwardBatch] = None

        logger.info(
            f"[MiniMaxSparse] Backend initialized "
            f"(score_type={self.score_type!r}, "
            f"kv={'int2 (BF16 staging)' if self.int2 else str(self.kv_pool.main_pool.dtype)}, "
            f"main_attn={'MSA' if self.use_msa else 'triton'}, "
            f"index_topk_freq={self.index_topk_freq}, "
            f"msa_decode={self._use_msa_decode}, "
            f"msa_owns_decode={self._msa_owns_decode}, "
            f"decode_cuda_graph={_decode_cuda_graph}, "
            f"fp8_attn_gemm={self.fp8_attn_gemm}, "
            f"hisparse={'enabled' if self._loc_mapping is not None else 'disabled'}, "
            f"npu_native_attn={'on' if (self._native_sparse_ok and _native_attn_enabled()) else 'off'}, "
            f"disable_value_layers={sorted(self.disable_value_layer_ids)})"
        )
        if self.fp8_attn_gemm and self.use_msa:
            logger.info(
                "[MiniMaxSparse] fp8 MSA active: the first forward may "
                "JIT-compile fmha_sm100 fp8 kernel variants (cold cache can "
                "take minutes; compiles serialize across TP ranks)."
            )

    def _hisparse_swap_in_blocks(
        self,
        forward_batch: ForwardBatch,
        topk_idx: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        assert topk_idx.size(0) == 1
        top_k_device_locs = self.hisparse_coordinator.swap_in_selected_blocks(
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            top_k_blocks=topk_idx[0],
            layer_id=layer_id,
            sparse_block_size=self.block_size_k,
        )
        return top_k_device_locs.unsqueeze(0)

    @staticmethod
    def _choose_decode_score_max_chunks(batch_size: int) -> int:
        """Score chunk count per graph bucket.

        bs=1 uses 16 chunks; larger buckets keep 32. Verify has its own tuning.
        """
        return 16 if int(batch_size) == 1 else 32

    @staticmethod
    def _choose_block_size_q(max_seqlen_k: int) -> int:
        """Pick block_size_q from max KV length (MINIMAX_NPU_PREFILL_BSQ overrides)."""
        _forced = os.environ.get("MINIMAX_NPU_PREFILL_BSQ")
        if _forced:
            try:
                _v = int(_forced)
                if _v > 0:
                    return _v
            except ValueError:
                pass
        if max_seqlen_k >= _BSQ_THRESHOLD_64:
            return 64
        if max_seqlen_k >= _BSQ_THRESHOLD_32:
            return 32
        if max_seqlen_k >= _BSQ_THRESHOLD_16:
            return 16
        return 1

    # ------------------------------------------------------------------
    # Delegation helpers
    # ------------------------------------------------------------------

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        # getattr covers replay views lacking extend_seq_lens_cpu and TARGET_VERIFY.
        self._msa_dec_meta = None
        # New forward -> drop the per-forward index-cache top-k (prefill only).
        if self.index_cache_enabled:
            self._topk_cache = {}
            self._topk_cache_owner = None
        # Decode top-k reuse: pre-allocate the per-bs persistent buffer so graph
        # capture never allocates. num_kv_heads == 1 at TP>=4 for M3.
        if self.index_cache_enabled and (
            forward_batch.forward_mode.is_decode_or_idle()
            or (self.is_hip and forward_batch.forward_mode.is_target_verify())
        ):
            bs = forward_batch.seq_lens.shape[0]
            if forward_batch.forward_mode.is_target_verify():
                bs *= self.speculative_num_draft_tokens
            if bs > 0 and bs not in self._decode_topk_buf:
                _nkv = self.kv_pool.main_pool.head_num
                self._decode_topk_buf[bs] = torch.empty(
                    (_nkv, bs, self.topk_blocks * self._idx_group_size),
                    dtype=torch.int32,
                    device=forward_batch.seq_lens.device,
                )
        # Per-forward cache of the layer-invariant prefill seqblock trio.
        self._prefill_seqblock_meta = None
        if self.is_npu:
            # Invalidate cached prefill/extend metadata; rebuilt on first sparse layer.
            self._prefill_meta = None
            self._extend_meta = None
            self._extend_meta_key = None
        extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if extend_lens is not None:
            self._max_seqlen_q = int(max(extend_lens))
        else:
            self._max_seqlen_q = 1
        if in_capture and (
            forward_batch.forward_mode.is_decode_or_idle()
            or (
                (self.is_npu or self.is_hip)
                and forward_batch.forward_mode.is_target_verify()
            )
        ):
            # Capture uses tiny dummy seq_lens; bound by full context so replay
            # (longer sequences) does not miss KV blocks.
            self._max_seqlen_k = self.max_context_len
        else:
            self._max_seqlen_k = int(forward_batch.seq_lens_cpu.max().item())
            if self.is_hip and forward_batch.forward_mode.is_target_verify():
                self._max_seqlen_k += self.speculative_num_draft_tokens

        # Build plan + page table eager (outside capture) so captured forward_decode
        # runs only device-side ops; host-side code can't be captured.
        if self._msa_owns_decode and forward_batch.forward_mode.is_decode_or_idle():
            self._prepare_msa_decode_meta(forward_batch)

        # ---- REPLAY-FRESH native verify block_table ----
        if (
            self.is_npu
            and forward_batch.forward_mode.is_target_verify()
            and self.speculative_num_draft_tokens
        ):
            _ndt = self.speculative_num_draft_tokens
            _bs = forward_batch.seq_lens.shape[0]
            _key = (_bs, int(_ndt))
            _vmeta = self._verify_meta_cg.get(_key)
            if _vmeta is not None:
                _vmeta.per_query_req.copy_(
                    forward_batch.req_pool_indices.long().repeat_interleave(int(_ndt))
                )
                _prefix = (forward_batch.seq_lens.to(torch.long) - int(_ndt)).clamp(
                    min=0
                )
                _offs = torch.arange(
                    1,
                    int(_ndt) + 1,
                    device=forward_batch.seq_lens.device,
                    dtype=torch.long,
                )
                _vmeta.per_query_seq_lens.copy_(
                    (_prefix.unsqueeze(1) + _offs.unsqueeze(0))
                    .reshape(-1)
                    .to(torch.int32)
                )
                _mb = self.req_to_token.shape[1] // self.page_size
                _bt_cols = (
                    torch.arange(
                        _mb, device=_vmeta.per_query_req.device, dtype=torch.long
                    )
                    * self.page_size
                ).clamp(max=self.req_to_token.shape[1] - 1)
                _vmeta.native_bt = (
                    self.req_to_token[_vmeta.per_query_req][:, _bt_cols]
                    // self.page_size
                ).to(torch.int32)

        if self.int2:
            # Prefill staging tables are rebuilt per ForwardBatch by the first
            # sparse layer that sees it; decode staging buffers are grown here,
            # outside any capture, so forward_decode never allocates them.
            self._int2_ext = None
            if forward_batch.forward_mode.is_decode_or_idle():
                self._ensure_int2_decode_buffers(forward_batch.seq_lens.shape[0])

    def _prepare_msa_decode_meta(self, forward_batch: ForwardBatch):
        """Refresh the persistent per-batch-size MSA decode plan + page table in place."""
        from sglang.srt.layers.attention.minimax_sparse_ops.msa import (
            build_msa_decode_cg_plan,
            update_msa_decode_cg_meta,
        )

        bs = forward_batch.seq_lens.shape[0]
        if bs == 0:
            return
        entry = self._msa_cg.get(bs)
        if entry is None:
            device = forward_batch.seq_lens.device
            plan = build_msa_decode_cg_plan(
                self.num_q_heads,
                self.num_kv_heads,
                self.block_size_k,
                self.topk_blocks,
                bs,
                device=device,
                is_fp8=self.fp8_attn_gemm,
            )
            kv_indices_buf = torch.zeros(
                bs * self._msa_nb_max, dtype=torch.int32, device=device
            )
            entry = (plan, kv_indices_buf)
            self._msa_cg[bs] = entry
        plan, kv_indices_buf = entry
        update_msa_decode_cg_meta(
            plan,
            kv_indices_buf,
            self.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            self.block_size_k,
            self.topk_blocks,
            self.num_q_heads,
            self.num_kv_heads,
        )
        self._msa_dec_meta = (kv_indices_buf, plan)

    def _init_rocm_linear_verify_metadata(self, forward_batch: ForwardBatch):
        ndt = self.speculative_num_draft_tokens
        # GPU seq_lens are the accepted prefix lengths. A linear EAGLE
        # chain exposes one more KV token to each successive query.
        offsets = torch.arange(
            1,
            ndt + 1,
            device=forward_batch.seq_lens.device,
            dtype=forward_batch.seq_lens.dtype,
        )
        self._linear_verify_meta = SimpleNamespace(
            seq_lens=(forward_batch.seq_lens[:, None] + offsets[None, :]).reshape(-1),
            req_pool_indices=forward_batch.req_pool_indices.repeat_interleave(ndt),
        )

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        if not self.is_npu:
            if self.is_hip and forward_batch.forward_mode.is_target_verify():
                self._init_rocm_linear_verify_metadata(forward_batch)
            return
        # Layer-invariant decode/verify metadata as captured ops (re-read at replay).
        fm = forward_batch.forward_mode
        if fm.is_target_verify():
            ndt = self.speculative_num_draft_tokens
            if ndt:
                prefix = (forward_batch.seq_lens.to(torch.long) - int(ndt)).clamp(min=0)
                offsets = torch.arange(
                    1,
                    int(ndt) + 1,
                    device=forward_batch.seq_lens.device,
                    dtype=torch.long,
                )
                per_query_seq_lens = (
                    (prefix.unsqueeze(1) + offsets.unsqueeze(0))
                    .reshape(-1)
                    .to(torch.int32)
                )
                per_query_req = forward_batch.req_pool_indices.long().repeat_interleave(
                    int(ndt)
                )
                # Captured block_table for the native verify op (re-runs at replay).
                _mb = self.req_to_token.shape[1] // self.page_size
                _bt_cols = (
                    torch.arange(_mb, device=per_query_req.device, dtype=torch.long)
                    * self.page_size
                ).clamp(max=self.req_to_token.shape[1] - 1)
                _native_bt = (
                    self.req_to_token[per_query_req][:, _bt_cols] // self.page_size
                ).to(torch.int32)
                self._verify_meta_cg[(forward_batch.seq_lens.shape[0], int(ndt))] = (
                    SimpleNamespace(
                        per_query_seq_lens=per_query_seq_lens,
                        per_query_req=per_query_req,
                        native_bt=_native_bt,
                    )
                )
        elif fm.is_decode_or_idle():
            self._decode_seq_lens_i32_cg[forward_batch.seq_lens.shape[0]] = (
                forward_batch.seq_lens.to(torch.int32)
            )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        if self.int2:
            # Decode staging (fake page table, row ids, dequantized K/V) is
            # preallocated for the largest captured batch so capture and
            # replay only ever write into fixed addresses.
            self._ensure_int2_decode_buffers(max(int(max_bs), int(max_num_tokens)))

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def _merge_sparse_blocks(
        self,
        topk_blocks: torch.Tensor,
        query_positions: torch.Tensor,
        num_blocks: int,
    ) -> torch.Tensor:
        """Append forced init/local blocks to top-k block ids and deduplicate."""
        total = self.topk_blocks + self.init_blocks + self.local_blocks
        if self.init_blocks <= 0 and self.local_blocks <= 0:
            return topk_blocks

        block_size = self.block_size_k
        q_len = query_positions.shape[0]
        num_idx_heads = topk_blocks.shape[1]
        qcol = query_positions[:, None, None]

        if self.init_blocks == 0 and self.local_blocks == 1:
            local = (query_positions // block_size).clamp(
                min=0, max=max(num_blocks - 1, 0)
            )
            local = (
                local.to(topk_blocks.dtype)
                .view(q_len, 1, 1)
                .expand(-1, num_idx_heads, -1)
            )
            valid_topk = (topk_blocks >= 0) & (topk_blocks < num_blocks)
            valid_topk = valid_topk & (topk_blocks * block_size <= qcol)
            local_duplicate = ((topk_blocks == local) & valid_topk).any(
                dim=-1, keepdim=True
            )
            valid_local = (local >= 0) & (local < num_blocks)
            valid_local = valid_local & (local * block_size <= qcol) & ~local_duplicate
            return torch.cat(
                [
                    torch.where(
                        valid_topk, topk_blocks, torch.full_like(topk_blocks, -1)
                    ),
                    torch.where(valid_local, local, torch.full_like(local, -1)),
                ],
                dim=-1,
            )

        forced_parts = []
        if self.init_blocks > 0:
            forced_parts.append(
                torch.arange(
                    self.init_blocks,
                    device=topk_blocks.device,
                    dtype=topk_blocks.dtype,
                )
                .view(1, 1, -1)
                .expand(q_len, num_idx_heads, -1)
            )
        if self.local_blocks > 0:
            offsets = torch.arange(
                self.local_blocks,
                device=topk_blocks.device,
                dtype=query_positions.dtype,
            )
            block_ids = query_positions // block_size
            first = (block_ids - self.local_blocks + 1).clamp(min=0)
            forced_parts.append(
                (first[:, None] + offsets[None, :])
                .to(topk_blocks.dtype)
                .view(q_len, 1, -1)
                .expand(-1, num_idx_heads, -1)
            )

        forced = torch.cat(forced_parts, dim=-1)
        candidates = torch.cat([forced, topk_blocks], dim=-1)
        valid = (candidates >= 0) & (candidates < num_blocks)
        valid = valid & (candidates * block_size <= qcol)
        invalid_value = torch.full_like(candidates, num_blocks)
        sorted_candidates = torch.sort(
            torch.where(valid, candidates, invalid_value), dim=-1
        ).values
        sorted_valid = sorted_candidates < num_blocks
        previous = torch.cat(
            [
                torch.full_like(sorted_candidates[..., :1], -1),
                sorted_candidates[:, :, :-1],
            ],
            dim=-1,
        )
        keep = sorted_valid & (sorted_candidates != previous)
        ranks = torch.cumsum(keep.to(torch.int32), dim=-1) - 1
        output = torch.full(
            (q_len, num_idx_heads, total + 1),
            -1,
            dtype=topk_blocks.dtype,
            device=topk_blocks.device,
        )
        overflow_rank = torch.full_like(ranks, total)
        scatter_index = torch.where(keep & (ranks < total), ranks, overflow_rank).long()
        scatter_src = torch.where(keep, sorted_candidates, -1)
        output.scatter_(2, scatter_index, scatter_src)
        return output[:, :, :total]

    def _prepare_npu_triton_topk_idx(
        self,
        topk_idx: torch.Tensor,
        seq_lens: torch.Tensor,
        num_idx_heads: int,
        num_kv_heads: int,
        max_blocks: int,
    ) -> torch.Tensor:
        """Prepare NPU triton top-k ids in the GQA kernel layout. MiniMax-M3 (TP=16) emits it directly, skipping transpose+append+dedup."""
        if (
            self.init_blocks == 0
            and self.local_blocks == 1
            and num_idx_heads == num_kv_heads
            and topk_idx.shape[0] == num_kv_heads
            and topk_idx.dtype == torch.int32
            and topk_idx.is_contiguous()
            and seq_lens.is_contiguous()
        ):
            # Fused prefill topk already appended the causal local block ([..., topk+1]); decode/verify still need the append.
            if topk_idx.shape[2] == self.topk_blocks + 1:
                return topk_idx
            from sgl_kernel_npu.indexer.flash_block_score_decode import (
                append_local_block_to_topk_idx,
            )

            return append_local_block_to_topk_idx(
                topk_idx, seq_lens, self.block_size_k, max_blocks
            )

        if num_idx_heads > num_kv_heads:
            idx_group_size = num_idx_heads // num_kv_heads
            topk_idx = topk_index_reduce(
                topk_idx.view(num_kv_heads, idx_group_size, -1, self.topk_blocks),
                dim=1,
            )

        topk_2d = topk_idx.permute(1, 0, 2).contiguous()
        query_positions = (seq_lens.to(torch.long) - 1).clamp(min=0)
        topk_merged = self._merge_sparse_blocks(topk_2d, query_positions, max_blocks)
        return topk_merged.permute(1, 0, 2).contiguous()

    def _build_native_block_table(
        self, req_indices: torch.Tensor, max_blocks: int, device
    ) -> torch.Tensor:
        """Logical->physical page table for the native sparse main op."""
        blk_cols = (
            torch.arange(max_blocks, device=device, dtype=torch.long) * self.page_size
        ).clamp(max=self.req_to_token.shape[1] - 1)
        return (self.req_to_token[req_indices][:, blk_cols] // self.page_size).to(
            torch.int32
        )

    def _forward_npu_triton_decode(
        self,
        q: torch.Tensor,  # [B, num_q_heads, head_dim]
        k_cache: torch.Tensor,  # [num_slots, num_kv_heads, head_dim] (NHD)
        v_cache: torch.Tensor,  # [num_slots, num_kv_heads, head_dim]
        idx_q: torch.Tensor,  # [B, num_idx_heads, idx_dim]
        idx_k_cache: torch.Tensor,  # [num_slots, idx_kv_heads, idx_dim]
        idx_v_cache: Optional[
            torch.Tensor
        ],  # [num_slots, idx_kv_heads, idx_dim] or None
        forward_batch: ForwardBatch,
    ):
        """NPU decode via the ported triton kernels (BNSD paged).
        NHD paged KV reshapes to [pages, block_size, H, D]; block table from
        req_to_token.
        """
        from sgl_kernel_npu.attention.gqa_share_sparse_attention import (
            flash_decode_bnsd_with_gqa_share_sparse,
        )
        from sgl_kernel_npu.indexer.flash_block_score_decode import (
            flash_decode_bnsd_with_topk_idx,
        )

        page_size = self.page_size  # == block_size_k
        num_q_heads = q.shape[1]
        head_dim = q.shape[2]
        num_idx_heads = idx_q.shape[1]
        idx_dim = idx_q.shape[2]

        k_bnsd, v_bnsd, num_pages, num_kv_heads, head_dim = _kv_cache_to_bnsd(
            k_cache, v_cache, page_size
        )
        idx_k_bnsd, idx_v_bnsd, idx_kv_heads, idx_dim = _idx_cache_to_bnsd(
            idx_k_cache, idx_v_cache, page_size
        )

        # int32 seq_lens is layer-invariant: read the per-bs buffer built once
        # per forward (captured op), with an inline eager fallback.
        bs = forward_batch.seq_lens.shape[0]
        seq_lens = self._decode_seq_lens_i32_cg.get(bs)
        if seq_lens is None:
            seq_lens = forward_batch.seq_lens.to(torch.int32)
        max_seqlen = (
            int(self._max_seqlen_k)
            if self._max_seqlen_k
            else int(seq_lens.max().item())
        )
        max_blocks = (max_seqlen + page_size - 1) // page_size
        disable_index_value = idx_v_cache is None
        # Native main op takes a logical->physical block table, hoisted to
        # once per forward (cached by id(forward_batch)); triton falls back to req_to_token.
        _native_main_kwargs = None
        if self._native_sparse_ok and _native_attn_enabled():
            try:
                _fb_id = id(forward_batch)
                _bt = self._native_decode_bt.get(_fb_id)
                if _bt is None or _bt.shape[0] != q.shape[0]:
                    _bt = self._build_native_block_table(
                        forward_batch.req_pool_indices.long(), max_blocks, q.device
                    )
                    self._native_decode_bt = {_fb_id: _bt}  # single-entry: drop stale
                _native_main_kwargs = {"block_table": _bt}
            except Exception:
                _native_main_kwargs = None
        if disable_index_value:
            page_source_kwargs = dict(
                block_table=None,
                req_to_token=self.req_to_token,
                req_pool_indices=forward_batch.req_pool_indices,
                max_num_blocks=max_blocks,
                num_pages=num_pages,
                sanitize_page_ids=False,
            )
        else:
            # Legacy score+index-value contract for non-MiniMax-M3 sparse layouts.
            req_idx = forward_batch.req_pool_indices.long()
            max_cols = self.req_to_token.shape[1]
            blk_cols = (
                torch.arange(max_blocks, device=q.device, dtype=torch.long) * page_size
            ).clamp(max=max_cols - 1)
            token_slots = self.req_to_token[req_idx][:, blk_cols]
            page_source_kwargs = dict(
                block_table=(token_slots // page_size).to(torch.int32)
            )

        # 1) indexer: score idx_k + index attention + topk (init/local=0;
        # forced blocks are re-appended by _prepare_npu_triton_topk_idx).
        idx_o, topk_idx = flash_decode_bnsd_with_topk_idx(
            q=idx_q,
            sink=None,
            k_cache_bnsd=idx_k_bnsd,
            v_cache_bnsd=idx_v_bnsd,
            **page_source_kwargs,
            seq_lens=seq_lens,
            max_seqlen=max_seqlen,
            block_size=page_size,
            topk=self.topk_blocks,
            init_blocks=0,
            local_blocks=0,
            sm_scale=idx_dim**-0.5,
            score_type=self.score_type,
            disable_index_value=disable_index_value,
            runtime_fill_only=True,
            score_max_chunks=self._choose_decode_score_max_chunks(bs),
            fused_append_local=True,
            use_native=_native_indexer_enabled(),
        )

        # 2) Reduce heads and append forced blocks.
        topk_idx = self._prepare_npu_triton_topk_idx(
            topk_idx, seq_lens, num_idx_heads, num_kv_heads, max_blocks
        )

        # 4) Main sparse attention; native op uses the cached block table override.
        _main_kwargs = (
            {**page_source_kwargs, **_native_main_kwargs}
            if _native_main_kwargs is not None
            else page_source_kwargs
        )
        o = flash_decode_bnsd_with_gqa_share_sparse(
            q=q,
            sink=None,
            k_cache_bnsd=k_bnsd,
            v_cache_bnsd=v_bnsd,
            **_main_kwargs,
            seq_lens=seq_lens,
            block_size=page_size,
            topk_idx=topk_idx,
            sm_scale=head_dim**-0.5,
            use_native=_native_attn_enabled(),
        )

        return idx_o, o

    def _forward_npu_triton_verify(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        idx_q: torch.Tensor,
        idx_k_cache: torch.Tensor,
        idx_v_cache: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        prefix_lens: torch.Tensor,
    ):
        """Capture-safe sparse attention for TARGET_VERIFY.
        ndt queries per request, each causal (j attends KV[0:prefix+j+1]). Flatten
        to per-query rows, reuse the decode kernels (device ops only, no .item()).
        """
        from sgl_kernel_npu.attention.gqa_share_sparse_attention import (
            flash_decode_bnsd_with_gqa_share_sparse,
        )
        from sgl_kernel_npu.indexer.flash_block_score_decode import (
            flash_decode_bnsd_with_topk_idx,
        )

        page_size = self.page_size  # == block_size_k
        num_q_heads = q.shape[1]
        head_dim = q.shape[2]
        num_idx_heads = idx_q.shape[1]
        idx_dim = idx_q.shape[2]
        num_tokens = q.shape[0]
        bs = forward_batch.seq_lens.shape[0]
        ndt = num_tokens // max(bs, 1)

        k_bnsd, v_bnsd, num_pages, num_kv_heads, head_dim = _kv_cache_to_bnsd(
            k_cache, v_cache, page_size
        )
        idx_k_bnsd, idx_v_bnsd, idx_kv_heads, idx_dim = _idx_cache_to_bnsd(
            idx_k_cache, idx_v_cache, page_size
        )

        # Per-query causal seq_lens + req are layer-invariant, built once per
        # forward as captured ops; inline fallback for the eager path.
        vmeta = self._verify_meta_cg.get((bs, ndt))
        if vmeta is None:
            prefix = (forward_batch.seq_lens.to(torch.long) - int(ndt)).clamp(min=0)
            offsets = torch.arange(1, int(ndt) + 1, device=q.device, dtype=torch.long)
            per_query_seq_lens = (
                (prefix.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1).to(torch.int32)
            )
            per_query_req = forward_batch.req_pool_indices.long().repeat_interleave(
                int(ndt)
            )
        else:
            per_query_seq_lens = vmeta.per_query_seq_lens
            per_query_req = vmeta.per_query_req

        # ``max_seqlen`` comes from the capture-safe ``_max_seqlen_k`` (host-derived
        # in init_forward_metadata_out_graph) so no device->host sync here.
        max_seqlen = (
            int(self._max_seqlen_k)
            if self._max_seqlen_k
            else int(per_query_seq_lens.max().item())
        )
        max_blocks = (max_seqlen + page_size - 1) // page_size
        disable_index_value = idx_v_cache is None
        # Native verify-main: per-query block_table. CUDA-graph path uses the
        # captured vmeta.native_bt (refreshed on replay); eager builds it per call.
        _native_main_kwargs = None
        if self._native_sparse_ok and _native_attn_enabled():
            try:
                _bt = (
                    vmeta.native_bt
                    if (
                        vmeta is not None
                        and getattr(vmeta, "native_bt", None) is not None
                    )
                    else None
                )
                if _bt is None:
                    _bt = self._build_native_block_table(
                        per_query_req.long(), max_blocks, q.device
                    )
                _native_main_kwargs = {"block_table": _bt}
            except Exception:
                _native_main_kwargs = None
        if disable_index_value:
            # Keep verify's page-id range guard in the direct-map kernel.
            page_source_kwargs = dict(
                block_table=None,
                req_to_token=self.req_to_token,
                req_pool_indices=per_query_req,
                max_num_blocks=max_blocks,
                num_pages=num_pages,
                sanitize_page_ids=True,
            )
        else:
            max_cols = self.req_to_token.shape[1]
            blk_cols = (
                torch.arange(max_blocks, device=q.device, dtype=torch.long) * page_size
            ).clamp(max=max_cols - 1)
            token_slots = self.req_to_token[per_query_req][:, blk_cols]
            block_table = (token_slots // page_size).to(torch.int32)
            block_table = block_table.clamp(min=0, max=num_pages - 1)
            page_source_kwargs = dict(block_table=block_table)

        # 1) indexer: score idx_k + index attention + topk (init/local=0).
        # Pack each request's ndt draft queries into the gqa row dim.
        pack_verify = (
            disable_index_value and int(ndt) > 1 and num_idx_heads == idx_kv_heads
        )
        if pack_verify:
            idx_q_score = idx_q.reshape(bs, ndt * num_idx_heads, idx_dim)
            if num_idx_heads == 1:
                # Row order == flat query order (request-major), so the
                # per-query lengths double as the packed per-row lengths.
                score_seq_lens = per_query_seq_lens
            else:
                score_seq_lens = (
                    per_query_seq_lens.view(bs, ndt, 1)
                    .expand(bs, ndt, num_idx_heads)
                    .reshape(-1)
                )
            score_page_source_kwargs = dict(
                block_table=None,
                req_to_token=self.req_to_token,
                req_pool_indices=forward_batch.req_pool_indices,
                max_num_blocks=max_blocks,
                num_pages=num_pages,
                sanitize_page_ids=True,
            )
        else:
            idx_q_score = idx_q
            score_seq_lens = per_query_seq_lens
            score_page_source_kwargs = page_source_kwargs
        idx_o, topk_idx = flash_decode_bnsd_with_topk_idx(
            q=idx_q_score,
            sink=None,
            k_cache_bnsd=idx_k_bnsd,
            v_cache_bnsd=idx_v_bnsd,
            **score_page_source_kwargs,
            seq_lens=score_seq_lens,
            max_seqlen=max_seqlen,
            block_size=page_size,
            topk=self.topk_blocks,
            init_blocks=0,
            local_blocks=0,
            sm_scale=idx_dim**-0.5,
            score_type=self.score_type,
            disable_index_value=disable_index_value,
            packed_seq_lens=pack_verify,
            # 64-chunk graph for long contexts; runtime uses 16 chunks while
            # <=256 blocks to cut short-context score work. Runtime direct-fill
            # removes register TopK maintenance.
            score_blocks_per_chunk=8 if pack_verify else 16,
            score_max_chunks=64 if pack_verify else 32,
            runtime_fill_only=pack_verify,
            runtime_score_short_max_blocks=256 if pack_verify else 0,
            runtime_score_short_chunks=16 if pack_verify else 0,
            fused_append_local=True,
            use_native=_native_indexer_enabled(),
        )
        if pack_verify:
            # [ndt*H, bs, K] -> [H, bs*ndt, K] (request-major rows).
            k_last = topk_idx.shape[-1]
            topk_idx = (
                topk_idx.view(ndt, num_idx_heads, bs, k_last)
                .permute(1, 2, 0, 3)
                .reshape(num_idx_heads, bs * ndt, k_last)
                .contiguous()
            )

        # 2) Reduce heads and append forced blocks in the GQA kernel layout.
        topk_idx = self._prepare_npu_triton_topk_idx(
            topk_idx,
            per_query_seq_lens,
            num_idx_heads,
            num_kv_heads,
            max_blocks,
        )

        # 4) Main sparse attention; native op uses the cached block table override.
        _vmain_kwargs = (
            {**page_source_kwargs, **_native_main_kwargs}
            if _native_main_kwargs is not None
            else page_source_kwargs
        )
        o = flash_decode_bnsd_with_gqa_share_sparse(
            q=q,
            sink=None,
            k_cache_bnsd=k_bnsd,
            v_cache_bnsd=v_bnsd,
            **_vmain_kwargs,
            seq_lens=per_query_seq_lens,
            block_size=page_size,
            topk_idx=topk_idx,
            sm_scale=head_dim**-0.5,
            use_native=_native_attn_enabled(),
        )
        return idx_o, o

    def _build_prefill_meta(
        self,
        forward_batch: ForwardBatch,
        cu_seqlens: torch.Tensor,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        device,
        page_size: int,
        num_pages: int,
        total_q: int,
    ) -> SimpleNamespace:
        """Build layer-invariant prefill metadata once per forward.
        Depends only on batch shape + req_to_token (invariant across layers).
        per_query_req is the direct-page-lookup map, kept live (no per-query table).
        """
        seq_lens_l = seq_lens.to(device=device, dtype=torch.long)
        prefix_lens_l = prefix_lens.to(device=device, dtype=torch.long)
        cu_q = cu_seqlens.to(device=device, dtype=torch.long)
        extend_lens = (seq_lens_l - prefix_lens_l).clamp(min=0)  # [bs]
        per_query_req = forward_batch.req_pool_indices.long().repeat_interleave(
            extend_lens
        )  # [total_q]
        # Query j of request r sits at position prefix_r + j and causally attends to
        # KV[0 : prefix_r + j + 1], so its seq_len = prefix_r + j + 1.
        per_query_prefix = prefix_lens_l.repeat_interleave(extend_lens)  # [total_q]
        per_query_within = torch.arange(
            total_q, device=device, dtype=torch.long
        ) - cu_q[:-1].repeat_interleave(extend_lens)  # 0-indexed within each request
        per_query_seq_lens = (per_query_prefix + per_query_within + 1).to(torch.int32)

        max_seqlen = (
            int(self._max_seqlen_k)
            if self._max_seqlen_k
            else int(per_query_seq_lens.max().item())
        )
        max_blocks = (max_seqlen + page_size - 1) // page_size
        block_size_q = self._choose_block_size_q(max_seqlen)

        # Score-path qblock mappings (layer-invariant), built once per forward.
        from sgl_kernel_npu.indexer.flash_block_score_prefill import (
            _build_qblock_mappings as _build_score_qblock_mappings,
        )

        qblock_mappings = _build_score_qblock_mappings(
            cu_seqlens,
            seq_lens,
            self.req_to_token,
            forward_batch.req_pool_indices,
            block_size_q,
            page_size,
            max_blocks,
            device,
        )

        # FIA prep workspace (layer-invariant shape, reused across layers).
        topk1 = self.topk_blocks + 1
        fia_block_table_ws = torch.empty(
            (total_q, topk1), dtype=torch.int32, device=device
        )
        fia_actual_kvlen_ws = torch.empty((total_q,), dtype=torch.int32, device=device)

        return SimpleNamespace(
            per_query_req=per_query_req,
            # Pre-cast int32 for the FIA prep kernel (avoids a per-layer cast).
            per_query_req_i32=per_query_req.to(torch.int32),
            per_query_seq_lens=per_query_seq_lens,
            max_seqlen=max_seqlen,
            max_blocks=max_blocks,
            block_size_q=block_size_q,
            qblock_mappings=qblock_mappings,
            fia_block_table_ws=fia_block_table_ws,
            fia_actual_kvlen_ws=fia_actual_kvlen_ws,
        )

    def _forward_npu_triton_prefill(
        self,
        q: torch.Tensor,  # [total_extend_tokens, num_q_heads, head_dim]
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        idx_q: torch.Tensor,  # [total_extend_tokens, num_idx_heads, idx_dim]
        idx_k_cache: torch.Tensor,
        idx_v_cache: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        cu_seqlens: torch.Tensor,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        # Prefill main-attention launch tuning (decode path is unaffected).
        main_num_warps: int = 4,
        main_num_stages: int = 2,
        # Fuse this many selected blocks per loop step of the main kernel. Same
        # block set per query -> same math, only online-softmax regrouping. Use
        # num_stages=1 when >1 (larger K/V tiles pressure the UB).
        main_blocks_per_step: int = 1,
    ):
        """NPU block-sparse PREFILL via the ported triton decode kernels.
        Generalizes verify to variable per-request extend lengths: each token becomes
        a per-query row with a causal seq_len; decode kernels attend selected blocks.
        """
        from sgl_kernel_npu.attention.gqa_share_sparse_attention import (
            flash_decode_bnsd_with_gqa_share_sparse,
        )

        page_size = self.page_size  # == block_size_k
        num_q_heads = q.shape[1]
        head_dim = q.shape[2]
        num_idx_heads = idx_q.shape[1]
        idx_dim = idx_q.shape[2]
        total_q = q.shape[0]

        k_bnsd, v_bnsd, num_pages, num_kv_heads, head_dim = _kv_cache_to_bnsd(
            k_cache, v_cache, page_size
        )
        idx_k_bnsd, idx_v_bnsd, idx_kv_heads, idx_dim = _idx_cache_to_bnsd(
            idx_k_cache, idx_v_cache, page_size
        )

        # Layer-invariant metadata: built once per forward (first layer builds).
        meta = self._prefill_meta
        if meta is None:
            meta = self._build_prefill_meta(
                forward_batch,
                cu_seqlens,
                seq_lens,
                prefix_lens,
                q.device,
                page_size,
                num_pages,
                total_q,
            )
            self._prefill_meta = meta
        per_query_seq_lens = meta.per_query_seq_lens
        max_seqlen = meta.max_seqlen
        max_blocks = meta.max_blocks
        block_size_q = meta.block_size_q
        per_query_req = meta.per_query_req

        disable_index_value = idx_v_cache is None

        # 1) indexer: score idx_k + index attention + topk (init/local=0).
        # Batched varlen indexer tiles queries into block_size_q blocks and
        # scores every query-block x kv-block in one 2D dot. Fused topk +
        # causal-local append yields [..., topk+1]; the prepare helper skips
        # the duplicate append.
        from sgl_kernel_npu.indexer.flash_block_score_prefill import (
            flash_prefill_bnsd_indexer,
        )
        from sgl_kernel_npu.indexer.flash_block_score_prefill import (
            flash_prefill_bnsd_with_topk_idx as _flash_prefill_score_topk,
        )

        topk_per_query_seq_lens = per_query_seq_lens

        if disable_index_value:
            idx_o = None
            topk_idx = _flash_prefill_score_topk(
                idx_q,
                idx_k_bnsd,
                cu_seqlens,
                seq_lens,
                self.req_to_token,
                forward_batch.req_pool_indices,
                block_size_q,
                page_size,
                self.topk_blocks,
                idx_dim**-0.5,
                self.score_type,
                qblock_mappings=meta.qblock_mappings,
                per_query_seq_lens=topk_per_query_seq_lens,
            )
        else:
            idx_o, topk_idx = flash_prefill_bnsd_indexer(
                idx_q,
                idx_k_bnsd,
                idx_v_bnsd,
                cu_seqlens,
                seq_lens,
                self.req_to_token,
                forward_batch.req_pool_indices,
                block_size_q,
                page_size,
                self.topk_blocks,
                idx_dim**-0.5,
                self.score_type,
                qblock_mappings=meta.qblock_mappings,
                per_query_seq_lens=topk_per_query_seq_lens,
            )

        # 2) Reduce heads and append forced blocks in the GQA kernel layout.
        topk_idx = self._prepare_npu_triton_topk_idx(
            topk_idx,
            per_query_seq_lens,
            num_idx_heads,
            num_kv_heads,
            max_blocks,
        )
        # No range/dtype guard needed: _prepare_npu_triton_topk_idx emits {-1} U
        # [0, max_blocks-1] as int32 on both paths, and the main kernel masks
        # logical_block < 0 and sanitizes physical ids to [0, num_pages-1].

        # 4) main sparse attention over the selected blocks.
        # BPS>1 fuses blocks per step of the decode-main kernel.
        main_bps = int(
            os.environ.get(
                "SGLANG_MINIMAX_NPU_PREFILL_MAIN_BPS", str(main_blocks_per_step)
            )
        )
        main_ns = main_num_stages if main_bps == 1 else min(main_num_stages, 1)

        def _decode_main():
            # Use the request-token map directly in the decode-main kernel.  This
            # avoids materializing a [total_q, max_blocks] page table for every
            # sparse layer and keeps the per-query mapping live for graph replay.
            return flash_decode_bnsd_with_gqa_share_sparse(
                q=q,
                sink=None,
                k_cache_bnsd=k_bnsd,
                v_cache_bnsd=v_bnsd,
                block_table=None,
                req_to_token=self.req_to_token,
                req_pool_indices=per_query_req,
                max_num_blocks=max_blocks,
                num_pages=num_pages,
                sanitize_page_ids=True,
                seq_lens=per_query_seq_lens,
                block_size=page_size,
                topk_idx=topk_idx,
                sm_scale=head_dim**-0.5,
                topk_blocks_per_step=main_bps,
                num_warps=main_num_warps,
                num_stages=main_ns,
            )

        def _fia_main():
            # Native Ascend FA (FIA) with a per-query custom block_table.
            from sgl_kernel_npu.attention.fia_blockq_attention import (
                flash_prefill_bnsd_blockq_sparse_fia,
            )

            return flash_prefill_bnsd_blockq_sparse_fia(
                q=q,
                k_cache_bnsd=k_bnsd,
                v_cache_bnsd=v_bnsd,
                topk_idx=topk_idx,
                seq_lens=per_query_seq_lens,
                per_query_req=meta.per_query_req_i32,
                req_to_token=self.req_to_token,
                block_size=page_size,
                sm_scale=head_dim**-0.5,
                num_pages=num_pages,
                max_num_blocks=max_blocks,
                block_table_out=meta.fia_block_table_ws,
                actual_kvlen_out=meta.fia_actual_kvlen_ws,
            )

        use_fia = envs.SGLANG_MINIMAX_NPU_PREFILL_FIA.get() and num_kv_heads == 1
        o = _fia_main() if use_fia else _decode_main()

        return idx_o, o

    @staticmethod
    def _is_sparse_kv_cached_by_fusion(
        forward_batch: ForwardBatch, layer_id: int
    ) -> bool:
        layer_ids = forward_batch.minimax_m3_precached_sparse_layers
        return layer_ids is not None and layer_id in layer_ids

    def forward(
        self,
        q,
        k,
        v,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if forward_batch.forward_mode.is_idle():
            idx_q = kwargs.get("idx_q")
            num_idx_heads = idx_q.shape[1]
            disable_value = layer.layer_id in self.disable_value_layer_ids
            idx_out: Optional[torch.Tensor] = (
                None
                if disable_value
                else q.new_zeros(q.shape[0], num_idx_heads * self.idx_head_dim)
            )
            out = q.new_zeros(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
            return idx_out, out
        else:
            return super().forward(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )

    def _resolve_extend_meta(self, forward_batch: ForwardBatch, q: torch.Tensor):
        """Return (cu_seqlens, seq_lens, prefix_lens); NPU caches per-forward casts."""
        # NPU TARGET_VERIFY has extend_seq_lens=None (seq_lens=prefix+draft);
        # reconstruct per-seq extend lengths + prefix_lens for cu_seqlens.
        if self.is_npu and forward_batch.extend_seq_lens is None:
            _bs = forward_batch.seq_lens.shape[0]
            _ndt = self.speculative_num_draft_tokens or (q.shape[0] // max(_bs, 1))
            forward_batch.extend_seq_lens = torch.full(
                (_bs,),
                int(_ndt),
                dtype=torch.int32,
                device=forward_batch.seq_lens.device,
            )
            forward_batch.extend_seq_lens_cpu = [int(_ndt)] * _bs
            if forward_batch.extend_prefix_lens is None:
                forward_batch.extend_prefix_lens = (
                    forward_batch.seq_lens.to(torch.int32) - int(_ndt)
                ).clamp(min=0)

        # NPU cache hit (same forward_batch).
        if (
            self.is_npu
            and self._extend_meta_key == id(forward_batch)
            and self._extend_meta is not None
        ):
            m = self._extend_meta
            return m.cu_seqlens, m.seq_lens, m.prefix_lens

        cu_seqlens = torch.cat(
            [
                torch.zeros(
                    1, dtype=torch.int32, device=forward_batch.extend_seq_lens.device
                ),
                forward_batch.extend_seq_lens.to(torch.int32).cumsum(0).to(torch.int32),
            ]
        )
        seq_lens = forward_batch.seq_lens.to(torch.int32)
        if forward_batch.extend_prefix_lens is not None:
            prefix_lens = forward_batch.extend_prefix_lens.to(torch.int32)
        else:
            prefix_lens = torch.zeros_like(seq_lens)

        # NPU cache write.
        if self.is_npu:
            self._extend_meta = SimpleNamespace(
                cu_seqlens=cu_seqlens, seq_lens=seq_lens, prefix_lens=prefix_lens
            )
            self._extend_meta_key = id(forward_batch)
        return cu_seqlens, seq_lens, prefix_lens

    def _prefill_seqblock_meta_for(self, forward_batch: ForwardBatch, q: torch.Tensor):
        """Layer-invariant prefill metadata, built once per ForwardBatch object."""
        cached = self._prefill_seqblock_meta
        if cached is None or cached[0] is not forward_batch:
            cu_seqlens, seq_lens, prefix_lens = self._resolve_extend_meta(
                forward_batch, q
            )
            if self.is_npu:
                cu_seqblocks_q = max_seqblock_q = all_seqblock_q = None
            else:
                from sglang.kernels.ops.attention.minimax_sparse.common.utils import (
                    get_cu_seqblocks,
                )

                cu_seqblocks_q, max_seqblock_q, all_seqblock_q, _, _, _ = (
                    get_cu_seqblocks(
                        cu_seqlens,
                        self._max_seqlen_q,
                        self.block_size_q,
                        self.block_size_k,
                        forward_batch.extend_seq_lens_cpu,
                    )
                )
            cached = (
                forward_batch,
                cu_seqlens,
                seq_lens,
                prefix_lens,
                cu_seqblocks_q,
                max_seqblock_q,
                all_seqblock_q,
            )
            self._prefill_seqblock_meta = cached
        return cached[1:]

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        *,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        idx_v: Optional[torch.Tensor],
    ):
        if self.is_hip and forward_batch.forward_mode.is_target_verify():
            meta = self._linear_verify_meta
            if meta is None or meta.seq_lens.numel() != q.shape[0]:
                raise RuntimeError("Missing MiniMax-M3 linear verify metadata")
            # Keep the dense backend's original per-request metadata intact.
            # Sparse decode accepts one causal query per row; cache stores
            # still use the original flattened out_cache_loc exactly once.
            verify_batch = copy.copy(forward_batch)
            verify_batch.seq_lens = meta.seq_lens
            verify_batch.req_pool_indices = meta.req_pool_indices
            return self.forward_decode(
                q,
                k,
                v,
                layer,
                verify_batch,
                save_kv_cache,
                idx_q=idx_q,
                idx_k=idx_k,
                idx_v=idx_v,
            )
        disable_value = layer.layer_id in self.disable_value_layer_ids
        kv_cached_by_fusion = self._is_sparse_kv_cached_by_fusion(
            forward_batch, layer.layer_id
        )
        if self.int2:
            return self._forward_extend_int2(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                idx_q=idx_q,
                idx_k=idx_k,
                idx_v=idx_v,
                disable_value=disable_value,
                kv_cached_by_fusion=kv_cached_by_fusion,
            )
        if not kv_cached_by_fusion:
            self.kv_pool.set_fused_kv_index_buffer(
                layer,
                forward_batch.out_cache_loc,
                k,
                v,
                idx_k,
                None if disable_value else idx_v,
                layer.k_scale_float,
                layer.v_scale_float,
                layer.idx_k_scale_float,
                layer.idx_v_scale_float,
            )
        k_cache, v_cache = self.kv_pool.get_kv_buffer(layer.layer_id)
        if disable_value:
            idx_k_cache = self.kv_pool.get_index_k_buffer(layer.layer_id)
            idx_v_cache = None
        else:
            idx_k_cache, idx_v_cache = self.kv_pool.get_index_kv_buffer(layer.layer_id)

        (
            cu_seqlens,
            seq_lens,
            prefix_lens,
            cu_seqblocks_q,
            max_seqblock_q,
            all_seqblock_q,
        ) = self._prefill_seqblock_meta_for(forward_batch, q)

        # DP attention pads q beyond real tokens; trim (CPU list avoids a sync).
        if forward_batch.extend_seq_lens_cpu is not None:
            actual_num_tokens = int(sum(forward_batch.extend_seq_lens_cpu))
        else:
            actual_num_tokens = int(cu_seqlens[-1].item())
        original_num_tokens = q.shape[0]
        if actual_num_tokens < original_num_tokens:
            q = q[:actual_num_tokens]
            idx_q = idx_q[:actual_num_tokens]

        if self.is_npu:
            if forward_batch.forward_mode.is_target_verify():
                # TARGET_VERIFY runs under cuda-graph capture; use the
                # capture-safe verify path (decode kernels, no .item()).
                idx_o, o = self._forward_npu_triton_verify(
                    q,
                    k_cache,
                    v_cache,
                    idx_q,
                    idx_k_cache,
                    idx_v_cache,
                    forward_batch,
                    prefix_lens,
                )
            else:
                idx_o, o = self._forward_npu_triton_prefill(
                    q,
                    k_cache,
                    v_cache,
                    idx_q,
                    idx_k_cache,
                    idx_v_cache,
                    forward_batch,
                    cu_seqlens,
                    seq_lens,
                    prefix_lens,
                )
        else:
            # fp8 attention GEMMs: quantize q/idx_q AFTER the KV store (which reads
            # the bf16 k/v) and the DP trim.
            if self.fp8_attn_gemm:
                q = _quant_q_fp8(q, layer.q_scale_float)
                idx_q = _quant_q_fp8(idx_q, layer.idx_q_scale_float)

            # GPU (CUDA/ROCm) sparse path; imported here so NPU never touches it.
            from sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse import (
                minimax_sparse_prefill,
            )

            # Index cache: only for disable_value layers (idx_o is None,
            # so skipping the indexer has no output side effect). A group's source
            # layer computes + stores the reduced top-k; the other layers reuse it.
            use_index_cache = self.index_cache_enabled and disable_value
            cached_topk_idx = None
            want_topk = False
            if use_index_cache:
                if self._topk_cache_owner is not forward_batch:
                    self._topk_cache = {}
                    self._topk_cache_owner = forward_batch
                group = self._topk_group_of_layer[layer.layer_id]
                if self._topk_is_source[layer.layer_id]:
                    want_topk = True  # compute and store for this group
                else:
                    cached_topk_idx = self._topk_cache.get(group)
                    # Miss (e.g. source layer chunked differently) -> recompute safely.

            result = minimax_sparse_prefill(
                q,
                k_cache,
                v_cache,
                None,
                idx_q,
                idx_k_cache,
                idx_v_cache,
                None,
                self.req_to_token,
                forward_batch.req_pool_indices,
                cu_seqlens,
                seq_lens,
                prefix_lens,
                self._max_seqlen_q,
                self._max_seqlen_k,
                self.block_size_q,
                self.block_size_k,
                self.topk_blocks,
                self.init_blocks,
                self.local_blocks,
                score_type=self.score_type,
                disable_index_value=disable_value,
                use_msa=self.use_msa,
                seqlens_cpu=forward_batch.extend_seq_lens_cpu,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
                cu_seqblocks_q=cu_seqblocks_q,
                max_seqblock_q=max_seqblock_q,
                all_seqblock_q=all_seqblock_q,
                q_scale=layer.q_scale_float,
                k_scale=layer.k_scale_float,
                v_scale=layer.v_scale_float,
                idx_q_scale=layer.idx_q_scale_float,
                idx_k_scale=layer.idx_k_scale_float,
                idx_v_scale=layer.idx_v_scale_float,
                page_size=self.page_size,
                cached_topk_idx=cached_topk_idx,
                return_topk_idx=want_topk,
                loc_mapping=self._loc_mapping,
            )
            if want_topk:
                idx_o, o, reduced_topk_idx = result
                self._topk_cache[group] = reduced_topk_idx
            else:
                idx_o, o = result
        if actual_num_tokens < original_num_tokens:
            pad_len = original_num_tokens - actual_num_tokens
            o = torch.cat([o, o.new_zeros(pad_len, *o.shape[1:])], dim=0)
            if idx_o is not None:
                idx_o = torch.cat(
                    [idx_o, idx_o.new_zeros(pad_len, *idx_o.shape[1:])], dim=0
                )

        return (
            (
                None
                if idx_o is None
                else idx_o.reshape(original_num_tokens, -1).contiguous()
            ),
            o.reshape(original_num_tokens, -1).contiguous(),
        )

    def _dense_sparse_main_decode(
        self,
        q: torch.Tensor,
        page_table: torch.Tensor,
        real_seq_lens: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend

        if isinstance(self.dense_backend, TRTLLMHAAttnBackend):
            import flashinfer

            ps = self.page_size
            nkv = 1
            head_dim = q.size(-1)
            # [max_slots, nkv, D] -> [num_pages, page_size, nkv, D]
            #                     -> [num_pages, nkv, page_size, D] (HND, trtllm default)
            kc = k_cache.view(-1, ps, nkv, head_dim).permute(0, 2, 1, 3)
            vc = v_cache.view(-1, ps, nkv, head_dim).permute(0, 2, 1, 3)
            return flashinfer.decode.trtllm_batch_decode_with_kv_cache(  # type: ignore
                query=q.contiguous(),
                kv_cache=(kc, vc),
                workspace_buffer=self.dense_backend.workspace_buffer,
                block_tables=page_table,
                seq_lens=real_seq_lens,
                max_seq_len=self.topk_blocks * self.block_size_k,
                bmm1_scale=layer.scaling,
                bmm2_scale=1.0,
            )
        raise NotImplementedError(
            "dense sparse decode currently supports trtllm_mha only (fa3 is TODO)"
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        *,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        idx_v: Optional[torch.Tensor],
        **kwargs,
    ):
        assert len(kwargs) == 0
        disable_value = layer.layer_id in self.disable_value_layer_ids
        if self.int2:
            return self._forward_decode_int2(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                idx_q=idx_q,
                idx_k=idx_k,
                idx_v=idx_v,
                disable_value=disable_value,
            )
        if not self._is_sparse_kv_cached_by_fusion(forward_batch, layer.layer_id):
            self.kv_pool.set_fused_kv_index_buffer(
                layer,
                forward_batch.out_cache_loc,
                k,
                v,
                idx_k,
                None if disable_value else idx_v,
                layer.k_scale_float,
                layer.v_scale_float,
                layer.idx_k_scale_float,
                layer.idx_v_scale_float,
            )
        k_cache, v_cache = self.kv_pool.get_kv_buffer(layer.layer_id)
        if disable_value:
            idx_k_cache = self.kv_pool.get_index_k_buffer(layer.layer_id)
            idx_v_cache = None
        else:
            idx_k_cache, idx_v_cache = self.kv_pool.get_index_kv_buffer(layer.layer_id)

        attn_fn = None
        if self.use_dense_sparse_decode and k_cache.shape[1] == 1:

            def attn_fn(main_q, page_table, real_seq_lens):
                return self._dense_sparse_main_decode(
                    main_q,
                    page_table,
                    real_seq_lens,
                    k_cache,
                    v_cache,
                    layer,
                    forward_batch,
                )

        msa_kv_indices = msa_plan = None
        if self._use_msa_decode and attn_fn is None:
            if self._msa_dec_meta is not None:
                msa_kv_indices, msa_plan = self._msa_dec_meta
            elif q.shape[0] > 0:
                # Rebuilding the plan inline would run host-side code inside
                # CUDA-graph capture; fail loudly instead.
                raise RuntimeError(
                    "MSA decode metadata missing: init_forward_metadata_out_graph "
                    "did not prepare the plan for this forward (gate mismatch)."
                )

        if self.is_npu:
            idx_o, o = self._forward_npu_triton_decode(
                q,
                k_cache,
                v_cache,
                idx_q,
                idx_k_cache,
                idx_v_cache,
                forward_batch,
            )
        else:
            # fp8 attn-GEMM: quantize q/idx_q after the KV store (reads bf16 k/v).
            if self.fp8_attn_gemm:
                q = _quant_q_fp8(q, layer.q_scale_float)
                idx_q = _quant_q_fp8(idx_q, layer.idx_q_scale_float)

            # GPU (CUDA/ROCm) sparse path; imported here so NPU never touches it.
            from sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse import (
                minimax_sparse_decode,
            )

            # Decode top-k reuse: group source layer computes+stores; skips reuse.
            _use_reuse = self.index_cache_enabled and disable_value and attn_fn is None
            _topk_buf = self._decode_topk_buf.get(q.shape[0]) if _use_reuse else None
            _cached_topk = None
            _want_topk = False
            if _use_reuse and _topk_buf is not None:
                if self._topk_is_source.get(layer.layer_id, True):
                    _want_topk = True
                else:
                    _cached_topk = _topk_buf

            hisparse_swap_in_fn = None
            if self.hisparse_coordinator is not None:

                def hisparse_swap_in_fn(topk_idx):
                    return self._hisparse_swap_in_blocks(
                        forward_batch=forward_batch,
                        topk_idx=topk_idx,
                        layer_id=layer.layer_id,
                    )

            idx_o, o = minimax_sparse_decode(
                q,
                None,
                k_cache,
                v_cache,
                idx_q,
                None,
                idx_k_cache,
                idx_v_cache,
                self.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                self._max_seqlen_k,
                1,
                self.block_size_k,
                self.topk_blocks,
                self.init_blocks,
                self.local_blocks,
                score_type=self.score_type,
                disable_index_value=disable_value,
                dense_main_attn_fn=attn_fn,
                page_size=self.page_size,
                use_msa=self._use_msa_decode,
                msa_kv_indices=msa_kv_indices,
                msa_plan=msa_plan,
                q_scale=layer.q_scale_float,
                k_scale=layer.k_scale_float,
                v_scale=layer.v_scale_float,
                idx_q_scale=layer.idx_q_scale_float,
                idx_k_scale=layer.idx_k_scale_float,
                idx_v_scale=layer.idx_v_scale_float,
                cached_topk_idx=_cached_topk,
                topk_out=_topk_buf if _want_topk else None,
                hisparse_swap_in_fn=hisparse_swap_in_fn,
            )
        return (
            None if idx_o is None else idx_o.reshape(q.shape[0], -1).contiguous(),
            o.reshape(q.shape[0], -1).contiguous(),
        )

    # ------------------------------------------------------------------
    # OSCAR INT2 pool: the sparse kernels read staged BF16 rows
    # ------------------------------------------------------------------
    # The kernels gather ``cache[req_to_token[slot_ids[b], pos]]``; the int2
    # pool has no BF16 cache, so each sparse layer dequantizes the rows a
    # forward attends (both tiers, rotated frame) into a staging buffer that the
    # kernels read through a fake req_to_token keeping the REAL positions, so
    # their causal / seq_len masking is untouched. Prefill stages every token of
    # the batch in ragged order; decode stages the selected blocks into
    # persistent buffers (static shapes, graph-capturable). The index cache has
    # one row per slot and is read over the real table.

    def _init_int2(
        self, *, model_dtype: torch.dtype, device, sparse_cfg: dict
    ) -> None:
        pool = self.kv_pool
        if not isinstance(pool, MiniMaxInt2SparseKVPool):
            raise TypeError(
                "MiniMax sparse attention on int2 KV needs the index-key cache "
                "beside the pool (MiniMaxInt2SparseKVPool); the configurator "
                f"built {type(pool).__name__}."
            )
        if self.is_npu:
            raise NotImplementedError(
                "MiniMax sparse attention on the int2 KV pool runs the CUDA/ROCm "
                "Triton sparse kernels only."
            )
        if self.hisparse_coordinator is not None:
            raise NotImplementedError(
                "MiniMax sparse attention on the int2 KV pool does not support "
                "HiSparse offload."
            )
        if get_spec().speculative_algorithm is not None:
            raise NotImplementedError(
                "MiniMax sparse attention on the int2 KV pool does not support "
                "speculative decoding."
            )
        if self.fp8_attn_gemm:
            raise NotImplementedError(
                "fp8 attention GEMMs cannot be combined with the int2 KV pool."
            )
        n_kv = int(pool.head_num)
        if n_kv != 1:
            raise NotImplementedError(
                "int2 staging keeps one selected-block set per request, which "
                f"needs exactly one KV head per rank; this rank holds {n_kv}. "
                "Raise the attention TP size."
            )
        self.model_dtype = model_dtype
        # Index heads per rank follow the model's split (replicated when there
        # are fewer index heads than ranks); the block set per KV head is the
        # union over the index heads sharing it, hence group * topk blocks.
        total_idx_heads = int(sparse_cfg["sparse_num_index_heads"])
        tp = get_parallel().attn_tp_size
        n_idx = total_idx_heads // max(1, min(tp, total_idx_heads))
        self._int2_group = max(1, n_idx // n_kv)
        self._int2_n_blocks = self._int2_group * self.topk_blocks
        self._int2_ar_block = torch.arange(
            self.block_size_k, device=device, dtype=torch.int64
        )
        self._int2_slot_to_ragged = torch.full(
            (pool.index_cache_slots,), -1, dtype=torch.int32, device=device
        )
        # The decode fake table has one spare column past every legal position.
        self._int2_dump_col = int(self.req_to_token.shape[1])
        self._int2_ext: Optional[_Int2PrefillStaging] = None
        self._int2_dec: Optional[_Int2DecodeBuffers] = None
        # Fused decode staging (one launch per layer); the four-step path is
        # kept as the reference and selectable for A/B.
        self._int2_fused_staging = envs.SGLANG_MINIMAX_INT2_FUSED_STAGING.get()
        logger.info(
            "[MiniMaxSparse] int2 staging: block %d, %d blocks per request "
            "(group %d x topk %d), index cache slots %d, fake-table columns %d",
            self.block_size_k,
            self._int2_n_blocks,
            self._int2_group,
            self.topk_blocks,
            int(self._int2_slot_to_ragged.numel()),
            self._int2_dump_col + 1,
        )

    def _ensure_int2_decode_buffers(self, bs: int) -> None:
        bs = int(bs)
        dec = self._int2_dec
        if bs <= 0 or (dec is not None and dec.max_bs >= bs):
            return
        # Growing here would hand out graph-pool memory; init_cuda_graph_state
        # sizes the buffers for the largest captured batch beforehand.
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "int2 decode staging buffers hold "
                f"{0 if dec is None else dec.max_bs} requests but a captured "
                f"batch has {bs}; init_cuda_graph_state must size them first."
            )
        pool = self.kv_pool
        device = self.req_to_token.device
        n_rows = bs * self._int2_n_blocks * self.block_size_k
        self._int2_dec = _Int2DecodeBuffers(
            max_bs=bs,
            fake=torch.zeros(
                (bs, self._int2_dump_col + 1), dtype=torch.int32, device=device
            ),
            # ``rows[:bs]`` enumerates exactly the row order decode_block_rows
            # produces for bs requests.
            rows=torch.arange(n_rows, dtype=torch.int32, device=device).view(
                bs, self._int2_n_blocks, self.block_size_k
            ),
            slot_ids=torch.arange(bs, dtype=torch.int32, device=device),
            k=torch.empty(
                (n_rows, pool.head_num, pool.head_dim),
                dtype=self.model_dtype,
                device=device,
            ),
            v=torch.empty(
                (n_rows, pool.head_num, pool.v_head_dim),
                dtype=self.model_dtype,
                device=device,
            ),
        )

    def _int2_extend_staging(
        self, forward_batch: ForwardBatch, seq_lens: torch.Tensor
    ) -> _Int2PrefillStaging:
        # Per ForwardBatch, not per layer (keyed like _prefill_seqblock_meta).
        ext = self._int2_ext
        if ext is not None and ext.owner is forward_batch:
            return ext
        device = seq_lens.device
        seq_lens_cpu = [int(x) for x in forward_batch.seq_lens_cpu.tolist()]
        bs = len(seq_lens_cpu)
        # Every token of every request (prefix and this chunk) in ragged order.
        flat = build_prefix_indices_from_req_to_token(
            req_to_token=self.req_to_token,
            req_pool_indices=forward_batch.req_pool_indices,
            cache_seqlens=seq_lens,
            cache_seqlens_cpu=seq_lens_cpu,
        )
        build_slot_to_ragged(flat_slots=flat, slot_to_ragged=self._int2_slot_to_ragged)
        cu_k_cpu = torch.zeros((bs + 1,), dtype=torch.int32)
        cu_k_cpu[1:] = torch.cumsum(torch.tensor(seq_lens_cpu, dtype=torch.int32), 0)
        ext = _Int2PrefillStaging(
            owner=forward_batch,
            flat=flat,
            fake=prefill_fake_table(
                cu_seqlens_k=cu_k_cpu.to(device, non_blocking=True),
                max_seqlen_k=self._max_seqlen_k,
            ),
            slot_ids=torch.arange(bs, dtype=torch.int32, device=device),
        )
        self._int2_ext = ext
        return ext

    def _int2_write_index(
        self,
        *,
        layer,
        loc: torch.Tensor,
        idx_k: torch.Tensor,
        idx_v: Optional[torch.Tensor],
        disable_value: bool,
    ) -> None:
        if disable_value:
            self.kv_pool.set_index_k_buffer(
                layer=layer, loc=loc, cache_idx_k=idx_k, k_scale=layer.idx_k_scale_float
            )
        else:
            self.kv_pool.set_index_kv_buffer(
                layer=layer,
                loc=loc,
                cache_idx_k=idx_k,
                cache_idx_v=idx_v,
                k_scale=layer.idx_k_scale_float,
                v_scale=layer.idx_v_scale_float,
            )

    def _int2_index_caches(self, *, layer_id: int, disable_value: bool):
        if disable_value:
            return self.kv_pool.get_index_k_buffer(layer_id), None
        return self.kv_pool.get_index_kv_buffer(layer_id)

    def _int2_reduce_topk(
        self, *, topk_idx: torch.Tensor, num_kv_heads: int
    ) -> torch.Tensor:
        n_idx = topk_idx.shape[0]
        if n_idx == num_kv_heads:
            return topk_idx
        from sglang.kernels.ops.attention.minimax_sparse.common.index import (
            topk_index_reduce,
        )

        return topk_index_reduce(
            topk_idx.view(num_kv_heads, n_idx // num_kv_heads, -1, topk_idx.shape[-1]),
            dim=1,
        )

    def _int2_prefill_topk(
        self,
        *,
        layer,
        idx_q: torch.Tensor,
        idx_k_cache: torch.Tensor,
        idx_v_cache: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        meta: tuple,
        disable_value: bool,
        num_kv_heads: int,
    ):
        from sglang.kernels.ops.attention.minimax_sparse.prefill.flash_with_topk_idx import (
            flash_prefill_with_topk_index,
        )

        cu_seqlens, seq_lens, prefix_lens, cu_seqblocks_q, max_seqblock_q, all_seqblock_q = (
            meta
        )
        idx_o, topk_idx = flash_prefill_with_topk_index(
            q=idx_q,
            k_cache=idx_k_cache,
            v_cache=idx_v_cache,
            sink=None,
            req_to_token=self.req_to_token,
            slot_ids=forward_batch.req_pool_indices,
            cu_seqlens=cu_seqlens,
            seq_lens=seq_lens,
            prefix_lens=prefix_lens,
            max_seqlen_q=self._max_seqlen_q,
            max_seqlen_k=self._max_seqlen_k,
            block_size_q=self.block_size_q,
            block_size_k=self.block_size_k,
            topk=self.topk_blocks,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            score_type=self.score_type,
            disable_index_value=disable_value,
            cu_seqblocks_q=cu_seqblocks_q,
            max_seqblock_q=max_seqblock_q,
            all_seqblock_q=all_seqblock_q,
            # int2 slot ids are per token (window slots are not page-contiguous).
            page_size=1,
            q_scale=layer.idx_q_scale_float,
            k_scale=layer.idx_k_scale_float,
            v_scale=layer.idx_v_scale_float,
        )
        return idx_o, self._int2_reduce_topk(topk_idx=topk_idx, num_kv_heads=num_kv_heads)

    def _int2_decode_topk(
        self,
        *,
        layer,
        idx_q: torch.Tensor,
        idx_k_cache: torch.Tensor,
        idx_v_cache: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        disable_value: bool,
        num_kv_heads: int,
    ):
        from sglang.kernels.ops.attention.minimax_sparse.decode.flash_with_topk_idx import (
            flash_decode_with_topk_idx,
        )

        idx_o, topk_idx, _ = flash_decode_with_topk_idx(
            q=idx_q,
            sink=None,
            k_cache=idx_k_cache,
            v_cache=idx_v_cache,
            req_to_token=self.req_to_token,
            seq_lens=forward_batch.seq_lens,
            max_seqlen=self._max_seqlen_k,
            slot_ids=forward_batch.req_pool_indices,
            block_size=self.block_size_k,
            topk=self.topk_blocks,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            score_type=self.score_type,
            disable_index_value=disable_value,
            use_dense_main_attn=False,
            page_size=1,
            q_scale=layer.idx_q_scale_float,
            k_scale=layer.idx_k_scale_float,
            v_scale=layer.idx_v_scale_float,
        )
        return idx_o, self._int2_reduce_topk(topk_idx=topk_idx, num_kv_heads=num_kv_heads)

    def _int2_stage_decode_blocks(
        self, *, layer_id: int, topk_idx: torch.Tensor, forward_batch: ForwardBatch, bs: int
    ):
        # Device-side only, static shapes, writes into the preallocated buffers.
        dec = self._int2_dec
        if dec is None or dec.max_bs < bs:
            raise RuntimeError(
                "int2 decode staging buffers were not sized for this batch "
                f"(have {0 if dec is None else dec.max_bs}, need {bs}); "
                "init_forward_metadata_out_graph did not run for this forward."
            )
        n_blocks = topk_idx.shape[-1]
        if n_blocks != self._int2_n_blocks:
            raise RuntimeError(
                f"reduced top-k width {n_blocks} != {self._int2_n_blocks} the "
                "decode staging buffers were sized for"
            )
        n_rows = bs * n_blocks * self.block_size_k
        k_st, v_st = dec.k[:n_rows], dec.v[:n_rows]
        fake = dec.fake[:bs]
        if self._int2_fused_staging:
            # One launch for slot lookup, K/V dequant of every head and the
            # fake table; the four-step path below is the reference it is
            # tested against (rotation/tests/test_minimax_fused_staging_gpu.py).
            pool = self.kv_pool
            stage_decode_blocks_fused(
                topk_blk=topk_idx[0],
                req_to_token=self.req_to_token,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                block_size=self.block_size_k,
                quant_k=pool.get_raw_key_buffer(layer_id),
                scales_zeros_k=pool.get_key_scales_zeros(layer_id),
                hp_k=pool.get_hp_key_buffer(layer_id),
                quant_v=pool.get_raw_value_buffer(layer_id),
                scales_zeros_v=pool.get_value_scales_zeros(layer_id),
                hp_v=pool.get_hp_value_buffer(layer_id),
                hp_global_offset=pool.hp_global_offset,
                out_k=k_st,
                out_v=v_st,
                fake=fake,
                dump_col=self._int2_dump_col,
            )
            return k_st, v_st, fake, dec.slot_ids[:bs]
        slots, pos, valid = decode_block_rows(
            topk_blk=topk_idx[0],
            req_to_token=self.req_to_token,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            block_size=self.block_size_k,
            ar_block=self._int2_ar_block,
        )
        dequantize_prefix_kv(
            kv_pool=self.kv_pool,
            layer_id=layer_id,
            prefix_indices=slots.reshape(-1),
            model_dtype=self.model_dtype,
            out_k=k_st,
            out_v=v_st,
        )
        fill_decode_fake_table(
            fake=fake, pos=pos, valid=valid, rows=dec.rows[:bs], dump_col=self._int2_dump_col
        )
        return k_st, v_st, fake, dec.slot_ids[:bs]

    def _int2_prefill_main(
        self,
        *,
        layer,
        q3: torch.Tensor,
        own_loc: torch.Tensor,
        own_k: torch.Tensor,
        own_v: torch.Tensor,
        topk_idx: torch.Tensor,
        staging: _Int2PrefillStaging,
        meta: tuple,
    ) -> torch.Tensor:
        from sglang.kernels.ops.attention.minimax_sparse.prefill.topk_sparse import (
            flash_prefill_with_gqa_share_sparse,
        )

        cu_seqlens, seq_lens, prefix_lens, cu_seqblocks_q, max_seqblock_q, _ = meta
        k_st, v_st = dequantize_prefix_kv(
            kv_pool=self.kv_pool,
            layer_id=layer.layer_id,
            prefix_indices=staging.flat,
            model_dtype=self.model_dtype,
        )
        # This forward's own tokens attend to their exact rows, as in the
        # triton extend path.
        own = self._int2_slot_to_ragged[own_loc.to(torch.int64)].to(torch.int64)
        k_st[own] = own_k
        v_st[own] = own_v
        return flash_prefill_with_gqa_share_sparse(
            q=q3.contiguous(),
            k_cache=k_st,
            v_cache=v_st,
            sink=None,
            req_to_token=staging.fake,
            slot_ids=staging.slot_ids,
            topk_idx=topk_idx,
            block_size_q=self.block_size_q,
            block_size_k=self.block_size_k,
            cu_seqlens=cu_seqlens,
            seq_lens=seq_lens,
            prefix_lens=prefix_lens,
            max_seqlen_q=self._max_seqlen_q,
            sm_scale=layer.scaling,
            cu_seqblocks_q=cu_seqblocks_q,
            max_seqblock_q=max_seqblock_q,
        )

    def _forward_extend_int2(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool,
        *,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        idx_v: Optional[torch.Tensor],
        disable_value: bool,
        kv_cached_by_fusion: bool,
    ):
        if kv_cached_by_fusion:
            raise RuntimeError(
                "The fused norm+rope+cache kernel wrote bf16 K/V rows into the "
                "int2 pool's packed buffers; the int2 pool must not take that path."
            )
        pool = self.kv_pool
        num_tokens = q.shape[0]
        n_q_heads, head_dim = layer.tp_q_head_num, layer.qk_head_dim
        n_kv_heads, v_head_dim = layer.tp_k_head_num, layer.v_head_dim
        # One rotation serves the cache write and the staged rows.
        q3, k3, v3, need_v_inverse = prepare_quantized_extend_qkv(
            kv_pool=pool,
            layer=layer,
            q=q.reshape(num_tokens, n_q_heads, head_dim),
            k=k.reshape(-1, n_kv_heads, head_dim),
            v=v.reshape(-1, n_kv_heads, v_head_dim),
        )
        q3 = q3.to(self.model_dtype)
        k3 = k3.to(self.model_dtype)
        v3 = v3.to(self.model_dtype)

        loc = forward_batch.out_cache_loc
        if save_kv_cache:
            pool.set_kv_buffer(
                layer=layer,
                loc=loc,
                cache_k=k3,
                cache_v=v3,
                already_hadamard_transformed=True,
                is_decode=False,
            )
            self._int2_write_index(
                layer=layer, loc=loc, idx_k=idx_k, idx_v=idx_v, disable_value=disable_value
            )
        idx_k_cache, idx_v_cache = self._int2_index_caches(
            layer_id=layer.layer_id, disable_value=disable_value
        )
        meta = self._prefill_seqblock_meta_for(forward_batch, q)
        cu_seqlens, seq_lens = meta[0], meta[1]
        staging = self._int2_extend_staging(forward_batch, seq_lens)

        # DP attention pads q beyond real tokens; trim (CPU list avoids a sync).
        # k/v and out_cache_loc keep the padded length for the cache write.
        if forward_batch.extend_seq_lens_cpu is not None:
            actual_num_tokens = int(sum(forward_batch.extend_seq_lens_cpu))
        else:
            actual_num_tokens = int(cu_seqlens[-1].item())
        q3 = q3[:actual_num_tokens]
        idx_q = idx_q[:actual_num_tokens].reshape(actual_num_tokens, -1, self.idx_head_dim)

        idx_o, topk_idx = self._int2_prefill_topk(
            layer=layer,
            idx_q=idx_q,
            idx_k_cache=idx_k_cache,
            idx_v_cache=idx_v_cache,
            forward_batch=forward_batch,
            meta=meta,
            disable_value=disable_value,
            num_kv_heads=n_kv_heads,
        )
        o = self._int2_prefill_main(
            layer=layer,
            q3=q3,
            own_loc=loc[:actual_num_tokens],
            own_k=k3[:actual_num_tokens],
            own_v=v3[:actual_num_tokens],
            topk_idx=topk_idx,
            staging=staging,
            meta=meta,
        )
        o = apply_inverse_v_rotation(
            result=o.view(actual_num_tokens, n_q_heads, v_head_dim),
            kv_pool=pool,
            layer=layer,
            need_v_inverse=need_v_inverse,
        ).to(q.dtype)
        o = o.reshape(actual_num_tokens, -1)
        if idx_o is not None:
            idx_o = idx_o.reshape(actual_num_tokens, -1)
        if actual_num_tokens < num_tokens:
            pad_len = num_tokens - actual_num_tokens
            o = torch.cat([o, o.new_zeros(pad_len, o.shape[1])], dim=0)
            if idx_o is not None:
                idx_o = torch.cat([idx_o, idx_o.new_zeros(pad_len, idx_o.shape[1])], dim=0)
        return (
            None if idx_o is None else idx_o.contiguous(),
            o.contiguous(),
        )

    def _forward_decode_int2(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool,
        *,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        idx_v: Optional[torch.Tensor],
        disable_value: bool,
    ):
        if self._is_sparse_kv_cached_by_fusion(forward_batch, layer.layer_id):
            raise RuntimeError(
                "The fused norm+rope+cache kernel wrote bf16 K/V rows into the "
                "int2 pool's packed buffers; the int2 pool must not take that path."
            )
        from sglang.kernels.ops.attention.minimax_sparse.decode.topk_sparse import (
            flash_decode_with_gqa_share_sparse,
        )

        pool = self.kv_pool
        bs = q.shape[0]
        n_q_heads, head_dim = layer.tp_q_head_num, layer.qk_head_dim
        n_kv_heads, v_head_dim = layer.tp_k_head_num, layer.v_head_dim
        q3, k3, v3, need_v_inverse = prepare_quantized_extend_qkv(
            kv_pool=pool,
            layer=layer,
            q=q.reshape(bs, n_q_heads, head_dim),
            k=k.reshape(bs, n_kv_heads, head_dim),
            v=v.reshape(bs, n_kv_heads, v_head_dim),
        )
        q3 = q3.to(self.model_dtype)

        loc = forward_batch.out_cache_loc
        if save_kv_cache:
            # is_decode=True: the unified pool routes a single-token write to
            # the HP-recent ring with no boolean masking (capture-safe).
            pool.set_kv_buffer(
                layer=layer,
                loc=loc,
                cache_k=k3,
                cache_v=v3,
                already_hadamard_transformed=True,
                is_decode=True,
            )
            self._int2_write_index(
                layer=layer, loc=loc, idx_k=idx_k, idx_v=idx_v, disable_value=disable_value
            )
        idx_k_cache, idx_v_cache = self._int2_index_caches(
            layer_id=layer.layer_id, disable_value=disable_value
        )
        idx_o, topk_idx = self._int2_decode_topk(
            layer=layer,
            idx_q=idx_q.reshape(bs, -1, self.idx_head_dim),
            idx_k_cache=idx_k_cache,
            idx_v_cache=idx_v_cache,
            forward_batch=forward_batch,
            disable_value=disable_value,
            num_kv_heads=n_kv_heads,
        )
        k_st, v_st, fake, slot_ids = self._int2_stage_decode_blocks(
            layer_id=layer.layer_id, topk_idx=topk_idx, forward_batch=forward_batch, bs=bs
        )
        o = flash_decode_with_gqa_share_sparse(
            q=q3.contiguous(),
            sink=None,
            k_cache=k_st,
            v_cache=v_st,
            req_to_token=fake,
            seq_lens=forward_batch.seq_lens,
            slot_ids=slot_ids,
            block_size=self.block_size_k,
            topk_idx=topk_idx,
            sm_scale=layer.scaling,
        )
        o = apply_inverse_v_rotation(
            result=o.view(bs, n_q_heads, v_head_dim),
            kv_pool=pool,
            layer=layer,
            need_v_inverse=need_v_inverse,
        ).to(q.dtype)
        return (
            None if idx_o is None else idx_o.reshape(bs, -1).contiguous(),
            o.reshape(bs, -1).contiguous(),
        )


class MiniMaxHybridAttnBackend(AttentionBackend):
    """Combines a dense backend and a sparse backend, routing by call site."""

    def __init__(
        self,
        dense_backend: AttentionBackend,
        sparse_backend: MiniMaxSparseAttnBackend,
        sparse_layer_ids: list[int],
    ):
        self.dense = dense_backend
        self.sparse = sparse_backend
        self.kv_index_translator = dense_backend.kv_index_translator
        self.sparse_layer_ids = sparse_layer_ids
        # Let the sparse decode reuse the dense paged backend (page table + workspace).
        self.sparse.dense_backend = dense_backend
        self.extend_dummy_seqs_capped_by_req_pool = getattr(
            dense_backend, "extend_dummy_seqs_capped_by_req_pool", False
        ) or getattr(sparse_backend, "extend_dummy_seqs_capped_by_req_pool", False)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # delegate so the dense (FlashInfer) backend keeps its own eager init.
        self.sparse.init_forward_metadata(forward_batch)
        self.dense.init_forward_metadata(forward_batch)

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        self.sparse.init_forward_metadata_out_graph(forward_batch, in_capture)
        self.dense.init_forward_metadata_out_graph(forward_batch, in_capture)

    def shared_read_ends(self, fm: ForwardMode) -> SharedReadEnds:
        return SharedReadEnds.max_of(
            b.shared_read_ends(fm) for b in (self.sparse, self.dense)
        )

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        self.sparse.init_forward_metadata_in_graph(forward_batch)
        self.dense.init_forward_metadata_in_graph(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.dense.init_cuda_graph_state(max_bs, max_num_tokens)
        self.sparse.init_cuda_graph_state(max_bs, max_num_tokens)

    def get_cuda_graph_seq_len_fill_value(self):
        return self.sparse.get_cuda_graph_seq_len_fill_value()

    def get_verify_buffers_to_fill_after_draft(self):
        # EAGLE3 verify buffer interface: the dense (ascend) backend owns the
        # tree-mask/position buffers consumed by the verify forward. The base
        # AttentionBackend raises NotImplementedError, so delegate to dense.
        return self.dense.get_verify_buffers_to_fill_after_draft()

    def update_verify_buffers_to_fill_after_draft(self, spec_info, cuda_graph_bs=None):
        return self.dense.update_verify_buffers_to_fill_after_draft(
            spec_info, cuda_graph_bs
        )

    def forward(
        self,
        q,
        k,
        v,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if layer.layer_id in self.sparse_layer_ids:
            return self.sparse.forward(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )

        # DP attention pads q to an even length but flashinfer builds qo_indptr from
        # extend_seq_lens, so padded q.shape[0] != qo_indptr[-1] and paged-prefill
        # raises. Trim q and re-pad output; k/v stay untrimmed so KV-cache writes
        # align with out_cache_loc.
        mode = forward_batch.forward_mode
        if mode.is_extend() and forward_batch.extend_seq_lens_cpu is not None:
            actual_num_tokens = int(sum(forward_batch.extend_seq_lens_cpu))
            original_num_tokens = q.shape[0]
            if actual_num_tokens < original_num_tokens:
                o = self.dense.forward(
                    q[:actual_num_tokens],
                    k,
                    v,
                    layer,
                    forward_batch,
                    save_kv_cache,
                    **kwargs,
                )
                pad_len = original_num_tokens - actual_num_tokens
                return torch.cat([o, o.new_zeros(pad_len, *o.shape[1:])], dim=0)

        return self.dense.forward(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def forward_extend(
        self,
        q,
        k,
        v,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if layer.layer_id in self.sparse_layer_ids:
            return self.sparse.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        else:
            return self.dense.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )

    def forward_decode(
        self,
        q,
        k,
        v,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if layer.layer_id in self.sparse_layer_ids:
            return self.sparse.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        else:
            return self.dense.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
