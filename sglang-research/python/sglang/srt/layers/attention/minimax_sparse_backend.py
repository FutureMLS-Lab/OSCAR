"""MiniMax-M3 sparse attention (MSA) backend for the OSCAR fork.

MiniMax-M3 is a block-sparse GQA model: on every sparse layer a lightning
indexer (``index_q_proj`` / ``index_k_proj``, Gemma-normed, partial RoPE)
scores each 128-token key block against the query -- the max over the block's
tokens -- and the main attention is restricted to the top-k blocks plus the
query's own block. This fork used to skip the indexer and run every layer as
dense GQA; that is a different operator from the one the model was trained
with, so it is no longer the default (``SGLANG_MINIMAX_SPARSE_ATTENTION=0``
brings it back).

Kernels are upstream's Triton block-sparse ops, ported unchanged into
``minimax_sparse_kernels``. They read K/V as
``cache[req_to_token[slot_ids[b], pos]]``, which the per-head INT2 pool cannot
serve directly -- its rows are 2-bit codes plus a BF16 window arena. For that
pool the backend dequantizes the rows a forward attends into a BF16 staging
buffer and hands the kernels a *fake* ``req_to_token`` pointing into it:

  prefill  every query token has its own block set, so the union is every
           token of every request; dequantize them all once per layer in ragged
           order, overwrite this forward's own tokens with their exact BF16
           values (as the triton extend path does), and give the kernel a
           table ``fake[b, pos] = cu_seqlens_k[b] + pos``;
  decode   dequantize only the selected blocks (topk+local, 128 tokens each)
           and scatter their rows into a persistent table at their real
           positions -- static shapes, so the whole step captures into the
           CUDA graph.

Positions are preserved in both cases, so the kernels' causal and seq_len
masking is untouched. The index-key cache itself is BF16 and shared by both
arms; it is the dominant per-token cost once K/V are 2-bit (see
``attach_index_k_cache``).

OSCAR rotations are handled exactly as the triton INT2 paths do it: K/V are
stored rotated, the query is rotated with R_k before attention and the output
un-rotated with R_v after it.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Optional, Tuple

import torch

from sglang.srt.configs.model_config import (
    get_minimax_sparse_attention_config,
    get_minimax_sparse_disable_value_layer_ids,
    get_minimax_sparse_layer_ids,
    get_minimax_sparse_score_type,
    is_minimax_sparse,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.attention.minimax_sparse_staging import (
    decode_block_rows,
    fill_decode_fake_table,
    prefill_fake_table,
)
from sglang.srt.layers.attention.nsa.packed_staging import build_slot_to_ragged
from sglang.srt.layers.attention.quantized_kv_prefill import (
    apply_inverse_v_rotation,
    dequantize_prefix_kv,
    prepare_quantized_extend_qkv,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

logger = logging.getLogger(__name__)

# block_size_q for the prefill kernels, by KV length (upstream's thresholds).
_BSQ_THRESHOLD_64 = 4096
_BSQ_THRESHOLD_32 = 1024
_BSQ_THRESHOLD_16 = 512


def minimax_sparse_enabled(hf_config) -> bool:
    return is_minimax_sparse(hf_config) and envs.SGLANG_MINIMAX_SPARSE_ATTENTION.get()


def index_cache_slots(pool) -> int:
    """Number of slot ids the page table can hold for this pool.

    The unified INT2 pool lays out quant slots first and the BF16 window arena
    after them (``hp_global_offset`` onwards); a plain pool has size+page rows.
    """
    if hasattr(pool, "hp_global_offset"):
        return int(pool.hp_global_offset) + int(
            pool.get_hp_key_buffer(pool.start_layer).shape[0]
        )
    return int(pool.size) + int(pool.page_size)


def sparse_layers_in_range(hf_config, start_layer: int, end_layer: int):
    cfg = get_minimax_sparse_attention_config(hf_config)
    _, sparse_ids = get_minimax_sparse_layer_ids(cfg)
    return [l for l in sparse_ids if start_layer <= l < end_layer]


def attach_index_k_cache(model_runner) -> None:
    """Allocate the lightning indexer's key cache beside the KV pool.

    One ``[slots, 1, index_dim]`` tensor in the model dtype per sparse layer,
    indexed by the same slot ids as K/V so prefix-cache hits and the INT2
    pool's HP/quant tiering need nothing extra. BF16 rather than FP8: both
    arms share it, and the point of the comparison is the K/V bit width.

    It is not small. At 128 dims it is 256 B/token/layer, which on this
    model's 1 KV head per rank is 3.6x the INT2 K+V row (72 B): once K/V are
    2-bit the indexer cache dominates the per-token footprint. The pool
    configurator prices it so the pool is sized for it.
    """
    pool = model_runner.token_to_kv_pool
    cfg = get_minimax_sparse_attention_config(model_runner.model_config.hf_config)
    ids = sparse_layers_in_range(
        model_runner.model_config.hf_config, model_runner.start_layer, model_runner.end_layer
    )
    idx_dim = int(cfg["sparse_index_dim"])
    n_slots = index_cache_slots(pool)
    dtype = model_runner.dtype
    pool.msa_index_k = {
        l: torch.zeros((n_slots, 1, idx_dim), dtype=dtype, device=model_runner.device)
        for l in ids
    }
    nbytes = n_slots * idx_dim * torch.empty(0, dtype=dtype).element_size() * len(ids)
    logger.info(
        "[MiniMaxSparse] index-key cache: %d sparse layers x %d slots x %d dims (%s) = %.2f GB",
        len(ids), n_slots, idx_dim, str(dtype), nbytes / 2**30,
    )


class MiniMaxSparseAttnBackend(AttentionBackend):
    def __init__(self, model_runner):
        super().__init__()
        self.device = model_runner.device
        self.pool = model_runner.token_to_kv_pool
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.max_context_len = int(model_runner.model_config.context_len)
        self.model_dtype = model_runner.dtype
        hf = model_runner.model_config.hf_config
        cfg = get_minimax_sparse_attention_config(hf)
        self.dense_layer_ids, self.sparse_layer_ids = get_minimax_sparse_layer_ids(cfg)
        self.disable_value_layer_ids = set(get_minimax_sparse_disable_value_layer_ids(cfg))
        self.score_type = get_minimax_sparse_score_type(cfg)
        self.idx_head_dim = int(cfg["sparse_index_dim"])
        self.block_size_k = int(cfg["sparse_block_size"])
        self.topk_blocks = int(cfg["sparse_topk_blocks"])
        if "sparse_init_block" in cfg:
            self.init_blocks = int(cfg["sparse_init_block"])
        else:
            self.init_blocks = -(-int(cfg["sparse_init_tokens"]) // self.block_size_k)
        if "sparse_local_block" in cfg:
            self.local_blocks = int(cfg["sparse_local_block"])
        else:
            self.local_blocks = -(-int(cfg["sparse_local_tokens"]) // self.block_size_k) + 1
        with_value = [l for l in self.sparse_layer_ids if l not in self.disable_value_layer_ids]
        if with_value:
            raise NotImplementedError(
                "MiniMax sparse layers with an index-value path (index_v_proj / "
                f"index_o_proj) are not ported; layers {with_value[:8]}..."
            )
        if model_runner.server_args.speculative_algorithm is not None:
            raise NotImplementedError("MiniMax sparse attention: speculative decoding not supported")
        if not hasattr(self.pool, "msa_index_k"):
            raise RuntimeError(
                "MiniMax sparse attention needs the index-key cache; "
                "attach_index_k_cache() was not called when the KV pool was built"
            )
        self.int2 = getattr(self.pool, "dtype", None) == "int2"
        if self.int2:
            if not hasattr(self.pool, "mixed_kv_enabled"):
                raise NotImplementedError(
                    "MiniMax sparse attention on int2 KV is staged for the unified "
                    "mixed HP+int2 pool only (SGLANG_ENABLE_MIXED_KV_WINDOWS=1)"
                )
            n_kv = int(self.pool.head_num)
            if n_kv != 1:
                raise NotImplementedError(
                    "int2 staging keeps one selected-block set per request, which "
                    f"needs one KV head per rank; this rank holds {n_kv}. Raise TP."
                )
        self.n_sparse = len(self.sparse_layer_ids)
        tp = get_attention_tp_size()
        self._gqa_group_size = max(
            1,
            model_runner.model_config.num_attention_heads // tp
            // max(1, model_runner.model_config.get_num_kv_heads(tp)),
        )
        self._max_seqlen_q = 1
        self._max_seqlen_k = 1
        self._ext: Optional[SimpleNamespace] = None
        self._ar_block = torch.arange(self.block_size_k, device=self.device, dtype=torch.int64)
        self._slot_to_ragged: Optional[torch.Tensor] = None
        self._dec_fake: Optional[torch.Tensor] = None
        self._dec_rows: Optional[torch.Tensor] = None
        self._dec_slot_ids: Optional[torch.Tensor] = None
        logger.info(
            "[MiniMaxSparse] backend: %d sparse / %d dense layers, block %d, topk %d, "
            "init %d, local %d, score=%s, kv=%s%s",
            len(self.sparse_layer_ids), len(self.dense_layer_ids), self.block_size_k,
            self.topk_blocks, self.init_blocks, self.local_blocks, self.score_type,
            "int2 (BF16 staging)" if self.int2 else str(getattr(self.pool, "dtype", "?")),
            "" if self.int2 else " (direct)",
        )

    # ── metadata ────────────────────────────────────────────────────────────

    def _choose_block_size_q(self, max_seqlen_k: int) -> int:
        if max_seqlen_k >= _BSQ_THRESHOLD_64:
            bsq = 64
        elif max_seqlen_k >= _BSQ_THRESHOLD_32:
            bsq = 32
        elif max_seqlen_k >= _BSQ_THRESHOLD_16:
            bsq = 16
        else:
            bsq = 1
        # The main sparse kernel tiles gqa_group_size * block_size_q query rows
        # together and asserts that product <= 128. Upstream's thresholds assume
        # its TP=16 deployment (group 4); at TP=8 the group is 8, so a 1.3K-token
        # prompt picked 32 and died on the assert mid-prefill.
        return max(1, min(bsq, 128 // max(1, self._gqa_group_size)))

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self._ext = None
        mode = forward_batch.forward_mode
        if mode.is_decode_or_idle():
            self._max_seqlen_q = 1
            self._max_seqlen_k = (
                int(forward_batch.seq_lens_cpu.max().item())
                if forward_batch.batch_size > 0
                else 1
            )
            return
        if not mode.is_extend() or mode.is_target_verify() or mode.is_draft_extend(include_v2=True):
            raise NotImplementedError(f"MiniMax sparse attention: unsupported {mode=}")
        device = forward_batch.seq_lens.device
        extend_lens = forward_batch.extend_seq_lens.to(torch.int32)
        cu_seqlens = torch.cat(
            [torch.zeros(1, dtype=torch.int32, device=device), extend_lens.cumsum(0).to(torch.int32)]
        )
        seq_lens = forward_batch.seq_lens.to(torch.int32)
        if forward_batch.extend_prefix_lens is not None:
            prefix_lens = forward_batch.extend_prefix_lens.to(torch.int32)
        else:
            prefix_lens = torch.zeros_like(seq_lens)
        self._max_seqlen_q = int(max(forward_batch.extend_seq_lens_cpu))
        self._max_seqlen_k = int(forward_batch.seq_lens_cpu.max().item())
        meta = SimpleNamespace(
            cu_seqlens=cu_seqlens,
            seq_lens=seq_lens,
            prefix_lens=prefix_lens,
            block_size_q=self._choose_block_size_q(self._max_seqlen_k),
            flat=None,
            fake=None,
            slot_ids=None,
        )
        if self.int2:
            lens = forward_batch.seq_lens_cpu.tolist()
            bs = len(lens)
            page_table = self.req_to_token[forward_batch.req_pool_indices, : self._max_seqlen_k]
            flat = torch.cat([page_table[i, :l] for i, l in enumerate(lens)]).to(torch.int32)
            n_slots = index_cache_slots(self.pool)
            if self._slot_to_ragged is None or self._slot_to_ragged.numel() < n_slots:
                self._slot_to_ragged = torch.full((n_slots,), -1, dtype=torch.int32, device=device)
            build_slot_to_ragged(flat, self._slot_to_ragged)
            cu_k = torch.cat(
                [torch.zeros(1, dtype=torch.int32, device=device), seq_lens.cumsum(0).to(torch.int32)]
            )
            meta.flat = flat
            meta.fake = prefill_fake_table(cu_k, self._max_seqlen_k)
            meta.slot_ids = torch.arange(bs, dtype=torch.int32, device=device)
        self._ext = meta

    def _ensure_decode_buffers(self, bs: int) -> None:
        if self._dec_fake is not None and self._dec_fake.shape[0] >= bs:
            return
        sel = self.topk_blocks * self.block_size_k
        # One spare column past every legal position collects the dead entries.
        self._dec_fake = torch.zeros(
            (bs, self.max_context_len + 1), dtype=torch.int32, device=self.device
        )
        self._dec_rows = torch.arange(bs * sel, dtype=torch.int32, device=self.device).view(
            bs, self.topk_blocks, self.block_size_k
        )
        self._dec_slot_ids = torch.arange(bs, dtype=torch.int32, device=self.device)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        if self.int2:
            self._ensure_decode_buffers(max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(
        self, bs, num_tokens, req_pool_indices, seq_lens, encoder_lens, forward_mode, spec_info
    ):
        assert forward_mode.is_decode_or_idle(), "MiniMax sparse: graphs are decode-only"
        self._ext = None
        self._max_seqlen_q = 1
        # Capture sees dummy seq_lens; bound the scan by the full context so a
        # replay with longer sequences misses no block.
        self._max_seqlen_k = self.max_context_len

    def init_forward_metadata_replay_cuda_graph(
        self, bs, req_pool_indices, seq_lens, seq_lens_sum, encoder_lens, forward_mode,
        spec_info, seq_lens_cpu,
    ):
        self._ext = None
        self._max_seqlen_q = 1
        self._max_seqlen_k = self.max_context_len

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    # ── helpers ─────────────────────────────────────────────────────────────

    def _index_k(self, layer_id: int) -> torch.Tensor:
        return self.pool.msa_index_k[layer_id]

    def _reduce_topk(self, topk_idx: torch.Tensor, n_kv: int) -> torch.Tensor:
        # One selection per index head; with more index heads than KV heads on
        # this rank, take the union per KV head (upstream's topk_index_reduce).
        n_idx = topk_idx.shape[0]
        if n_idx == n_kv:
            return topk_idx
        from sglang.srt.layers.attention.minimax_sparse_kernels.common.index import (
            topk_index_reduce,
        )

        return topk_index_reduce(
            topk_idx.view(n_kv, n_idx // n_kv, -1, topk_idx.shape[-1]), dim=1
        )

    # ── prefill ─────────────────────────────────────────────────────────────

    def forward_extend(
        self, q, k, v, layer, forward_batch: ForwardBatch, save_kv_cache: bool = True,
        *, idx_q: torch.Tensor, idx_k: torch.Tensor, idx_v: Optional[torch.Tensor] = None,
    ):
        from sglang.srt.layers.attention.minimax_sparse_kernels.prefill.flash_with_topk_idx import (
            flash_prefill_with_topk_index,
        )
        from sglang.srt.layers.attention.minimax_sparse_kernels.prefill.topk_sparse import (
            flash_prefill_with_gqa_share_sparse,
        )

        meta = self._ext
        assert meta is not None, "init_forward_metadata did not see this extend batch"
        T = q.shape[0]
        Hq, D, Hkv, Dv = layer.tp_q_head_num, layer.qk_head_dim, layer.tp_k_head_num, layer.v_head_dim
        q3 = q.reshape(T, Hq, D)
        k3 = k.reshape(T, Hkv, D)
        v3 = v.reshape(T, Hkv, Dv)
        need_v_inverse = False
        if self.int2:
            # Rotate once, for the cache write and for the staged rows alike.
            q3, k3, v3, need_v_inverse = prepare_quantized_extend_qkv(self.pool, layer, q3, k3, v3)
            q3 = q3.to(self.model_dtype)
        if save_kv_cache:
            loc = forward_batch.out_cache_loc
            if self.int2:
                self.pool.set_kv_buffer(
                    layer, loc, k3, v3, already_hadamard_transformed=True, is_decode=False
                )
            else:
                self.pool.set_kv_buffer(layer, loc, k3, v3)
            self._index_k(layer.layer_id)[loc.to(torch.int64)] = idx_k.reshape(
                T, 1, self.idx_head_dim
            ).to(self.model_dtype)

        idx_k_cache = self._index_k(layer.layer_id)
        n_idx = idx_q.numel() // (T * self.idx_head_dim)
        _, topk_idx = flash_prefill_with_topk_index(
            q=idx_q.reshape(T, n_idx, self.idx_head_dim).to(self.model_dtype),
            k_cache=idx_k_cache,
            v_cache=None,
            sink=None,
            req_to_token=self.req_to_token,
            slot_ids=forward_batch.req_pool_indices,
            cu_seqlens=meta.cu_seqlens,
            seq_lens=meta.seq_lens,
            prefix_lens=meta.prefix_lens,
            max_seqlen_q=self._max_seqlen_q,
            max_seqlen_k=self._max_seqlen_k,
            block_size_q=meta.block_size_q,
            block_size_k=self.block_size_k,
            topk=self.topk_blocks,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            score_type=self.score_type,
            disable_index_value=True,
            page_size=1,
        )
        topk_idx = self._reduce_topk(topk_idx, Hkv)

        if self.int2:
            k_cache, v_cache = dequantize_prefix_kv(
                self.pool, layer.layer_id, meta.flat, self.model_dtype
            )
            # This forward's own tokens at full precision, in the stored frame.
            pos = self._slot_to_ragged[forward_batch.out_cache_loc.to(torch.int64)].to(torch.int64)
            k_cache[pos] = k3.to(self.model_dtype)
            v_cache[pos] = v3.to(self.model_dtype)
            rtt, slot_ids = meta.fake, meta.slot_ids
        else:
            k_cache, v_cache = self.pool.get_kv_buffer(layer.layer_id)
            rtt, slot_ids = self.req_to_token, forward_batch.req_pool_indices

        o = flash_prefill_with_gqa_share_sparse(
            q=q3.contiguous(),
            k_cache=k_cache,
            v_cache=v_cache,
            sink=None,
            req_to_token=rtt,
            slot_ids=slot_ids,
            topk_idx=topk_idx,
            block_size_q=meta.block_size_q,
            block_size_k=self.block_size_k,
            cu_seqlens=meta.cu_seqlens,
            seq_lens=meta.seq_lens,
            prefix_lens=meta.prefix_lens,
            max_seqlen_q=self._max_seqlen_q,
            sm_scale=layer.scaling,
        )
        if self.int2:
            o = apply_inverse_v_rotation(o.view(T, Hq, Dv), self.pool, layer, need_v_inverse)
        return None, o.reshape(T, Hq * Dv).to(q.dtype)

    # ── decode ──────────────────────────────────────────────────────────────

    def forward_decode(
        self, q, k, v, layer, forward_batch: ForwardBatch, save_kv_cache: bool = True,
        *, idx_q: torch.Tensor, idx_k: torch.Tensor, idx_v: Optional[torch.Tensor] = None,
    ):
        from sglang.srt.layers.attention.minimax_sparse_kernels.decode.flash_with_topk_idx import (
            flash_decode_with_topk_idx,
        )
        from sglang.srt.layers.attention.minimax_sparse_kernels.decode.topk_sparse import (
            flash_decode_with_gqa_share_sparse,
        )

        bs = q.shape[0]
        Hq, D, Hkv, Dv = layer.tp_q_head_num, layer.qk_head_dim, layer.tp_k_head_num, layer.v_head_dim
        q3 = q.reshape(bs, Hq, D)
        k3 = k.reshape(bs, Hkv, D)
        v3 = v.reshape(bs, Hkv, Dv)
        need_v_inverse = False
        if self.int2:
            q3, k3, v3, need_v_inverse = prepare_quantized_extend_qkv(self.pool, layer, q3, k3, v3)
            q3 = q3.to(self.model_dtype)
        if save_kv_cache:
            loc = forward_batch.out_cache_loc
            if self.int2:
                self.pool.set_kv_buffer(
                    layer, loc, k3, v3, already_hadamard_transformed=True, is_decode=True
                )
            else:
                self.pool.set_kv_buffer(layer, loc, k3, v3)
            self._index_k(layer.layer_id)[loc.to(torch.int64)] = idx_k.reshape(
                bs, 1, self.idx_head_dim
            ).to(self.model_dtype)

        n_idx = idx_q.numel() // (bs * self.idx_head_dim) if bs > 0 else 1
        _, topk_idx, _ = flash_decode_with_topk_idx(
            q=idx_q.reshape(bs, n_idx, self.idx_head_dim).to(self.model_dtype),
            sink=None,
            k_cache=self._index_k(layer.layer_id),
            v_cache=None,
            req_to_token=self.req_to_token,
            seq_lens=forward_batch.seq_lens,
            max_seqlen=self._max_seqlen_k,
            slot_ids=forward_batch.req_pool_indices,
            block_size=self.block_size_k,
            topk=self.topk_blocks,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            score_type=self.score_type,
            disable_index_value=True,
            use_dense_main_attn=False,
            page_size=1,
        )
        topk_idx = self._reduce_topk(topk_idx, Hkv)

        if self.int2:
            self._ensure_decode_buffers(bs)
            slots, pos, valid = decode_block_rows(
                topk_idx[0], self.req_to_token, forward_batch.req_pool_indices,
                forward_batch.seq_lens, self.block_size_k, self._ar_block,
            )
            k_cache, v_cache = dequantize_prefix_kv(
                self.pool, layer.layer_id, slots.reshape(-1), self.model_dtype
            )
            fake = self._dec_fake[:bs]
            fill_decode_fake_table(
                fake, pos, valid, self._dec_rows[:bs], dump_col=self.max_context_len
            )
            rtt, slot_ids = fake, self._dec_slot_ids[:bs]
        else:
            k_cache, v_cache = self.pool.get_kv_buffer(layer.layer_id)
            rtt, slot_ids = self.req_to_token, forward_batch.req_pool_indices

        o = flash_decode_with_gqa_share_sparse(
            q=q3.contiguous(),
            sink=None,
            k_cache=k_cache,
            v_cache=v_cache,
            req_to_token=rtt,
            seq_lens=forward_batch.seq_lens,
            slot_ids=slot_ids,
            block_size=self.block_size_k,
            topk_idx=topk_idx,
            sm_scale=layer.scaling,
        )
        if self.int2:
            o = apply_inverse_v_rotation(o.view(bs, Hq, Dv), self.pool, layer, need_v_inverse)
        return None, o.reshape(bs, Hq * Dv).to(q.dtype)


class MiniMaxHybridAttnBackend(AttentionBackend):
    """Routes sparse layers to the MSA backend and the rest to the dense one."""

    def __init__(self, dense_backend: AttentionBackend, sparse_backend: MiniMaxSparseAttnBackend,
                 sparse_layer_ids):
        super().__init__()
        self.dense = dense_backend
        self.sparse = sparse_backend
        self.sparse_layer_ids = set(sparse_layer_ids)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self.sparse.init_forward_metadata(forward_batch)
        self.dense.init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.dense.init_cuda_graph_state(max_bs, max_num_tokens)
        self.sparse.init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(self, *args, **kwargs):
        self.sparse.init_forward_metadata_capture_cuda_graph(*args, **kwargs)
        self.dense.init_forward_metadata_capture_cuda_graph(*args, **kwargs)

    def init_forward_metadata_replay_cuda_graph(self, *args, **kwargs):
        self.sparse.init_forward_metadata_replay_cuda_graph(*args, **kwargs)
        self.dense.init_forward_metadata_replay_cuda_graph(*args, **kwargs)

    def get_cuda_graph_seq_len_fill_value(self):
        return self.dense.get_cuda_graph_seq_len_fill_value()

    def __getattr__(self, name):
        # Anything the model runner asks of "the" backend that only the dense
        # one defines (e.g. data_type, kv_cache_dtype).
        dense = self.__dict__.get("dense")
        if dense is None:
            raise AttributeError(name)
        return getattr(dense, name)

    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        if layer.layer_id in self.sparse_layer_ids:
            return self.sparse.forward_extend(q, k, v, layer, forward_batch, save_kv_cache, **kwargs)
        return self.dense.forward_extend(q, k, v, layer, forward_batch, save_kv_cache, **kwargs)

    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        if layer.layer_id in self.sparse_layer_ids:
            return self.sparse.forward_decode(q, k, v, layer, forward_batch, save_kv_cache, **kwargs)
        return self.dense.forward_decode(q, k, v, layer, forward_batch, save_kv_cache, **kwargs)
