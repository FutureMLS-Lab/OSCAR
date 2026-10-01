"""MiniMax-M3 sparse-attention KV pool on the OSCAR per-head INT2 tiers.

``MiniMaxSparseKVPool`` (memory_pool.py) keeps a dense main pool for K/V and
one slot-indexed side pool per sparse layer for the lightning indexer's keys.
This pool is the same thing on 2-bit storage: K/V of every layer -- the dense
layers and the sparse layers alike -- live in ``UnifiedInt2HPKVPool`` (packed
INT2 codes plus the BF16 prefix / recent windows, one flat slot id space), and
the index-key cache is carried beside it with one row per slot of that space,
window slots included. Slot ids come straight out of ``req_to_token``, so a
prefix-cache hit, a window promotion or a flush needs no bookkeeping here.

The index cache is NOT quantized: the indexer reads it through the ordinary
sparse kernels over the real page table, and it is not what attention reads.
Its dtype follows ``get_minimax_sparse_index_dtype`` (the model dtype unless an
fp8 index cache is requested), exactly as for the dense pool, so the pricing in
``pool_configurator`` stays one formula. With 2-bit K/V this cache is the larger
per-token cost (at 128 dims it is 256 B/token/layer in bf16), which is why the
configurator reserves it for the window arena's slots too.

Writes keep the unified pool's contract: the attention backend rotates K/V and
calls ``set_kv_buffer(..., already_hadamard_transformed=True, is_decode=...)``,
then stores the index key with ``set_index_k_buffer``. The fused
``set_fused_kv_index_buffer`` of the dense pool cannot be offered here because
it carries no ``is_decode`` and would route decode writes down the extend path.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import get_tensor_size_bytes, unwrap_write_loc
from sglang.srt.mem_cache.unified_kv_pool import UnifiedInt2HPKVPool

logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz)


def index_cache_slots(pool: UnifiedInt2HPKVPool) -> int:
    # The slot id space is the quant tier [0, hp_global_offset) followed by
    # every row of the BF16 window arena.
    return int(pool.hp_global_offset) + int(
        pool.get_hp_key_buffer(pool.start_layer).shape[0]
    )


class MiniMaxInt2SparseKVPool(UnifiedInt2HPKVPool):
    """``UnifiedInt2HPKVPool`` plus the lightning indexer's key (and optional
    value) cache for MiniMax sparse layers. Keyword arguments other than the
    ones named below are the unified pool's geometry."""

    def __init__(
        self,
        *,
        idx_head_dim: int,
        dense_layer_ids: List[int],
        sparse_layer_ids: List[int],
        disable_value_sparse_layer_ids: Optional[List[int]] = None,
        index_dtype: Optional[torch.dtype] = None,
        **unified_kwargs,
    ):
        # The parent's allocation log calls get_kv_size_bytes before the index
        # caches exist; they start empty.
        self.index_k_buffer: Dict[int, torch.Tensor] = {}
        self.index_v_buffer: Dict[int, torch.Tensor] = {}
        super().__init__(**unified_kwargs)

        self.idx_head_dim = int(idx_head_dim)
        self.index_dtype = (
            index_dtype if index_dtype is not None else self.model_dtype
        )
        # fp8 cannot be index_put directly (the dense pools store it as uint8
        # for the same reason), so storage and logical dtype may differ.
        self.index_store_dtype = (
            torch.uint8 if self.index_dtype in _FP8_DTYPES else self.index_dtype
        )

        local_dense = [l for l in dense_layer_ids if self.start_layer <= l < self.end_layer]
        local_sparse = [l for l in sparse_layer_ids if self.start_layer <= l < self.end_layer]
        disable_set = set(disable_value_sparse_layer_ids or [])
        local_kv_sparse = [l for l in local_sparse if l not in disable_set]
        local_k_only_sparse = [l for l in local_sparse if l in disable_set]

        # Same mapping surface as MiniMaxSparseKVPool.
        self._dense_layer_ids = set(local_dense)
        self.sparse_layer_id_mapping: Dict[int, int] = {
            gid: i for i, gid in enumerate(local_sparse)
        }
        self.index_kv_layer_id_mapping: Dict[int, int] = {
            gid: i for i, gid in enumerate(local_kv_sparse)
        }
        self.index_k_layer_id_mapping: Dict[int, int] = {
            gid: i for i, gid in enumerate(local_k_only_sparse)
        }
        self.dense_pool = None

        n_slots = self.index_cache_slots
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                for lid in local_sparse:
                    self.index_k_buffer[lid] = torch.zeros(
                        (n_slots, 1, self.idx_head_dim),
                        dtype=self.index_store_dtype,
                        device=self.device,
                    )
                for lid in local_kv_sparse:
                    self.index_v_buffer[lid] = torch.zeros(
                        (n_slots, 1, self.idx_head_dim),
                        dtype=self.index_store_dtype,
                        device=self.device,
                    )

        index_bytes = self.get_index_cache_size_bytes()
        self.mem_usage += index_bytes / GB
        logger.info(
            "MiniMaxInt2SparseKVPool: index cache %d K layers + %d V layers x %d "
            "slots (quant %d + window %d) x %d dims (%s) = %.2f GB; dense layers "
            "%d, sparse layers %d",
            len(self.index_k_buffer),
            len(self.index_v_buffer),
            n_slots,
            self.quant_size,
            self.hp_size,
            self.idx_head_dim,
            str(self.index_dtype),
            index_bytes / GB,
            len(local_dense),
            len(local_sparse),
        )

    # -- Surface shared with MiniMaxSparseKVPool ------------------------------

    @property
    def main_pool(self):
        # The dense pool reads main_pool.dtype / head_num off a sub-pool; here
        # K/V of every layer live in this object.
        return self

    @property
    def index_cache_slots(self) -> int:
        return index_cache_slots(self)

    def get_index_cache_size_bytes(self) -> int:
        return sum(get_tensor_size_bytes(t) for t in self.index_k_buffer.values()) + sum(
            get_tensor_size_bytes(t) for t in self.index_v_buffer.values()
        )

    def get_kv_size_bytes(self):
        k, v = super().get_kv_size_bytes()
        # Index keys ride on the K side, index values on the V side.
        k += sum(get_tensor_size_bytes(t) for t in self.index_k_buffer.values())
        v += sum(get_tensor_size_bytes(t) for t in self.index_v_buffer.values())
        return k, v

    def _view_index(self, buf: torch.Tensor) -> torch.Tensor:
        if self.index_store_dtype != self.index_dtype:
            return buf.view(self.index_dtype)
        return buf

    def get_index_k_buffer(self, layer_id: int) -> torch.Tensor:
        buf = self.index_k_buffer.get(layer_id)
        if buf is None:
            raise ValueError(
                f"layer_id={layer_id} is not a sparse attention layer; "
                f"sparse layers: {list(self.sparse_layer_id_mapping.keys())}"
            )
        return self._view_index(buf)

    def get_index_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        buf_v = self.index_v_buffer.get(layer_id)
        if buf_v is None:
            raise ValueError(
                f"layer_id={layer_id} does not have an index V cache "
                f"(either dense, or in the K-only group). "
                f"index_kv layers: {list(self.index_kv_layer_id_mapping.keys())}"
            )
        return self.get_index_k_buffer(layer_id), self._view_index(buf_v)

    def _store_index(
        self,
        *,
        buf: torch.Tensor,
        loc: torch.Tensor,
        values: torch.Tensor,
        scale: Optional[float],
    ) -> None:
        values = values.reshape(-1, 1, self.idx_head_dim)
        if values.dtype != self.index_dtype:
            # None means unit scale; a scale applies before the cast only
            # (MiniMaxSparseKVPool.set_index_k_buffer semantics).
            if scale is not None:
                values = values / scale
            values = values.to(self.index_dtype)
        if self.index_store_dtype != self.index_dtype:
            values = values.view(self.index_store_dtype)
        buf[loc.to(torch.int64)] = values

    def set_index_k_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_idx_k: torch.Tensor,
        k_scale: Optional[float] = None,
    ) -> None:
        loc, _, _ = unwrap_write_loc(loc)
        if loc.numel() == 0:
            return
        buf = self.index_k_buffer.get(layer.layer_id)
        if buf is None:
            raise ValueError(
                f"layer.layer_id={layer.layer_id} is not a sparse attention "
                f"layer; sparse layers: {list(self.sparse_layer_id_mapping.keys())}"
            )
        self._store_index(buf=buf, loc=loc, values=cache_idx_k, scale=k_scale)

    def set_index_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_idx_k: torch.Tensor,
        cache_idx_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
    ) -> None:
        loc, _, _ = unwrap_write_loc(loc)
        if loc.numel() == 0:
            return
        buf_v = self.index_v_buffer.get(layer.layer_id)
        if buf_v is None:
            raise ValueError(
                f"layer.layer_id={layer.layer_id} does not have an index V "
                f"cache (either dense, or in the K-only group). "
                f"index_kv layers: {list(self.index_kv_layer_id_mapping.keys())}"
            )
        self._store_index(
            buf=self.index_k_buffer[layer.layer_id], loc=loc, values=cache_idx_k, scale=k_scale
        )
        self._store_index(buf=buf_v, loc=loc, values=cache_idx_v, scale=v_scale)

    def set_fused_kv_index_buffer(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "MiniMaxInt2SparseKVPool has no fused K/V + index store: the int2 "
            "tiers need the rotated K/V through set_kv_buffer(..., "
            "already_hadamard_transformed=True, is_decode=<by forward mode>) "
            "followed by set_index_k_buffer / set_index_kv_buffer."
        )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        super().move_kv_cache(tgt_loc, src_loc)
        if tgt_loc.numel() == 0:
            return
        # The index caches span the whole slot space, so a move is tier-agnostic.
        tgt = tgt_loc.to(torch.int64)
        src = src_loc.to(torch.int64)
        for buf in self.index_k_buffer.values():
            buf[tgt] = buf[src]
        for buf in self.index_v_buffer.values():
            buf[tgt] = buf[src]
