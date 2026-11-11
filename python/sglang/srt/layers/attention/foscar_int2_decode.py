"""FOSCAR decode over the INT2 tier of the unified mixed-KV pool (``SGLANG_OSCAR_FOSCAR``).

One decode step: the HP tiers (prefix sink + recent ring) keep the Triton stage-1 and write
splits ``[0, hp_max_kv_splits)`` of the mixed scratch; FOSCAR decodes the request's INT2 pages
and its normalized output and natural-log LSE become split ``hp_max_kv_splits``; the unified
stage-2 then merges them (both sides use ``lse = max + log(sum exp(sm_scale * q.k - max))``).

Eager steps plan once per step from the token-level INT2 indices, which hold whole 64-token
pages per request in logical order, so ``indices[::64] // 64`` are the page ids and a request
with no INT2 tokens is a zero-length row (FOSCAR returns ``lse = -inf`` for it). CUDA-graph
steps use one graph-mode wrapper per captured batch size; their lengths must be at least 1, so
an empty row reads the tier's dummy page with length 1 and its LSE is masked to ``-inf``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import triton
import triton.language as tl

from sglang.srt.mem_cache.foscar_int2_tier import FOSCAR_PAGE_TOKENS, FoscarInt2Tier

# fp8 e4m3 largest finite value; SM100 decodes an fp8 query.
_E4M3_MAX = 448.0


@triton.jit
def _fill_page_table_kernel(
    token_indptr,
    token_indices,
    page_table,
    seq_lens,
    valid,
    max_pages,
    dummy_page,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per request row: page ids of its INT2 tokens into ``page_table[row]``, its
    token count into ``seq_lens`` (1 and the dummy page when it has none)."""
    row = tl.program_id(0)
    start = tl.load(token_indptr + row)
    n = tl.load(token_indptr + row + 1) - start
    n_pages = n // PAGE
    # A padded graph row carries a single slot-0 index; only whole pages are an INT2 tier.
    tl.store(seq_lens + row, tl.maximum(n_pages * PAGE, 1).to(tl.int32))
    tl.store(valid + row, (n_pages > 0).to(tl.int32))
    for off in range(0, max_pages, BLOCK):
        j = off + tl.arange(0, BLOCK)
        in_row = j < n_pages
        slot = tl.load(token_indices + start + j * PAGE, mask=in_row, other=0)
        page = tl.where(in_row, slot // PAGE, dummy_page)
        tl.store(page_table + row * max_pages + j, page.to(tl.int32), mask=j < max_pages)


class _GraphBucket:
    def __init__(self, wrapper, page_table: torch.Tensor, seq_lens: torch.Tensor, valid: torch.Tensor):
        self.wrapper = wrapper
        self.page_table = page_table
        self.seq_lens = seq_lens
        self.valid = valid


class FoscarInt2Decode:
    """Plans and runs FOSCAR's paged decode over a ``FoscarInt2Tier`` for one decode step."""

    def __init__(
        self,
        tier: FoscarInt2Tier,
        *,
        num_qo_heads: int,
        sm_scale: float,
        max_context_len: int,
        device: torch.device,
        eager_workspace_bytes: int = 256 << 20,
    ):
        from foscar import PagedDecodeWrapper, get_capabilities

        self.tier = tier
        self.num_qo_heads = int(num_qo_heads)
        self.sm_scale = float(sm_scale)
        self.max_context_len = int(max_context_len)
        self.max_pages = (self.max_context_len + FOSCAR_PAGE_TOKENS - 1) // FOSCAR_PAGE_TOKENS
        self.device = torch.device(device)
        self.fp8_query = get_capabilities()["arch"] == "sm100"
        self._eager = PagedDecodeWrapper(
            torch.empty(eager_workspace_bytes, dtype=torch.uint8, device=self.device)
        )
        self._buckets: Dict[int, _GraphBucket] = {}
        self._graph_workspace: Optional[torch.Tensor] = None
        self._active: Optional[_GraphBucket] = None
        self._eager_empty = False

    # -- eager ----------------------------------------------------------------------------

    def plan_eager(
        self,
        token_indptr: torch.Tensor,
        token_indices: torch.Tensor,
        num_tokens: int,
        max_tokens: int,
        batch_size: int,
    ) -> None:
        """Plan a step from token-level INT2 indices. ``token_indptr`` int [B + 1] (multiples of
        64), ``token_indices`` holds ``num_tokens`` quant slot ids; ``max_tokens`` is the
        longest row (host value, so the plan does not read the device)."""
        self._active = None
        # No INT2 token anywhere in the batch: run() returns -inf LSE without a launch.
        self._eager_empty = int(num_tokens) == 0
        if self._eager_empty:
            return
        indptr = token_indptr[: batch_size + 1]
        page_indptr = torch.div(indptr, FOSCAR_PAGE_TOKENS, rounding_mode="floor").to(torch.int32)
        page_indices = torch.div(
            token_indices[:num_tokens:FOSCAR_PAGE_TOKENS], FOSCAR_PAGE_TOKENS, rounding_mode="floor"
        ).to(torch.int32)
        seq_lens = (indptr[1:] - indptr[:-1]).to(torch.int32)
        self._eager.plan(
            page_indptr,
            page_indices,
            seq_lens,
            num_qo_heads=self.num_qo_heads,
            layout=self.tier.layout,
            max_kv_len=max(int(max_tokens), 1),
            sm_scale=self.sm_scale,
        )

    # -- CUDA graph -----------------------------------------------------------------------

    def _workspace_for_graphs(self, max_batch: int) -> torch.Tensor:
        from foscar import graph_workspace_bytes

        need = graph_workspace_bytes(
            max_batch, self.num_qo_heads, self.tier.layout, max_kv_len=self.max_context_len, device=self.device
        )
        if self._graph_workspace is None:
            self._graph_workspace = torch.empty(need, dtype=torch.uint8, device=self.device)
        elif self._graph_workspace.numel() < need:
            raise RuntimeError(
                f"FOSCAR graph workspace was reserved for {self._graph_workspace.numel()} bytes, a bucket "
                f"needs {need}; reserve_graph_workspace(max_batch) must run before the first capture"
            )
        return self._graph_workspace

    def reserve_graph_workspace(self, max_batch: int) -> None:
        """Size the workspace every graph bucket shares once, before the first bucket: a
        bucket's plan records the workspace address for the wrapper's life."""
        self._workspace_for_graphs(int(max_batch))

    def prepare_graph_buckets(self, batch_sizes: List[int]) -> None:
        """Build one graph-mode wrapper per captured batch size, outside any capture (plan()
        compiles and allocates). Buckets share one workspace: no two graphs run at once."""
        from foscar import PagedDecodeWrapper

        if self._graph_workspace is None:
            self._workspace_for_graphs(max(batch_sizes))
        ws = self._graph_workspace
        for bs in sorted(set(int(b) for b in batch_sizes)):
            if bs in self._buckets:
                continue
            page_table = torch.full(
                (bs, self.max_pages), self.tier.dummy_page, dtype=torch.int32, device=self.device
            )
            seq_lens = torch.ones(bs, dtype=torch.int32, device=self.device)
            valid = torch.zeros(bs, dtype=torch.int32, device=self.device)
            wrapper = PagedDecodeWrapper(ws, use_cuda_graph=True)
            wrapper.plan(
                page_table=page_table,
                seq_lens=seq_lens,
                num_qo_heads=self.num_qo_heads,
                layout=self.tier.layout,
                max_kv_len=self.max_context_len,
                sm_scale=self.sm_scale,
                lse=True,
                q_descale=self.fp8_query,
            )
            self._buckets[bs] = _GraphBucket(wrapper, page_table, seq_lens, valid)

    def bind_graph_bucket(self, batch_size: int) -> None:
        """Select the bucket a capture records (its buffers are what the graph reads)."""
        self._active = self._buckets[int(batch_size)]

    def _fill(self, bucket: _GraphBucket, batch_size: int, token_indptr: torch.Tensor, token_indices: torch.Tensor):
        _fill_page_table_kernel[(batch_size,)](
            token_indptr,
            token_indices,
            bucket.page_table,
            bucket.seq_lens,
            bucket.valid,
            self.max_pages,
            self.tier.dummy_page,
            PAGE=FOSCAR_PAGE_TOKENS,
            BLOCK=256,
        )

    def capture_graph_bucket(self, batch_size: int, token_indptr: torch.Tensor, token_indices: torch.Tensor) -> None:
        """Before a capture of ``batch_size``: plan its bucket (outside the capture), fill its
        page table and bind it. FOSCAR's work list is not updated here: ``update()`` must always
        run on the stream the graph replays on, and the plan's initial work list (one chunk of
        length 1 per row) already serves a capture."""
        self.prepare_graph_buckets([batch_size])
        bucket = self._buckets[int(batch_size)]
        self._fill(bucket, batch_size, token_indptr, token_indices)
        self._active = bucket

    def update_graph_bucket(
        self,
        batch_size: int,
        token_indptr: torch.Tensor,
        token_indices: torch.Tensor,
        lengths_host: List[int],
    ) -> None:
        """Before a replay, outside the graph and on the replay stream: refresh the bucket's
        page table, lengths and valid mask from the step's token-level INT2 indices, and
        FOSCAR's work list from the host lengths (padded rows included, each at least 1)."""
        bucket = self._buckets[int(batch_size)]
        self._fill(bucket, batch_size, token_indptr, token_indices)
        bucket.wrapper.update([max(1, int(n)) for n in lengths_host])
        self._active = bucket

    # -- run --------------------------------------------------------------------------------

    def run(
        self, layer: int, q: torch.Tensor, out: torch.Tensor, lse: torch.Tensor, sm_scale: Optional[float] = None
    ) -> None:
        """Decode one layer: ``q`` bf16 [B, Hq, D] in the OSCAR frame, ``out`` bf16 [B, Hq, D],
        ``lse`` fp32 [B, Hq] (natural log; -inf for rows without INT2 tokens)."""
        if self._active is None and self._eager_empty:
            lse.fill_(float("-inf"))
            out.zero_()
            return
        wrapper = self._eager if self._active is None else self._active.wrapper
        cache = self.tier.caches[layer]
        scale = self.sm_scale if sm_scale is None else float(sm_scale)
        if self.fp8_query:
            q_descale = q.float().abs().amax(dim=-1).clamp_min(1e-12) / _E4M3_MAX
            q8 = (q.float() / q_descale.unsqueeze(-1)).to(torch.float8_e4m3fn)
            wrapper.run(q8, cache, out=out, lse=lse, q_descale=q_descale, sm_scale=scale)
        else:
            wrapper.run(q.contiguous(), cache, out=out, lse=lse, sm_scale=scale)
        if self._active is not None:
            lse.masked_fill_(self._active.valid[: lse.shape[0]].unsqueeze(-1) == 0, float("-inf"))


def decode_attention_fwd_int2_foscar(
    q: torch.Tensor,
    hp_k_buffer: torch.Tensor,
    hp_v_buffer: torch.Tensor,
    o: torch.Tensor,
    hp_kv_indptr: torch.Tensor,
    hp_kv_indices: torch.Tensor,
    attn_logits: torch.Tensor,
    attn_lse: torch.Tensor,
    hp_num_kv_splits: torch.Tensor,
    hp_max_kv_splits: int,
    sm_scale: float,
    *,
    foscar: FoscarInt2Decode,
    layer_index: int,
) -> torch.Tensor:
    """HP + INT2 decode with FOSCAR on the INT2 tier.

    ``attn_logits`` / ``attn_lse`` are the unified mixed scratch
    ``[bs, heads, hp_max + quant_max, Lv]`` / ``[bs, heads, hp_max + quant_max]``: the HP
    stage-1 writes splits ``[0, hp_max)``, FOSCAR's normalized output and LSE go to split
    ``hp_max``, and the unified stage-2 reduces splits ``[0, hp_max]``."""
    from sglang.srt.layers.attention.triton_ops.decode_attention import (
        _decode_att_m_fwd,
        _decode_grouped_att_m_fwd,
        _unified_stage2,
    )

    # Splits a short request does not use keep -inf so the stage-2 skips them.
    attn_lse.fill_(float("-inf"))
    if hp_kv_indices.numel() > 0:
        stage1 = _decode_att_m_fwd if q.shape[1] == hp_k_buffer.shape[1] else _decode_grouped_att_m_fwd
        stage1(
            q,
            hp_k_buffer,
            hp_v_buffer,
            attn_logits[:, :, :hp_max_kv_splits, :],
            attn_lse[:, :, :hp_max_kv_splits],
            hp_kv_indptr,
            hp_kv_indices,
            hp_num_kv_splits,
            hp_max_kv_splits,
            sm_scale,
            0.0,
        )
    bs, heads, dim = q.shape
    out = torch.empty((bs, heads, dim), dtype=torch.bfloat16, device=q.device)
    lse = torch.empty((bs, heads), dtype=torch.float32, device=q.device)
    foscar.run(layer_index, q.to(torch.bfloat16), out, lse, sm_scale=sm_scale)
    attn_logits[:, :, hp_max_kv_splits, :].copy_(out)
    attn_lse[:, :, hp_max_kv_splits].copy_(lse)
    _unified_stage2(attn_logits, attn_lse, o, total_splits=hp_max_kv_splits + 1)
    return o
