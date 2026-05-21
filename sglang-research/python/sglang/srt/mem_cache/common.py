from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, EvictParams
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import support_triton
from sglang.srt.utils.common import ceil_align
from sglang.QuantKernel.gpu_flush_int2 import (
    gpu_flush_int2,
    gpu_flush_int2_apply,
    gpu_flush_int2_plan,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

# Needs 2 + 1 slots for mamba request with prefix cache. 2 for ping pong cache, 1 for running mamba state.
MAMBA_STATE_PER_REQ_PREFIX_CACHE = 3
MAMBA_STATE_PER_REQ_NO_CACHE = 1

logger = logging.getLogger(__name__)


@triton.jit
def write_req_to_token_pool_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices,
    prefix_tensors,
    pre_lens,
    seq_lens,
    extend_lens,
    out_cache_loc,
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(0)

    req_pool_index = tl.load(req_pool_indices + pid)
    pre_len = tl.load(pre_lens + pid)
    seq_len = tl.load(seq_lens + pid)
    prefix_tensor = tl.load(prefix_tensors + pid).to(tl.pointer_type(tl.int64))

    # write prefix
    num_loop = tl.cdiv(pre_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < pre_len
        value = tl.load(prefix_tensor + offset, mask=mask)
        tl.store(
            req_to_token_ptr + req_pool_index * req_to_token_ptr_stride + offset,
            value,
            mask=mask,
        )

    # NOTE: This can be slow for large bs
    cumsum_start = tl.cast(0, tl.int64)
    for i in range(pid):
        cumsum_start += tl.load(extend_lens + i)

    num_loop = tl.cdiv(seq_len - pre_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < (seq_len - pre_len)
        value = tl.load(out_cache_loc + cumsum_start + offset, mask=mask)
        tl.store(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + offset
            + pre_len,
            value,
            mask=mask,
        )


def write_cache_indices(
    out_cache_loc: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    req_pool_indices_cpu: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens_tensor: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    extend_lens_tensor: torch.Tensor,
    extend_lens_cpu: torch.Tensor,
    prefix_tensors: list[torch.Tensor],
    req_to_token_pool: ReqToTokenPool,
):
    if support_triton(get_global_server_args().attention_backend):
        prefix_pointers = torch.tensor(
            [t.data_ptr() for t in prefix_tensors],
            device=req_to_token_pool.device,
            dtype=torch.uint64,
        )
        # TODO: some tensors can be reused for ForwardBatchInfo (e.g., extend_lens, cumsum_start)
        write_req_to_token_pool_triton[(req_pool_indices_tensor.shape[0],)](
            req_to_token_pool.req_to_token,
            req_pool_indices_tensor,
            prefix_pointers,
            prefix_lens_tensor,
            seq_lens_tensor,
            extend_lens_tensor,
            out_cache_loc,
            req_to_token_pool.req_to_token.shape[1],
        )
    else:
        pt = 0
        for i in range(req_pool_indices_cpu.shape[0]):
            req_idx = req_pool_indices_cpu[i].item()
            prefix_len = prefix_lens_cpu[i].item()
            seq_len = seq_lens_cpu[i].item()
            extend_len = extend_lens_cpu[i].item()

            req_to_token_pool.write(
                (req_idx, slice(0, prefix_len)),
                prefix_tensors[i],
            )
            req_to_token_pool.write(
                (req_idx, slice(prefix_len, seq_len)),
                out_cache_loc[pt : pt + extend_len],
            )
            pt += extend_len


def get_last_loc(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    if (
        get_global_server_args().attention_backend != "ascend"
        and get_global_server_args().attention_backend != "torch_native"
    ):
        impl = get_last_loc_triton
    else:
        impl = get_last_loc_torch

    return impl(req_to_token, req_pool_indices_tensor, prefix_lens_tensor)


def get_last_loc_torch(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    return torch.where(
        prefix_lens_tensor > 0,
        req_to_token[req_pool_indices_tensor, prefix_lens_tensor - 1],
        torch.full_like(prefix_lens_tensor, -1),
    )


@triton.jit
def get_last_loc_kernel(
    req_to_token,
    req_pool_indices_tensor,
    prefix_lens_tensor,
    result,
    num_tokens,
    req_to_token_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    mask = offset < num_tokens

    prefix_lens = tl.load(prefix_lens_tensor + offset, mask=mask, other=0)
    req_pool_indices = tl.load(req_pool_indices_tensor + offset, mask=mask, other=0)

    token_mask = prefix_lens > 0
    token_index = req_pool_indices * req_to_token_stride + (prefix_lens - 1)
    tokens = tl.load(req_to_token + token_index, mask=token_mask, other=-1)

    tl.store(result + offset, tokens, mask=mask)


def get_last_loc_triton(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    BLOCK_SIZE = 256
    num_tokens = prefix_lens_tensor.shape[0]
    result = torch.empty_like(prefix_lens_tensor)
    grid = (triton.cdiv(num_tokens, BLOCK_SIZE),)

    get_last_loc_kernel[grid](
        req_to_token,
        req_pool_indices_tensor,
        prefix_lens_tensor,
        result,
        num_tokens,
        req_to_token.stride(0),
        BLOCK_SIZE,
    )
    return result


def alloc_token_slots(
    tree_cache: BasePrefixCache,
    num_tokens: int,
    backup_state: bool = False,
):
    allocator = tree_cache.token_to_kv_pool_allocator
    evict_from_tree_cache(tree_cache, num_tokens)

    state = None
    if backup_state:
        state = allocator.backup_state()

    out_cache_loc = allocator.alloc(num_tokens)

    if out_cache_loc is None:
        error_msg = (
            f"Out of memory. Try to lower your batch size.\n"
            f"Try to allocate {num_tokens} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(error_msg)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_msg)

    return (out_cache_loc, state) if backup_state else out_cache_loc


def evict_from_tree_cache(tree_cache: BasePrefixCache | None, num_tokens: int):
    if tree_cache is None:
        return

    if tree_cache.is_chunk_cache():
        return

    allocator = tree_cache.token_to_kv_pool_allocator

    if isinstance(allocator, SWATokenToKVPoolAllocator):
        # Hybrid allocator
        full_available_size = allocator.full_available_size()
        swa_available_size = allocator.swa_available_size()

        if full_available_size < num_tokens or swa_available_size < num_tokens:
            full_num_tokens = max(0, num_tokens - full_available_size)
            swa_num_tokens = max(0, num_tokens - swa_available_size)
            tree_cache.evict(
                EvictParams(num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
            )
    else:
        # Standard allocator
        if allocator.available_size() < num_tokens:
            tree_cache.evict(EvictParams(num_tokens=num_tokens))


def alloc_paged_token_slots_extend(
    tree_cache: BasePrefixCache,
    prefix_lens: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    extend_num_tokens: int,
    backup_state: bool = False,
):
    # Over estimate the number of tokens: assume each request needs a new page.
    allocator = tree_cache.token_to_kv_pool_allocator
    num_tokens = extend_num_tokens + len(seq_lens_cpu) * allocator.page_size
    evict_from_tree_cache(tree_cache, num_tokens)

    state = None
    if backup_state:
        state = allocator.backup_state()

    out_cache_loc = allocator.alloc_extend(
        prefix_lens,
        prefix_lens_cpu,
        seq_lens,
        seq_lens_cpu,
        last_loc,
        extend_num_tokens,
    )

    if out_cache_loc is None:
        error_msg = (
            f"Prefill out of memory. Try to lower your batch size.\n"
            f"Try to allocate {extend_num_tokens} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(error_msg)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_msg)

    return (out_cache_loc, state) if backup_state else out_cache_loc


def alloc_req_slots(
    req_to_token_pool: ReqToTokenPool,
    reqs: list[Req],
    tree_cache: BasePrefixCache | None,
) -> list[int]:
    """Allocate request slots from the pool."""
    num_reqs = len(reqs)
    if isinstance(req_to_token_pool, HybridReqToTokenPool):
        mamba_available_size = req_to_token_pool.mamba_pool.available_size()
        factor = (
            MAMBA_STATE_PER_REQ_PREFIX_CACHE
            if tree_cache.supports_mamba()
            else MAMBA_STATE_PER_REQ_NO_CACHE
        )
        mamba_state_needed = num_reqs * factor
        if mamba_available_size < mamba_state_needed:
            if tree_cache is not None and tree_cache.supports_mamba():
                mamba_num = max(0, mamba_state_needed - mamba_available_size)
                tree_cache.evict(EvictParams(num_tokens=0, mamba_num=mamba_num))
    req_pool_indices = req_to_token_pool.alloc(reqs)

    if req_pool_indices is None:
        raise RuntimeError(
            "alloc_req_slots runs out of memory. "
            "Please set a smaller number for `--max-running-requests`. "
            f"{req_to_token_pool.available_size()=}, "
            f"{num_reqs=}, "
        )
    return req_pool_indices


def _is_mixed_kv_enabled(batch: ScheduleBatch) -> bool:
    allocator = batch.token_to_kv_pool_allocator
    kvcache = allocator.get_kvcache()
    return getattr(kvcache, "mixed_kv_enabled", None) is not None and kvcache.mixed_kv_enabled()


def _mixed_window_lengths(
    seq_len: int, hp_prefix_tokens: int, hp_recent_tokens: int
) -> tuple[int, int, int]:
    prefix_len = min(seq_len, hp_prefix_tokens)
    recent_len = min(max(seq_len - prefix_len, 0), hp_recent_tokens)
    quant_len = seq_len - prefix_len - recent_len
    return prefix_len, recent_len, quant_len


def _mixed_extend_layout_counts(
    pre_len: int,
    seq_len: int,
    hp_prefix_tokens: int,
    hp_recent_tokens: int,
    n_q: int,
) -> tuple[int, int, int, int, int]:
    """Return per-request mixed-KV extend counts.

    The logical layout is ``[HP-prefix][quant-middle][HP-recent]``. Quant
    middle may end with a partially occupied atomic quant page; we allocate the
    whole page but write only the live quant slots to ``req_to_token``.
    """
    prefix_keep, recent_keep, _ = _mixed_window_lengths(
        seq_len, hp_prefix_tokens, hp_recent_tokens
    )
    recent_start = seq_len - recent_keep
    hp_prefix_count = max(0, min(prefix_keep, seq_len) - pre_len)
    quant_count = max(0, recent_start - max(pre_len, prefix_keep))
    hp_recent_count = max(0, seq_len - max(pre_len, recent_start))
    quant_alloc_count = ceil_align(quant_count, n_q)

    # Per-request flush counter: count steps until this request's next flush.
    # ``H_0`` is the current HP-recent size after this extend chunk is admitted.
    # Quant tails remain quant, so they no longer increase ``H_0``.
    h0_total = (
        hp_recent_tokens
        if seq_len >= hp_prefix_tokens + hp_recent_tokens
        else max(0, seq_len - hp_prefix_tokens)
    )
    counter_init = max(0, (hp_recent_tokens + n_q - 1) - h0_total)
    return (
        hp_prefix_count,
        hp_recent_count,
        quant_count,
        quant_alloc_count,
        counter_init,
    )


def _alloc_for_extend_mixed(
    batch: ScheduleBatch,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    kv_pool = batch.token_to_kv_pool_allocator.get_kvcache()
    prefix_lens_cpu = torch.tensor(batch.prefix_lens, dtype=torch.int64)
    extend_lens_cpu = torch.tensor(batch.extend_lens, dtype=torch.int64)
    prefix_lens_device = prefix_lens_cpu.to(batch.device, non_blocking=True)
    extend_lens_device = extend_lens_cpu.to(batch.device, non_blocking=True)

    req_pool_indices = alloc_req_slots(
        batch.req_to_token_pool, batch.reqs, batch.tree_cache
    )
    req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
    req_pool_indices_device = req_pool_indices_cpu.to(batch.device, non_blocking=True)

    n_q = int(kv_pool.N_Q)
    hp_recent_target = int(kv_pool.hp_recent_tokens)

    hp_prefix_counts: list[int] = []
    hp_recent_counts: list[int] = []
    quant_counts: list[int] = []
    quant_alloc_counts: list[int] = []
    flush_counter_inits: list[int] = []
    for pre_len, seq_len in zip(batch.prefix_lens, batch.seq_lens_cpu.tolist()):
        (
            hp_prefix_count,
            hp_recent_count,
            quant_count,
            quant_alloc_count,
            counter_init,
        ) = _mixed_extend_layout_counts(
            int(pre_len),
            int(seq_len),
            int(kv_pool.hp_prefix_tokens),
            hp_recent_target,
            n_q,
        )

        hp_prefix_counts.append(hp_prefix_count)
        hp_recent_counts.append(hp_recent_count)
        quant_counts.append(quant_count)
        quant_alloc_counts.append(quant_alloc_count)
        flush_counter_inits.append(counter_init)

    total_quant_alloc = sum(quant_alloc_counts)
    # HP-prefix is N_Q-paged in the shared pool; round each per-req count up.
    hp_prefix_alloc_counts = [
        ceil_align(int(c), n_q) for c in hp_prefix_counts
    ]
    total_hp_prefix_alloc = sum(hp_prefix_alloc_counts)

    allocator = batch.token_to_kv_pool_allocator
    pooled_need = total_quant_alloc + total_hp_prefix_alloc
    if pooled_need > 0:
        evict_from_tree_cache(batch.tree_cache, pooled_need)

    quant_alloc = (
        allocator.alloc_quant(total_quant_alloc)
        if total_quant_alloc > 0
        else torch.empty((0,), dtype=torch.int64, device=batch.device)
    )
    if total_quant_alloc > 0 and quant_alloc is None:
        raise RuntimeError(
            "Mixed KV windows failed to allocate quant slots after eviction. "
            f"{allocator.debug_print()}"
        )
    hp_prefix_alloc = allocator.alloc_hp_prefix(
        req_pool_indices_device, hp_prefix_alloc_counts
    )
    hp_recent_alloc = allocator.alloc_hp_recent(
        req_pool_indices_device, hp_recent_counts
    )

    per_req_locs = []
    hp_prefix_pt = 0
    hp_recent_pt = 0
    quant_pt = 0
    for (
        req,
        pre_len,
        hp_prefix_count,
        hp_prefix_alloc_count,
        hp_recent_count,
        quant_count,
        quant_alloc_count,
    ) in zip(
        batch.reqs,
        batch.prefix_lens,
        hp_prefix_counts,
        hp_prefix_alloc_counts,
        hp_recent_counts,
        quant_counts,
        quant_alloc_counts,
    ):
        req_parts = []
        req_hp_prefix = None
        req_hp_recent = None
        if hp_prefix_count > 0:
            req_hp_prefix = hp_prefix_alloc[
                hp_prefix_pt : hp_prefix_pt + hp_prefix_count
            ]
        if hp_prefix_alloc_count > hp_prefix_count:
            # HP-prefix slack: trailing slots of a partially-filled page;
            # request-owned until release, freed by tier-routing in `free`.
            req_hp_prefix_slack = hp_prefix_alloc[
                hp_prefix_pt + hp_prefix_count : hp_prefix_pt + hp_prefix_alloc_count
            ]
            req.mixed_kv_quant_slack_indices = torch.cat(
                [
                    req.mixed_kv_quant_slack_indices.to(batch.device),
                    req_hp_prefix_slack,
                ]
            )
            slack_page_start = int(pre_len) + (hp_prefix_count // n_q) * n_q
            existing_cutoff = getattr(req, "mixed_kv_quant_slack_cutoff_len", None)
            req.mixed_kv_quant_slack_cutoff_len = (
                slack_page_start
                if existing_cutoff is None
                else min(existing_cutoff, slack_page_start)
            )
        hp_prefix_pt += hp_prefix_alloc_count

        if hp_recent_count > 0:
            req_hp_recent = hp_recent_alloc[
                hp_recent_pt : hp_recent_pt + hp_recent_count
            ]
            hp_recent_pt += hp_recent_count
        if quant_count > 0:
            req_quant = quant_alloc[quant_pt : quant_pt + quant_count]
        else:
            req_quant = None
        if quant_alloc_count > quant_count:
            req_quant_slack = quant_alloc[
                quant_pt + quant_count : quant_pt + quant_alloc_count
            ]
            req.mixed_kv_quant_slack_indices = torch.cat(
                [req.mixed_kv_quant_slack_indices.to(batch.device), req_quant_slack]
            )
            quant_start = max(int(pre_len), int(kv_pool.hp_prefix_tokens))
            slack_page_start = quant_start + (quant_count // n_q) * n_q
            existing_cutoff = getattr(req, "mixed_kv_quant_slack_cutoff_len", None)
            req.mixed_kv_quant_slack_cutoff_len = (
                slack_page_start
                if existing_cutoff is None
                else min(existing_cutoff, slack_page_start)
            )
        quant_pt += quant_alloc_count

        # Reconstruct logical order: [hp-prefix][quant-middle][hp-recent].
        if req_hp_prefix is not None:
            req_parts.append(req_hp_prefix)
        if req_quant is not None:
            req_parts.append(req_quant)
        if req_hp_recent is not None:
            req_parts.append(req_hp_recent)
        valid_parts = [part for part in req_parts if part is not None]
        per_req_locs.append(
            torch.cat(valid_parts)
            if valid_parts
            else torch.empty((0,), dtype=torch.int64, device=batch.device)
        )

    out_cache_loc = (
        torch.cat(per_req_locs)
        if per_req_locs
        else torch.empty((0,), dtype=torch.int64, device=batch.device)
    )
    prefix_tensors = [r.prefix_indices for r in batch.reqs]
    write_cache_indices(
        out_cache_loc,
        req_pool_indices_device,
        req_pool_indices_cpu,
        prefix_lens_device,
        prefix_lens_cpu,
        batch.seq_lens,
        batch.seq_lens_cpu,
        extend_lens_device,
        extend_lens_cpu,
        prefix_tensors,
        batch.req_to_token_pool,
    )

    # Seed each request's per-request flush counter. This is overwritten on
    # every chunk of a chunked extend (the latest H_0 is what matters), and
    # on the first chunk for a fresh admission. We do a single async H2D
    # copy + scatter so the decode hot path sees the counter without a
    # later sync.
    counter_inits_cpu = torch.tensor(flush_counter_inits, dtype=torch.int32)
    counter_inits_device = counter_inits_cpu.to(batch.device, non_blocking=True)
    kv_pool._flush_counter[req_pool_indices_device] = counter_inits_device

    return out_cache_loc, req_pool_indices_device, req_pool_indices

def _alloc_for_decode_mixed(batch: ScheduleBatch, token_per_req: int) -> torch.Tensor:
    # One HP-recent slot per req per step; over-provision bs*N_Q quant slots
    # per step and let the per-req flush counter decide which use them as
    # demote targets vs return them via ``returned_slot_ids``.
    if token_per_req != 1:
        raise NotImplementedError(
            "Mixed KV decode currently supports token_per_req=1 only."
        )

    allocator = batch.token_to_kv_pool_allocator
    kv_pool = allocator.get_kvcache()
    bs = batch.seq_lens.shape[0]
    # ``flush_interval`` is hardcoded to ``N_Q`` in ``UnifiedInt2HPKVPool``.
    flush_interval = int(kv_pool.N_Q)
    assert kv_pool.flush_interval == flush_interval

    req_pool_indices_int64 = batch.req_pool_indices.to(torch.int64)

    # Per-request flush gating: shape-static RMW, no host sync.
    counters = kv_pool._flush_counter[req_pool_indices_int64]
    flush_mask = counters == 0
    new_counters = torch.where(
        flush_mask,
        torch.full_like(counters, flush_interval - 1),
        counters - 1,
    )
    kv_pool._flush_counter[req_pool_indices_int64] = new_counters

    # Worst case: every req flushes -> bs*N_Q quant slots needed.
    quant_need = bs * flush_interval
    # ``evict_from_tree_cache`` gates on ``allocator.available_size()`` which
    # for the unified pool sums quant + HP-prefix free slots. When quant is
    # drained but HP-prefix has slack, the combined check skips eviction and
    # ``alloc_quant`` below crashes. Force quant-tier-specific eviction here.
    quant_pages_have = (
        allocator.free_pages.numel() + allocator.release_pages.numel()
    )
    if (
        quant_pages_have < bs
        and batch.tree_cache is not None
        and not batch.tree_cache.is_chunk_cache()
    ):
        # Tree leaves may be quant or HP-prefix; some leaves are big. Loop a
        # few times in case the first leaves popped are HP-prefix, but cap
        # work so we don't spin if everything left is pinned.
        for attempt in range(8):
            prev_quant = quant_pages_have
            prev_hp = (
                allocator.hp_prefix_free_pages.numel()
                + allocator.hp_prefix_release_pages.numel()
            )
            # Ramp up the budget each attempt: 1x, 2x, 4x ... up to 16x.
            mult = 1 << min(attempt, 4)
            evict_slots = max(bs - quant_pages_have, 1) * flush_interval * mult
            batch.tree_cache.evict(EvictParams(num_tokens=evict_slots))
            quant_pages_have = (
                allocator.free_pages.numel() + allocator.release_pages.numel()
            )
            if quant_pages_have >= bs:
                break
            cur_hp = (
                allocator.hp_prefix_free_pages.numel()
                + allocator.hp_prefix_release_pages.numel()
            )
            if quant_pages_have == prev_quant and cur_hp == prev_hp:
                # Tree had nothing to evict — leaves all pinned. Stop.
                break

    out_cache_loc = allocator.alloc_hp_recent(
        req_pool_indices_int64, [token_per_req] * bs
    )

    dst_quant_slots = allocator.alloc_quant(quant_need)
    if dst_quant_slots is None:
        raise RuntimeError(
            "Mixed KV windows failed to allocate quant flush slots. "
            f"{allocator.debug_print()}"
        )

    # Build the protected boundary on device in one go.  This is the
    # tree-owned prefix that flush must not overwrite; ``prefix_indices`` may
    # additionally contain request-owned tail slots for chunk continuation.
    prefix_lens_cpu = torch.tensor(
        [int(r.cache_protected_len) for r in batch.reqs], dtype=torch.int32
    )
    prefix_lens_gpu = prefix_lens_cpu.to(batch.device, non_blocking=True)
    seq_lens_int32 = batch.seq_lens.to(torch.int32)

    # Phase 1 (no-race with previous forward): plan kernel reads
    # ``req_to_token`` and produces ``returned_slot_ids`` etc. Followed by
    # ``allocator.free``, whose ``torch.unique`` host-syncs only against this
    # short pre-wait prefix instead of the previous forward. See
    # plan-for-a-fix-starry-russell.md.
    plan = gpu_flush_int2_plan(
        seq_lens=seq_lens_int32,
        prefix_lens=prefix_lens_gpu,
        req_pool_indices=req_pool_indices_int64,
        dst_quant_slots=dst_quant_slots,
        req_to_token=batch.req_to_token_pool.req_to_token,
        flush_mask=flush_mask,
        hp_prefix_tokens=kv_pool.hp_prefix_tokens,
        hp_recent_tokens=kv_pool.hp_recent_tokens,
        hp_global_offset=kv_pool.hp_global_offset,
        flush_interval=flush_interval,
    )

    if plan is not None:
        # Free everything returned by the kernel in one call: flushed HP
        # slots (freed from HP tier) and unused quant slots from
        # non-flushing requests (whole pages, since per-request
        # all-or-nothing). The allocator decodes tier from each global slot
        # id.
        allocator.free(plan.returned_slot_ids)

    # Phase 2 (must wait): the apply kernels write ``req_to_token`` at
    # positions inside the previous forward's read range. Order
    # schedule_stream after ``forward_done`` here, not at the top of the
    # event loop, so the host syncs above and any retract/eviction frees
    # don't stall behind the previous forward.
    wait_pending_forward = getattr(kv_pool, "wait_pending_forward", None)
    if wait_pending_forward is not None:
        wait_pending_forward()

    if plan is not None:
        gpu_flush_int2_apply(
            plan,
            req_pool_indices=req_pool_indices_int64,
            req_to_token=batch.req_to_token_pool.req_to_token,
            hp_k_ptrs=kv_pool._flush_hp_k_ptrs,
            hp_v_ptrs=kv_pool._flush_hp_v_ptrs,
            quant_k_ptrs=kv_pool._flush_quant_k_ptrs,
            quant_v_ptrs=kv_pool._flush_quant_v_ptrs,
            k_sz_ptrs=kv_pool._flush_k_sz_ptrs,
            v_sz_ptrs=kv_pool._flush_v_sz_ptrs,
            hp_k_sample=kv_pool.hp_k_buffer[0],
            hp_v_sample=kv_pool.hp_v_buffer[0],
            quant_k_sample=kv_pool.k_buffer[0],
            quant_v_sample=kv_pool.v_buffer[0],
            k_sz_sample=kv_pool.k_scales_zeros[0],
            v_sz_sample=kv_pool.v_scales_zeros[0],
            hp_k_strides=kv_pool._flush_hp_k_stride,
            hp_v_strides=kv_pool._flush_hp_v_stride,
            quant_k_strides=kv_pool._flush_quant_k_stride,
            quant_v_strides=kv_pool._flush_quant_v_stride,
            k_sz_strides=kv_pool._flush_k_sz_stride,
            v_sz_strides=kv_pool._flush_v_sz_stride,
            num_heads=kv_pool.head_num,
            head_dim=kv_pool.head_dim,
            v_head_dim=kv_pool.v_head_dim,
            k_num_scale_groups=kv_pool.k_num_scale_groups,
            v_num_scale_groups=kv_pool.v_num_scale_groups,
            num_layers=kv_pool.layer_num,
            k_clip_ratio=kv_pool._k_clip_ratio,
            v_clip_ratio=kv_pool._v_clip_ratio,
        )

    if batch.model_config.is_encoder_decoder:
        locs = batch.encoder_lens + batch.seq_lens
    else:
        locs = batch.seq_lens.clone()

    batch.req_to_token_pool.write(
        (batch.req_pool_indices, locs), out_cache_loc.to(torch.int32)
    )
    return out_cache_loc


def alloc_for_extend(
    batch: ScheduleBatch,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """
    Allocate KV cache for extend batch and write to req_to_token_pool.

    Returns:
        out_cache_loc: allocated cache locations
        req_pool_indices_device: request pool indices at a device tensor
        req_pool_indices: request pool indices as list
    """
    # free out-of-window swa tokens
    batch.maybe_evict_swa()

    if _is_mixed_kv_enabled(batch):
        return _alloc_for_extend_mixed(batch)

    prefix_tensors = [r.prefix_indices for r in batch.reqs]

    # Create tensors for allocation
    prefix_lens_cpu = torch.tensor(batch.prefix_lens, dtype=torch.int64)
    extend_lens_cpu = torch.tensor(batch.extend_lens, dtype=torch.int64)
    prefix_lens_device = prefix_lens_cpu.to(batch.device, non_blocking=True)
    extend_lens_device = extend_lens_cpu.to(batch.device, non_blocking=True)

    # Allocate req slots
    req_pool_indices = alloc_req_slots(
        batch.req_to_token_pool, batch.reqs, batch.tree_cache
    )
    req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
    req_pool_indices_device = req_pool_indices_cpu.to(batch.device, non_blocking=True)

    # Allocate KV cache (throws exception on failure)
    if batch.tree_cache.page_size == 1:
        out_cache_loc = alloc_token_slots(batch.tree_cache, batch.extend_num_tokens)
    else:
        # Paged allocation - build last_loc
        last_loc = [
            (t[-1:] if len(t) > 0 else torch.tensor([-1], device=batch.device))
            for t in prefix_tensors
        ]
        out_cache_loc = alloc_paged_token_slots_extend(
            tree_cache=batch.tree_cache,
            prefix_lens=prefix_lens_device,
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=batch.seq_lens,
            seq_lens_cpu=batch.seq_lens_cpu,
            last_loc=torch.cat(last_loc),
            extend_num_tokens=batch.extend_num_tokens,
        )

    # Write to req_to_token_pool
    write_cache_indices(
        out_cache_loc,
        req_pool_indices_device,
        req_pool_indices_cpu,
        prefix_lens_device,
        prefix_lens_cpu,
        batch.seq_lens,
        batch.seq_lens_cpu,
        extend_lens_device,
        extend_lens_cpu,
        prefix_tensors,
        batch.req_to_token_pool,
    )

    return out_cache_loc, req_pool_indices_device, req_pool_indices


def alloc_paged_token_slots_decode(
    tree_cache: BasePrefixCache,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    token_per_req: int = 1,
) -> torch.Tensor:
    """Allocate paged KV cache for decode batch."""
    allocator = tree_cache.token_to_kv_pool_allocator
    # Over estimate the number of tokens: assume each request needs a new page.
    num_tokens = len(seq_lens) * allocator.page_size
    evict_from_tree_cache(tree_cache, num_tokens)

    out_cache_loc = allocator.alloc_decode(seq_lens, seq_lens_cpu, last_loc)

    if out_cache_loc is None:
        error_msg = (
            f"Decode out of memory. Try to lower your batch size.\n"
            f"Try to allocate {len(seq_lens) * token_per_req} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(error_msg)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_msg)

    return out_cache_loc


def alloc_for_decode(batch: ScheduleBatch, token_per_req: int) -> torch.Tensor:
    """
    Allocate KV cache for decode batch and write to req_to_token_pool.

    Returns:
        out_cache_loc: allocated cache locations
    """

    batch.maybe_evict_swa()

    if _is_mixed_kv_enabled(batch):
        return _alloc_for_decode_mixed(batch, token_per_req)

    bs = batch.seq_lens.shape[0]

    if batch.tree_cache.page_size == 1:
        # Non-paged allocation
        out_cache_loc = alloc_token_slots(batch.tree_cache, bs * token_per_req)
    else:
        # Paged allocation
        last_loc = batch.req_to_token_pool.req_to_token[
            batch.req_pool_indices, batch.seq_lens - 1
        ]
        seq_lens_next = batch.seq_lens + token_per_req
        out_cache_loc = alloc_paged_token_slots_decode(
            tree_cache=batch.tree_cache,
            seq_lens=seq_lens_next,
            seq_lens_cpu=batch.seq_lens_cpu + token_per_req,
            last_loc=last_loc,
            token_per_req=token_per_req,
        )

    # Write to req_to_token_pool
    if batch.model_config.is_encoder_decoder:
        locs = batch.encoder_lens + batch.seq_lens
    else:
        locs = batch.seq_lens.clone()

    batch.req_to_token_pool.write(
        (batch.req_pool_indices, locs), out_cache_loc.to(torch.int32)
    )

    return out_cache_loc


def release_kv_cache(req: Req, tree_cache: BasePrefixCache, is_insert: bool = True):
    # MambaRadixCache may alloc mamba state before alloc KV cache
    if req.req_pool_idx is None:
        assert (
            tree_cache.supports_mamba()
        ), "Only MambaRadixCache allow freeing before alloc"
        # TODO (csy, hanming): clean up this early allocation logic
        if req.mamba_pool_idx is not None:
            tree_cache.req_to_token_pool.mamba_pool.free(
                req.mamba_pool_idx.unsqueeze(-1)
            )
            req.mamba_pool_idx = None
        return

    tree_cache.cache_finished_req(req, is_insert=is_insert)

    # FIXME: SessionAwareCache.cache_finished_req sets req_pool_idx = None to
    # transfer KV ownership to the SessionSlot, so we skip the remaining
    # cleanup (overalloc free + pool slot free). This means over-allocated
    # tokens from speculative decoding are NOT freed between turns.
    if req.req_pool_idx is None:
        return

    start_p, end_p = req.pop_overallocated_kv_cache()

    global_server_args = get_global_server_args()
    page_size = global_server_args.page_size
    spec_algo = global_server_args.speculative_algorithm

    if spec_algo is None:
        assert (
            start_p == end_p
        ), f"Unexpected overallocated KV cache, {req.kv_committed_len=}, {req.kv_allocated_len=}"

    if page_size > 1:
        start_p = ceil_align(start_p, page_size)

    if start_p < end_p:
        indices_to_free = tree_cache.req_to_token_pool.req_to_token[req.req_pool_idx][
            start_p:end_p
        ]
        tree_cache.token_to_kv_pool_allocator.free(indices_to_free)
    # If the prefix cache doesn't manage mamba states, we must free them here.
    if isinstance(tree_cache.req_to_token_pool, HybridReqToTokenPool) and (
        not tree_cache.supports_mamba()
    ):
        assert (
            req.mamba_pool_idx is not None
        ), "mamba state is freed while the tree cache does not manage mamba states"
        tree_cache.req_to_token_pool.free_mamba_cache(req)

    # Mixed-KV: reset the per-req HP-recent cursor before the req_pool_idx
    # is recycled, so the next request starts fresh.
    kvcache = tree_cache.token_to_kv_pool_allocator.get_kvcache()
    release_slab = getattr(kvcache, "release_req_slab", None)
    if release_slab is not None:
        release_slab(req.req_pool_idx)

    tree_cache.req_to_token_pool.free(req)


def available_and_evictable_str(tree_cache: BasePrefixCache) -> str:
    return tree_cache.available_and_evictable_str()
