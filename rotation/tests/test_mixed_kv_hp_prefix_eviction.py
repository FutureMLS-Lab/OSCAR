"""HP-prefix pool pressure must evict the radix tree, not kill the scheduler.

With the radix cache on, every cached prefix pins its BF16 sink pages in the
shared HP-prefix pool. ``evict_from_tree_cache`` checks the pooled total
(quant + HP-prefix), which never runs short while the quant tier is large, so
before ``_evict_until_tier_slots_free`` a server that had seen more distinct
prompts than the pool holds sinks (~208 at the default sizing) died in
``alloc_hp_prefix`` with "HP-prefix pool exhausted" -- all three MATH500
(500 distinct prompts) runs crashed at ~463 on 2026-10-12.

Runs on CPU inside the serving image::

    docker run --rm -v "$PWD:/sgl-workspace/sglang" --entrypoint python3 \\
        oscar-env:v120 -m pytest -q rotation/tests/test_mixed_kv_hp_prefix_eviction.py
"""

from __future__ import annotations

import pytest
import torch

from sglang.srt.mem_cache.allocation import _evict_until_tier_slots_free
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.unified_kv_allocator import UnifiedInt2HPKVAllocator

N_Q = 8
HP_PREFIX = 64
HP_RECENT = 64
RING = HP_RECENT + N_Q - 1


def _make_allocator(hp_slots: int = 10 * N_Q) -> UnifiedInt2HPKVAllocator:
    return UnifiedInt2HPKVAllocator(
        num_quant_pages=64,
        quant_tokens_per_page=N_Q,
        hp_prefix_tokens=HP_PREFIX,
        hp_recent_tokens=HP_RECENT,
        hp_recent_ring_size=RING,
        max_req_slots=2,
        num_hp_prefix_slots=hp_slots + N_Q,  # + the reserved sink page
        dtype="int2",
        hp_dtype=torch.bfloat16,
        device="cpu",
        kvcache=None,
        need_sort=False,
    )


class _TreeStub:
    """The two calls the eviction loop makes, over branches that free real
    allocator slots. Each branch holds a quant tail and an HP-prefix page, and
    like leaf-first LRU eviction it hands back the quant tail cheaply while
    the HP-prefix page only comes back when the whole branch is evicted."""

    def __init__(self, allocator, branches):
        self.allocator = allocator
        self.branches = list(branches)
        self.evict_calls = 0

    def is_chunk_cache(self) -> bool:
        return False

    def evict(self, params: EvictParams):
        self.evict_calls += 1
        freed = 0
        while self.branches and freed < params.num_tokens:
            slots = self.branches.pop(0)
            self.allocator.free(slots)
            freed += slots.numel()
        return type("R", (), {"num_tokens_evicted": freed})()


def _drain_hp_pool(alloc):
    """Hand out every HP-prefix page, returning the slot batches (one page each)."""
    batches = []
    while alloc._hp_prefix_free_slots() > 0:
        batches.append(alloc.alloc_hp_prefix(torch.tensor([0]), [N_Q]))
    return batches


def test_hp_pressure_evicts_tree_and_alloc_succeeds():
    alloc = _make_allocator()
    pages = _drain_hp_pool(alloc)
    assert alloc._hp_prefix_free_slots() == 0
    # Each cached branch = one HP-prefix page + a 3-page quant tail.
    branches = []
    for hp in pages:
        quant = alloc.alloc_quant(3 * N_Q)
        branches.append(torch.cat([hp, quant]))
    tree = _TreeStub(alloc, branches)

    need = 2 * N_Q
    _evict_until_tier_slots_free(
        tree,
        free_slots=lambda: (
            alloc.hp_prefix_free_pages.numel() + alloc.hp_prefix_release_pages.numel()
        )
        * N_Q,
        need_slots=need,
    )
    assert alloc._hp_prefix_free_slots() >= need
    got = alloc.alloc_hp_prefix(torch.tensor([0, 1]), [N_Q, N_Q])
    assert got.numel() == need
    # LRU semantics: only as many branches evicted as the shortfall required.
    assert tree.branches, "the loop evicted the whole tree for a 2-page need"


def test_exhausted_and_unevictable_still_raises():
    alloc = _make_allocator()
    _drain_hp_pool(alloc)
    tree = _TreeStub(alloc, branches=[])  # everything locked / nothing cached
    _evict_until_tier_slots_free(
        tree,
        free_slots=lambda: alloc.hp_prefix_free_pages.numel() * N_Q,
        need_slots=N_Q,
    )
    assert tree.evict_calls == 1  # one probe, then the zero-evicted early return
    with pytest.raises(RuntimeError, match="HP-prefix pool exhausted"):
        alloc.alloc_hp_prefix(torch.tensor([0]), [N_Q])


def test_no_need_or_chunk_cache_is_a_no_op():
    alloc = _make_allocator()
    tree = _TreeStub(alloc, branches=[])
    _evict_until_tier_slots_free(tree, free_slots=lambda: 0, need_slots=0)
    assert tree.evict_calls == 0
    chunk = _TreeStub(alloc, branches=[])
    chunk.is_chunk_cache = lambda: True
    _evict_until_tier_slots_free(chunk, free_slots=lambda: 0, need_slots=N_Q)
    assert chunk.evict_calls == 0
