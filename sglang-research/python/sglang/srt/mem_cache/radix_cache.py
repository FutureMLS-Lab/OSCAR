from __future__ import annotations

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.utils import convert_to_bigram_key

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
The radix tree data structure for managing the KV cache.
"""

import heapq
import logging
import math
import sys
import time
from collections import defaultdict
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)

from sglang.srt.disaggregation.kv_events import (
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.evict_policy import (
    EvictionStrategy,
    FIFOStrategy,
    FILOStrategy,
    LFUStrategy,
    LRUStrategy,
    MRUStrategy,
    PriorityStrategy,
    SLRUStrategy,
)
from sglang.srt.mem_cache.utils import get_hash_str, hash_str_to_int64

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class RadixKey:
    def __init__(
        self,
        token_ids: List[int],
        extra_key: Optional[str] = None,
        is_bigram: bool = False,
    ):
        # token ids sequence
        self.token_ids = token_ids
        # extra key (e.g. lora_id, cache_salt)
        self.extra_key = extra_key
        # is bigram key
        self.is_bigram = is_bigram

    def __len__(self) -> int:
        return len(self.token_ids)

    def __iter__(self) -> Iterator[int]:
        return iter(self.token_ids)

    def __getitem__(self, idx: Union[int, slice]) -> "RadixKey":
        if isinstance(idx, slice):
            return RadixKey(self.token_ids[idx], self.extra_key)
        return RadixKey([self.token_ids[idx]], self.extra_key)

    def __repr__(self) -> str:
        preview = self.token_ids[:10]
        return f"RadixKey(extra_key={self.extra_key!r}, token_ids={preview}{'...' if len(self.token_ids) > 10 else ''})"


def maybe_bigram_convert(
    is_eagle: bool,
    key: RadixKey,
    value: Optional[torch.Tensor] = None,
) -> Tuple[RadixKey, Optional[torch.Tensor]]:
    if is_eagle and not key.is_bigram:
        key.token_ids = convert_to_bigram_key(key.token_ids)
        key.is_bigram = True
        if value is not None:
            value = value[: len(key)]
    return key, value


def page_align_keys(key: list, page_size) -> list:
    if page_size == 1:
        return key
    page_aligned_len = len(key) // page_size * page_size
    return key[:page_aligned_len]


class TreeNode:

    counter = 0

    def __init__(self, id: Optional[int] = None, priority: int = 0):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: RadixKey = None
        self.value: Optional[torch.Tensor] = None
        self.lock_ref = 0
        self.last_access_time = time.monotonic()
        self.creation_time = time.monotonic()

        self.hit_count = 0
        # indicating the node is locked to protect from eviction
        # incremented when the node is referenced by a storage operation
        self.host_ref_counter = 0
        # store the host indices of KV cache
        self.host_value: Optional[torch.Tensor] = None
        # store hash values of each pages
        self.hash_value: Optional[List[str]] = None
        # priority for priority-aware eviction
        self.priority = priority

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None

    def protect_host(self):
        """Protect the host value from eviction."""
        self.host_ref_counter += 1

    def release_host(self):
        """Release the host value, allowing it to be evicted."""
        if self.host_ref_counter > 0:
            self.host_ref_counter -= 1
        else:
            raise RuntimeError("Host reference counter is already zero.")

    def get_last_hash_value(self) -> Optional[str]:
        """Returns the hash value of the last page in this node."""
        if self.hash_value is None or len(self.hash_value) == 0:
            return None
        return self.hash_value[-1]

    @lru_cache(maxsize=1)
    def get_prefix_hash_values(self, node: TreeNode) -> List[str]:
        if node is None or node.hash_value is None:
            return []

        return node.get_prefix_hash_values(node.parent) + node.hash_value

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


def _check_extra_key(key0: RadixKey, key1: RadixKey):
    if key0.extra_key != key1.extra_key:
        raise ValueError(
            f"_key_match should be run on the same extra key, but got key0.extra_key={key0.extra_key} != key1.extra_key={key1.extra_key}"
        )


def _key_match_page_size1(key0: RadixKey, key1: RadixKey):
    _check_extra_key(key0, key1)
    i = 0
    for k0, k1 in zip(key0.token_ids, key1.token_ids):
        if k0 != k1:
            break
        i += 1
    return i


def _key_match_paged(key0: RadixKey, key1: RadixKey, page_size: int):
    _check_extra_key(key0, key1)
    min_len = min(len(key0), len(key1))

    i = 0
    while i < min_len:
        if key0.token_ids[i : i + page_size] != key1.token_ids[i : i + page_size]:
            break
        i += page_size

    return i


def get_child_key(key: RadixKey, page_size: int = 1):
    if page_size == 1:
        plain_key = key.token_ids[0]
    else:
        plain_key = tuple(key.token_ids[:page_size])
    if key.extra_key is None:
        return plain_key
    else:
        return (key.extra_key, plain_key)


def compute_node_hash_values(node: "TreeNode", page_size: int) -> List[str]:
    """Compute SHA256-based hash values for position-aware identification.

    Args:
        node: The TreeNode to compute hash values for
        page_size: The page size for chunking tokens

    Returns:
        List of SHA256 hex strings, one per page
    """
    hash_values = []

    # Get parent's last hash value if parent exists
    parent_hash = None
    if node.parent is not None and node.parent.hash_value is not None:
        # Check if parent is root by checking if it has empty key
        if len(node.parent.key) > 0 and len(node.parent.hash_value) > 0:
            parent_hash = node.parent.hash_value[-1]

    # Iterate through node's pages
    for start in range(0, len(node.key), page_size):
        page_tokens = node.key.token_ids[start : start + page_size]
        if not page_tokens:
            continue

        # Use SHA256-based chaining via get_hash_str
        hash_val = get_hash_str(page_tokens, prior_hash=parent_hash)
        hash_values.append(hash_val)
        parent_hash = hash_val

    return hash_values


def split_node_hash_value(
    child_hash_value: Optional[List[str]], split_len: int, page_size: int
) -> tuple[Optional[List[str]], Optional[List[str]]]:
    """Split hash_value between parent and child nodes during node splitting.

    Args:
        child_hash_value: The hash_value list from the child node being split
        split_len: The length at which to split (in tokens)
        page_size: The page size for calculating number of pages

    Returns:
        Tuple of (new_node_hash_value, updated_child_hash_value)
    """
    if child_hash_value is None:
        return None, None

    if page_size == 1:
        split_pages = split_len
    else:
        split_pages = split_len // page_size

    new_node_hash = child_hash_value[:split_pages]
    child_hash = child_hash_value[split_pages:]

    return new_node_hash, child_hash


class RadixCache(BasePrefixCache):
    def __init__(self, params: CacheInitParams):
        self.disable = params.disable
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        self.enable_kv_cache_events = params.enable_kv_cache_events
        self.is_eagle = params.is_eagle
        self.disable_finished_insert = params.disable_finished_insert
        self.eviction_policy = params.eviction_policy.lower()

        self.kv_event_queue = []

        if params.enable_metrics:
            self.init_metrics_collector()

        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        if self.page_size == 1:
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = get_child_key
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=self.page_size)
            self.get_child_key_fn = partial(get_child_key, page_size=self.page_size)

        if self.eviction_policy == "lru":
            self.eviction_strategy: EvictionStrategy = LRUStrategy()
        elif self.eviction_policy == "lfu":
            self.eviction_strategy: EvictionStrategy = LFUStrategy()
        elif self.eviction_policy == "fifo":
            self.eviction_strategy: EvictionStrategy = FIFOStrategy()
        elif self.eviction_policy == "mru":
            self.eviction_strategy: EvictionStrategy = MRUStrategy()
        elif self.eviction_policy == "filo":
            self.eviction_strategy: EvictionStrategy = FILOStrategy()
        elif self.eviction_policy == "priority":
            self.eviction_strategy: EvictionStrategy = PriorityStrategy()
        elif self.eviction_policy == "slru":
            self.eviction_strategy: EvictionStrategy = SLRUStrategy()

        else:
            raise ValueError(
                f"Unknown eviction policy: {self.eviction_policy}. Supported policies: 'lru', 'lfu', 'fifo', 'mru', 'filo', 'priority', 'slru'."
            )

        self.evictable_leaves = set()

        # Mixed-KV: tree caches the shared HP-prefix pool + quant slots;
        # per-request HP-recent slots are excluded by ``_mixed_kv_tail_to_drop``
        # (correctness — they alias across requests) and ``match_prefix`` is
        # capped via ``_mixed_kv_match_cap_overhead``.
        self._mixed_kv_enabled = False
        self._mixed_kv_hp_prefix_tokens = 0
        self._mixed_kv_match_cap_overhead = 0
        if self.token_to_kv_pool_allocator is not None:
            kvc_getter = getattr(self.token_to_kv_pool_allocator, "get_kvcache", None)
            kvc = kvc_getter() if kvc_getter is not None else None
            if kvc is not None:
                mixed_kv_enabled_fn = getattr(kvc, "mixed_kv_enabled", None)
                if mixed_kv_enabled_fn is not None and mixed_kv_enabled_fn():
                    self._mixed_kv_enabled = True
                    self._mixed_kv_hp_prefix_tokens = int(
                        getattr(kvc, "hp_prefix_tokens", 0)
                    )
                    hp_recent = int(getattr(kvc, "hp_recent_tokens", 0))
                    flush_overflow = max(
                        0, int(getattr(kvc, "flush_interval", 1)) - 1
                    )
                    self._mixed_kv_match_cap_overhead = hp_recent + flush_overflow

        self.reset()

    @classmethod
    def create_simulated(
        self,
        disable: bool = False,
        mock_allocator: Optional[Any] = None,
        page_size: int = 1,
        enable_kv_cache_events: bool = False,
    ) -> RadixCache:
        """Init a radix cache without memory pools for simulation purpose."""
        params = CacheInitParams(
            disable=disable,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=page_size,
            enable_kv_cache_events=enable_kv_cache_events,
        )
        return RadixCache(params)

    ##### Public API #####

    def reset(self):
        # Initialize root with minimum priority so any real priority overrides it
        self.root_node = TreeNode(priority=-sys.maxsize)
        self.root_node.key = RadixKey(token_ids=[], extra_key=None)
        self.root_node.value = []
        self.root_node.host_value = []
        self.root_node.lock_ref = 1
        self.root_node.hash_value = []
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self.evictable_leaves.clear()
        self._record_all_cleared_event()

    def maybe_bigram_convert(
        self, key: RadixKey, value: Optional[torch.Tensor] = None
    ) -> Tuple[RadixKey, Optional[torch.Tensor]]:
        return maybe_bigram_convert(self.is_eagle, key, value)

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """Find the longest cached prefix of ``key`` in the radix tree.

        The logical namespace for prefix matching is determined by both the
        token id sequence and the optional ``extra_key`` carried by ``RadixKey``.
        Entries that share identical leading token ids but have *different*
        ``extra_key`` values are intentionally kept disjoint and never share
        prefix nodes. This is useful to:

        * Isolate KV cache lines for different LoRA / adapter IDs.
        * Separate requests that intentionally should not share state (e.g.,
          different sampling salt, cache version, or retrieval augmentation
          context) by supplying a distinct ``extra_key``.

        Args:
            params (MatchPrefixParams): Parameters containing the lookup key
                with a list of token ids and an optional ``extra_key`` namespace tag.
                If ``page_size > 1`` the length is internally truncated to a multiple
                of ``page_size`` before matching. Passing an empty key returns an
                empty result with the root as the last node.

        Returns:
            MatchResult: ``device_indices`` is a 1-D ``torch.int64`` tensor of
            the concatenated KV cache indices corresponding to the longest
            cached prefix (may be length 0). ``last_device_node`` and
            ``last_host_node`` (currently the same) are the tree node objects
            representing the terminal node of the matched prefix. This method
            may mutate internal structure by splitting an existing node if the
            match ends inside a stored segment.

        Internal updates:
            * Refreshes access metadata (timestamps) used by the
                configured eviction strategy.
            * If the lookup ends inside a stored segment the node is split once
                to expose a precise boundary; this structural refinement improves
                subsequent match efficiency and does not duplicate data.
        """
        key = params.key
        key, _ = self.maybe_bigram_convert(key)

        def empty_match_result():
            return MatchResult(
                device_indices=torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                last_device_node=self.root_node,
                last_host_node=self.root_node,
            )

        if self.disable or len(key) == 0:
            return empty_match_result()

        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        # Mixed-KV: cap the match so positions in the request's HP-recent
        # window are NEVER returned from cache. A full match would cover
        # those positions with quant slots from a deeper inserter,
        # breaking the HP-precision invariant for the recent window.
        # Cacheable positions = [0, max(hp_prefix, len - hp_recent_overflow))
        # — for very short requests where the post-prefix region is
        # entirely HP-recent, only the HP-prefix portion is cacheable;
        # for long requests, everything before the trailing HP-recent
        # band is cacheable. Page-aligned to keep the radix tree happy.
        #
        # Internal callers (``cache_unfinished_req``'s post-insert
        # sibling-coverage match) pass ``bypass_mixed_kv_cap=True``: that
        # call needs the FULL match length to stay consistent with the
        # ``cache_protected_len`` admission set. Capping it desyncs the
        # ``prefix_indices`` reconstruction at line ~725 (Python slicing
        # silently truncates ``new_indices[:cache_protected_len]`` when
        # ``len(new_indices) < cache_protected_len``), which silently
        # drops slot ids and leaks them (the leak detector flags it as
        # ~32-slot per-request residue under stress).
        if (
            self._mixed_kv_enabled
            and len(key) > 0
            and not params.bypass_mixed_kv_cap
        ):
            n = len(key)
            cap = min(
                n,
                max(
                    self._mixed_kv_hp_prefix_tokens,
                    n - self._mixed_kv_match_cap_overhead,
                ),
            )
            if self.page_size > 1:
                cap = cap // self.page_size * self.page_size
            if cap < n:
                key = key[:cap]

        if len(key) == 0:
            return empty_match_result()

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = torch.empty((0,), dtype=torch.int64, device=self.device)
        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
        )

    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0)

        key = params.key
        value = params.value
        priority = params.priority
        chunked = params.chunked

        if value is None:
            value = torch.tensor(key.token_ids, dtype=torch.int64)

        key, value = self.maybe_bigram_convert(key, value)

        prefix_len = self._insert_helper(self.root_node, key, value, priority, chunked)
        return InsertResult(prefix_len=prefix_len)

    def _with_mixed_quant_slack(self, req: Req, indices: torch.Tensor) -> torch.Tensor:
        slack = getattr(req, "mixed_kv_quant_slack_indices", None)
        if slack is None or slack.numel() == 0:
            return indices

        req.mixed_kv_quant_slack_indices = torch.empty((0,), dtype=torch.int64)
        req.mixed_kv_quant_slack_cutoff_len = None
        return torch.cat([indices.to(torch.int64), slack.to(indices.device)])

    def _mixed_kv_slack_insert_limit(self, req: Req, key_len: int) -> int:
        """Keep radix ownership below any request-owned partial quant page."""
        cutoff_len = getattr(req, "mixed_kv_quant_slack_cutoff_len", None)
        if cutoff_len is None:
            return key_len
        return max(0, min(key_len, int(cutoff_len)))

    def _committed_fill_ids(self, req: Req) -> list[int]:
        committed_len = min(int(req.kv_committed_len), len(req.fill_ids))
        return req.fill_ids[:committed_len]

    def cache_finished_req(self, req: Req, is_insert: bool = True):
        """Cache request when it finishes."""
        # In deterministic mode, disable finished request insertion to radix cache
        if self.disable_finished_insert:
            is_insert = False

        kv_committed_len = req.pop_committed_kv_cache()
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(
                self._with_mixed_quant_slack(req, kv_indices)
            )
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]
        protected_len = min(req.cache_protected_len, len(kv_indices))
        if protected_len != req.cache_protected_len:
            req.cache_protected_len = protected_len

        if self._mixed_kv_enabled:
            # Mixed-KV: the radix tree is populated ONLY by
            # ``cache_unfinished_req`` (called once after each request's
            # prefill via the scheduler output-processor mixin).
            # ``cache_finished_req`` deliberately returns early for both
            # is_insert=True (natural finish) and is_insert=False (retract):
            # finished/retracted requests just free their tail slots and
            # ``dec_lock_ref`` — they do NOT extend the tree.
            #
            # The naive "insert + bypass-cap re-match + inc_lock_ref(new)
            # → dec_lock_ref(old)" pattern is unsafe under mixed-KV +
            # retract: the new leaf has ``lock_ref=0`` immediately and
            # gets evicted under concurrent retract memory pressure,
            # freeing slot ids that other live requests' ``req_to_token``
            # mappings still reference → corrupted reads → gibberish
            # (~0.5% rate at this batch size, confirmed against the
            # mixed-pool reference implementation which has the same
            # early-return).
            self.token_to_kv_pool_allocator.free(
                self._with_mixed_quant_slack(req, kv_indices[protected_len:])
            )
            self.dec_lock_ref(req.last_node)
            return

        # Maybe convert to bigram keys for EAGLE
        keys = convert_to_bigram_key(token_ids) if self.is_eagle else token_ids
        keys = page_align_keys(keys, self.page_size)
        values = kv_indices[: len(keys)].to(dtype=torch.int64, copy=True)
        radix_key = RadixKey(keys, req.extra_key, is_bigram=self.is_eagle)

        # Drop the mixed-KV HP-recent tail before insert. Here the request is
        # finishing, so the trimmed tail is reclaimed by the existing
        # ``free(kv_indices[len(keys):])`` call below -- which, once ``keys``
        # is shortened, covers the HP-recent tail, the live slots in any
        # unaligned quant tail page, and that page's owned slack slots in one
        # call.
        mixed_trim = self._mixed_kv_tail_to_drop(len(keys))
        if mixed_trim > 0:
            keys = keys[: len(keys) - mixed_trim]
            values = values[: len(keys)]
            radix_key = RadixKey(keys, req.extra_key, is_bigram=self.is_eagle)

        insert_limit = self._mixed_kv_slack_insert_limit(req, len(keys))
        if insert_limit < len(keys):
            keys = keys[:insert_limit]
            values = values[:insert_limit]
            radix_key = RadixKey(keys, req.extra_key, is_bigram=self.is_eagle)

        # Radix Cache takes one ref in memory pool
        if is_insert:
            priority = getattr(req, "priority", 0) or 0
            result = self.insert(
                InsertParams(key=radix_key, value=values, priority=priority)
            )
            new_prefix_len = result.prefix_len
            # Free the duplicates that were already in the tree
            self.token_to_kv_pool_allocator.free(
                kv_indices[protected_len:new_prefix_len]
            )
        else:
            self.token_to_kv_pool_allocator.free(
                kv_indices[protected_len : len(keys)]
            )

        # Free the unaligned tail (includes mixed-KV HP-recent tail and any
        # request-owned quant slack when trimmed). CRITICAL: skip positions in
        # ``[len(keys), cache_protected_len)`` -- those hold tree-owned
        # slot ids (written into req_to_token by cache_unfinished_req's
        # match_prefix step when mixed_trim > 0 and the tree's full-key
        # match exceeded the post-trim length). Those slots are still
        # owned by the tree and will be released when the tree evicts
        # their node; freeing them here would be a double-free, visible
        # to the scheduler as a phantom-slot leak.
        tail_start = max(len(keys), protected_len)
        self.token_to_kv_pool_allocator.free(
            self._with_mixed_quant_slack(req, kv_indices[tail_start:])
        )

        # Remove req slot release the cache lock
        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, chunked=False):
        """Cache request when it is unfinished."""
        if self.disable:
            return

        token_ids = self._committed_fill_ids(req)
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]
        protected_len = min(req.cache_protected_len, len(kv_indices))
        if protected_len != req.cache_protected_len:
            req.cache_protected_len = protected_len

        # Maybe convert to bigram keys for EAGLE
        keys = convert_to_bigram_key(token_ids) if self.is_eagle else token_ids
        keys = page_align_keys(keys, self.page_size)
        values = kv_indices[: len(keys)].to(dtype=torch.int64, copy=True)
        full_radix_key = RadixKey(keys, req.extra_key, is_bigram=self.is_eagle)

        # Drop the mixed-KV HP-recent tail before insert. The live request
        # still owns those HP slots (they stay addressable via the
        # ``torch.cat`` branch below that rebuilds ``req.prefix_indices``),
        # so we must not free them here. See :meth:`_mixed_kv_tail_to_drop`.
        mixed_trim = self._mixed_kv_tail_to_drop(len(keys))
        if mixed_trim > 0:
            insert_keys = keys[: len(keys) - mixed_trim]
            insert_values = values[: len(insert_keys)]
            insert_radix_key = RadixKey(
                insert_keys, req.extra_key, is_bigram=self.is_eagle
            )
        else:
            insert_keys = keys
            insert_values = values
            insert_radix_key = full_radix_key

        insert_limit = self._mixed_kv_slack_insert_limit(req, len(insert_keys))
        if insert_limit < len(insert_keys):
            insert_keys = insert_keys[:insert_limit]
            insert_values = insert_values[:insert_limit]
            insert_radix_key = RadixKey(
                insert_keys, req.extra_key, is_bigram=self.is_eagle
            )

        # Radix Cache takes one ref in memory pool
        result = self.insert(
            InsertParams(
                key=insert_radix_key,
                value=insert_values,
                chunked=chunked,
                priority=getattr(req, "priority", 0) or 0,
            )
        )
        new_prefix_len = result.prefix_len

        # Free our own slot ids that duplicated tree's pre-existing nodes
        # at the time of insert (positions in
        # ``[cache_protected_len, new_prefix_len)`` were covered by the
        # tree before our insert, and our kv_indices there are our-own
        # allocations from extend).
        #
        # CRITICAL: Clamp by ``_mixed_kv_slack_insert_limit``. If
        # ``new_prefix_len`` overlaps a request-owned partial quant page
        # (whose remaining slots are tracked in
        # ``req.mixed_kv_quant_slack_indices`` and freed later by
        # ``_with_mixed_quant_slack``), freeing slots there now would put
        # the page into ``free_pages`` while the request still owns the
        # rest of it via slack. The next ``alloc_quant`` would re-issue
        # the page to another request → multiple requests writing to the
        # same physical slots, and on their finish, the same page gets
        # added to ``free_pages`` twice. Restricting the free to the
        # slack-cutoff range keeps partial-page ownership intact.
        free_end = self._mixed_kv_slack_insert_limit(req, new_prefix_len)
        if free_end > protected_len:
            self.token_to_kv_pool_allocator.free(
                kv_indices[protected_len:free_end]
            )

        # For mixed-KV we match as deeply as it is safe to share. That is
        # usually the FULL (untrimmed) key so ``cache_protected_len`` never
        # regresses, but it must stop before a request-owned partial quant
        # page; otherwise radix can retain the live slots while the request
        # frees that page's slack later.
        match_len = len(keys)
        slack_insert_limit = self._mixed_kv_slack_insert_limit(req, match_len)
        if slack_insert_limit < match_len:
            match_len = slack_insert_limit
        match_key = keys[:match_len]
        full_radix_key = RadixKey(match_key, req.extra_key, is_bigram=self.is_eagle)
        match_result = self.match_prefix(
            MatchPrefixParams(key=full_radix_key, bypass_mixed_kv_cap=True)
        )
        new_indices, new_last_node = (
            match_result.device_indices,
            match_result.last_device_node,
        )
        full_match_len = len(new_indices)
        # The tree must contain at least what we just inserted.
        assert full_match_len >= new_prefix_len, (
            f"{full_match_len=} regressed below {new_prefix_len=}; tree "
            f"must cover at least what we inserted"
        )
        # `match_prefix` on the full key can extend past the trimmed
        # insert via tree coverage from sibling requests. The upper
        # bound is `len(keys)`.
        assert full_match_len <= match_len, f"{full_match_len=}, {match_len=}"

        # Only advance the protected region; never regress. ``cache_protected_len``
        # MUST be monotonic across calls to cache_unfinished_req for a given
        # request: if the tree's full match is not strictly greater than the
        # current value, the current state is already at least as conservative
        # and we must leave it (and ``last_node``) alone. Regressing it -- as
        # the old code did when mixed_trim > 0 -- causes a downstream
        # cross-request double-free at cache_finished_req time.
        if full_match_len > req.cache_protected_len:
            # Free our-own slots at positions that became tree-covered AFTER
            # our insert (i.e., via a sibling-added node past
            # ``max(new_prefix_len, insert_len)``). For those positions,
            # kv_indices holds our extend-allocated slots (admission only
            # wrote tree ids up to the old cache_protected_len), and we are
            # about to overwrite req_to_token with tree's ids, so the
            # originals must be reclaimed to avoid a leak.
            insert_len = len(insert_keys)
            extra_free_start = max(protected_len, new_prefix_len, insert_len)
            # Same partial-page clamp as the dup-free above: the free
            # range must not cross the slack boundary, or it will
            # release slots in a page the request still owns via
            # ``mixed_kv_quant_slack_indices``.
            extra_free_end = self._mixed_kv_slack_insert_limit(
                req, full_match_len
            )
            if extra_free_start < extra_free_end:
                self.token_to_kv_pool_allocator.free(
                    kv_indices[extra_free_start:extra_free_end]
                )

            self.req_to_token_pool.write(
                (
                    req.req_pool_idx,
                    slice(protected_len, full_match_len),
                ),
                new_indices[protected_len:],
            )
            req.cache_protected_len = full_match_len
            self.dec_lock_ref(req.last_node)
            self.inc_lock_ref(new_last_node)
            req.last_node = new_last_node

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        # - page_size != 1: there is a partial page at the end, keep the full kv_indices
        # - eagle case: bigram keys will only cache len - 1 kv indices
        protected_len = req.cache_protected_len
        if protected_len <= len(new_indices):
            protected_indices = new_indices[:protected_len]
        else:
            protected_indices = kv_indices[:protected_len].to(dtype=torch.int64)

        if req.cache_protected_len < len(kv_indices):
            req.prefix_indices = torch.cat(
                [
                    protected_indices,
                    kv_indices[protected_len:],
                ]
            )
        else:
            req.prefix_indices = protected_indices

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        return self._total_size_helper()

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()

        start_time = time.perf_counter()
        num_tokens = params.num_tokens
        leaves = list(self.evictable_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            self.token_to_kv_pool_allocator.free(x.value)
            num_evicted += len(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

            self._record_remove_event(x)

        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
        if self.disable:
            return IncLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            self._update_leaf_status(node)
            node = node.parent
        return IncLockRefResult(delta=delta)

    def dec_lock_ref(
        self, node: TreeNode, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if self.disable:
            return DecLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            self._update_leaf_status(node)
            if node.parent is None:
                assert (
                    node is self.root_node
                ), f"This request holds the node from another tree"
            node = node.parent
        return DecLockRefResult(delta=delta)

    def evictable_size(self):
        return self.evictable_size_

    def recoverable_size(self):
        """Capacity recoverable via eviction.

        Under the unified mixed KV pool the radix tree only stores quant
        slots (HP slots are per-request and never enter the tree), so this
        is identical to ``evictable_size_`` — the standard SGLang shape.
        """
        return self.evictable_size_

    def protected_size(self):
        # protected size refers to the size of the cache that is locked
        return self.protected_size_

    def all_values_flatten(self):
        values = []

        def _dfs_helper(node: TreeNode):
            for _, child in node.children.items():
                values.append(child.value)
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values)

    ##### Internal Helper Functions #####

    def _mixed_kv_tail_to_drop(self, committed_len: int) -> int:
        # HP-recent slot ids are per-request and must not enter the tree.
        # Trim a fixed ``hp_recent + flush_overflow`` window from the
        # tail (page-aligned, ceil'd), which fully covers the worst-case
        # HP-recent span at any time.
        allocator = self.token_to_kv_pool_allocator
        if allocator is None:
            return 0
        kvcache = allocator.get_kvcache()
        mixed_kv_enabled_fn = getattr(kvcache, "mixed_kv_enabled", None)
        if mixed_kv_enabled_fn is None or not mixed_kv_enabled_fn():
            return 0
        hp_prefix = int(getattr(kvcache, "hp_prefix_tokens", 0))
        hp_recent = int(getattr(kvcache, "hp_recent_tokens", 0))
        flush_overflow = max(1, int(getattr(kvcache, "flush_interval", 1))) - 1
        if hp_recent <= 0 or committed_len <= hp_prefix:
            return 0

        if kv_indices is not None and kv_indices.numel() > 0:
            hp_offset = int(getattr(kvcache, "hp_global_offset", 0))
            num_hp_prefix_slots = int(getattr(kvcache, "num_hp_prefix_slots", 0))
            hp_recent_threshold = hp_offset + num_hp_prefix_slots
            window = kv_indices[: committed_len].to(torch.int64)
            is_recent = (window >= hp_recent_threshold).to(torch.int32)
            # Forward cumprod on the NOT-recent mask: stays 1 until the first
            # HP-recent slot, then drops to 0 permanently.  This handles both
            # the normal case (HP-recent only at the tail) and the chunked-
            # prefill case where stale HP-recent IDs appear in the interior
            # (second+ extend chunk slides the HP-recent window but leaves the
            # first chunk's ring slot IDs in req_to_token).
            safe_prefix = torch.cumprod(1 - is_recent, dim=0)
            trim = committed_len - int(safe_prefix.sum().item())
        else:
            if committed_len <= hp_prefix:
                return 0
            trim = min(hp_recent + flush_overflow, committed_len - hp_prefix)

        if self.page_size > 1:
            trim = math.ceil(trim / self.page_size) * self.page_size
        # Clip back if ceil pushed past the available range.
        trim = min(trim, committed_len - hp_prefix)
        return trim

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        access_time = time.monotonic()
        node.last_access_time = access_time

        child_key = self.get_child_key_fn(key)

        value = []
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = access_time
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                node = new_node
                break
            else:
                value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)

        return value, node

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        # new_node -> child
        # New node inherits child's priority (represents shared prefix)
        new_node = TreeNode(priority=child.priority)
        new_node.hit_count = child.hit_count
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref

        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len].clone()
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:].clone()
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        # Split hash_value if it was already computed, otherwise leave as None
        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )

        return new_node

    def _inc_hit_count(self, node: TreeNode, chunked: bool = False):
        # Skip the hit count update for chunked requests to avoid self-referencing
        # inflation where a chunked request increments hit_count on nodes it created
        # in previous chunks.
        if chunked:
            return
        node.hit_count += 1

    def _insert_helper(
        self,
        node: TreeNode,
        key: RadixKey,
        value,
        priority: int = 0,
        chunked: bool = False,
    ):
        # Convert None priority to 0
        if priority is None:
            priority = 0
        access_time = time.monotonic()
        node.last_access_time = access_time
        # Update priority along the path (take max to propagate higher priority)
        node.priority = max(node.priority, priority)
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = access_time
            prefix_len = self.key_match_fn(node.key, key)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                new_node.priority = max(new_node.priority, priority)
                self._inc_hit_count(new_node, chunked)
                node = new_node
            else:
                node.priority = max(node.priority, priority)
                self._inc_hit_count(node, chunked)
            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode(priority=priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            self._inc_hit_count(new_node, chunked)
            node.children[child_key] = new_node
            self.evictable_size_ += len(key)
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)
            # Hash will be computed lazily during event emission
            self._record_store_event(new_node)
        return total_prefix_length

    def _print_helper(self, node: TreeNode, indent: int):
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                len(current_node.key),
                current_node.key.token_ids[:10],
                f"r={current_node.lock_ref}",
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                assert key == self.get_child_key_fn(
                    child.key
                ), f"{key=}, {self.get_child_key_fn(child.key)=}"

    def _delete_leaf(self, node):
        key = self.get_child_key_fn(node.key)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        self.evictable_size_ -= len(node.key)
        if node in self.evictable_leaves:
            self.evictable_leaves.remove(node)
        self._update_leaf_status(node.parent)

    def _update_leaf_status(self, node: TreeNode):
        if node.evicted or node.lock_ref > 0:
            if node in self.evictable_leaves:
                self.evictable_leaves.remove(node)
            return

        for child in node.children.values():
            if not child.evicted:
                if node in self.evictable_leaves:
                    self.evictable_leaves.remove(node)
                return

        if node not in self.evictable_leaves:
            self.evictable_leaves.add(node)

    def _total_size_helper(self):
        total_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size

    def _record_store_event(self, node: TreeNode):
        # One BlockStored per ``page_size`` chunk.
        if self.enable_kv_cache_events:
            # Compute hash_value lazily if not already set
            if node.hash_value is None:
                node.hash_value = compute_node_hash_values(node, self.page_size)

            # Get parent's last hash value for first page
            parent_block_hash = None
            if node.parent is not None and node.parent != self.root_node:
                if (
                    node.parent.hash_value is not None
                    and len(node.parent.hash_value) > 0
                ):
                    parent_block_hash = hash_str_to_int64(node.parent.hash_value[-1])

            page_index = 0
            for start in range(0, len(node.key), self.page_size):
                page_tokens = node.key.token_ids[start : start + self.page_size]
                if not page_tokens:
                    continue

                block_hash = hash_str_to_int64(node.hash_value[page_index])

                self.kv_event_queue.append(
                    BlockStored(
                        block_hashes=[block_hash],
                        parent_block_hash=parent_block_hash,
                        token_ids=page_tokens,
                        block_size=len(page_tokens),
                        lora_id=None,
                        medium=MEDIUM_GPU,
                    )
                )

                parent_block_hash = block_hash
                page_index += 1

    def _record_remove_event(self, node: TreeNode):
        # One BlockRemoved per chunk.
        if self.enable_kv_cache_events:
            # Compute hash_value lazily if not already set (must match what was stored)
            if node.hash_value is None:
                node.hash_value = compute_node_hash_values(node, self.page_size)

            page_index = 0
            for start in range(0, len(node.key), self.page_size):
                page_tokens = node.key.token_ids[start : start + self.page_size]
                if not page_tokens:
                    continue

                block_hash = hash_str_to_int64(node.hash_value[page_index])

                self.kv_event_queue.append(
                    BlockRemoved(block_hashes=[block_hash], medium=MEDIUM_GPU)
                )

                page_index += 1

    def _record_all_cleared_event(self):
        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

    def take_events(self):
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events


if __name__ == "__main__":
    tree = RadixCache.create_simulated()

    # Example token id sequences (as lists of ints)
    tree.insert(InsertParams(key=RadixKey(token_ids=[1, 2, 3], extra_key=None)))
    tree.insert(InsertParams(key=RadixKey(token_ids=[1, 2, 3], extra_key=None)))
    tree.insert(InsertParams(key=RadixKey(token_ids=[1, 2, 4, 5], extra_key=None)))
    tree.insert(
        InsertParams(key=RadixKey(token_ids=[1, 2, 4, 5, 6, 7], extra_key=None))
    )
    tree.insert(
        InsertParams(key=RadixKey(token_ids=[8, 9, 10, 11, 12], extra_key=None))
    )
    tree.pretty_print()

    print(
        tree.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids=[1, 2, 3, 13, 14], extra_key=None))
        )
    )
