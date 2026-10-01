"""Shared mixed-KV prefix-cache invariants.

:class:`RadixCache`, :class:`ChunkCache` and :class:`UnifiedRadixCache` (and
``MambaRadixCache`` in the tree this was ported from) must honour the same tier
rules, and they must honour them at *every* call site. Bug 2 in this project
was exactly one call site missing the cap while the other had it, so these
live in one place rather than being copied: a copy that drifts is the failure
mode this file exists to prevent.

Requirements on the host class: ``self.page_size``,
``self.token_to_kv_pool_allocator``, and a ``BasePrefixCache`` further down the
MRO (``on_release`` chains to it; a host that defines its own ``on_release``
must call :meth:`_mixed_kv_drop_quant_slack` itself, as ``UnifiedRadixCache``
does).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


def mixed_kv_pool_of(allocator) -> object | None:
    """The OSCAR mixed-KV pool behind ``allocator``, or None.

    The one probe every cache and the registries share. Duck-typed on
    purpose: stock allocators and pools, the mock allocator of
    ``create_simulated`` and SimpleNamespace test stubs know nothing about
    mixed KV. ``is True`` rather than truthiness: a ``Mock`` answers every
    attribute probe with a callable, truthy ``Mock``.
    """
    if allocator is None:
        return None
    kvc_getter = getattr(allocator, "get_kvcache", None)
    kvc = kvc_getter() if kvc_getter is not None else None
    mixed_kv_enabled_fn = getattr(kvc, "mixed_kv_enabled", None)
    if mixed_kv_enabled_fn is None or mixed_kv_enabled_fn() is not True:
        return None
    return kvc


class MixedKVPrefixMixin:
    # Class-level defaults: a host built without ``__init__`` (unit tests
    # construct the cache through ``__new__``) reads a plain-pool configuration.
    _mixed_kv_enabled: bool = False
    _mixed_kv_hp_prefix_tokens: int = 0
    _mixed_kv_hp_recent_tokens: int = 0
    _mixed_kv_flush_overflow: int = 0
    _mixed_kv_match_cap_overhead: int = 0

    def _init_mixed_kv(self) -> None:
        """Probe the allocator's pool for mixed-KV geometry (duck-typed).

        The geometry is fixed when the pool is built, so it is read once here
        and every later check uses the stored values.
        """
        self._mixed_kv_enabled = False
        self._mixed_kv_hp_prefix_tokens = 0
        self._mixed_kv_hp_recent_tokens = 0
        self._mixed_kv_flush_overflow = 0
        self._mixed_kv_match_cap_overhead = 0
        kvc = mixed_kv_pool_of(self.token_to_kv_pool_allocator)
        if kvc is None:
            return
        # Every pool that answers True is a UnifiedInt2HPKVPool, directly or
        # behind HybridLinearKVPool's attribute forwarding.
        self._mixed_kv_enabled = True
        self._mixed_kv_hp_prefix_tokens = int(kvc.hp_prefix_tokens)
        self._mixed_kv_hp_recent_tokens = int(kvc.hp_recent_tokens)
        # Between two flushes up to ``flush_interval - 1`` positions past the
        # nominal HP-recent window are still BF16.
        self._mixed_kv_flush_overflow = max(0, int(kvc.flush_interval) - 1)
        self._mixed_kv_match_cap_overhead = (
            self._mixed_kv_hp_recent_tokens + self._mixed_kv_flush_overflow
        )

    def _mixed_kv_tier_cap(self, key_len: int) -> int:
        """Longest prefix of a ``key_len``-token sequence that may be shared.

        Mixed-KV tiers a sequence as
        ``[HP-prefix][quant][HP-recent ring]``. The HP-prefix pool is shared
        and its slots are never rewritten, so that window is safe to hand to
        another request. The HP-recent ring is not: it is the BF16 window that
        carries the newest ``hp_recent`` tokens, it is allocated per request
        (``_mixed_extend_layout_counts`` sizes it as
        ``seq_len - max(prefix_len, seq_len - hp_recent)``), and the flush
        demotes out of it from the request's own HP-recent start upward.

        A match that reaches *into* that window is therefore not a cache hit,
        it is a downgrade: every covered position is served from the tree as
        packed INT2 instead of BF16, and ``_mixed_extend_layout_counts``
        allocates a correspondingly shorter ring. The request decodes with
        almost no high-precision recent window -- the one component 2-bit KV
        quality depends on most. In the CPU model in
        ``tests/test_mixed_kv_radix.py`` a 300-token request behind a
        1400-token donor keeps 4 BF16 recent positions instead of 236.
        (It is not memory corruption: those positions' ring slots are orphaned
        at the same time the flush stops demoting them, so ring accounting
        stays balanced. It is a silent, large quality regression.)
        Single-turn evals barely notice -- their shared prefix is shorter than
        ``hp_prefix`` -- but multi-turn hits it on every turn, because turn
        N+1's prompt is almost entirely cached: Qwen3-8B BFCL 38.4 -> 14.6
        with the cache on.

        The cap therefore keeps every shared prefix at or below
        ``max(hp_prefix, key_len - hp_recent - flush_overflow)``, i.e. strictly
        below the HP-recent start the allocator will use
        (``max(hp_prefix, seq_len - hp_recent)``, see
        ``_mixed_extend_layout_counts``), page-aligned so the radix tree keeps
        whole pages. For a short request whose whole post-prefix region is
        HP-recent this degenerates to the HP-prefix window, which is still
        shareable because those slots live in the shared HP-prefix pool and are
        never demoted.
        """
        if not self._mixed_kv_enabled or key_len <= 0:
            return key_len
        cap = min(
            key_len,
            max(
                self._mixed_kv_hp_prefix_tokens,
                key_len - self._mixed_kv_match_cap_overhead,
            ),
        )
        if self.page_size > 1:
            cap = cap // self.page_size * self.page_size
        return cap

    def _mixed_kv_tail_to_drop(self, committed_len: int) -> int:
        # HP-recent slot ids are per-request and must not enter the tree.
        # Trim a fixed ``hp_recent + flush_overflow`` window from the
        # tail (page-aligned, ceil'd), which fully covers the worst-case
        # HP-recent span at any time.
        if not self._mixed_kv_enabled:
            return 0
        hp_prefix = self._mixed_kv_hp_prefix_tokens
        hp_recent = self._mixed_kv_hp_recent_tokens
        if hp_recent <= 0 or committed_len <= hp_prefix:
            return 0
        trim = min(
            hp_recent + self._mixed_kv_flush_overflow, committed_len - hp_prefix
        )
        if self.page_size > 1:
            trim = math.ceil(trim / self.page_size) * self.page_size
        # Clip back if ceil pushed past the available range.
        trim = min(trim, committed_len - hp_prefix)
        return trim

    def _mixed_kv_slack_insert_limit(self, req: Req, key_len: int) -> int:
        """Keep radix ownership below any request-owned partial quant page."""
        if not self._mixed_kv_enabled:
            return key_len
        cutoff_len = req.mixed_kv_quant_slack_cutoff_len
        if cutoff_len is None:
            return key_len
        return max(0, min(key_len, int(cutoff_len)))

    def _mixed_kv_insert_ceiling(self, req: Req, key_len: int) -> int:
        """Longest page-aligned prefix of a ``key_len``-unit committed key that
        ``cache_unfinished_req`` may hand to the tree.

        Two bounds, both required: the HP-recent tail is dropped
        (:meth:`_mixed_kv_tail_to_drop` -- those slot ids are per-request and
        alias across requests, so they must never enter the tree) and radix
        ownership stays below any request-owned partial quant page
        (:meth:`_mixed_kv_slack_insert_limit`). ``insert`` page-aligns its key
        anyway; aligning here too makes the free ranges derived from the
        result start on a page boundary.
        """
        if not self._mixed_kv_enabled:
            return key_len
        limit = key_len - self._mixed_kv_tail_to_drop(key_len)
        limit = self._mixed_kv_slack_insert_limit(req, limit)
        if self.page_size > 1:
            limit = limit // self.page_size * self.page_size
        return max(0, limit)

    def _mixed_kv_share_ceiling(self, req: Req, key_len: int, insert_len: int) -> int:
        """How deep ``cache_unfinished_req``'s post-insert match may let the
        tree cover this request. Three bounds, all required:

        1. :meth:`_mixed_kv_tier_cap`: never past this request's own HP-recent
           start, or the tree serves at 2 bits what the request was supposed
           to keep in BF16 (and the flush can never demote it). This is the
           same cap admission gets; applying it here too is what keeps
           ``cache_protected_len`` tier-stable. Without it a sibling (or, in
           multi-turn, this request's own previous turn) covers the whole
           prompt and pushes ``cache_protected_len`` above the HP-recent
           start, wiping out the recent window.
        2. the request-owned partial quant page cutoff; otherwise radix can
           retain the live slots while the request frees that page's slack.
        3. never below what was just inserted (``insert_len``), so
           ``cache_protected_len`` is monotonic and the ``prefix_indices``
           rebuild cannot silently truncate (which would leak slot ids).

        The caller passes the result with ``bypass_mixed_kv_cap=True``: the cap
        was applied here, to this key length, before the other two clamps.
        """
        match_len = key_len
        tier_cap = self._mixed_kv_tier_cap(match_len)
        if tier_cap < match_len:
            match_len = tier_cap
        slack_insert_limit = self._mixed_kv_slack_insert_limit(req, match_len)
        if slack_insert_limit < match_len:
            match_len = slack_insert_limit
        match_len = max(match_len, insert_len)
        if self.page_size > 1:
            match_len = match_len // self.page_size * self.page_size
        return match_len

    def on_release(self, req: Req, *, inserted: bool) -> None:
        # ``release_kv_cache`` calls this after ``free_kv_row`` gave back
        # ``[cache_protected_len, owned_kv_len)`` and ``unpin`` dropped the lock.
        if self._mixed_kv_enabled:
            self._mixed_kv_drop_quant_slack(req)
        super().on_release(req, inserted=inserted)

    @staticmethod
    def _mixed_kv_drop_quant_slack(req: Req) -> None:
        # The row free covered the live slots of every request-owned partial
        # page (radix ownership stays below ``mixed_kv_quant_slack_cutoff_len``,
        # so they sit above ``cache_protected_len``), and the mixed allocator's
        # ``free`` returns whole pages, so the slack went back with them. A
        # second ``free`` for the slack would push the same pages onto the free
        # list twice (the double-add that corrupts KV); the old tree avoided
        # that by concatenating the slack into the tail free, which the split
        # ``free_kv_row`` / ``on_release`` flow cannot do. So the slack is only
        # forgotten here, never issued on its own.
        req.mixed_kv_quant_slack_indices = torch.empty((0,), dtype=torch.int64)
        req.mixed_kv_quant_slack_cutoff_len = None
