"""Shared mixed-KV prefix-cache invariants.

Both :class:`RadixCache` and :class:`MambaRadixCache` must honour the same
tier rules, and they must honour them at *every* call site. Bug 2 in this
project was exactly one call site missing the cap while the other had it, so
these live in one place rather than being copied: a copy that drifts is the
failure mode this file exists to prevent.

Requirements on the host class: ``self.page_size`` and
``self.token_to_kv_pool_allocator``.
"""

import math


class MixedKVPrefixMixin:
    def _init_mixed_kv(self) -> None:
        """Probe the allocator's pool for mixed-KV geometry (duck-typed)."""
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
        trim = min(hp_recent + flush_overflow, committed_len - hp_prefix)
        if self.page_size > 1:
            trim = math.ceil(trim / self.page_size) * self.page_size
        # Clip back if ceil pushed past the available range.
        trim = min(trim, committed_len - hp_prefix)
        return trim
