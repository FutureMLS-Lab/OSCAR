#!/usr/bin/env python3
"""Add init/engagement diagnostics to MixedKVPrefixMixin (workspace-local only).

Carried over verbatim in intent from the 4B campaign's patch_diag.py. Two log
lines, both at INFO:

  * one per cache construction, naming the concrete class and the probed
    geometry -- this is what proves the duck-typed probe returned True on the
    ``MambaRadixCache`` path rather than silently falling through to a no-op cap;
  * the first three times the tier cap actually truncates a match, with the
    numbers -- a cache that reports "enabled" but never bites would still make
    every downstream number worthless.

Applied to both caches (the mixin is shared), so no arm is privileged.

This is deliberately NOT committed to the branch and NOT applied to the
workspace the scored arms run from: it adds logging inside ``match_prefix``,
which is on the hot path, and the scored arms must run pristine HEAD. Stage a
second workspace, patch that, and run the engagement probe there.

Usage: patch_mixin_diag.py <path/to/mixed_kv_prefix_mixin.py>
"""
import sys

P = sys.argv[1]
src = open(P).read()

if "mixed-kv-prefix" in src:
    print("already patched")
    sys.exit(0)

src = src.replace(
    "import math\n",
    "import logging\nimport math\n\nlogger = logging.getLogger(__name__)\n",
    1,
)

# 1) init diagnostic, at the end of _init_mixed_kv
anchor = "                    self._mixed_kv_match_cap_overhead = hp_recent + flush_overflow\n"
assert anchor in src, "init anchor missing"
src = src.replace(
    anchor,
    anchor
    + """        self._mixed_kv_cap_hits = 0
        logger.info(
            "[mixed-kv-prefix] %s: enabled=%s hp_prefix_tokens=%d "
            "match_cap_overhead=%d page_size=%s",
            type(self).__name__,
            self._mixed_kv_enabled,
            self._mixed_kv_hp_prefix_tokens,
            self._mixed_kv_match_cap_overhead,
            getattr(self, "page_size", None),
        )
""",
    1,
)

# 2) cap-engagement diagnostic, inside _mixed_kv_tier_cap
anchor2 = """        if self.page_size > 1:
            cap = cap // self.page_size * self.page_size
        return cap
"""
assert anchor2 in src, "cap anchor missing"
src = src.replace(
    anchor2,
    """        if self.page_size > 1:
            cap = cap // self.page_size * self.page_size
        if cap < key_len:
            n = getattr(self, "_mixed_kv_cap_hits", 0) + 1
            self._mixed_kv_cap_hits = n
            if n <= 3 or n % 500 == 0:
                logger.info(
                    "[mixed-kv-prefix] %s: tier cap #%d truncated match "
                    "%d -> %d (hp_prefix=%d overhead=%d page=%d)",
                    type(self).__name__,
                    n,
                    key_len,
                    cap,
                    self._mixed_kv_hp_prefix_tokens,
                    self._mixed_kv_match_cap_overhead,
                    self.page_size,
                )
        return cap
""",
    1,
)

open(P, "w").write(src)
print("patched", P)
