"""Mixed-KV tiering must stay consistent when the radix (prefix) cache is on.

The unified pool tiers every sequence as
``[BF16 HP-prefix][INT2 quant][BF16 HP-recent ring]``. Two of those tiers are
*positional*: the HP-recent ring holds only the newest ``hp_recent`` tokens and
recycles its slots, and tokens are physically rewritten (demoted into packed
INT2) as they age out of it. The radix cache, meanwhile, hands one request's
slot ids to another request. The two only coexist if the cached span never
overlaps a tier that the borrowing request still needs to own.

This module drives the *real* ``RadixCache`` against a CPU shadow-memory model
of the pool: every physical slot records which ``(token id, position)`` was
written into it, and every read of position ``p`` asserts the slot still holds
the value written for that position. That is corruption detection, not policy
checking -- a lapped ring slot or a double-allocated quant page shows up as a
shadow mismatch regardless of which bookkeeping rule caused it.

The flush model mirrors the three load-bearing lines of
``_flush_plan_kernel`` (``QuantKernel/gpu_flush_int2.py``)::

    fp = seq_len - HP_RECENT_TOKENS - (FLUSH_INTERVAL - 1) + j
    if fp >= prefix_len and fp >= 0:
        if req_to_token[req][fp] >= HP_OFFSET:  demote

and the per-request counter RMW from ``_alloc_for_decode_mixed``
(``mem_cache/common.py``). ``prefix_len`` is the value the real radix cache
computed (``req.cache_protected_len``), so a wrong cached-prefix length shows
up here as real corruption.
"""

import types

import torch

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import _mixed_extend_layout_counts
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey

N_Q = 8               # int2 page size == --page-size for bf16 HP dtype
HP_PREFIX = 64        # SGLANG_MIXED_KV_PREFIX_TOKENS
HP_RECENT = 256       # SGLANG_MIXED_KV_RECENT_TOKENS
RING = HP_RECENT + N_Q - 1
MAX_REQ = 8
NUM_QUANT_PAGES = 4096
NUM_HP_PREFIX_SLOTS = 1024
MAX_CTX = 8192

HP_OFFSET = NUM_QUANT_PAGES * N_Q
HP_RECENT_BASE = HP_OFFSET + NUM_HP_PREFIX_SLOTS


class _Pool:
    """Duck-typed ``UnifiedInt2HPKVPool`` + allocator with shadow memory."""

    def __init__(self):
        self.hp_prefix_tokens = HP_PREFIX
        self.hp_recent_tokens = HP_RECENT
        self.hp_recent_ring_size = RING
        self.num_hp_prefix_slots = NUM_HP_PREFIX_SLOTS
        self.flush_interval = N_Q
        self.N_Q = N_Q
        self.device = torch.device("cpu")
        # shadow: slot id -> (token_id, position) last written there
        self.shadow: dict[int, tuple[int, int]] = {}
        self.quant_free = list(range(1, NUM_QUANT_PAGES))       # page 0 reserved
        self.hp_prefix_free = list(range(NUM_HP_PREFIX_SLOTS // N_Q))
        self.ring_cursor = [0] * MAX_REQ
        self.flush_counter = [0] * MAX_REQ
        self.freed: list[int] = []
        self.double_freed: list[int] = []

    # -- pool-side accessors the radix cache probes for ---------------------
    def mixed_kv_enabled(self):
        return True

    @property
    def hp_global_offset(self):
        return HP_OFFSET

    def get_kvcache(self):
        return self

    # -- allocation --------------------------------------------------------
    def alloc_quant(self, n):
        assert n % N_Q == 0
        pages = [self.quant_free.pop(0) for _ in range(n // N_Q)]
        return [p * N_Q + i for p in pages for i in range(N_Q)]

    def alloc_hp_prefix(self, n):
        assert n % N_Q == 0
        pages = [self.hp_prefix_free.pop(0) for _ in range(n // N_Q)]
        return [HP_OFFSET + p * N_Q + i for p in pages for i in range(N_Q)]

    def alloc_hp_recent(self, rpi, n):
        base = HP_RECENT_BASE + rpi * RING
        out = []
        for j in range(n):
            out.append(base + (self.ring_cursor[rpi] + j) % RING)
        self.ring_cursor[rpi] = (self.ring_cursor[rpi] + n) % RING
        return out

    def free(self, free_index):
        """Mirrors ``UnifiedInt2HPKVAllocator.free``: whole-page aggregation."""
        if isinstance(free_index, torch.Tensor):
            ids = [int(x) for x in free_index.flatten().tolist()]
        else:
            ids = [int(x) for x in free_index]
        for s in ids:
            self.freed.append(s)
        quant_pages = {s // N_Q for s in ids if s < HP_OFFSET}
        for p in quant_pages:
            if p in self.quant_free:
                self.double_freed.append(p)
            else:
                self.quant_free.append(p)
        hp_pages = {
            (s - HP_OFFSET) // N_Q
            for s in ids
            if HP_OFFSET <= s < HP_RECENT_BASE
        }
        for p in hp_pages:
            if p in self.hp_prefix_free:
                self.double_freed.append(HP_OFFSET + p)
            else:
                self.hp_prefix_free.append(p)


class _ReqToTokenPool:
    def __init__(self):
        self.req_to_token = torch.zeros(
            (MAX_REQ, MAX_CTX), dtype=torch.int64
        )

    def write(self, indices, values):
        self.req_to_token[indices] = values


class _Req:
    """The attribute surface ``RadixCache`` touches on a ``Req``."""

    def __init__(self, rid, rpi, token_ids):
        self.rid = rid
        self.req_pool_idx = rpi
        self.origin_input_ids = list(token_ids)
        self.output_ids: list[int] = []
        self.fill_ids = list(token_ids)
        self.kv_committed_len = len(token_ids)
        self.cache_protected_len = 0
        self.extra_key = None
        self.last_node = None
        self.prefix_indices = torch.empty((0,), dtype=torch.int64)
        self.mixed_kv_quant_slack_indices = torch.empty((0,), dtype=torch.int64)
        self.mixed_kv_quant_slack_cutoff_len = None
        self.priority = 0
        # sim-only bookkeeping
        self.seq_len = 0

    def pop_committed_kv_cache(self):
        return self.kv_committed_len


class Sim:
    """Prefill / decode / finish driver over the real radix cache."""

    def __init__(self):
        self.pool = _Pool()
        self.rtt = _ReqToTokenPool()
        self.tree = RadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=self.rtt,
                token_to_kv_pool_allocator=self.pool,
                page_size=N_Q,
            )
        )
        self.violations: list[str] = []
        self.free_rpi = list(range(MAX_REQ))
        self._admitted: list[_Req] = []

    # ------------------------------------------------------------------
    def _write(self, req, pos, slot):
        self.rtt.req_to_token[req.req_pool_idx, pos] = slot
        self.pool.shadow[slot] = (req.fill_ids[pos], pos)

    def _check_reads(self, req, tag):
        """Every live position must still read back its own token/position."""
        row = self.rtt.req_to_token[req.req_pool_idx, : req.seq_len].tolist()
        for pos, slot in enumerate(row):
            got = self.pool.shadow.get(int(slot))
            if got is None:
                self.violations.append(
                    f"{tag} rid={req.rid} pos={pos} slot={slot} never written"
                )
            elif got != (req.fill_ids[pos], pos):
                self.violations.append(
                    f"{tag} rid={req.rid} pos={pos} slot={slot} holds "
                    f"token/pos {got}, expected {(req.fill_ids[pos], pos)}"
                )
            if len(self.violations) > 20:
                return

    # ------------------------------------------------------------------
    def admit(self, rid, token_ids):
        req = _Req(rid, self.free_rpi.pop(0), token_ids)
        # --- scheduler admission: capped match_prefix on fill_ids[:-1] ----
        key = RadixKey(token_ids=req.fill_ids[: len(req.fill_ids) - 1], extra_key=None)
        m = self.tree.match_prefix(MatchPrefixParams(key=key, req=req))
        req.prefix_indices = m.device_indices
        req.last_node = m.last_device_node
        req.cache_protected_len = (
            m.cache_protected_len
            if getattr(m, "cache_protected_len", None) is not None
            else len(m.device_indices)
        )
        self.tree.inc_lock_ref(req.last_node)
        pre_len = len(req.prefix_indices)
        seq_len = len(req.fill_ids)

        # --- prefill tier layout (production helper) ----------------------
        (
            hp_prefix_count,
            hp_recent_count,
            quant_count,
            quant_alloc_count,
            counter_init,
        ) = _mixed_extend_layout_counts(
            pre_len, seq_len, HP_PREFIX, HP_RECENT, N_Q, is_final_chunk=True
        )
        assert hp_prefix_count + quant_count + hp_recent_count == seq_len - pre_len, (
            "layout counts must cover exactly the extend range"
        )
        hp_prefix_slots = self.pool.alloc_hp_prefix(
            ((hp_prefix_count + N_Q - 1) // N_Q) * N_Q
        )
        quant_slots = self.pool.alloc_quant(quant_alloc_count)
        recent_slots = self.pool.alloc_hp_recent(req.req_pool_idx, hp_recent_count)
        locs = (
            hp_prefix_slots[:hp_prefix_count]
            + quant_slots[:quant_count]
            + recent_slots
        )
        # request-owned slack (partial pages) is tracked exactly as production
        slack = hp_prefix_slots[hp_prefix_count:] + quant_slots[quant_count:]
        if slack:
            req.mixed_kv_quant_slack_indices = torch.tensor(slack, dtype=torch.int64)
            cut = max(pre_len, HP_PREFIX) + (quant_count // N_Q) * N_Q
            req.mixed_kv_quant_slack_cutoff_len = cut

        # prefix positions read the borrowed slots
        pfx = req.prefix_indices.tolist()
        for pos, slot in enumerate(pfx):
            self.rtt.req_to_token[req.req_pool_idx, pos] = int(slot)
        for i, slot in enumerate(locs):
            self._write(req, pre_len + i, slot)
        req.seq_len = seq_len
        req.kv_committed_len = seq_len
        self.pool.flush_counter[req.req_pool_idx] = counter_init

        self.tree.cache_unfinished_req(req)
        self._check_reads(req, "after-prefill")
        self._admitted.append(req)
        return req

    # ------------------------------------------------------------------
    def decode_step(self, req, token_id):
        rpi = req.req_pool_idx
        seq_len = req.seq_len  # pre-increment, as ``locs = batch.seq_lens``
        prefix_len = int(req.cache_protected_len)

        # per-request flush gate (RMW from _alloc_for_decode_mixed)
        counter = self.pool.flush_counter[rpi]
        do_flush = counter == 0
        self.pool.flush_counter[rpi] = N_Q - 1 if do_flush else counter - 1

        # one HP-recent slot for the new token
        new_slot = self.pool.alloc_hp_recent(rpi, 1)[0]

        # flush plan + apply (mirror of _flush_plan_kernel)
        dst = self.pool.alloc_quant(N_Q)
        used = []
        for j in range(N_Q):
            fp = seq_len - HP_RECENT - (N_Q - 1) + j
            if not do_flush or fp < prefix_len or fp < 0:
                continue
            src = int(self.rtt.req_to_token[rpi, fp])
            if src < HP_OFFSET:
                continue
            # demote: copy the shadow value, repoint req_to_token, free the ring slot
            self.pool.shadow[dst[j]] = self.pool.shadow[src]
            self.rtt.req_to_token[rpi, fp] = dst[j]
            used.append(j)
        self.pool.free([dst[j] for j in range(N_Q) if j not in used])

        req.fill_ids.append(token_id)
        req.output_ids.append(token_id)
        req.seq_len += 1
        req.kv_committed_len = req.seq_len
        self._write(req, seq_len, new_slot)
        return len(used)

    def finish(self, req):
        self.tree.cache_finished_req(req)
        self.free_rpi.append(req.req_pool_idx)
        self.pool.ring_cursor[req.req_pool_idx] = 0
        self.pool.flush_counter[req.req_pool_idx] = 0


# ----------------------------------------------------------------------
# Scenarios
# ----------------------------------------------------------------------
SHARED = list(range(1000, 1000 + 96))     # 96-token shared instruction prefix


def _prompt(seed, length):
    body = [(seed * 7919 + i) % 30000 + 2 for i in range(length - len(SHARED))]
    return SHARED + body


def _run(prompts, gen_tokens, check_every=32):
    sim = Sim()
    reqs = []
    for i, p in enumerate(prompts):
        reqs.append(sim.admit(f"r{i}", p))
    for step in range(gen_tokens):
        for r in reqs:
            sim.decode_step(r, 5000 + step)
        if (step % check_every) == 0:
            for r in reqs:
                sim._check_reads(r, f"decode-step{step}")
            if sim.violations:
                break
    for r in reqs:
        sim.finish(r)
    return sim


def test_hp_prefix_padding_sink_is_not_allocatable():
    """The slot CUDA-graph padding writes to must not belong to anyone.

    ``CudaGraphRunner`` points padded decode ``out_cache_loc`` entries at
    ``token_to_kv_pool.hp_global_offset`` and ``UnifiedInt2HPKVPool.set_kv_buffer``
    writes ``hp_buffer[loc - hp_offset]`` with no mask (masking is unsafe under
    capture), so every replay whose batch size is not exactly a captured size
    writes a dummy token's K/V into HP-prefix slot 0. While that page was in
    ``hp_prefix_free_pages`` it was handed to a real request -- and with the
    radix cache on it is typically part of the *shared* prefix node, so one
    padded replay corrupted the cached prefix every live request reads (that is
    the GPQA 57.07 -> 27.78 collapse: coherent generations that lose the
    instruction mid-stream and run to max_tokens).
    """
    from sglang.srt.mem_cache.unified_kv_allocator import UnifiedInt2HPKVAllocator

    requested_hp_slots = 64
    alloc = UnifiedInt2HPKVAllocator(
        num_quant_pages=64,
        quant_tokens_per_page=N_Q,
        hp_prefix_tokens=HP_PREFIX,
        hp_recent_tokens=HP_RECENT,
        hp_recent_ring_size=RING,
        max_req_slots=2,
        # what UnifiedInt2HPKVPool passes: requested capacity + the sink page
        num_hp_prefix_slots=requested_hp_slots + N_Q,
        dtype="int2",
        hp_dtype=torch.bfloat16,
        device="cpu",
        kvcache=None,
        need_sort=False,
    )
    sink = alloc.hp_global_offset
    handed_out = []
    # drain the whole shared HP-prefix pool
    while alloc._hp_prefix_free_slots() > 0:
        got = alloc.alloc_hp_prefix(torch.tensor([0]), [N_Q])
        handed_out.extend(int(x) for x in got.tolist())
    assert sink not in handed_out, (
        f"hp_global_offset ({sink}) -- the slot CUDA-graph padding writes to -- "
        "was handed to a request; padded replays will overwrite its KV"
    )
    assert len(handed_out) == requested_hp_slots, (
        f"usable shared-prefix capacity is {len(handed_out)}, expected "
        f"{requested_hp_slots}: the reserved sink page must be extra, not "
        "carved out of the operator's budget"
    )
    # freeing everything must not put the sink page back into circulation
    alloc.free(torch.tensor(handed_out, dtype=torch.int64))
    assert sink not in [
        alloc.hp_global_offset + int(p) * N_Q + i
        for p in alloc.hp_prefix_free_pages.tolist()
        for i in range(N_Q)
    ], "the reserved sink page re-entered the free list after free()"


def _tier_counts(sim, req):
    """(hp_prefix, quant, hp_recent) position counts as the reader sees them."""
    row = sim.rtt.req_to_token[req.req_pool_idx, : req.seq_len].tolist()
    hp_prefix = sum(1 for s in row if HP_OFFSET <= s < HP_RECENT_BASE)
    recent = sum(1 for s in row if s >= HP_RECENT_BASE)
    return hp_prefix, req.seq_len - hp_prefix - recent, recent


def test_cached_prefix_does_not_cannibalize_hp_recent():
    """A borrowed prefix must not eat the borrower's BF16 HP-recent window.

    This is the multi-turn shape: an earlier, longer sequence has donated a
    deep prefix (its quant middle) to the tree, and a later request whose
    tokens match it arrives. Nothing stops the match from covering the later
    request's *whole* prompt -- including the tail it was supposed to keep in
    BF16 -- because the post-insert match in ``cache_unfinished_req`` used to
    bypass the tier cap. The request then decodes with an HP-recent window of
    a handful of tokens instead of ``hp_recent``, i.e. with essentially the
    whole context at 2 bits. That is the Qwen3-8B BFCL 38.4 -> 14.6 collapse,
    and the reason the GPQA harness had to pass --disable-radix-cache.
    """
    sim = Sim()
    long_req = sim.admit("long", _prompt(3, 1400))
    short_req = sim.admit("short", _prompt(3, 300))
    _, _, recent = _tier_counts(sim, short_req)
    want = min(HP_RECENT, short_req.seq_len - HP_PREFIX)
    assert recent >= want, (
        f"borrower kept only {recent} BF16 HP-recent positions, expected "
        f"{want} (cache_protected_len={short_req.cache_protected_len}, "
        f"donor tree depth={long_req.cache_protected_len}): the cached prefix "
        "reached into its HP-recent window"
    )


def test_protected_len_stays_below_hp_recent_start():
    """Tight regression guard on the boundary itself.

    ``cache_protected_len`` is the ``prefix_len`` the flush kernel receives
    and the boundary below which the request does not own its KV. It must
    never exceed the request's own HP-recent start
    ``max(hp_prefix, committed - hp_recent)``.
    """
    sim = Sim()
    sim.admit("long", _prompt(3, 1400))          # donates a deep prefix
    sim.admit("mid", _prompt(3, 700))            # borrows across the boundary
    sim.admit("short", _prompt(3, 300))
    for r in sim_reqs(sim):
        recent_start = max(HP_PREFIX, r.seq_len - HP_RECENT)
        assert r.cache_protected_len <= recent_start, (
            f"rid={r.rid} cache_protected_len={r.cache_protected_len} > "
            f"HP-recent start {recent_start}: those positions can never be "
            "demoted and are served from the tree at 2 bits"
        )


def sim_reqs(sim):
    return getattr(sim, "_admitted", [])


def test_shared_prefix_does_not_corrupt_kv():
    """Shadow-memory check over a mixed batch: no position ever reads foreign KV."""
    sim = _run([_prompt(1, 200), _prompt(2, 200), _prompt(1, 900)], gen_tokens=600)
    assert not sim.violations, "KV corruption:\n" + "\n".join(sim.violations[:8])
    assert not sim.pool.double_freed, f"double-freed pages {sim.pool.double_freed[:8]}"


def test_long_prompt_prefix_reuse_is_consistent():
    """Long prompts (quant middle in the tree) plus a short sibling.

    The long request donates real quant slots to the tree; the short sibling
    matches them. Exercises the mixed ``[hp-prefix][quant][ring]`` read path
    with a borrowed cross-tier prefix.
    """
    sim = _run(
        [_prompt(3, 1400), _prompt(3, 1400), _prompt(3, 300)],
        gen_tokens=400,
    )
    assert not sim.violations, "KV corruption:\n" + "\n".join(sim.violations[:8])
    assert not sim.pool.double_freed, f"double-freed pages {sim.pool.double_freed[:8]}"


def test_no_ring_slot_enters_the_tree():
    """HP-recent ids are per-request and recycled; they must never be cached."""
    sim = _run([_prompt(4, 300), _prompt(5, 300)], gen_tokens=300)
    values = sim.tree.all_values_flatten().tolist() if _tree_nonempty(sim) else []
    bad = [int(v) for v in values if int(v) >= HP_RECENT_BASE]
    assert not bad, f"HP-recent ring slots reachable from the radix tree: {bad[:8]}"


def _tree_nonempty(sim):
    return len(sim.tree.root_node.children) > 0
