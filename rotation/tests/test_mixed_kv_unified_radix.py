"""Mixed-KV tiering on ``UnifiedRadixCache`` -- FULL alone and FULL + MAMBA.

The OSCAR pool tiers every sequence as ``[BF16 HP-prefix][INT2 quant][BF16
HP-recent ring]``. Two tiers are positional: the HP-recent ring holds only the
newest ``hp_recent`` tokens and recycles its slots, and tokens are rewritten
(demoted into packed INT2) as they age out of it. The radix cache hands one
request's slot ids to another; the two coexist only if the cached span never
overlaps a tier the borrowing request still needs to own.

This module drives the *real* ``UnifiedRadixCache`` (Python tree core) against
a CPU shadow-memory model of the pool: every physical slot records which
``(token id, position)`` was written into it, and every read of position ``p``
asserts the slot still holds the value written for that position. The hybrid
variant adds a shadow for mamba slots -- which ``(prefix, depth)`` state each
slot holds -- and checks every copy-on-write a prefix hit performs against it.
The flush model mirrors ``_flush_plan_kernel`` and the per-request counter RMW
of ``_alloc_for_decode_mixed``; the prefill checkpoint placement mirrors
``ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend``. The reference
semantics are the ``RadixCache`` port (``rotation/tests/test_mixed_kv_radix.py``).

Runs on CPU inside the serving image::

    docker run --rm -v "$PWD:/sgl-workspace/sglang" --entrypoint python3 \\
        oscar-env:v56 -m pytest -q rotation/tests/test_mixed_kv_unified_radix.py
"""

from __future__ import annotations

import types
from array import array
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.srt.mem_cache.mixed_kv_prefix_mixin import mixed_kv_prefill_insert_ceiling
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.registry import (
    TreeCacheBuildContext,
    default_radix_cache_factory,
)
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import (
    get_context,
    get_server_args,
    mamba_cache_chunk_size,
    mamba_checkpoint_grid,
    mamba_track_grid,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

N_Q = 8  # int2 page size == --page-size for the bf16 HP dtype
HP_PREFIX = 64  # SGLANG_MIXED_KV_PREFIX_TOKENS
HP_RECENT = 256  # SGLANG_MIXED_KV_RECENT_TOKENS
RING = HP_RECENT + N_Q - 1
MAX_REQ = 8
NUM_QUANT_PAGES = 4096
NUM_HP_PREFIX_SLOTS = 1024
MAX_CTX = 8192
HP_OFFSET = NUM_QUANT_PAGES * N_Q
HP_RECENT_BASE = HP_OFFSET + NUM_HP_PREFIX_SLOTS
MAMBA_SLOTS = 64
MAMBA_CHUNK = 64  # FLA chunk size; the checkpoint grid is lcm(chunk, page)


@pytest.fixture(scope="module", autouse=True)
def _published_server_args():
    server_args = ServerArgs(model_path="dummy", page_size=N_Q)
    # Pre-seeded so the dummy model never loads an HF config (as the upstream
    # unified-cache fixture does).
    server_args._mamba_cache_chunk_size = MAMBA_CHUNK
    set_global_server_args_for_scheduler(server_args)
    yield server_args


def ceil_align(x, a):
    return (x + a - 1) // a * a


# --------------------------------------------------------------------------
# Pool / allocator models
# --------------------------------------------------------------------------
class _AllocatorAPI:
    """The segment-free surface ``free_kv_row`` / the tree's ``FreeDeviceKV``
    actions go through."""

    def free_segment(self, free_index, *, start_pos):
        assert (
            start_pos % self.page_size == 0
        ), f"segment start {start_pos} is not page-aligned"
        self.free(free_index)

    def free_segments(self, segments):
        prev_end = None
        for free_index, start_pos in segments:
            n = free_index.numel()
            if n == 0:
                continue
            assert (
                prev_end is None
                or start_pos // self.page_size > (prev_end - 1) // self.page_size
            ), f"segment at {start_pos} shares a page with one ending at {prev_end}"
            self.free_segment(free_index, start_pos=start_pos)
            prev_end = start_pos + n

    def free_full_segments(self, segments):
        self.free_segments(segments)


class _MixedPool(_AllocatorAPI):
    """Duck-typed ``UnifiedInt2HPKVPool`` + allocator with shadow memory."""

    def __init__(self):
        self.page_size = N_Q
        self.hp_prefix_tokens = HP_PREFIX
        self.hp_recent_tokens = HP_RECENT
        self.flush_interval = N_Q
        self.N_Q = N_Q
        self.device = torch.device("cpu")
        # slot id -> (token_id, position) last written there
        self.shadow = {}
        self.quant_free = list(range(1, NUM_QUANT_PAGES))  # page 0 reserved
        self.hp_prefix_free = list(range(NUM_HP_PREFIX_SLOTS // N_Q))
        self.ring_cursor = [0] * (MAX_REQ + 1)
        self.flush_counter = [0] * (MAX_REQ + 1)
        self.double_freed = []

    def mixed_kv_enabled(self):
        return True

    def get_kvcache(self):
        return self

    def release_req_slab(self, rpi):
        # ``release_kv_cache`` resets the per-request HP-recent cursor here.
        self.ring_cursor[rpi] = 0
        self.flush_counter[rpi] = 0

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
        out = [base + (self.ring_cursor[rpi] + j) % RING for j in range(n)]
        self.ring_cursor[rpi] = (self.ring_cursor[rpi] + n) % RING
        return out

    def free(self, free_index):
        """Mirrors ``UnifiedInt2HPKVAllocator.free``: whole-page aggregation,
        HP-recent ids are a no-op."""
        ids = (
            [int(x) for x in free_index.flatten().tolist()]
            if isinstance(free_index, torch.Tensor)
            else [int(x) for x in free_index]
        )
        for p in {s // N_Q for s in ids if s < HP_OFFSET}:
            if p in self.quant_free:
                self.double_freed.append(p)
            else:
                self.quant_free.append(p)
        for p in {
            (s - HP_OFFSET) // N_Q for s in ids if HP_OFFSET <= s < HP_RECENT_BASE
        }:
            if p in self.hp_prefix_free:
                self.double_freed.append(HP_OFFSET + p)
            else:
                self.hp_prefix_free.append(p)

    def live_quant_pages(self):
        return set(range(1, NUM_QUANT_PAGES)) - set(self.quant_free)

    def live_hp_prefix_pages(self):
        return set(range(NUM_HP_PREFIX_SLOTS // N_Q)) - set(self.hp_prefix_free)


class _PlainPool(_AllocatorAPI):
    """Stock paged allocator model (no mixed KV), page-granular free."""

    def __init__(self, page_size):
        self.page_size = page_size
        self.device = torch.device("cpu")
        self.shadow = {}
        self.num_pages = 4096
        self.free_pages = list(range(1, self.num_pages))
        self.double_freed = []

    def get_kvcache(self):
        return self

    def mixed_kv_enabled(self):
        return False

    def alloc_pages(self, n_pages):
        pages = [self.free_pages.pop(0) for _ in range(n_pages)]
        return [p * self.page_size + i for p in pages for i in range(self.page_size)]

    def free(self, free_index):
        ids = [int(x) for x in free_index.flatten().tolist()]
        for p in {s // self.page_size for s in ids}:
            if p in self.free_pages:
                self.double_freed.append(p)
            else:
                self.free_pages.append(p)

    def live_pages(self):
        return set(range(1, self.num_pages)) - set(self.free_pages)


class _MambaSlotLedger:
    """The mamba sub-allocator surface the MAMBA component and
    ``HybridReqToTokenPool`` use: slot ids only, with double-free detection."""

    def __init__(self, n):
        self.n = n
        self.free_slots = list(range(1, n))
        self.double_freed = []

    def alloc(self, k):
        if len(self.free_slots) < k:
            return None
        out, self.free_slots = self.free_slots[:k], self.free_slots[k:]
        return torch.tensor(out, dtype=torch.int64)

    def free(self, idx):
        for s in idx.flatten().tolist():
            s = int(s)
            if s in self.free_slots:
                self.double_freed.append(s)
            else:
                self.free_slots.append(s)

    def available_size(self):
        return len(self.free_slots)

    def live(self):
        return set(range(1, self.n)) - set(self.free_slots)


class _HybridRTT(HybridReqToTokenPool):
    """``HybridReqToTokenPool`` whose mamba pool is a slot ledger (no state
    tensors); the ping-pong bookkeeping is the real one."""

    def __init__(self, *, size, max_context_len, mamba_slots):
        ReqToTokenPool.__init__(
            self,
            size=size,
            max_context_len=max_context_len,
            device="cpu",
            enable_memory_saver=False,
        )
        self.mamba_ping_pong_track_buffer_size = 2
        self.enable_mamba_extra_buffer = True
        self.enable_mamba_extra_buffer_lazy = False
        self.mamba_allocator = _MambaSlotLedger(mamba_slots)
        self.mamba_pool = types.SimpleNamespace(replayssm_write_pos=None)
        self.req_index_to_mamba_index_mapping = torch.zeros(
            self._alloc_size, dtype=torch.int32
        )
        self.req_index_to_mamba_ping_pong_track_buffer_mapping = torch.full(
            (self._alloc_size, 2), -1, dtype=torch.int64
        )


# Pure copy of mem_cache.allocation._mixed_extend_layout_counts.
def _mixed_window_lengths(seq_len, hp_prefix_tokens, hp_recent_tokens):
    prefix_len = min(seq_len, hp_prefix_tokens)
    recent_len = min(max(seq_len - prefix_len, 0), hp_recent_tokens)
    return prefix_len, recent_len, seq_len - prefix_len - recent_len


def _mixed_extend_layout_counts(
    pre_len, seq_len, hp_prefix_tokens, hp_recent_tokens, n_q, is_final_chunk=True
):
    if is_final_chunk:
        prefix_keep, recent_keep, _ = _mixed_window_lengths(
            seq_len, hp_prefix_tokens, hp_recent_tokens
        )
        recent_start = seq_len - recent_keep
        hp_prefix_count = max(0, min(prefix_keep, seq_len) - pre_len)
        quant_count = max(0, recent_start - max(pre_len, prefix_keep))
        hp_recent_count = max(0, seq_len - max(pre_len, recent_start))
    else:
        prefix_keep = min(seq_len, hp_prefix_tokens)
        hp_prefix_count = max(0, min(prefix_keep, seq_len) - pre_len)
        quant_count = max(0, seq_len - max(pre_len, prefix_keep))
        hp_recent_count = 0
    quant_alloc_count = ceil_align(quant_count, n_q)
    counter_init = max(0, (hp_recent_tokens + n_q - 1) - hp_recent_count)
    return (
        hp_prefix_count,
        hp_recent_count,
        quant_count,
        quant_alloc_count,
        counter_init,
    )


# --------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------
class _Live:
    """Sim-side bookkeeping for one request (the ``Req`` stays untouched)."""

    __slots__ = ("req", "tokens", "seq_len", "admit_match_len")

    def __init__(self, req, tokens):
        self.req = req
        self.tokens = list(tokens)
        self.seq_len = 0
        self.admit_match_len = 0


def _new_req(rid, token_ids):
    return Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=array("q", token_ids),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
    )


class MixedSim:
    """Prefill / decode / finish driver over the real unified cache."""

    def __init__(
        self,
        *,
        hybrid: bool,
        disable: bool = False,
        clamp_prefill_checkpoint: bool = True,
    ):
        self.hybrid = hybrid
        # False models a scheduler that tracks the end-of-prefill checkpoint
        # (the pre-clamp behaviour / DCP), i.e. the FULL-only fallback path.
        self.clamp_prefill_checkpoint = clamp_prefill_checkpoint
        self.pool = _MixedPool()
        if hybrid:
            self.rtt = _HybridRTT(
                size=MAX_REQ, max_context_len=MAX_CTX, mamba_slots=MAMBA_SLOTS
            )
            components = (ComponentType.FULL, ComponentType.MAMBA)
        else:
            self.rtt = ReqToTokenPool(
                size=MAX_REQ,
                max_context_len=MAX_CTX,
                device="cpu",
                enable_memory_saver=False,
            )
            components = (ComponentType.FULL,)
        self.tree = UnifiedRadixCache(
            CacheInitParams(
                disable=disable,
                req_to_token_pool=self.rtt,
                token_to_kv_pool_allocator=self.pool,
                page_size=N_Q,
                tree_components=components,
                enable_mamba_extra_buffer=hybrid,
                tree_core_backend="python",
            )
        )
        self.live: dict[str, _Live] = {}
        self.admitted: list[_Live] = []
        self.violations: list[str] = []
        if hybrid:
            self.grid = mamba_checkpoint_grid(N_Q)
            self.chunk = mamba_cache_chunk_size()
            self.track_grid = mamba_track_grid(N_Q)
            # mamba slot -> (prefix signature, depth) of the state it holds
            self.mamba_shadow: dict[int, tuple[int, int]] = {}

    # -- shadow memory -----------------------------------------------------
    def _write(self, live, pos, slot):
        self.rtt.req_to_token[live.req.kv.req_pool_idx, pos] = slot
        self.pool.shadow[slot] = (live.tokens[pos], pos)

    def check_reads(self, live, tag):
        """Every live position must still read back its own token/position."""
        row = self.rtt.req_to_token[live.req.kv.req_pool_idx, : live.seq_len].tolist()
        for pos, slot in enumerate(row):
            got = self.pool.shadow.get(int(slot))
            want = (live.tokens[pos], pos)
            if got is None:
                self.violations.append(
                    f"{tag} rid={live.req.rid} pos={pos} slot={slot} never written"
                )
            elif got != want:
                self.violations.append(
                    f"{tag} rid={live.req.rid} pos={pos} slot={slot} holds {got}, "
                    f"expected {want}"
                )
            if len(self.violations) > 20:
                return

    @staticmethod
    def _sig(live, depth):
        return hash(tuple(live.tokens[:depth]))

    # -- scheduler admission -----------------------------------------------
    def _match_and_lock(self, live):
        req = live.req
        fill = req.full_untruncated_fill_ids
        # The scheduler matches all but the last token (``_compute_max_prefix_len``).
        key = RadixKey(token_ids=fill, extra_key=None, limit=max(len(fill) - 1, 0))
        m = self.tree.match_prefix(
            MatchPrefixParams(key=key, req=req, cow_mamba=self.hybrid)
        )
        req.prefix_indices = m.device_indices
        live.admit_match_len = len(m.device_indices)
        req.last_node = m.last_device_node
        req.last_host_node = m.last_host_node
        req.best_match_node = m.best_match_node
        req.kv.cache_protected_len = (
            m.cache_protected_len
            if m.cache_protected_len is not None
            else len(m.device_indices)
        )
        req.mamba_branching_seqlen = m.mamba_branching_seqlen
        req.lock_receipt = self.tree.inc_lock_ref(req.last_node).to_dec_params()
        if self.hybrid:
            self._check_mamba_cow(live)

    def _check_mamba_cow(self, live):
        req = live.req
        src = req.kv.mamba_cow_src_index
        if src is None:
            return
        depth = len(req.prefix_indices)
        got = self.mamba_shadow.get(int(src.item()))
        want = (self._sig(live, depth), depth)
        if got != want:
            self.violations.append(
                f"mamba COW rid={req.rid} src slot {int(src.item())} holds {got}, "
                f"expected the state at depth {depth}"
            )
        # The deferred COW: the request's own slot now holds that state.
        self.mamba_shadow[int(req.kv.mamba_pool_idx.item())] = want

    # -- prefill -------------------------------------------------------------
    def _extend(self, live, pre_len, seq_len, is_final):
        """One prefill chunk [pre_len, seq_len): production tier layout and
        slack rules."""
        req = live.req
        (
            hp_prefix_count,
            hp_recent_count,
            quant_count,
            quant_alloc_count,
            counter_init,
        ) = _mixed_extend_layout_counts(
            pre_len, seq_len, HP_PREFIX, HP_RECENT, N_Q, is_final
        )
        assert hp_prefix_count + quant_count + hp_recent_count == seq_len - pre_len
        if not is_final:
            tail_start = max(0, seq_len - HP_RECENT)
            cur = req.mixed_kv_quant_slack_cutoff_len
            req.mixed_kv_quant_slack_cutoff_len = (
                tail_start if cur is None else min(cur, tail_start)
            )
        hp_prefix_slots = self.pool.alloc_hp_prefix(ceil_align(hp_prefix_count, N_Q))
        if len(hp_prefix_slots) > hp_prefix_count:
            req.mixed_kv_quant_slack_indices = torch.cat(
                [
                    req.mixed_kv_quant_slack_indices,
                    torch.tensor(hp_prefix_slots[hp_prefix_count:], dtype=torch.int64),
                ]
            )
            cut = pre_len + (hp_prefix_count // N_Q) * N_Q
            cur = req.mixed_kv_quant_slack_cutoff_len
            req.mixed_kv_quant_slack_cutoff_len = cut if cur is None else min(cur, cut)
        quant_slots = self.pool.alloc_quant(quant_alloc_count)
        if quant_alloc_count > quant_count:
            req.mixed_kv_quant_slack_indices = torch.cat(
                [
                    req.mixed_kv_quant_slack_indices,
                    torch.tensor(quant_slots[quant_count:], dtype=torch.int64),
                ]
            )
            cut = max(pre_len, HP_PREFIX) + (quant_count // N_Q) * N_Q
            cur = req.mixed_kv_quant_slack_cutoff_len
            req.mixed_kv_quant_slack_cutoff_len = cut if cur is None else min(cur, cut)
        recent_slots = self.pool.alloc_hp_recent(req.kv.req_pool_idx, hp_recent_count)
        locs = (
            hp_prefix_slots[:hp_prefix_count] + quant_slots[:quant_count] + recent_slots
        )
        for pos, slot in enumerate(req.prefix_indices.tolist()):
            self.rtt.req_to_token[req.kv.req_pool_idx, pos] = int(slot)
        for i, slot in enumerate(locs):
            self._write(live, pre_len + i, slot)
        live.seq_len = seq_len
        req.kv.kv_committed_len = seq_len
        req.kv.kv_allocated_len = seq_len
        self.pool.flush_counter[req.kv.req_pool_idx] = counter_init
        if self.hybrid:
            self._mamba_prefill_track(live, pre_len, seq_len)

    def _mamba_prefill_track(self, live, pre_len, seq_end):
        """``_mamba_radix_cache_v2_req_prepare_for_extend`` (non-DCP): the
        tracked snapshot lands on the checkpoint grid relative to the prefix,
        clamped to the mixed-KV insert ceiling, or on the branching point the
        admission match asked for."""
        req = live.req
        extend_len = seq_end - pre_len
        live_slot = int(req.kv.mamba_pool_idx.item())
        self.mamba_shadow[live_slot] = (self._sig(live, seq_end), seq_end)
        mask = extend_len >= self.grid
        aligned = pre_len + (extend_len // self.grid) * self.grid
        if self.clamp_prefill_checkpoint:
            ceiling = mixed_kv_prefill_insert_ceiling(
                self.tree, req=req, seq_end=seq_end
            )
            assert ceiling is not None
            capped = pre_len
            if ceiling > pre_len:
                capped += ((ceiling - pre_len) // self.grid) * self.grid
            if capped <= pre_len:
                mask = False
            else:
                aligned = min(aligned, capped)
        if not mask:
            return
        req.kv.mamba_last_track_idx = req.kv.mamba_next_track_idx
        req.kv.mamba_next_track_idx = self.rtt.get_mamba_ping_pong_other_idx(
            req.kv.mamba_next_track_idx
        )
        branching = req.mamba_branching_seqlen
        if (
            branching is not None
            and pre_len < branching < seq_end
            and (branching - pre_len) % self.chunk == 0
        ):
            aligned = branching
        req.kv.mamba_last_track_seqlen = aligned
        slot = int(
            req.kv.mamba_ping_pong_track_buffer[req.kv.mamba_last_track_idx].item()
        )
        self.mamba_shadow[slot] = (self._sig(live, aligned), aligned)

    def admit(self, rid, token_ids, *, req=None, chunk_size=None):
        if req is None:
            req = _new_req(rid, token_ids)
            live = _Live(req, token_ids)
            self.live[rid] = live
        else:
            live = self.live[req.rid]
        req._refresh_fill_ids()
        self._match_and_lock(live)
        self.rtt.alloc([req])
        pre_len = len(req.prefix_indices)
        total = len(live.tokens)
        while True:
            end = total if chunk_size is None else min(total, pre_len + chunk_size)
            is_final = end >= total
            req.set_extend_range(pre_len, end)
            self._extend(live, pre_len, end, is_final)
            self.tree.cache_unfinished_req(req, chunked=not is_final)
            self.check_reads(live, f"after-chunk-{end}")
            if is_final:
                break
            pre_len = len(req.prefix_indices)
            assert pre_len == end, (pre_len, end)
        self.admitted.append(live)
        return req

    # -- decode --------------------------------------------------------------
    def decode_step(self, live, token_id):
        req = live.req
        rpi = req.kv.req_pool_idx
        seq_len = live.seq_len  # pre-increment, as ``locs = batch.seq_lens``
        prefix_len = int(req.kv.cache_protected_len)
        # per-request flush gate (RMW from _alloc_for_decode_mixed)
        counter = self.pool.flush_counter[rpi]
        do_flush = counter == 0
        self.pool.flush_counter[rpi] = N_Q - 1 if do_flush else counter - 1
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
            self.pool.shadow[dst[j]] = self.pool.shadow[src]
            self.rtt.req_to_token[rpi, fp] = dst[j]
            used.append(j)
        self.pool.free([dst[j] for j in range(N_Q) if j not in used])
        live.tokens.append(token_id)
        req.output_ids.append(token_id)
        live.seq_len += 1
        req.kv.kv_committed_len = live.seq_len
        req.kv.kv_allocated_len = live.seq_len
        self._write(live, seq_len, new_slot)
        if self.hybrid:
            self._mamba_decode_track(live)

    def _mamba_decode_track(self, live):
        req = live.req
        seq = live.seq_len
        self.mamba_shadow[int(req.kv.mamba_pool_idx.item())] = (
            self._sig(live, seq),
            seq,
        )
        if seq % self.track_grid != 0:
            return
        idx = req.kv.mamba_next_track_idx
        slot = int(req.kv.mamba_ping_pong_track_buffer[idx].item())
        self.mamba_shadow[slot] = (self._sig(live, seq), seq)
        req.kv.mamba_last_track_idx = idx
        req.kv.mamba_next_track_idx = self.rtt.get_mamba_ping_pong_other_idx(idx)
        req.kv.mamba_last_track_seqlen = seq

    # -- release ---------------------------------------------------------------
    def release(self, live, *, is_insert=True):
        req = live.req
        rpi = req.kv.req_pool_idx
        release_kv_cache(req, self.tree, is_insert=is_insert)
        assert self.pool.ring_cursor[rpi] == 0 and self.pool.flush_counter[rpi] == 0
        self.admitted.remove(live)
        if self.hybrid:
            assert req.kv.mamba_pool_idx is None
            assert req.kv.mamba_ping_pong_track_buffer is None

    # -- accounting --------------------------------------------------------------
    def tree_values(self):
        if not self.tree.root_node.children:
            return []
        return [int(v) for v in self.tree.all_values_flatten().tolist()]

    def tree_pages(self):
        vals = self.tree_values()
        q = {v // N_Q for v in vals if v < HP_OFFSET}
        h = {(v - HP_OFFSET) // N_Q for v in vals if HP_OFFSET <= v < HP_RECENT_BASE}
        return q, h

    def tree_mamba_states(self):
        return {int(v) for v in self.tree.all_mamba_values_flatten().tolist()}

    def assert_no_leak(self):
        """Every live quant / HP-prefix page is either in the tree, addressed by
        a live request past its watermark, or that request's partial-page
        slack -- and nothing is counted twice."""
        q, h = self.tree_pages()
        for live in self.admitted:
            req = live.req
            row = self.rtt.req_to_token[
                req.kv.req_pool_idx, req.kv.cache_protected_len : live.seq_len
            ].tolist()
            for s in row + req.mixed_kv_quant_slack_indices.tolist():
                s = int(s)
                if s < HP_OFFSET:
                    q.add(s // N_Q)
                elif s < HP_RECENT_BASE:
                    h.add((s - HP_OFFSET) // N_Q)
        assert self.pool.live_quant_pages() == q, (
            "quant pages held by nobody: "
            f"{sorted(self.pool.live_quant_pages() - q)[:8]} / referenced pages "
            f"already free: {sorted(q - self.pool.live_quant_pages())[:8]}"
        )
        assert self.pool.live_hp_prefix_pages() == h, (
            f"hp-prefix pages held by nobody: "
            f"{sorted(self.pool.live_hp_prefix_pages() - h)[:8]} / referenced pages "
            f"already free: {sorted(h - self.pool.live_hp_prefix_pages())[:8]}"
        )
        assert (
            not self.pool.double_freed
        ), f"double-freed pages {self.pool.double_freed[:8]}"
        # (f) the tree holds quant slots only, so recoverable == evictable.
        assert self.tree.recoverable_size() == self.tree.evictable_size()
        if self.hybrid:
            self.assert_mamba_balanced()

    def assert_mamba_balanced(self):
        held = set()
        for live in self.admitted:
            kv = live.req.kv
            held.add(int(kv.mamba_pool_idx.item()))
            held.update(int(x) for x in kv.mamba_ping_pong_track_buffer.tolist())
        tree_states = self.tree_mamba_states()
        assert held.isdisjoint(tree_states), held & tree_states
        assert self.rtt.mamba_allocator.live() == held | tree_states, (
            sorted(self.rtt.mamba_allocator.live() - (held | tree_states)),
            sorted((held | tree_states) - self.rtt.mamba_allocator.live()),
        )
        assert (
            not self.rtt.mamba_allocator.double_freed
        ), self.rtt.mamba_allocator.double_freed

    def evict_all_and_check(self):
        self.tree.evict(EvictParams(num_tokens=10**9, mamba_num=10**9))
        assert not self.tree.root_node.children
        assert not self.pool.live_quant_pages(), sorted(self.pool.live_quant_pages())[
            :8
        ]
        assert not self.pool.live_hp_prefix_pages()
        assert not self.pool.double_freed, self.pool.double_freed[:8]
        assert self.tree.evictable_size() == 0 and self.tree.protected_size() == 0
        if self.hybrid:
            assert not self.rtt.mamba_allocator.live()
            assert not self.rtt.mamba_allocator.double_freed


class PlainSim:
    """Stock (non-mixed) ``UnifiedRadixCache`` over a paged allocator model."""

    def __init__(self, page_size):
        self.ps = page_size
        self.pool = _PlainPool(page_size)
        self.rtt = ReqToTokenPool(
            size=MAX_REQ,
            max_context_len=MAX_CTX,
            device="cpu",
            enable_memory_saver=False,
        )
        self.tree = UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=self.rtt,
                token_to_kv_pool_allocator=self.pool,
                page_size=page_size,
                tree_components=(ComponentType.FULL,),
                tree_core_backend="python",
            )
        )
        assert not self.tree._mixed_kv_enabled
        self.violations = []
        self.live = {}

    def _write(self, live, pos, slot):
        self.rtt.req_to_token[live.req.kv.req_pool_idx, pos] = slot
        self.pool.shadow[slot] = (live.tokens[pos], pos)

    def check_reads(self, live, tag):
        row = self.rtt.req_to_token[live.req.kv.req_pool_idx, : live.seq_len].tolist()
        for pos, slot in enumerate(row):
            got = self.pool.shadow.get(int(slot))
            if got != (live.tokens[pos], pos):
                self.violations.append(
                    f"{tag} rid={live.req.rid} pos={pos} slot={slot} holds {got}"
                )

    def admit(self, rid, token_ids):
        req = _new_req(rid, token_ids)
        live = _Live(req, token_ids)
        self.live[rid] = live
        req._refresh_fill_ids()
        fill = req.full_untruncated_fill_ids
        key = RadixKey(token_ids=fill, extra_key=None, limit=len(fill) - 1)
        m = self.tree.match_prefix(MatchPrefixParams(key=key, req=req))
        req.prefix_indices = m.device_indices
        req.last_node = m.last_device_node
        req.kv.cache_protected_len = len(m.device_indices)
        req.lock_receipt = self.tree.inc_lock_ref(req.last_node).to_dec_params()
        self.rtt.alloc([req])
        pre_len = len(req.prefix_indices)
        seq_len = len(live.tokens)
        assert pre_len % self.ps == 0
        req.set_extend_range(pre_len, seq_len)
        slots = self.pool.alloc_pages(ceil_align(seq_len - pre_len, self.ps) // self.ps)
        for pos, slot in enumerate(req.prefix_indices.tolist()):
            self.rtt.req_to_token[req.kv.req_pool_idx, pos] = int(slot)
        for i in range(seq_len - pre_len):
            self._write(live, pre_len + i, slots[i])
        live.seq_len = seq_len
        req.kv.kv_committed_len = req.kv.kv_allocated_len = seq_len
        self.tree.cache_unfinished_req(req)
        self.check_reads(live, "after-prefill")
        return live

    def decode_step(self, live, token_id):
        req = live.req
        seq_len = live.seq_len
        if seq_len % self.ps == 0:
            slot = self.pool.alloc_pages(1)[0]
        else:
            slot = int(self.rtt.req_to_token[req.kv.req_pool_idx, seq_len - 1]) + 1
        live.tokens.append(token_id)
        req.output_ids.append(token_id)
        live.seq_len += 1
        req.kv.kv_committed_len = req.kv.kv_allocated_len = live.seq_len
        self._write(live, seq_len, slot)

    def release(self, live, *, is_insert=True):
        release_kv_cache(live.req, self.tree, is_insert=is_insert)

    def tree_pages(self):
        if not self.tree.root_node.children:
            return set()
        return {int(v) // self.ps for v in self.tree.all_values_flatten().tolist()}

    def assert_no_leak(self):
        assert self.pool.live_pages() == self.tree_pages(), (
            sorted(self.pool.live_pages() - self.tree_pages())[:8],
            sorted(self.tree_pages() - self.pool.live_pages())[:8],
        )
        assert not self.pool.double_freed, self.pool.double_freed[:8]


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------
SHARED = list(range(1000, 1000 + 96))  # 96-token shared instruction prefix


def _prompt(seed, length):
    body = [(seed * 7919 + i) % 30000 + 2 for i in range(length - len(SHARED))]
    return SHARED + body


def _tier_counts(sim, live):
    """(hp_prefix, quant, hp_recent) position counts as the reader sees them."""
    row = sim.rtt.req_to_token[live.req.kv.req_pool_idx, : live.seq_len].tolist()
    hp_prefix = sum(1 for s in row if HP_OFFSET <= s < HP_RECENT_BASE)
    recent = sum(1 for s in row if s >= HP_RECENT_BASE)
    return hp_prefix, live.seq_len - hp_prefix - recent, recent


def _run_mixed(hybrid, prompts, gen_tokens, check_every=32, chunk_size=None):
    sim = MixedSim(hybrid=hybrid)
    for i, p in enumerate(prompts):
        sim.admit(f"r{i}", p, chunk_size=chunk_size)
    lives = list(sim.admitted)
    for step in range(gen_tokens):
        for live in lives:
            sim.decode_step(live, 5000 + step)
        if step % check_every == 0:
            for live in lives:
                sim.check_reads(live, f"decode-step{step}")
            if sim.violations:
                break
    for live in lives:
        sim.check_reads(live, "before-finish")
    if not sim.violations:
        sim.assert_no_leak()
    for live in lives:
        sim.release(live)
    return sim


@pytest.fixture(params=[False, True], ids=["full", "full+mamba"])
def hybrid(request):
    return request.param


def test_cached_prefix_does_not_cannibalize_hp_recent(hybrid):
    """(a) A borrowed prefix must not eat the borrower's BF16 HP-recent window
    (the multi-turn shape behind the Qwen3-8B BFCL 38.4 -> 14.6 collapse)."""
    sim = MixedSim(hybrid=hybrid)
    sim.admit("long", _prompt(3, 1400))
    sim.admit("short", _prompt(3, 300))
    short = sim.live["short"]
    _, _, recent = _tier_counts(sim, short)
    want = min(HP_RECENT, short.seq_len - HP_PREFIX)
    assert recent >= want, (recent, want, short.req.kv.cache_protected_len)
    assert not sim.violations, "\n".join(sim.violations[:8])


def test_protected_len_stays_below_hp_recent_start(hybrid):
    """(a)+(e) ``cache_protected_len`` is the flush kernel's ``prefix_len``; it
    must never exceed the request's own HP-recent start, stay page-aligned, and
    the ``prefix_indices`` rebuild must cover the whole row."""
    sim = MixedSim(hybrid=hybrid)
    sim.admit("long", _prompt(3, 1400))
    sim.admit("mid", _prompt(3, 700))
    sim.admit("short", _prompt(3, 300))
    for live in sim.admitted:
        recent_start = max(HP_PREFIX, live.seq_len - HP_RECENT)
        protected = live.req.kv.cache_protected_len
        assert protected <= recent_start, (live.req.rid, protected, recent_start)
        assert protected % N_Q == 0
        assert len(live.req.prefix_indices) == live.seq_len
    assert not sim.violations, "\n".join(sim.violations[:8])


def test_match_cap_and_bypass():
    """(a) ``match_prefix`` caps at the tier cap; ``bypass_mixed_kv_cap`` returns
    the tree's real coverage, which still never contains a ring slot (b)."""
    sim = MixedSim(hybrid=False)
    donor = sim.admit("donor", _prompt(3, 1400))
    key = RadixKey(token_ids=array("q", _prompt(3, 700)), extra_key=None)
    capped = sim.tree.match_prefix(MatchPrefixParams(key=key))
    cap = sim.tree._mixed_kv_tier_cap(700)
    assert len(capped.device_indices) <= cap, (len(capped.device_indices), cap)
    full = sim.tree.match_prefix(MatchPrefixParams(key=key, bypass_mixed_kv_cap=True))
    assert len(full.device_indices) >= len(capped.device_indices)
    assert len(full.device_indices) == min(
        700 // N_Q * N_Q, donor.kv.cache_protected_len
    )
    assert all(int(v) < HP_RECENT_BASE for v in full.device_indices.tolist())


def test_shared_prefix_does_not_corrupt_kv(hybrid):
    """Shadow-memory check over a mixed batch: no position ever reads foreign KV,
    no page is held outside the tree or freed twice."""
    sim = _run_mixed(
        hybrid, [_prompt(1, 200), _prompt(2, 200), _prompt(1, 900)], gen_tokens=600
    )
    assert not sim.violations, "\n".join(sim.violations[:8])
    sim.assert_no_leak()
    sim.evict_all_and_check()


def test_long_prompt_prefix_reuse_is_consistent(hybrid):
    """Long prompts (quant middle in the tree) plus a short sibling borrowing a
    cross-tier prefix."""
    sim = _run_mixed(
        hybrid, [_prompt(3, 1400), _prompt(3, 1400), _prompt(3, 300)], gen_tokens=400
    )
    assert not sim.violations, "\n".join(sim.violations[:8])
    sim.assert_no_leak()
    sim.evict_all_and_check()


def test_no_ring_slot_enters_the_tree(hybrid):
    """(b) HP-recent ids are per-request and recycled; they must never be cached."""
    sim = _run_mixed(hybrid, [_prompt(4, 300), _prompt(5, 300)], gen_tokens=300)
    bad = [v for v in sim.tree_values() if v >= HP_RECENT_BASE]
    assert not bad, bad[:8]


def test_chunked_prefill_keeps_radix_below_cutoff(hybrid):
    """(c) Radix ownership stays below the request-owned partial-page cutoff
    across chunks, and a sibling behind the chunked donor keeps its window."""
    sim = MixedSim(hybrid=hybrid)
    req = sim.admit("chunked", _prompt(6, 1700), chunk_size=640)
    assert not sim.violations, "\n".join(sim.violations[:8])
    cut = req.mixed_kv_quant_slack_cutoff_len
    assert cut is not None and req.kv.cache_protected_len <= cut, (
        req.kv.cache_protected_len,
        cut,
    )
    sibling = sim.admit("sib", _prompt(6, 1700))
    assert sibling.kv.cache_protected_len <= max(
        HP_PREFIX, sim.live["sib"].seq_len - HP_RECENT
    )
    for step in range(300):
        for live in list(sim.admitted):
            sim.decode_step(live, 7000 + step)
    for live in list(sim.admitted):
        sim.check_reads(live, "chunked-decode")
    assert not sim.violations, "\n".join(sim.violations[:8])
    sim.assert_no_leak()
    for live in list(sim.admitted):
        sim.release(live)
    sim.assert_no_leak()
    sim.evict_all_and_check()


def test_retract_and_readmit_under_eviction(hybrid):
    """(c)+(d) A retracted request drops its slack without a second free, the
    tree holds only what ``cache_unfinished_req`` put there, and eviction under
    retract pressure never frees a slot a live request still references."""
    sim = MixedSim(hybrid=hybrid)
    sim.admit("a", _prompt(3, 1400))
    b_req = sim.admit("b", _prompt(3, 600))
    a, b = sim.live["a"], sim.live["b"]
    for step in range(100):
        for live in list(sim.admitted):
            sim.decode_step(live, 8000 + step)
    sim.check_reads(a, "pre-retract")
    sim.check_reads(b, "pre-retract")
    assert not sim.violations, "\n".join(sim.violations[:8])
    sim.release(b, is_insert=False)
    assert b_req.mixed_kv_quant_slack_indices.numel() == 0
    assert b_req.mixed_kv_quant_slack_cutoff_len is None
    b_req.reset_for_retract()
    sim.tree.evict(EvictParams(num_tokens=10**9, mamba_num=10**9))
    assert not sim.pool.double_freed, sim.pool.double_freed[:8]
    sim.check_reads(a, "after-evict")
    assert not sim.violations, "\n".join(sim.violations[:8])
    sim.admit("b", None, req=b_req)
    for step in range(100):
        for live in list(sim.admitted):
            sim.decode_step(live, 9000 + step)
    for live in list(sim.admitted):
        sim.check_reads(live, "post-readmit")
    assert not sim.violations, "\n".join(sim.violations[:8])
    sim.assert_no_leak()
    for live in list(sim.admitted):
        sim.release(live)
    sim.assert_no_leak()
    sim.evict_all_and_check()


def test_repeated_cache_unfinished_is_monotonic(hybrid):
    """(e) A second ``cache_unfinished_req`` with no new KV leaves ownership, the
    lock anchor and the row untouched."""
    sim = MixedSim(hybrid=hybrid)
    req = sim.admit("rep", _prompt(8, 900))
    before = req.kv.cache_protected_len
    before_node = req.last_node
    sim.tree.cache_unfinished_req(req, chunked=True)
    assert req.kv.cache_protected_len == before
    assert req.last_node == before_node
    assert len(req.prefix_indices) == sim.live["rep"].seq_len
    sim.check_reads(sim.live["rep"], "repeat")
    assert not sim.violations
    assert not sim.pool.double_freed
    sim.assert_no_leak()
    sim.release(sim.live["rep"])
    sim.assert_no_leak()


def test_finish_does_not_populate_tree_and_forgets_slack(hybrid):
    """(d)+(c) Under mixed KV ``insert_req`` inserts nothing (the decode tail
    stays request-owned and is freed), the lock is dropped, and the slack is
    forgotten rather than freed again."""
    sim = MixedSim(hybrid=hybrid)
    req = sim.admit("fin", _prompt(9, 900))
    live = sim.live["fin"]
    assert req.mixed_kv_quant_slack_indices.numel() > 0
    for step in range(50):
        sim.decode_step(live, 4000 + step)
    values_before = sorted(sim.tree_values())
    states_before = sim.tree_mamba_states() if hybrid else set()
    sim.release(live, is_insert=True)
    assert sorted(sim.tree_values()) == values_before
    if hybrid:
        assert sim.tree_mamba_states() == states_before
    assert req.mixed_kv_quant_slack_indices.numel() == 0
    assert req.mixed_kv_quant_slack_cutoff_len is None
    assert sim.tree.protected_size() == 0
    sim.assert_no_leak()
    sim.evict_all_and_check()


def _expected_prefill_checkpoint(sim, req, seq_end, prefix_len=0):
    """The scheduler's clamped checkpoint: the deepest grid position at or
    below the mixed-KV insert ceiling."""
    ceiling = mixed_kv_prefill_insert_ceiling(sim.tree, req=req, seq_end=seq_end)
    return prefix_len + ((ceiling - prefix_len) // sim.grid) * sim.grid


def test_first_request_donates_full_and_mamba_at_ceiling():
    """A fresh prefill's checkpoint is clamped to the insert ceiling, so the
    very first request donates FULL + mamba and its immediate repeat hits the
    whole shareable prefix with the state at exactly that depth."""
    sim = MixedSim(hybrid=True)
    first = sim.admit("first", _prompt(3, 1400))
    checkpoint = _expected_prefill_checkpoint(sim, first, 1400)
    assert HP_PREFIX < checkpoint <= sim.tree._mixed_kv_tier_cap(1399)
    assert first.kv.cache_protected_len == checkpoint
    assert len(sim.tree_mamba_states()) == 1
    sim.admit("repeat", _prompt(3, 1400))
    assert sim.live["repeat"].admit_match_len == checkpoint
    assert not sim.violations, "\n".join(sim.violations[:8])
    _, _, recent = _tier_counts(sim, sim.live["repeat"])
    assert recent >= HP_RECENT
    sim.assert_no_leak()
    for live in list(sim.admitted):
        sim.release(live)
    sim.assert_no_leak()
    sim.evict_all_and_check()


def test_short_prompt_repeat_hits_hp_prefix_window():
    """The GPU smoke shape: a prompt shorter than hp_prefix + hp_recent, sent
    twice, must hit the HP-prefix window on the repeat (not 0 cached tokens)."""
    sim = MixedSim(hybrid=True)
    first = sim.admit("first", _prompt(5, 253))
    assert _expected_prefill_checkpoint(sim, first, 253) == HP_PREFIX
    assert first.kv.cache_protected_len == HP_PREFIX
    assert len(sim.tree_mamba_states()) == 1
    sim.admit("repeat", _prompt(5, 253))
    assert sim.live["repeat"].admit_match_len == HP_PREFIX
    assert not sim.violations, "\n".join(sim.violations[:8])
    sim.assert_no_leak()
    for live in list(sim.admitted):
        sim.release(live)
    sim.evict_all_and_check()


def test_mamba_checkpoint_past_ceiling_is_not_donated():
    """Fallback path (an unclamped end-of-prefill checkpoint, as DCP tracks):
    the FULL KV below the window is cached alone and the state stays with the
    request."""
    sim = MixedSim(hybrid=True, clamp_prefill_checkpoint=False)
    req = sim.admit("first", _prompt(3, 1400))
    live = sim.live["first"]
    key_len = len(live.tokens) // N_Q * N_Q
    assert req.kv.cache_protected_len == sim.tree._mixed_kv_insert_ceiling(req, key_len)
    assert req.kv.cache_protected_len > 0
    assert sim.tree_mamba_states() == set()
    assert req.kv.mamba_last_track_seqlen is None
    assert all(int(x) > 0 for x in req.kv.mamba_ping_pong_track_buffer.tolist())
    sim.assert_no_leak()
    sim.release(live)
    sim.assert_no_leak()
    sim.evict_all_and_check()


def test_mamba_bootstrap_via_branching_point():
    """Fallback path: a FULL-only cache yields a branching point inside the
    shareable region; the next request checkpoints there and later requests hit
    FULL + mamba with the state at exactly that depth."""
    sim = MixedSim(hybrid=True, clamp_prefill_checkpoint=False)
    sim.admit("donor", _prompt(3, 1400))
    borrower = sim.admit("borrower", _prompt(3, 700))
    cap = sim.tree._mixed_kv_tier_cap(699)
    expected_branch = cap // sim.grid * sim.grid
    assert borrower.mamba_branching_seqlen == expected_branch
    assert len(sim.tree_mamba_states()) == 1
    assert borrower.kv.cache_protected_len >= expected_branch
    sim.admit("third", _prompt(3, 700))
    assert sim.live["third"].admit_match_len == expected_branch
    assert not sim.violations, "\n".join(sim.violations[:8])
    _, _, recent = _tier_counts(sim, sim.live["third"])
    assert recent >= min(HP_RECENT, 700 - HP_PREFIX)
    sim.assert_no_leak()
    for live in list(sim.admitted):
        sim.release(live)
    sim.assert_no_leak()
    sim.evict_all_and_check()


def test_mutation_without_tier_cap_borrower_loses_hp_recent():
    """Guard value of the cap tests: with the cap disabled the borrower's BF16
    window collapses, so the assertions above would catch its loss."""
    sim = MixedSim(hybrid=False)
    sim.tree._mixed_kv_tier_cap = lambda key_len: key_len
    sim.admit("long", _prompt(3, 1400))
    sim.admit("short", _prompt(3, 300))
    short = sim.live["short"]
    _, _, recent = _tier_counts(sim, short)
    assert recent < min(HP_RECENT, short.seq_len - HP_PREFIX)


def test_plain_pool_keeps_upstream_semantics():
    """Plain pools: every mixed-KV hook is a no-op, so finished requests are
    inserted and the watermark never regresses."""
    for ps in (1, 8):
        sim = PlainSim(ps)
        lives = [
            sim.admit(f"p{i}", p)
            for i, p in enumerate([_prompt(1, 200), _prompt(1, 500), _prompt(2, 333)])
        ]
        for step in range(150):
            for live in lives:
                sim.decode_step(live, 6000 + step)
        for live in lives:
            sim.check_reads(live, f"ps{ps}")
        assert not sim.violations, "\n".join(sim.violations[:8])
        for live in lives:
            before = live.req.kv.cache_protected_len
            sim.release(live)
            assert live.req.kv.cache_protected_len >= before
        sim.assert_no_leak()
        assert sim.tree.evictable_size() > 0
        sim.tree.evict(EvictParams(num_tokens=10**9))
        assert not sim.pool.live_pages() and not sim.pool.double_freed


def test_plain_pool_retract_path():
    sim = PlainSim(8)
    live = sim.admit("x", _prompt(1, 300))
    for step in range(20):
        sim.decode_step(live, 100 + step)
    sim.release(live, is_insert=False)
    sim.assert_no_leak()


# --------------------------------------------------------------------------
# Registry routing (mixed KV x hybrid SSM x extra buffer)
# --------------------------------------------------------------------------
def _registry_ctx(*, allocator, is_hybrid_ssm, extra_buffer=True):
    override = get_context().override_server_args(
        radix_cache_backend=None,
        enable_streaming_session=False,
        enable_lmcache=False,
        enable_flexkv=False,
        enable_unified_cache_external_linker=False,
    )
    override.install()
    params = MagicMock()
    params.token_to_kv_pool_allocator = allocator
    params.enable_mamba_extra_buffer = extra_buffer
    params.page_size = N_Q
    params.req_to_token_pool = types.SimpleNamespace()
    params.component_registry_override = None
    ctx = TreeCacheBuildContext(
        server_args=get_server_args(),
        params=params,
        is_hybrid_swa=False,
        is_hybrid_ssm=is_hybrid_ssm,
        enable_hierarchical_cache=False,
        disable_radix_cache=False,
        effective_chunked_prefill_size=None,
        tp_worker=MagicMock(),
        model_config=MagicMock(),
        tp_size=1,
        tp_rank=0,
        tp_group=MagicMock(),
    )
    return ctx, override


def test_registry_routes_hybrid_ssm_mixed_kv_to_unified_cache():
    ctx, override = _registry_ctx(allocator=_MixedPool(), is_hybrid_ssm=True)
    fake_components = MagicMock()
    fake_radix = MagicMock()
    try:
        with patch.dict(
            "sys.modules",
            {
                "sglang.srt.mem_cache.unified_cache.components": fake_components,
                "sglang.srt.mem_cache.unified_radix_cache": fake_radix,
            },
        ):
            result = default_radix_cache_factory(ctx)
    finally:
        override.restore()
    fake_radix.UnifiedRadixCache.assert_called_once_with(ctx.params)
    assert result is fake_radix.UnifiedRadixCache.return_value
    assert ctx.params.tree_components == (
        fake_components.ComponentType.FULL,
        fake_components.ComponentType.MAMBA,
    )


def test_registry_requires_mamba_extra_buffer_for_hybrid_mixed_kv():
    ctx, override = _registry_ctx(
        allocator=_MixedPool(), is_hybrid_ssm=True, extra_buffer=False
    )
    try:
        with pytest.raises(ValueError, match="extra buffer"):
            default_radix_cache_factory(ctx)
    finally:
        override.restore()


def test_registry_keeps_non_hybrid_mixed_kv_on_radix_cache():
    ctx, override = _registry_ctx(allocator=_MixedPool(), is_hybrid_ssm=False)
    try:
        with patch("sglang.srt.mem_cache.radix_cache.RadixCache") as radix_cache:
            radix_cache.return_value = MagicMock()
            result = default_radix_cache_factory(ctx)
    finally:
        override.restore()
    radix_cache.assert_called_once_with(ctx.params)
    assert result is radix_cache.return_value


def test_registry_probe_ignores_mock_pools():
    """A ``Mock`` answers every probe with a truthy ``Mock``; only a literal
    ``True`` from ``mixed_kv_enabled`` selects the mixed-KV route."""
    ctx, override = _registry_ctx(allocator=MagicMock(), is_hybrid_ssm=True)
    fake_components = MagicMock()
    fake_radix = MagicMock()
    try:
        with patch.dict(
            "sys.modules",
            {
                "sglang.srt.mem_cache.unified_cache.components": fake_components,
                "sglang.srt.mem_cache.unified_radix_cache": fake_radix,
            },
        ):
            default_radix_cache_factory(ctx)
    finally:
        override.restore()
    # Not the mixed-KV route: plain hybrid SSM gets FULL + MAMBA too, but the
    # extra-buffer requirement would have raised above with extra_buffer left
    # at its mock default had the probe fired -- so assert on that directly.
    ctx2, override2 = _registry_ctx(
        allocator=MagicMock(), is_hybrid_ssm=True, extra_buffer=False
    )
    try:
        with patch.dict(
            "sys.modules",
            {
                "sglang.srt.mem_cache.unified_cache.components": fake_components,
                "sglang.srt.mem_cache.unified_radix_cache": fake_radix,
            },
        ):
            default_radix_cache_factory(ctx2)  # no ValueError: not mixed KV
    finally:
        override2.restore()
