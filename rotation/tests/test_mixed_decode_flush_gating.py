"""CPU: the host-side flush countdown reproduces the device formula the
decode step used to run every step (flush when the counter is 0 and restart
at N_Q - 1, else count down), per request row, across admission re-seeds
and slab release."""
import types

import torch

from sglang.srt.mem_cache.unified_kv_pool import UnifiedInt2HPKVPool


def _pool(n_req=6, n_q=4):
    # only the pieces advance/seed/release touch
    p = UnifiedInt2HPKVPool.__new__(UnifiedInt2HPKVPool)
    p.flush_interval = n_q
    p.max_req_slots = n_req
    p._flush_counter_host = [0] * n_req
    p._next_slab_offset = torch.zeros(n_req, dtype=torch.int32)
    p._flush_stage = [torch.empty((n_req, 3), dtype=torch.int64) for _ in range(2)]
    p._flush_stage_i = 0
    return p


def _device_formula(counters, rpi, fi):
    c = counters[rpi]
    flush = c == 0
    new = torch.where(flush, torch.full_like(c, fi - 1), c - 1)
    counters[rpi] = new
    return flush


def test_host_countdown_matches_device_formula():
    n_q = 4
    p = _pool(n_q=n_q)
    p.seed_flush_counters([0, 2, 5], [3, 0, 1])
    dev = torch.zeros(6, dtype=torch.int32)
    dev[[0, 2, 5]] = torch.tensor([3, 0, 1], dtype=torch.int32)
    rpi = torch.tensor([5, 0, 2])  # batch rows in some order
    for _step in range(13):
        rows = p.advance_flush_counters(rpi.tolist())
        flush = _device_formula(dev, rpi, n_q)
        assert rows == torch.nonzero(flush).flatten().tolist()
        assert p._flush_counter_host == dev.tolist()


def test_release_and_reseed_reset_the_countdown():
    p = _pool(n_q=4)
    p.seed_flush_counters([1], [2])
    assert p.advance_flush_counters([1]) == []
    p.release_req_slab(1)
    assert p._flush_counter_host[1] == 0
    p.release_req_slab(torch.tensor([1]))
    # a fresh occupant flushes on its first step when seeded to 0
    p.seed_flush_counters([1], [0])
    assert p.advance_flush_counters([1]) == [0]
    assert p._flush_counter_host[1] == 3


def test_stage_flush_rows_packs_the_three_columns():
    p = _pool()
    req, seq, pre = p.stage_flush_rows([4, 1], [700, 33], [64, 0], "cpu")
    assert req.tolist() == [4, 1] and req.dtype == torch.int64
    assert seq.tolist() == [700, 33] and seq.dtype == torch.int32
    assert pre.tolist() == [64, 0] and pre.dtype == torch.int32
    assert p._flush_stage_i == 1
