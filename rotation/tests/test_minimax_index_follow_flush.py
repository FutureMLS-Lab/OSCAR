"""The MiniMax indexer keys must follow a decode flush: a flushed token's
req_to_token entry moves from its window slot to a quant slot, and the index
row written at the window slot has to move with it (GPQA@64K 56.6 vs BF16 90.9
when it did not)."""
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.minimax_int2_kv_pool import follow_flush


def test_follow_flush_moves_demoted_rows_and_nothing_else():
    layers, n_slots, dim, hp_offset = 3, 40, 4, 24  # quant [0, 24), window [24, 40)
    cache = torch.arange(layers * n_slots * dim, dtype=torch.float32).reshape(
        layers, n_slots, 1, dim
    )
    before = cache.clone()
    # bs=2, flush_interval=2: request 0 demotes window slots 24, 25 into quant
    # slots 0, 1; request 1 does not flush this step (its quant slots 2, 3 are
    # returned to the allocator untouched).
    plan = SimpleNamespace(
        valid_mask=torch.tensor([1, 1, 0, 0], dtype=torch.int8),
        src_hp_slot=torch.tensor([0, 1, -1, -1], dtype=torch.int64),
        dst_quant_slots=torch.tensor([0, 1, 2, 3], dtype=torch.int64),
    )
    follow_flush(cache, plan, hp_offset)
    assert torch.equal(cache[:, 0], before[:, 24])
    assert torch.equal(cache[:, 1], before[:, 25])
    assert torch.equal(cache[:, 2:], before[:, 2:])


def test_follow_flush_is_a_no_op_without_a_plan_or_layers():
    cache = torch.zeros((0, 8, 1, 4))
    follow_flush(cache, None, 4)
    follow_flush(cache, SimpleNamespace(valid_mask=torch.ones(2, dtype=torch.int8),
                                        src_hp_slot=torch.zeros(2, dtype=torch.int64),
                                        dst_quant_slots=torch.zeros(2, dtype=torch.int64)), 4)
