"""The MiniMax INT2 staging tables must route every live (request, position)
the sparse kernels read to the row holding that slot's K/V, and a -1 block
(clamped to block 0) must never overwrite a live block-0 entry."""

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "python"))

from sglang.srt.layers.attention.minimax_sparse_staging import (  # noqa: E402
    build_slot_to_ragged,
    decode_block_rows,
    fill_decode_fake_table,
    prefill_fake_table,
)

BLOCK = 8  # small block keeps the tables readable; the arithmetic is size-agnostic


def _page_table(seq_lens, max_ctx, hp_offset, seed=0):
    # Distinct slots per token: quant slots early, window (>= hp_offset) slots
    # for the last three positions, like the mixed pool.
    g = torch.Generator().manual_seed(seed)
    rtt = torch.zeros((len(seq_lens), max_ctx), dtype=torch.int32)
    perm = torch.randperm(10_000, generator=g)
    k = 0
    for r, L in enumerate(seq_lens):
        for p in range(L):
            slot = int(perm[k])
            k += 1
            if p >= L - 3:
                slot = hp_offset + slot
            rtt[r, p] = slot
    return rtt


def test_prefill_fake_table_and_slot_map():
    seq_lens = [5, 13, 1]
    hp_offset = 20_000
    rtt = _page_table(seq_lens, max_ctx=16, hp_offset=hp_offset)
    req_pool_indices = torch.tensor([2, 0, 1])  # requests in a scrambled order
    lens = [seq_lens[int(i)] for i in req_pool_indices]
    flat = torch.cat([rtt[int(r), :L] for r, L in zip(req_pool_indices, lens)])
    cu_k = torch.zeros(len(lens) + 1, dtype=torch.int32)
    cu_k[1:] = torch.cumsum(torch.tensor(lens, dtype=torch.int32), 0)
    fake = prefill_fake_table(cu_seqlens_k=cu_k, max_seqlen_k=max(lens))
    assert fake.shape == (3, max(lens)) and fake.dtype == torch.int32
    for b, (r, L) in enumerate(zip(req_pool_indices, lens)):
        for pos in range(L):
            assert int(flat[int(fake[b, pos])]) == int(rtt[int(r), pos])

    slot_to_ragged = torch.full((hp_offset + 10_001,), -1, dtype=torch.int32)
    build_slot_to_ragged(flat_slots=flat, slot_to_ragged=slot_to_ragged)
    # This forward's own tokens: the last two of every request (window tier).
    own = torch.cat([rtt[int(r), L - 2 : L] for r, L in zip(req_pool_indices, lens)])
    rows = slot_to_ragged[own.to(torch.int64)]
    assert (rows >= 0).all()
    assert torch.equal(flat[rows.to(torch.int64)], own)


def test_decode_block_rows_and_fake_table():
    max_ctx = 64
    seq_lens = torch.tensor([37, 9, 1])  # 9 half-fills block 1; 1 is a fresh request
    hp_offset = 20_000
    rtt = _page_table(seq_lens.tolist(), max_ctx=max_ctx, hp_offset=hp_offset)
    req_pool_indices = torch.tensor([0, 1, 2])
    topk = 4
    # Block 0 is selected by every request, and every request also carries -1
    # entries (which clamp to block 0): the clobber case.
    topk_blk = torch.tensor(
        [[4, 0, 2, -1], [1, 0, -1, -1], [0, -1, -1, -1]], dtype=torch.int32
    )
    ar = torch.arange(BLOCK, dtype=torch.int64)
    slots, pos, valid = decode_block_rows(
        topk_blk=topk_blk,
        req_to_token=rtt,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        block_size=BLOCK,
        ar_block=ar,
    )
    bs = 3
    assert slots.shape == pos.shape == valid.shape == (bs, topk, BLOCK)
    for b in range(bs):
        for j in range(topk):
            blk = int(topk_blk[b, j])
            for i in range(BLOCK):
                p = blk * BLOCK + i if blk >= 0 else i
                live = blk >= 0 and p < int(seq_lens[b])
                assert bool(valid[b, j, i]) == live, (b, j, i)
                if live:
                    assert int(slots[b, j, i]) == int(rtt[b, p])
                else:
                    assert int(slots[b, j, i]) == 0  # a real row the kernel never reads

    staged = slots.reshape(-1)  # (request, block, token) order
    rows = torch.arange(bs * topk * BLOCK, dtype=torch.int32).view(bs, topk, BLOCK)
    dump_col = max_ctx
    fake = torch.full((bs, max_ctx + 1), -7, dtype=torch.int32)  # stale garbage
    fill_decode_fake_table(fake=fake, pos=pos, valid=valid, rows=rows, dump_col=dump_col)
    for b in range(bs):
        L = int(seq_lens[b])
        for j in range(topk):
            blk = int(topk_blk[b, j])
            if blk < 0:
                continue
            for i in range(BLOCK):
                p = blk * BLOCK + i
                if p >= L:
                    continue
                assert int(staged[int(fake[b, p])]) == int(rtt[b, p]), (b, blk, p)
    assert int((~valid).sum()) > 0
    # Positions outside every selected block are never read and may stay stale.
    assert int(fake[2, 8]) == -7

    # CUDA-graph padding: seq_len fill value 1 on the padding request row.
    s, p, v = decode_block_rows(
        topk_blk=torch.tensor([[0, -1, -1, -1]], dtype=torch.int32),
        req_to_token=torch.zeros((1, max_ctx), dtype=torch.int32),
        req_pool_indices=torch.tensor([0]),
        seq_lens=torch.tensor([1]),
        block_size=BLOCK,
        ar_block=ar,
    )
    assert int(v.sum()) == 1 and int(s[0, 0, 0]) == 0 and int(p[0, 0, 0]) == 0


def test_rows_prefix_matches_smaller_batch():
    # The backend slices a max_bs ``rows`` buffer to ``rows[:bs]``; the slice
    # must enumerate exactly the first bs*topk*BLOCK staged rows in order.
    max_bs, topk = 5, 3
    rows = torch.arange(max_bs * topk * BLOCK, dtype=torch.int32).view(max_bs, topk, BLOCK)
    for bs in range(1, max_bs + 1):
        assert torch.equal(
            rows[:bs].reshape(-1), torch.arange(bs * topk * BLOCK, dtype=torch.int32)
        )


if __name__ == "__main__":
    test_prefill_fake_table_and_slot_map()
    test_decode_block_rows_and_fake_table()
    test_rows_prefix_matches_smaller_batch()
    print("ok")
