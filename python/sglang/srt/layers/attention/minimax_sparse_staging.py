"""Index arithmetic for serving MiniMax-M3 sparse attention off the per-head
INT2 pool.

The block-sparse kernels read K/V as ``cache[req_to_token[slot_ids[b], pos]]``.
The INT2 pool has no BF16 cache to read, so the backend dequantizes the rows a
forward attends into a BF16 staging buffer and hands the kernels a *fake*
``req_to_token`` whose entries point into that buffer. Positions are preserved
(the table is still indexed by sequence position), so the kernels' causal and
seq_len masking is untouched.

Prefill stages every token of the batch in ragged request order
(``prefill_fake_table`` / ``build_slot_to_ragged``); decode stages only the
selected blocks at their real positions (``decode_block_rows`` /
``fill_decode_fake_table``).

Pure tensor arithmetic, static shapes -- decode is CUDA-graph capturable -- and
no sglang imports, so it is unit-tested on CPU.
"""

from typing import Tuple

import torch


def prefill_fake_table(cu_seqlens_k: torch.Tensor, max_seqlen_k: int) -> torch.Tensor:
    """``fake[b, pos] = cu_seqlens_k[b] + pos``: request b's tokens occupy a
    contiguous run of the ragged staging buffer. Entries at pos >= seq_len[b]
    are never read (the kernels mask by seq_len), so no masking is needed."""
    ar = torch.arange(max_seqlen_k, dtype=torch.int32, device=cu_seqlens_k.device)
    return (cu_seqlens_k[:-1].to(torch.int32)[:, None] + ar[None, :]).contiguous()


def build_slot_to_ragged(flat_slots: torch.Tensor, slot_to_ragged: torch.Tensor) -> None:
    """``slot_to_ragged[flat_slots[i]] = i``: where a pool slot landed in the
    ragged staging buffer. Used to overwrite this forward's own tokens with
    their exact rows. Entries for slots outside the batch are left stale;
    nothing in the batch can reference them."""
    n = flat_slots.numel()
    slot_to_ragged[flat_slots.to(torch.int64)] = torch.arange(
        n, dtype=slot_to_ragged.dtype, device=slot_to_ragged.device
    )


def decode_block_rows(
    topk_blk: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    ar_block: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For each request, the slots of every token in its selected blocks.

    ``topk_blk``: ``[bs, topk]`` block ids, -1 = unused. Returns
    ``(slots, pos, valid)`` each ``[bs, topk, block_size]``: the pool slot of the
    token (0 where invalid, which is a real row the kernel never attends), its
    sequence position (clamped into the table), and whether it is a live token
    (block selected and pos < seq_len).
    """
    bs, topk = topk_blk.shape
    blk_valid = topk_blk >= 0
    pos = topk_blk.clamp(min=0).to(torch.int64)[:, :, None] * block_size + ar_block[None, None, :]
    valid = blk_valid[:, :, None] & (pos < seq_lens.to(torch.int64)[:, None, None])
    pos_c = pos.clamp(max=req_to_token.shape[1] - 1)
    slots = req_to_token[req_pool_indices.to(torch.int64)[:, None, None], pos_c]
    slots = torch.where(valid, slots, torch.zeros_like(slots))
    return slots, pos_c, valid


def fill_decode_fake_table(
    fake: torch.Tensor,
    pos: torch.Tensor,
    valid: torch.Tensor,
    rows: torch.Tensor,
    dump_col: int,
) -> None:
    """``fake[b, pos] = row`` for live tokens. Dead entries are steered into a
    spare column (``dump_col``, beyond any legal position) instead of being
    skipped, so the scatter has a static shape and can be captured: a masked
    scatter would need a host-side nonzero.

    Writing a dead entry over a live one is the bug this guards against -- a
    -1 block is clamped to block 0, which may itself be selected.
    """
    bs = pos.shape[0]
    b_idx = torch.arange(bs, device=pos.device, dtype=torch.int64)[:, None, None].expand_as(pos)
    col = torch.where(valid, pos, torch.full_like(pos, dump_col))
    fake[b_idx.reshape(-1), col.reshape(-1)] = rows.reshape(-1).to(fake.dtype)
