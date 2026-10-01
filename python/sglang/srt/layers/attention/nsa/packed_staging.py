"""BF16 staging of the OSCAR packed-INT2 latent pool for the sparse (DSA) kernels.

The DSA kernels (flashmla_sparse for prefill, trtllm-gen sparse MLA for decode)
gather KV rows by absolute token slot out of one BF16/FP8 buffer. The packed
pool has no such buffer -- it holds 2-bit codes plus a BF16 window arena, and
its ``get_key_buffer`` raises on purpose. Rather than teach two closed-source
kernels to read codes, dequantize exactly the rows a forward attends into a
BF16 staging buffer and point the kernels at that:

* decode: every request reads its top-k rows, so the buffer is ``bs * topk``
  rows and the table becomes a plain arange with the -1 holes preserved;
* prefill: every query token has its own top-k, so the union is every token of
  every request (``seq_lens_sum`` rows). Dequantize them all once per layer,
  in the ragged order of the flattened page table, and remap slots through an
  inverse ``slot -> ragged position`` table.

Everything here is tensor arithmetic on device with static shapes, so the
decode path is CUDA-graph capturable. These functions take the dequantizer as
a callable and know nothing about sglang, so they are unit-tested on CPU.
"""

from typing import Callable, Optional, Tuple

import torch

Materialize = Callable[[torch.Tensor, torch.Tensor], None]


def stage_decode(
    materialize: Materialize,
    page_table_1: torch.Tensor,
    arange_i32: torch.Tensor,
    out: torch.Tensor,
    row_multiple: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dequantize each request's top-k rows into ``out`` and remap the table.

    ``page_table_1``: ``[bs, topk]`` int32 absolute token slots, -1 where a
    request holds fewer than ``topk`` tokens. ``out``: at least ``bs*topk``
    rows of ``[rows, 1, D]``, rounded up to ``row_multiple``. Returns
    ``(buf, table)`` with ``buf`` the first ``bs*topk`` rows -- padded to a
    multiple of ``row_multiple`` so a paged view of it is legal for the
    kernels that address the buffer as pages; the padding rows are never
    referenced -- and ``table`` an arange into it, -1 kept where it was.
    """
    bs, topk = page_table_1.shape
    n = bs * topk
    n_alloc = -(-n // row_multiple) * row_multiple
    page_table_1 = page_table_1.to(torch.int32)
    valid = page_table_1 >= 0
    # -1 would read before the buffer; row 0 is a real row and is simply
    # ignored by the kernel through the -1 kept in the remapped table.
    safe = torch.where(valid, page_table_1, torch.zeros_like(page_table_1))
    buf = out[:n]
    materialize(safe.reshape(-1), buf)
    table = torch.where(valid, arange_i32[:n].view(bs, topk), page_table_1)
    return out[:n_alloc], table


def build_slot_to_ragged(flat_slots: torch.Tensor, slot_to_ragged: torch.Tensor) -> None:
    """``slot_to_ragged[flat_slots[i]] = i``. Entries for slots not in this
    batch are left stale: nothing in the batch can reference them."""
    n = flat_slots.numel()
    slot_to_ragged[flat_slots.to(torch.int64)] = torch.arange(
        n, dtype=slot_to_ragged.dtype, device=slot_to_ragged.device
    )


def stage_prefill(
    materialize: Materialize,
    flat_slots: torch.Tensor,
    slot_to_ragged: torch.Tensor,
    page_table_1: torch.Tensor,
    out: torch.Tensor,
    fresh_slots: Optional[torch.Tensor] = None,
    fresh_rows: Optional[torch.Tensor] = None,
    row_multiple: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dequantize every token of the batch in ragged order and remap the table.

    ``flat_slots``: ``[n]`` absolute slots of all tokens of all requests, in
    request order (prefix first). ``page_table_1``: ``[num_q, topk]`` absolute
    slots, -1 = hole. ``fresh_slots``/``fresh_rows``: this forward's own tokens
    and their exact BF16 rows (already in the pool's stored frame); they
    overwrite the dequantized copies so the current chunk attends to itself at
    full precision, exactly as the triton extend path does. The buffer is
    padded to a multiple of ``row_multiple`` rows so a paged view of it is
    legal; the padding rows are never referenced.
    """
    n = flat_slots.numel()
    n_alloc = -(-n // row_multiple) * row_multiple
    buf = out[:n_alloc]
    materialize(flat_slots, buf[:n])
    if fresh_rows is not None:
        pos = slot_to_ragged[fresh_slots.to(torch.int64)].to(torch.int64)
        buf[pos, 0, :] = fresh_rows.reshape(pos.numel(), -1).to(buf.dtype)
    page_table_1 = page_table_1.to(torch.int32)
    valid = page_table_1 >= 0
    safe = torch.where(valid, page_table_1, torch.zeros_like(page_table_1))
    table = torch.where(valid, slot_to_ragged[safe.to(torch.int64)], page_table_1)
    return buf, table
