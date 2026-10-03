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


# ---------------------------------------------------------------------------
# Fused decode staging: one launch per (layer) instead of the dozen-plus small
# ops of decode_block_rows + two dequant launches + fill_decode_fake_table.
# Same arithmetic, same outputs; the Python path above stays as the reference
# the GPU equivalence test compares against.
# ---------------------------------------------------------------------------
try:  # keep the module importable on CPU-only test hosts
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _stage_decode_blocks_kernel(
        topk_blk_ptr,          # [bs, topk] block ids, -1 = unused
        req_to_token_ptr,      # [num_req_slots, max_ctx] int32
        req_pool_indices_ptr,  # [bs] int64
        seq_lens_ptr,          # [bs]
        quant_k_ptr, sz_k_ptr, hp_k_ptr,
        quant_v_ptr, sz_v_ptr, hp_v_ptr,
        out_k_ptr, out_v_ptr,
        fake_ptr,              # [max_bs, fake_cols] int32
        topk, block_size, max_ctx, rtt_stride, fake_stride, dump_col,
        qk_s_tok, qk_s_head, qk_s_dim, szk_s_tok, szk_s_head, szk_s_dim, hpk_s_tok, hpk_s_head, hpk_s_dim,
        qv_s_tok, qv_s_head, qv_s_dim, szv_s_tok, szv_s_head, szv_s_dim, hpv_s_tok, hpv_s_head, hpv_s_dim,
        ok_s_tok, ok_s_head, ok_s_dim, ov_s_tok, ov_s_head, ov_s_dim,
        HP_OFFSET: tl.constexpr,
        HEAD_DIM_K: tl.constexpr, GROUP_K: tl.constexpr, BLOCK_DIM_K: tl.constexpr,
        HEAD_DIM_V: tl.constexpr, GROUP_V: tl.constexpr, BLOCK_DIM_V: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        pid = tl.program_id(0)          # b * topk + k
        head = tl.program_id(1)
        b = pid // topk
        blk = tl.load(topk_blk_ptr + pid).to(tl.int64)
        blk_valid = blk >= 0
        blk0 = tl.maximum(blk, 0)
        offs_t = tl.arange(0, BLOCK_T)
        t_mask = offs_t < block_size
        pos = blk0 * block_size + offs_t
        seq_len = tl.load(seq_lens_ptr + b).to(tl.int64)
        valid = blk_valid & (pos < seq_len) & t_mask
        pos_c = tl.minimum(pos, max_ctx - 1)
        req = tl.load(req_pool_indices_ptr + b).to(tl.int64)
        slots = tl.load(req_to_token_ptr + req * rtt_stride + pos_c, mask=t_mask, other=0).to(tl.int64)
        slots = tl.where(valid, slots, 0)
        rows = pid.to(tl.int64) * block_size + offs_t
        # fake[b, pos] = row for live tokens; dead entries go to the dump column
        if head == 0:
            col = tl.where(valid, pos_c, dump_col)
            tl.store(fake_ptr + b * fake_stride + col, rows.to(tl.int32), mask=t_mask)
        is_hp = slots >= HP_OFFSET
        hp_slot = slots - HP_OFFSET
        # ---- K rows ----------------------------------------------------------
        offs_d = tl.arange(0, BLOCK_DIM_K)
        d_mask = offs_d < HEAD_DIM_K
        m2 = t_mask[:, None] & d_mask[None, :]
        quarter = HEAD_DIM_K // 4
        packed = tl.load(
            quant_k_ptr + slots[:, None] * qk_s_tok + head * qk_s_head + (offs_d % quarter)[None, :] * qk_s_dim,
            mask=m2 & (~is_hp)[:, None], other=0)
        q = ((packed >> ((offs_d // quarter) * 2)[None, :]) & 0x03).to(tl.float32)
        gid = offs_d // GROUP_K
        scale = tl.load(sz_k_ptr + slots[:, None] * szk_s_tok + head * szk_s_head + (gid * 2)[None, :] * szk_s_dim,
                        mask=m2 & (~is_hp)[:, None], other=1.0).to(tl.float32)
        zero = tl.load(sz_k_ptr + slots[:, None] * szk_s_tok + head * szk_s_head + (gid * 2 + 1)[None, :] * szk_s_dim,
                       mask=m2 & (~is_hp)[:, None], other=0.0).to(tl.float32)
        quant_val = (q - zero) * scale
        hp_val = tl.load(hp_k_ptr + hp_slot[:, None] * hpk_s_tok + head * hpk_s_head + offs_d[None, :] * hpk_s_dim,
                         mask=m2 & is_hp[:, None], other=0.0)
        out = tl.where(is_hp[:, None], hp_val, quant_val)
        tl.store(out_k_ptr + rows[:, None] * ok_s_tok + head * ok_s_head + offs_d[None, :] * ok_s_dim, out, mask=m2)
        # ---- V rows ----------------------------------------------------------
        offs_dv = tl.arange(0, BLOCK_DIM_V)
        dv_mask = offs_dv < HEAD_DIM_V
        m2v = t_mask[:, None] & dv_mask[None, :]
        quarter_v = HEAD_DIM_V // 4
        packed_v = tl.load(
            quant_v_ptr + slots[:, None] * qv_s_tok + head * qv_s_head + (offs_dv % quarter_v)[None, :] * qv_s_dim,
            mask=m2v & (~is_hp)[:, None], other=0)
        qv = ((packed_v >> ((offs_dv // quarter_v) * 2)[None, :]) & 0x03).to(tl.float32)
        gidv = offs_dv // GROUP_V
        scale_v = tl.load(sz_v_ptr + slots[:, None] * szv_s_tok + head * szv_s_head + (gidv * 2)[None, :] * szv_s_dim,
                          mask=m2v & (~is_hp)[:, None], other=1.0).to(tl.float32)
        zero_v = tl.load(sz_v_ptr + slots[:, None] * szv_s_tok + head * szv_s_head + (gidv * 2 + 1)[None, :] * szv_s_dim,
                         mask=m2v & (~is_hp)[:, None], other=0.0).to(tl.float32)
        quant_vv = (qv - zero_v) * scale_v
        hp_vv = tl.load(hp_v_ptr + hp_slot[:, None] * hpv_s_tok + head * hpv_s_head + offs_dv[None, :] * hpv_s_dim,
                        mask=m2v & is_hp[:, None], other=0.0)
        outv = tl.where(is_hp[:, None], hp_vv, quant_vv)
        tl.store(out_v_ptr + rows[:, None] * ov_s_tok + head * ov_s_head + offs_dv[None, :] * ov_s_dim, outv, mask=m2v)


def _num_scale_groups(scales_zeros: torch.Tensor) -> int:
    return max(1, scales_zeros.shape[-1] // 2)


def stage_decode_blocks_fused(
    *,
    topk_blk: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    quant_k: torch.Tensor, scales_zeros_k: torch.Tensor, hp_k: torch.Tensor,
    quant_v: torch.Tensor, scales_zeros_v: torch.Tensor, hp_v: torch.Tensor,
    hp_global_offset: int,
    out_k: torch.Tensor,
    out_v: torch.Tensor,
    fake: torch.Tensor,
    dump_col: int,
) -> None:
    """One launch: for every (request, selected block) resolve the token slots,
    dequantize the K and V rows of every head into ``out_k`` / ``out_v`` (rows
    in ``(b * topk + k) * block_size + t`` order, as the reference path lays
    them out) and write ``fake[b, pos] = row`` (dead entries to ``dump_col``).
    ``out_k`` / ``out_v`` must hold ``bs * topk * block_size`` rows."""
    assert triton is not None, "fused staging needs triton"
    bs, topk = topk_blk.shape
    n_rows = bs * topk * block_size
    assert out_k.shape[0] >= n_rows and out_v.shape[0] >= n_rows, (out_k.shape, n_rows)
    num_heads, head_dim_k = quant_k.shape[1], out_k.shape[2]
    head_dim_v = out_v.shape[2]
    assert fake.shape[1] > dump_col, (fake.shape, dump_col)
    grid = (bs * topk, num_heads)
    _stage_decode_blocks_kernel[grid](
        topk_blk, req_to_token, req_pool_indices, seq_lens,
        quant_k, scales_zeros_k, hp_k, quant_v, scales_zeros_v, hp_v,
        out_k, out_v, fake,
        topk, block_size, req_to_token.shape[1], req_to_token.stride(0), fake.stride(0), dump_col,
        quant_k.stride(0), quant_k.stride(1), quant_k.stride(2),
        scales_zeros_k.stride(0), scales_zeros_k.stride(1), scales_zeros_k.stride(2),
        hp_k.stride(0), hp_k.stride(1), hp_k.stride(2),
        quant_v.stride(0), quant_v.stride(1), quant_v.stride(2),
        scales_zeros_v.stride(0), scales_zeros_v.stride(1), scales_zeros_v.stride(2),
        hp_v.stride(0), hp_v.stride(1), hp_v.stride(2),
        out_k.stride(0), out_k.stride(1), out_k.stride(2),
        out_v.stride(0), out_v.stride(1), out_v.stride(2),
        HP_OFFSET=int(hp_global_offset),
        HEAD_DIM_K=head_dim_k, GROUP_K=head_dim_k // _num_scale_groups(scales_zeros_k),
        BLOCK_DIM_K=triton.next_power_of_2(head_dim_k),
        HEAD_DIM_V=head_dim_v, GROUP_V=head_dim_v // _num_scale_groups(scales_zeros_v),
        BLOCK_DIM_V=triton.next_power_of_2(head_dim_v),
        BLOCK_T=triton.next_power_of_2(block_size),
        num_warps=4, num_stages=1,
    )
