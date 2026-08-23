"""MLA decode stage-1 that reads a **packed INT2 latent** cache.

This is the read half of ``mem_cache/mla_packed_kv_pool.py`` and the reason the
MLA pool could store packed codes at all: the stock kernel
(``decode_attention._fwd_grouped_kernel_stage1``) loads a 576-wide BF16 row per
token, so a pool that does not hold one cannot serve it.

It is a specialisation, not an edit of the shared kernel. ``decode_attention``
serves every model in the tree and has already produced one silent-wrong-answer
bug from a head-tiling change (``_safe_block_h``); a separate entry point that
only a packed MLA pool can reach cannot regress anything else.

What changes versus the stock kernel, and nothing else does:

* ``k`` comes from ``codes``/``params`` (2 bits + a per-group scale/zero) rather
  than from a BF16 row, dequantized into registers inside the KV loop. The
  arithmetic is the same ``q * scale + zero`` / ``(q - zero) * scale`` pair the
  torch reference and the pack kernel use, chosen by ``LLOYD``.
* ``k_pe`` comes from its own BF16 buffer, because the rope half is never
  quantized.
* ``v`` is the same dequantized latent as ``k``'s first 512 lanes -- in MLA the
  value *is* the latent -- so it is computed once and transposed, not loaded
  twice. The stock kernel reads V a second time from an aliased view; here that
  would mean dequantizing twice.
* rows still inside the BF16 window arena override the dequantized value. The
  arena load is guarded by a block-level ``tl.max`` so the 98%+ of KV blocks
  containing no window row never pay for it -- without that guard the extra
  1024 B/token would put the packed path *above* the BF16 baseline it is
  supposed to beat.

Bandwidth, which is the point: 160 B/token of latent instead of 1024, plus the
unchanged 128 B of ``k_pe``. 288 versus 1152.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.layers.attention.triton_ops.decode_attention import (
    _MIN_BLOCK_KV,
    _decode_softmax_reducev_fwd,
    _safe_block_h,
    tanh,
)


@triton.jit
def _fwd_packed_mla_stage1(
    Q,
    Codes,
    Params,
    Rope,
    HP,
    HpRowOfSlot,
    HpOwnerOfRow,
    sm_scale_withk,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    D: tl.constexpr,          # kv_lora_rank (512)
    DPE: tl.constexpr,        # qk_rope_head_dim (64)
    GS: tl.constexpr,         # quant group size
    NG: tl.constexpr,         # groups per row = D // GS
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    LLOYD: tl.constexpr,
    HAS_HP: tl.constexpr,
    PARAM_BCAST: tl.constexpr,
    logit_cap: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    # MLA is single-KV-head by construction (the latent is shared across query
    # heads), so the stock kernel's cur_kv_head arithmetic collapses to 0 and
    # the head-tiling hazard it carries -- deriving cur_kv_head from the block
    # index while addressing heads by flat index -- cannot arise here.
    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, D)
    offs_dpe = D + tl.arange(0, DPE)
    offs_pe = tl.arange(0, DPE)

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
    off_qpe = (
        cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
    )

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, D], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        q = tl.load(Q + offs_q, mask=mask_h[:, None], other=0.0)
        qpe = tl.load(Q + off_qpe, mask=mask_h[:, None], other=0.0)

        gid = offs_d // GS

        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_ok = offs_n < split_kv_end
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n, mask=n_ok, other=0
            ).to(tl.int64)

            # ---- dequantize the latent for this block: [BLOCK_N, D] --------
            byte = tl.load(
                Codes + kv_loc[:, None] * (D // 4) + (offs_d // 4)[None, :],
                mask=n_ok[:, None],
                other=0,
            ).to(tl.int32)
            code = (byte >> (2 * (offs_d % 4))[None, :]) & 0x3
            if PARAM_BCAST:
                # scale/zero are per (token, group): [BLOCK_N, NG] of unique
                # values, 8 floats per token at NG=4. Addressing them as a
                # [BLOCK_N, D] tile via ``gid`` asks the LSU to compute 8192
                # addresses per tile to move 64 distinct floats -- 128x
                # redundant. L1 serves the repeats, so this never showed up as
                # DRAM traffic (and narrowing the *traffic* alone measured
                # 1.3x SLOWER, which is what ruled out a bandwidth story);
                # the cost is address computation and load issue.
                #
                # Loading the unique tile and expanding it in registers is
                # exact rather than approximate: the pack layout is
                # d = g * GS + j, so broadcasting [BLOCK_N, NG] over a trailing
                # GS axis and folding that axis into D reproduces gid's mapping
                # element for element. Output is bit-identical; only the
                # address arithmetic goes away.
                offs_g = tl.arange(0, NG)
                s_ng = tl.load(
                    Params + kv_loc[:, None] * (2 * NG) + (2 * offs_g)[None, :],
                    mask=n_ok[:, None],
                    other=0.0,
                )
                z_ng = tl.load(
                    Params + kv_loc[:, None] * (2 * NG) + (2 * offs_g + 1)[None, :],
                    mask=n_ok[:, None],
                    other=0.0,
                )
                scale = tl.reshape(
                    tl.broadcast_to(s_ng[:, :, None], (BLOCK_N, NG, GS)),
                    (BLOCK_N, D),
                )
                zero = tl.reshape(
                    tl.broadcast_to(z_ng[:, :, None], (BLOCK_N, NG, GS)),
                    (BLOCK_N, D),
                )
            else:
                scale = tl.load(
                    Params + kv_loc[:, None] * (2 * NG) + (2 * gid)[None, :],
                    mask=n_ok[:, None],
                    other=0.0,
                )
                zero = tl.load(
                    Params + kv_loc[:, None] * (2 * NG) + (2 * gid + 1)[None, :],
                    mask=n_ok[:, None],
                    other=0.0,
                )
            if LLOYD:
                cv = (code.to(tl.float32) - zero) * scale
            else:
                cv = code.to(tl.float32) * scale + zero

            # Narrow to the compute dtype immediately. The dequant chain is
            # elementwise-fused, so the fp32 code/scale/zero values never form a
            # second full tile; ``cvq`` is the only [BLOCK_N, D] tile alive, the
            # same one the BF16 kernel loads.
            cvq = tl.where(n_ok[:, None], cv, 0.0).to(q.dtype)

            if HAS_HP:
                r = tl.load(HpRowOfSlot + kv_loc, mask=n_ok, other=-1).to(tl.int32)
                rr = tl.where(r >= 0, r, 0).to(tl.int64)
                owner = tl.load(HpOwnerOfRow + rr, mask=n_ok, other=-1).to(tl.int32)
                use_hp = (r >= 0) & (owner == kv_loc.to(tl.int32)) & n_ok
                # A KV block with no window row -- the overwhelming majority --
                # skips the arena read entirely. Paying for it unconditionally
                # would cost 1024 B/token and erase the packed read's advantage.
                #
                # The select happens on the *narrowed* tile, and the arena load
                # stays in its stored dtype rather than being widened to fp32.
                # The compiler has to reserve registers for this branch in every
                # block even though ~2% of blocks take it, so an fp32 staging
                # tile here was costing 32 KiB of reservation to move 16 KiB of
                # bf16 -- measured at 1.54x on the whole kernel. Rounding is
                # unaffected: the arena stores bf16 and cvq is already bf16.
                if tl.max(use_hp.to(tl.int32)) > 0:
                    hpv = tl.load(
                        HP + rr[:, None] * D + offs_d[None, :],
                        mask=use_hp[:, None],
                        other=0.0,
                    )
                    cvq = tl.where(use_hp[:, None], hpv.to(q.dtype), cvq)
            qk = tl.dot(q, tl.trans(cvq))                      # [BLOCK_H, BLOCK_N]

            kpe = tl.load(
                Rope + kv_loc[None, :] * DPE + offs_pe[:, None],
                mask=n_ok[None, :],
                other=0.0,
            )
            qk += tl.dot(qpe, kpe.to(qpe.dtype))
            qk *= sm_scale_withk

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(mask_h[:, None] & n_ok[None, :], qk, float("-inf"))

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(cvq.dtype), cvq)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_d[None, :]
        )
        tl.store(Att_Out + offs_mid_o, acc / e_sum[:, None], mask=mask_h[:, None])

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // D
        tl.store(Att_Lse + offs_mid_o_1, e_max + tl.log(e_sum), mask=mask_h)


def packed_mla_decode_stage1(
    q,
    operands,
    att_out,
    att_lse,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    max_kv_splits,
    sm_scale_withk,
    logit_cap,
    block_n: int = 0,
    num_warps: int = 0,
    num_stages: int = 0,
    param_bcast: bool | None = None,
):
    """``q`` is ``[batch, head, D+DPE]``; ``operands`` from ``packed_read_operands``."""
    codes, params, rope, hp, hp_row, hp_owner, group_size, lloyd = operands
    # BLOCK_N 16 / 8 warps rather than the BF16 kernel's 32 / 4. The dequant
    # keeps three extra values live per element (the code byte and the group's
    # scale and zero) on top of the output tile, and at 32x512 that is past the
    # register file -- the kernel still runs, but every spill is a round trip to
    # local memory, which is precisely the bandwidth this path exists to save.
    block_n = block_n or envs.SGLANG_OSCAR_MLA_PACKED_BLOCK_N.get()
    num_warps = num_warps or envs.SGLANG_OSCAR_MLA_PACKED_WARPS.get()
    num_stages = num_stages or envs.SGLANG_OSCAR_MLA_PACKED_STAGES.get()
    d_pe = rope.shape[-1]
    d = codes.shape[-1] * 4
    assert q.shape[-1] == d + d_pe, (q.shape, d, d_pe)

    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = head_num  # MLA: one KV head
    block_h = _safe_block_h(16, kv_group_num)
    grid = (batch, triton.cdiv(head_num, min(block_h, kv_group_num)), max_kv_splits)

    n_groups = d // group_size
    # The register-side expansion needs ``tl.arange(0, NG)`` and a trailing-axis
    # fold, so NG must be a power of two. Every shipped MLA config is (512/128
    # = 4), but a group size that does not divide D into a power of two falls
    # back to the gid-indexed load rather than failing to compile.
    if param_bcast is None:
        param_bcast = envs.SGLANG_OSCAR_MLA_PACKED_PARAM_BCAST.get()
    param_bcast = bool(
        param_bcast and n_groups > 0 and (n_groups & (n_groups - 1)) == 0
    )

    has_hp = hp is not None
    _fwd_packed_mla_stage1[grid](
        q,
        codes,
        params,
        rope,
        hp if has_hp else rope,
        hp_row if has_hp else kv_indices,
        hp_owner if has_hp else kv_indices,
        sm_scale_withk,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        D=d,
        DPE=d_pe,
        GS=group_size,
        NG=n_groups,
        BLOCK_N=block_n,
        BLOCK_H=block_h,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        LLOYD=bool(lloyd),
        HAS_HP=has_hp,
        PARAM_BCAST=param_bcast,
        logit_cap=logit_cap,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def packed_mla_decode_fwd(
    q,
    pool,
    layer_id,
    o,
    attn_logits,
    attn_lse,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    max_kv_splits,
    sm_scale_withk,
    logit_cap=0.0,
):
    """Full two-stage MLA decode against a packed pool.

    Stage 2 is the stock softmax/reduce kernel -- it never touches the KV cache,
    only the split partials, so there is nothing to specialise.
    """
    operands = pool.packed_read_operands(layer_id)
    packed_mla_decode_stage1(
        q,
        operands,
        attn_logits,
        attn_lse,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        max_kv_splits,
        sm_scale_withk,
        logit_cap,
    )
    # Stage 2 reads ``v_buffer`` only for ``shape[-1]``; the packed pool has no
    # dense V to hand it, so a zero-row tensor of the right width stands in.
    _decode_softmax_reducev_fwd(
        attn_logits,
        attn_lse,
        q,
        o,
        1.0,  # v_scale: a scalar multiplier on the reduced output, not a tensor
        _v_shape_proxy(o, pool.kv_lora_rank),
        kv_indptr,
        num_kv_splits,
        max_kv_splits,
    )


_PROXY_CACHE: dict = {}


def _v_shape_proxy(ref: torch.Tensor, lv: int) -> torch.Tensor:
    key = (ref.device, ref.dtype, lv)
    t = _PROXY_CACHE.get(key)
    if t is None:
        t = torch.empty((0, 1, lv), dtype=ref.dtype, device=ref.device)
        _PROXY_CACHE[key] = t
    return t
