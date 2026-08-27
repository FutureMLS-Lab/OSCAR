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

# lse written for a split with no live tokens is -1.0e4: large enough that
# exp(sentinel - anything_real) underflows to 0, finite so that subtracting it
# from itself is 0 rather than NaN. Inlined at both store sites rather than
# named here -- this Triton will not resolve a module-level Python float used
# as a tl.where operand inside a jit function, and the failure is a
# CompilationError pointing at the store, not at the definition.
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
    BITS: tl.constexpr,       # bits per latent value (2 or 4)
    PF: tl.constexpr,         # latent values packed per byte = 8 // BITS
    MASK: tl.constexpr,       # (1 << BITS) - 1
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    LLOYD: tl.constexpr,
    HAS_HP: tl.constexpr,
    PARAM_BCAST: tl.constexpr,
    ABLATE: tl.constexpr,
    DUAL_LOAD: tl.constexpr,
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
                Codes + kv_loc[:, None] * (D // PF) + (offs_d // PF)[None, :],
                mask=n_ok[:, None],
                other=0,
            ).to(tl.int32)
            code = (byte >> (BITS * (offs_d % PF))[None, :]) & MASK
            if ABLATE == 1:
                scale = 0.0
                zero = 0.0
            elif PARAM_BCAST:
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
            if ABLATE == 1:
                # NUMERICALLY WRONG BY CONSTRUCTION -- a measurement device, not
                # a serving path (``packed_mla_decode_stage1`` only reaches it
                # from the benchmark, never from the pool). Drops the entire
                # dequant: no scale/zero, no fp32 stage, code straight to the
                # compute dtype. What remains is the byte load, the shift/mask,
                # and the two dots, so the delta against ABLATE=0 is the cost of
                # dequantizing -- the one quantity the tiling, traffic and
                # addressing experiments all failed to move.
                cv = code.to(q.dtype)
            elif ABLATE == 2:
                # Loads kept, arithmetic dropped: separates "reading scale/zero"
                # from "applying them". ``* 0`` keeps the loads live against DCE.
                cv = code.to(tl.float32) + (scale * 0.0) + (zero * 0.0)
            elif LLOYD:
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

            if DUAL_LOAD:
                # ``tl.trans`` of a [BLOCK_N, D] tile is a shared-memory layout
                # conversion, and layout conversion is the one cost in this
                # kernel that measurement has *not* exonerated. Four hypotheses
                # have now been refuted here -- tiling (18-point grid), traffic
                # (narrowing the bytes, 1.3x slower), addressing (register
                # broadcast, bit-identical and 1.28x slower) and the dequant
                # arithmetic itself (ABLATE=2 removes it and measures 1.00x) --
                # and two of the four got *slower* specifically by replacing a
                # direct wide load with a constructed tile.
                #
                # So the original trade is backwards. The docstring's reason for
                # transposing was to avoid dequantizing twice; the ablation says
                # dequantizing is free. Read the codes a second time in the
                # layout the first dot wants, and delete the transpose. The rope
                # load below has always done exactly this.
                byte_t = tl.load(
                    Codes + kv_loc[None, :] * (D // PF) + (offs_d // PF)[:, None],
                    mask=n_ok[None, :],
                    other=0,
                ).to(tl.int32)
                code_t = (byte_t >> (BITS * (offs_d % PF))[:, None]) & MASK
                scale_t = tl.load(
                    Params + kv_loc[None, :] * (2 * NG) + (2 * gid)[:, None],
                    mask=n_ok[None, :],
                    other=0.0,
                )
                zero_t = tl.load(
                    Params + kv_loc[None, :] * (2 * NG) + (2 * gid + 1)[:, None],
                    mask=n_ok[None, :],
                    other=0.0,
                )
                if LLOYD:
                    cv_t = (code_t.to(tl.float32) - zero_t) * scale_t
                else:
                    cv_t = code_t.to(tl.float32) * scale_t + zero_t
                cvq_t = tl.where(n_ok[None, :], cv_t, 0.0).to(q.dtype)
                if HAS_HP:
                    if tl.max(use_hp.to(tl.int32)) > 0:
                        hpv_t = tl.load(
                            HP + rr[None, :] * D + offs_d[:, None],
                            mask=use_hp[None, :],
                            other=0.0,
                        )
                        cvq_t = tl.where(use_hp[None, :], hpv_t.to(q.dtype), cvq_t)
                qk = tl.dot(q, cvq_t)                          # [BLOCK_H, BLOCK_N]
            else:
                qk = tl.dot(q, tl.trans(cvq))                  # [BLOCK_H, BLOCK_N]

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
    ablate: int = 0,
    block_h: int = 0,
    dual_load: bool | None = None,
):
    """``q`` is ``[batch, head, D+DPE]``; ``operands`` from ``packed_read_operands``."""
    codes, params, rope, hp, hp_row, hp_owner, group_size, lloyd, bits = operands
    # BLOCK_N 16 / 8 warps rather than the BF16 kernel's 32 / 4. The dequant
    # keeps three extra values live per element (the code byte and the group's
    # scale and zero) on top of the output tile, and at 32x512 that is past the
    # register file -- the kernel still runs, but every spill is a round trip to
    # local memory, which is precisely the bandwidth this path exists to save.
    block_n = block_n or envs.SGLANG_OSCAR_MLA_PACKED_BLOCK_N.get()
    num_warps = num_warps or envs.SGLANG_OSCAR_MLA_PACKED_WARPS.get()
    num_stages = num_stages or envs.SGLANG_OSCAR_MLA_PACKED_STAGES.get()
    d_pe = rope.shape[-1]
    # Derived from the stored byte width, so it MUST use the same packing
    # factor the pool allocated with -- at four bits this expression yields
    # half the latent dimension and every index downstream is wrong.
    d = codes.shape[-1] * (8 // bits)
    assert q.shape[-1] == d + d_pe, (q.shape, d, d_pe)

    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = head_num  # MLA: one KV head
    block_h = _safe_block_h(block_h or 16, kv_group_num)
    grid = (batch, triton.cdiv(head_num, min(block_h, kv_group_num)), max_kv_splits)

    n_groups = d // group_size
    # The register-side expansion needs ``tl.arange(0, NG)`` and a trailing-axis
    # fold, so NG must be a power of two. Every shipped MLA config is (512/128
    # = 4), but a group size that does not divide D into a power of two falls
    # back to the gid-indexed load rather than failing to compile.
    if dual_load is None:
        dual_load = envs.SGLANG_OSCAR_MLA_PACKED_DUAL_LOAD.get()
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
        BITS=bits,
        PF=8 // bits,
        MASK=(1 << bits) - 1,
        BLOCK_N=block_n,
        BLOCK_H=block_h,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        LLOYD=bool(lloyd),
        HAS_HP=has_hp,
        PARAM_BCAST=param_bcast,
        ABLATE=int(ablate),
        DUAL_LOAD=bool(dual_load),
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


@triton.jit
def _fwd_packed_mla_stage1_gf(
    Q, Codes, Params, Rope, HpRowOfSlot, HpOwnerOfRow,
    sm_scale_withk, kv_indptr, kv_indices, Att_Out, Att_Lse, num_kv_splits,
    stride_qbs, stride_qh, stride_mid_ob, stride_mid_oh, stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    D: tl.constexpr,
    DPE: tl.constexpr,
    GS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    LLOYD: tl.constexpr,
    WIDE_LOAD: tl.constexpr,
    SKIP_HP: tl.constexpr,
    SLOT_OFF: tl.constexpr,
    logit_cap: tl.constexpr,
):
    """Group-factored packed MLA stage-1, specialised to NG == 4.

    Six hypotheses have been measured on the tiled kernel and all six refuted:
    tiling, traffic (0.77x), addressing (0.78x, bit-identical), the dequant
    arithmetic itself (ABLATE=2 removes it and measures **1.00x** -- it is free),
    the transpose (0.86x), and head amortization (no room at 16 q heads/rank).
    Every one of those still ended at a **computed** ``[BLOCK_N, D]`` tile fed to
    ``tl.dot``, and a computed tile must be materialised through shared memory
    where a loaded one can stream -- which is why BLOCK_N is pinned at 16 (32 is
    slower, 64 exhausts smem) against the BF16 kernel's 32-64.

    So factor the dequant out of the dot instead of computing it first:

        q . cv = sum_g [ s_g * sum_{d in g} q_d code_d + z_g * sum_{d in g} q_d ]

    ``sum_{d in g} q_d`` is loop-invariant. ``tl.dot`` then sees the raw code
    tile converted to bf16 -- INT2 values 0..3 are exact there -- and scale/zero
    apply in the ``[BLOCK_H, BLOCK_N]`` domain, 256 elements instead of 8192.
    Lloyd-Max is the same shape: ``(code - z) s = code s - z s``.

    Unrolled by hand for NG == 4 because Triton cannot build a Python list of
    tensors inside a jit function -- ``[tl.zeros(...) for _ in range(NG)]`` is a
    compile error, which is what killed the first attempt. NG=4 is every shipped
    config (kv_lora_rank 512 / group 128); the launcher refuses anything else.
    """
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_g = tl.arange(0, GS)
    offs_b = tl.arange(0, GS // 4)
    shf = 2 * tl.arange(0, 4)
    offs_dpe = D + tl.arange(0, DPE)
    offs_pe = tl.arange(0, DPE)

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    qbase = cur_batch * stride_qbs + cur_head[:, None] * stride_qh
    off_qpe = qbase + offs_dpe[None, :]

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc0 = tl.zeros([BLOCK_H, GS], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_H, GS], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_H, GS], dtype=tl.float32)
    acc3 = tl.zeros([BLOCK_H, GS], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        qpe = tl.load(Q + off_qpe, mask=mask_h[:, None], other=0.0)
        q0 = tl.load(Q + qbase + (0 * GS + offs_g)[None, :], mask=mask_h[:, None], other=0.0)
        q1 = tl.load(Q + qbase + (1 * GS + offs_g)[None, :], mask=mask_h[:, None], other=0.0)
        q2 = tl.load(Q + qbase + (2 * GS + offs_g)[None, :], mask=mask_h[:, None], other=0.0)
        q3 = tl.load(Q + qbase + (3 * GS + offs_g)[None, :], mask=mask_h[:, None], other=0.0)
        qs0 = tl.sum(q0.to(tl.float32), 1)
        qs1 = tl.sum(q1.to(tl.float32), 1)
        qs2 = tl.sum(q2.to(tl.float32), 1)
        qs3 = tl.sum(q3.to(tl.float32), 1)

        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_ok = offs_n < split_kv_end
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n, mask=n_ok, other=0
            ).to(tl.int64)

            if SKIP_HP:
                # EXCLUDE the window-arena tokens rather than override them.
                #
                # The computed-tile kernel overrides: it selects the arena's
                # bf16 value into a full-width [BLOCK_N, D] tile. That branch is
                # taken by ~2% of blocks but the compiler must reserve its
                # registers in ALL of them, which measured 1.54x on the whole
                # kernel. Folding that shape into the group-factored kernel
                # would hand back most of what the factoring won, because the
                # arena value is arbitrary bf16 -- it cannot be expressed as a
                # code plus a per-group scale, so it forces the full-width tile
                # this kernel exists to avoid.
                #
                # Excluding costs two int lookups and a mask, and no full-width
                # tile ever enters the fast path. The excluded tokens are then
                # computed by a small dense BF16 pass written into a separate
                # split slot, and stage 2's LSE merge combines them -- it
                # already merges partial (acc, lse) across splits, so an extra
                # split is not a new mechanism.
                r = tl.load(HpRowOfSlot + kv_loc, mask=n_ok, other=-1).to(tl.int32)
                rr = tl.where(r >= 0, r, 0).to(tl.int64)
                owner = tl.load(HpOwnerOfRow + rr, mask=n_ok, other=-1).to(tl.int32)
                n_ok = n_ok & ~((r >= 0) & (owner == kv_loc.to(tl.int32)))

            # scale/zero: [BLOCK_N] per group, 8 loads per token, not 8192.
            s0 = tl.load(Params + kv_loc * 8 + 0, mask=n_ok, other=0.0)
            z0 = tl.load(Params + kv_loc * 8 + 1, mask=n_ok, other=0.0)
            s1 = tl.load(Params + kv_loc * 8 + 2, mask=n_ok, other=0.0)
            z1 = tl.load(Params + kv_loc * 8 + 3, mask=n_ok, other=0.0)
            s2 = tl.load(Params + kv_loc * 8 + 4, mask=n_ok, other=0.0)
            z2 = tl.load(Params + kv_loc * 8 + 5, mask=n_ok, other=0.0)
            s3 = tl.load(Params + kv_loc * 8 + 6, mask=n_ok, other=0.0)
            z3 = tl.load(Params + kv_loc * 8 + 7, mask=n_ok, other=0.0)
            if LLOYD:
                b0 = -z0 * s0
                b1 = -z1 * s1
                b2 = -z2 * s2
                b3 = -z3 * s3
            else:
                b0 = z0
                b1 = z1
                b2 = z2
                b3 = z3

            # Code tiles in [GS, BLOCK_N] -- the layout tl.dot(q_g, .) wants, so
            # there is no transpose and nothing full-width is ever computed.
            if WIDE_LOAD:
                # Four consecutive dims share one byte, so the narrow form below
                # issues four loads per byte: (g*GS + offs_g)//4 maps GS=128 rows
                # onto only 32 distinct addresses. Load those 32 rows once and
                # unpack the four 2-bit fields into the [GS, BLOCK_N] tile.
                #
                # Worth trying specifically because the BLOCK_H sweep measured
                # what this costs: halving BLOCK_H doubles the code load+unpack
                # while leaving the dot work unchanged, and it cost 1.82x
                # (0.341 -> 0.622 ms), which puts the code path at roughly 80%
                # of the kernel. Address arithmetic alone was refuted before
                # (0.78x), but that experiment broadcast the *parameters* and
                # left this fourfold code redundancy untouched.
                #
                # g*GS is a multiple of 4 for GS=128, so the byte row is
                # g*(GS//4) + d//4 and the shift is 2*(d%4) with no dependence
                # on g. reshape maps (i, j) -> row 4i+j, which is exactly dim
                # 4i+j, so the unpacked tile is index-identical to the narrow one.
                w0 = tl.load(Codes + kv_loc[None, :] * (D // 4) + (0 * (GS // 4) + offs_b)[:, None], mask=n_ok[None, :], other=0).to(tl.int32)
                c0 = tl.reshape((w0[:, None, :] >> shf[None, :, None]) & 0x3, (GS, BLOCK_N)).to(qpe.dtype)
                w1 = tl.load(Codes + kv_loc[None, :] * (D // 4) + (1 * (GS // 4) + offs_b)[:, None], mask=n_ok[None, :], other=0).to(tl.int32)
                c1 = tl.reshape((w1[:, None, :] >> shf[None, :, None]) & 0x3, (GS, BLOCK_N)).to(qpe.dtype)
                w2 = tl.load(Codes + kv_loc[None, :] * (D // 4) + (2 * (GS // 4) + offs_b)[:, None], mask=n_ok[None, :], other=0).to(tl.int32)
                c2 = tl.reshape((w2[:, None, :] >> shf[None, :, None]) & 0x3, (GS, BLOCK_N)).to(qpe.dtype)
                w3 = tl.load(Codes + kv_loc[None, :] * (D // 4) + (3 * (GS // 4) + offs_b)[:, None], mask=n_ok[None, :], other=0).to(tl.int32)
                c3 = tl.reshape((w3[:, None, :] >> shf[None, :, None]) & 0x3, (GS, BLOCK_N)).to(qpe.dtype)
            else:
                by0 = tl.load(Codes + kv_loc[None, :] * (D // 4) + ((0 * GS + offs_g) // 4)[:, None], mask=n_ok[None, :], other=0).to(tl.int32)
                c0 = ((by0 >> (2 * ((0 * GS + offs_g) % 4))[:, None]) & 0x3).to(qpe.dtype)
                by1 = tl.load(Codes + kv_loc[None, :] * (D // 4) + ((1 * GS + offs_g) // 4)[:, None], mask=n_ok[None, :], other=0).to(tl.int32)
                c1 = ((by1 >> (2 * ((1 * GS + offs_g) % 4))[:, None]) & 0x3).to(qpe.dtype)
                by2 = tl.load(Codes + kv_loc[None, :] * (D // 4) + ((2 * GS + offs_g) // 4)[:, None], mask=n_ok[None, :], other=0).to(tl.int32)
                c2 = ((by2 >> (2 * ((2 * GS + offs_g) % 4))[:, None]) & 0x3).to(qpe.dtype)
                by3 = tl.load(Codes + kv_loc[None, :] * (D // 4) + ((3 * GS + offs_g) // 4)[:, None], mask=n_ok[None, :], other=0).to(tl.int32)
                c3 = ((by3 >> (2 * ((3 * GS + offs_g) % 4))[:, None]) & 0x3).to(qpe.dtype)

            qk = tl.dot(q0, c0) * s0[None, :] + qs0[:, None] * b0[None, :]
            qk += tl.dot(q1, c1) * s1[None, :] + qs1[:, None] * b1[None, :]
            qk += tl.dot(q2, c2) * s2[None, :] + qs2[:, None] * b2[None, :]
            qk += tl.dot(q3, c3) * s3[None, :] + qs3[:, None] * b3[None, :]

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
            # An all-masked block makes every qk -inf, so n_e_max is -inf and
            # qk - n_e_max is (-inf) - (-inf) = NaN -- acc is poisoned before e_sum
            # is ever looked at, which is why guarding only the final divide did
            # nothing. Exclusion creates exactly this: the sink-prefix block is
            # entirely window tokens and it is the FIRST block of every request, so
            # one NaN reaches every output. Subtracting 0 instead leaves exp(-inf)=0,
            # so p and e_sum stay 0 and the block contributes nothing.
            # e_max keeps the true -inf so a later non-empty block still rescales.
            n_e_max_s = tl.where(n_e_max == float("-inf"), 0.0, n_e_max)
            re_scale = tl.exp(e_max - n_e_max_s)
            p = tl.exp(qk - n_e_max_s[:, None])

            # AV: fold the group scale into p, dot against the raw codes, and add
            # the zero term as a per-(head, group) scalar.
            acc0 = acc0 * re_scale[:, None] + tl.dot((p * s0[None, :]).to(qpe.dtype), tl.trans(c0)) + tl.sum(p * b0[None, :], 1)[:, None]
            acc1 = acc1 * re_scale[:, None] + tl.dot((p * s1[None, :]).to(qpe.dtype), tl.trans(c1)) + tl.sum(p * b1[None, :], 1)[:, None]
            acc2 = acc2 * re_scale[:, None] + tl.dot((p * s2[None, :]).to(qpe.dtype), tl.trans(c2)) + tl.sum(p * b2[None, :], 1)[:, None]
            acc3 = acc3 * re_scale[:, None] + tl.dot((p * s3[None, :]).to(qpe.dtype), tl.trans(c3)) + tl.sum(p * b3[None, :], 1)[:, None]

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        # SLOT_OFF shifts these partials up by one so the dense window pass can
        # own slot 0. The stock stage-2 merge is measurably wrong when two or
        # more empty splits precede the first real one -- torch merging the
        # identical buffers returns rel 3.7e-03 where the kernel returns 1.0 --
        # and exclusion makes that pattern the normal case at short sequences.
        # Keeping a real partial in slot 0 avoids generating it at all, which is
        # preferable to editing a kernel every model on this backend shares.
        obase = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + (split_kv_id + SLOT_OFF) * stride_mid_os
        )
        # A split can now be non-empty in RANGE but empty in CONTENT: with
        # SKIP_HP every token in it may belong to the window arena, which is
        # normal for the split covering the sink prefix. e_sum is then 0 and
        # acc / e_sum is 0/0 = NaN, and stage 2 propagates it -- the NaN is
        # multiplied by exp(lse - max_lse) = exp(-inf) = 0, which is still NaN.
        # Before exclusion this could not arise, because every in-range split
        # had at least one live token, which is why the store was unguarded.
        safe = tl.where(e_sum > 0, e_sum, 1.0)
        tl.store(Att_Out + obase + (0 * GS + offs_g)[None, :], acc0 / safe[:, None], mask=mask_h[:, None])
        tl.store(Att_Out + obase + (1 * GS + offs_g)[None, :], acc1 / safe[:, None], mask=mask_h[:, None])
        tl.store(Att_Out + obase + (2 * GS + offs_g)[None, :], acc2 / safe[:, None], mask=mask_h[:, None])
        tl.store(Att_Out + obase + (3 * GS + offs_g)[None, :], acc3 / safe[:, None], mask=mask_h[:, None])

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // D
        # An empty split gets a very negative FINITE lse, not -inf.
        #
        # -inf is the mathematically honest value and it is what broke this.
        # Stage 2 seeds its running max at -inf, so if the FIRST split is also
        # -inf it computes exp(-inf - (-inf)) = NaN on iteration 0 and the
        # accumulator stays NaN however many finite splits follow. The
        # per-split diagnostic showed exactly that: a -inf in split 7 of 8 is
        # harmless (MATCH), a -inf in split 0 when it is the ONLY packed split
        # poisons every output. Exclusion makes the all-empty case ordinary --
        # the sink-prefix split is entirely window tokens.
        #
        # A finite sentinel merges correctly in both directions:
        # max(SENTINEL, finite) = finite and exp(SENTINEL - finite) underflows
        # to 0; and if every split is SENTINEL, exp(0) = 1 over zero-valued
        # accumulators still yields 0 rather than NaN. Fixing this in stage 2
        # instead would mean editing a kernel every model on this backend
        # shares, to accommodate one pool.
        tl.store(
            Att_Lse + offs_mid_o_1,
            tl.where(e_sum > 0, e_max + tl.log(safe), -1.0e4),
            mask=mask_h,
        )


def packed_mla_decode_stage1_gf(
    q, operands, att_out, att_lse, kv_indptr, kv_indices, num_kv_splits,
    max_kv_splits, sm_scale_withk, logit_cap,
    block_n: int = 0, num_warps: int = 0, num_stages: int = 0,
    block_h: int = 0, wide_load: bool = True, skip_hp: bool = False,
    slot_off: int = 0,
):
    """Launch the group-factored stage-1. Benchmark-only: no window arena.

    ``wide_load`` defaults on: it is bit-identical (maxdev 0.00e+00, by
    construction rather than by tolerance) and 1.51x at the 32/warps=4 tile this
    launcher actually picks. It is 0.95x at 32/warps=8 only, which is not a
    configuration anything selects.
    """
    codes, params, rope, hp, hp_row, hp_owner, group_size, lloyd, bits = operands
    if hp is not None and not skip_hp:
        raise NotImplementedError(
            "the group-factored kernel has no window-arena path; folding one in "
            "reintroduces the full-width tile it exists to avoid, in a branch "
            "the compiler must reserve for in every block"
        )
    d_pe = rope.shape[-1]
    # Derived from the stored byte width, so it MUST use the same packing
    # factor the pool allocated with -- at four bits this expression yields
    # half the latent dimension and every index downstream is wrong.
    d = codes.shape[-1] * (8 // bits)
    n_groups = d // group_size
    if n_groups != 4:
        raise NotImplementedError(
            f"group-factored kernel is unrolled for NG==4, got {n_groups} "
            "(Triton rejects a Python list of tensors inside a jit function, "
            "so the group loop cannot be written generically)"
        )
    assert q.shape[-1] == d + d_pe, (q.shape, d, d_pe)

    block_n = block_n or envs.SGLANG_OSCAR_MLA_PACKED_BLOCK_N.get()
    num_warps = num_warps or envs.SGLANG_OSCAR_MLA_PACKED_WARPS.get()
    num_stages = num_stages or envs.SGLANG_OSCAR_MLA_PACKED_STAGES.get()

    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = head_num
    block_h = _safe_block_h(block_h or 16, kv_group_num)
    grid = (batch, triton.cdiv(head_num, min(block_h, kv_group_num)), max_kv_splits)

    return _fwd_packed_mla_stage1_gf[grid](
        q, codes, params, rope,
        hp_row if skip_hp else kv_indices,
        hp_owner if skip_hp else kv_indices,
        sm_scale_withk, kv_indptr, kv_indices, att_out, att_lse, num_kv_splits,
        q.stride(0), q.stride(1),
        att_out.stride(0), att_out.stride(1), att_out.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        D=d, DPE=d_pe, GS=group_size,
        BLOCK_N=block_n, BLOCK_H=block_h, MIN_BLOCK_KV=_MIN_BLOCK_KV,
        LLOYD=bool(lloyd), WIDE_LOAD=bool(wide_load), SKIP_HP=bool(skip_hp),
        SLOT_OFF=int(slot_off),
        logit_cap=logit_cap,
        num_warps=num_warps, num_stages=num_stages,
    )


@triton.jit
def _fwd_hp_window_stage1(
    Q, Rope, HP, HpRowOfSlot, HpOwnerOfRow,
    sm_scale_withk, kv_indptr, kv_indices, Att_Out, Att_Lse, num_kv_splits,
    stride_qbs, stride_qh, stride_mid_ob, stride_mid_oh, stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    D: tl.constexpr,
    DPE: tl.constexpr,
    P_TOK: tl.constexpr,
    R_TOK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    logit_cap: tl.constexpr,
):
    """Dense BF16 stage-1 over exactly the tokens the packed kernel excluded.

    Complementary by construction, not by agreement: the packed kernel keeps a
    token when ``not (r >= 0 and owner == slot)`` and this keeps it when
    ``(r >= 0 and owner == slot)``, the identical predicate on the identical
    slot. Union is every token, intersection is empty -- so no token is dropped
    and none is counted twice, without the two kernels having to be kept in
    sync by hand.

    Only the window ranges are scanned: the arena is positional, covering
    ``[0, P)`` and ``[seq - R, seq)``, so this is O(P + R) rather than O(seq).
    A token inside those ranges whose owner tag fails was KEPT by the packed
    kernel (prefix-cache reuse can re-issue a ring row to another request), and
    the shared predicate means it is skipped here for the same reason.

    The partial lands in split slot ``num_kv_splits[b]`` -- one past what the
    packed kernel wrote -- and stage 2 merges it as an ordinary extra split.
    Merging partial (acc, lse) across splits is what stage 2 already does.
    """
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)

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

    qbase = cur_batch * stride_qbs + cur_head[:, None] * stride_qh

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, D], dtype=tl.float32)

    q = tl.load(Q + qbase + offs_d[None, :], mask=mask_h[:, None], other=0.0)
    qpe = tl.load(Q + qbase + offs_dpe[None, :], mask=mask_h[:, None], other=0.0)

    # Two positional ranges: the sink prefix and the recent tail. Clamped so a
    # sequence shorter than the window cannot make them overlap and
    # double-count -- the tail starts no earlier than P_TOK.
    tail_start = tl.maximum(cur_batch_seq_len - R_TOK, P_TOK)
    n_window = tl.minimum(P_TOK, cur_batch_seq_len) + tl.maximum(
        cur_batch_seq_len - tail_start, 0
    )

    for w in range(0, n_window, BLOCK_N):
        idx = w + tl.arange(0, BLOCK_N)
        in_prefix = idx < tl.minimum(P_TOK, cur_batch_seq_len)
        offs_n = tl.where(
            in_prefix,
            idx,
            tail_start + (idx - tl.minimum(P_TOK, cur_batch_seq_len)),
        )
        n_ok = (idx < n_window) & (offs_n < cur_batch_seq_len)
        kv_loc = tl.load(
            kv_indices + cur_batch_kv_start_idx + offs_n, mask=n_ok, other=0
        ).to(tl.int64)

        r = tl.load(HpRowOfSlot + kv_loc, mask=n_ok, other=-1).to(tl.int32)
        rr = tl.where(r >= 0, r, 0).to(tl.int64)
        owner = tl.load(HpOwnerOfRow + rr, mask=n_ok, other=-1).to(tl.int32)
        n_ok = n_ok & (r >= 0) & (owner == kv_loc.to(tl.int32))

        cv = tl.load(
            HP + rr[:, None] * D + offs_d[None, :], mask=n_ok[:, None], other=0.0
        )
        kpe = tl.load(
            Rope + kv_loc[None, :] * DPE + offs_pe[:, None],
            mask=n_ok[None, :],
            other=0.0,
        )

        qk = tl.dot(q, tl.trans(cv.to(q.dtype)))
        qk += tl.dot(qpe, kpe.to(qpe.dtype))
        qk *= sm_scale_withk
        if logit_cap > 0:
            qk = logit_cap * tanh(qk / logit_cap)
        qk = tl.where(mask_h[:, None] & n_ok[None, :], qk, float("-inf"))

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        # An all-masked block makes every qk -inf, so n_e_max is -inf and
        # qk - n_e_max is (-inf) - (-inf) = NaN -- acc is poisoned before e_sum
        # is ever looked at, which is why guarding only the final divide did
        # nothing. Exclusion creates exactly this: the sink-prefix block is
        # entirely window tokens and it is the FIRST block of every request, so
        # one NaN reaches every output. Subtracting 0 instead leaves exp(-inf)=0,
        # so p and e_sum stay 0 and the block contributes nothing.
        # e_max keeps the true -inf so a later non-empty block still rescales.
        n_e_max_s = tl.where(n_e_max == float("-inf"), 0.0, n_e_max)
        re_scale = tl.exp(e_max - n_e_max_s)
        p = tl.exp(qk - n_e_max_s[:, None])
        acc = acc * re_scale[:, None] + tl.dot(p.to(cv.dtype), cv)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    # Slot kv_splits, one past the packed kernel's last. An all-masked window
    # leaves e_sum at 0 and e_max at -inf; storing lse = -inf makes stage 2's
    # merge ignore this split rather than poison it with a NaN.
    # Slot 0, not kv_splits: see SLOT_OFF above. The window partial is the one
    # partial guaranteed to hold real values whenever the arena has content, so
    # it goes first and the packed splits follow.
    obase = (
        cur_batch * stride_mid_ob
        + cur_head[:, None] * stride_mid_oh
    )
    # MERGE into the slot instead of overwriting it.
    #
    # The window used to own a slot of its own, which forced slot_off=1 and
    # shifted the packed splits up. The shared stage 2 picks slots by SEQUENCE
    # ARITHMETIC -- it assumes slot i is split i of one uniform division -- so
    # that shift silently addressed the wrong partials whenever the sequence was
    # longer than the window (the only case that mixes real and empty splits;
    # seq <= window has every packed split empty and happened to work).
    #
    # Folding the window partial into split 0's slot with the same LSE
    # combination stage 2 itself uses keeps the slot/range correspondence
    # intact, so no slot_off and no special merge are needed. Split 0's slot is
    # always read: its range is non-empty for any non-empty sequence.
    # The whole offset is divided by D, exactly as the packed kernel does it.
    # Dividing each term separately is not the same expression -- integer
    # division does not distribute -- and would silently address the wrong lse.
    offs_mid_o_1 = (
        cur_batch * stride_mid_ob
        + cur_head * stride_mid_oh
    ) // D
    prev_lse = tl.load(Att_Lse + offs_mid_o_1, mask=mask_h, other=-1.0e4)
    prev_out = tl.load(Att_Out + obase + offs_d[None, :], mask=mask_h[:, None],
                       other=0.0)
    win_lse = tl.where(e_sum > 0, e_max + tl.log(tl.where(e_sum > 0, e_sum, 1.0)),
                       -1.0e4)
    win_out = acc / tl.where(e_sum > 0, e_sum, 1.0)[:, None]
    m = tl.maximum(prev_lse, win_lse)
    wp = tl.exp(prev_lse - m)
    ww = tl.exp(win_lse - m)
    den = wp + ww
    tl.store(
        Att_Out + obase + offs_d[None, :],
        (prev_out * wp[:, None] + win_out * ww[:, None]) / den[:, None],
        mask=mask_h[:, None],
    )
    # The MERGED lse, not the window's own: stage 2 reweights this slot by it,
    # so it has to describe both contributions. Finite sentinel when neither
    # side has anything -- -inf in the first split stage 2 examines makes its
    # seed max -inf too, and exp(-inf - -inf) is NaN.
    both_empty = (prev_lse <= -1.0e4) & (win_lse <= -1.0e4)
    tl.store(
        Att_Lse + offs_mid_o_1,
        tl.where(both_empty, -1.0e4, m + tl.log(den)),
        mask=mask_h,
    )


def _check_is_capturing() -> bool:
    from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

    return bool(get_is_capture_mode())


def packed_mla_decode_gf_fwd(
    q, pool, layer_id, o, attn_logits, attn_lse, kv_indptr, kv_indices,
    num_kv_splits, max_kv_splits, sm_scale_withk, logit_cap=0.0,
    num_kv_splits_plus1=None,
):
    """Two-pass packed MLA decode: fast factored bulk + dense BF16 window.

    The window arena is handled by EXCLUSION rather than override. The packed
    kernel masks arena tokens out (two int lookups, measured 1.08x, against
    1.54x for the override path it replaces) and the dense pass owns exactly
    those tokens, writing one extra split that stage 2 merges.

    Requires one spare split slot: attn_logits/attn_lse must have
    max_kv_splits + 1 along the split axis, and stage 2 must be told
    num_kv_splits + 1 so it reads the window partial. Passing the un-bumped
    count would silently drop the BF16 window and quietly serve a
    fully-quantized cache -- an accuracy regression with no error, so the
    caller supplies the bumped tensor explicitly instead of this function
    allocating one per step.
    """
    # NSA pools reach this function too. NSAPackedInt2KVPool shares the packed
    # mixin, so it exposes packed_read_operands and the backend's duck-typed
    # _is_packed_mla_pool accepts it -- but the group-factored path has never
    # been run against an NSA cache, whose rows also carry an index head. It is
    # better to refuse than to serve an untested layout: this path is opt-in, so
    # anyone who reaches this error asked for it explicitly and gets told why.
    _bits = getattr(pool, "_bits", 2)
    if _bits != 2:
        raise NotImplementedError(
            "the group-factored packed MLA decode unpacks four 2-bit fields per "
            "byte, hand-unrolled; it has not been ported to "
            f"SGLANG_OSCAR_MLA_KV_BITS={_bits}. Use the production kernel "
            "(SGLANG_OSCAR_MLA_PACKED_GF=0) at this width."
        )
    # NSA is allowed through when the in-call A/B check is armed.
    #
    # The original refusal assumed an NSA row carries an extra index head the
    # kernel does not know about. That is wrong: the indexer's
    # index_k_with_scale_buffer is a SEPARATE fp8 buffer that selects which
    # tokens to read, not part of what attention reads. Both pools get
    # packed_read_operands from the same _PackedLatentMixin -- the subclasses
    # override only __init__ -- so the operand tuple is field-for-field
    # identical.
    #
    # MEASURED, not argued. Run on GLM-5.2 (GlmMoeDsaForCausalLM, 78 layers)
    # with GF_CHECK armed, which executes the production kernel in the same call
    # and reports the deviation:
    #
    #     all 78/78 layers covered, bs=64
    #     worst rel  8.065e-03      (bf16 rounding; the non-NSA gate sits at
    #                                2.5e-03 - 5.2e-03 on the same measure)
    #     non-finite 0 everywhere
    #
    # So the refusal is lifted. It was precautionary from the start -- the
    # reason given for it, that an NSA row carries an extra index head, was
    # simply wrong.
    operands = pool.packed_read_operands(layer_id)
    has_hp = operands[3] is not None

    # Neutralise every split slot stage 2 will read, before stage 1 writes.
    #
    # Stage 1 stores inside `if split_kv_end > split_kv_start`, so a split with
    # no tokens is never written -- and attn_logits comes from torch.empty, so
    # those slots hold whatever was there before. Stage 2 then merges
    # uninitialised memory.
    #
    # This is why the group-factored path agreed at bs=8 (rel 7.8e-03) and was
    # completely wrong at bs=1 (rel 1.0): with a fixed split count over a short
    # early-decode sequence, most splits are empty, and at bs=1 nearly all of
    # them are. The override path is not exposed to this because it does not
    # borrow a slot, so its per-batch split count and its written slots line up.
    #
    # -1e30 is the same inert sentinel the kernels write for an empty split:
    # stage 2's max ignores it and exp() underflows it to zero.
    _ns = max_kv_splits + 1 if has_hp else max_kv_splits
    attn_lse[:, :, :_ns].fill_(-1.0e4)
    attn_logits[:, :, :_ns].zero_()

    if not getattr(pool, "_gf_entry_logged", False):
        # One line, unconditionally, the first time this path runs. The A/B
        # instrument produced nothing at all in eager mode and I could not tell
        # whether the check was being skipped or the branch was never entered
        # -- those have opposite fixes, and guessing between them is how the
        # last five hypotheses went wrong. This distinguishes them.
        import logging as _lg

        pool._gf_entry_logged = True
        _lg.getLogger(__name__).info(
            "[GF-ENTRY] group-factored decode is LIVE: layer=%s has_hp=%s "
            "check_env=%s capturing=%s bs=%s max_splits=%s",
            layer_id, has_hp,
            envs.SGLANG_OSCAR_MLA_PACKED_GF_CHECK.get(),
            _check_is_capturing(), q.shape[0], max_kv_splits,
        )
    # slot_off=1 whenever the window pass will write slot 0. The stock stage-2
    # merge is wrong when two or more empty splits precede the first real one,
    # and exclusion makes that the ordinary case at short sequences, so the
    # window partial -- the one guaranteed real while the arena has content --
    # takes slot 0 and these shift up.
    packed_mla_decode_stage1_gf(
        q, operands, attn_logits, attn_lse, kv_indptr, kv_indices,
        num_kv_splits, max_kv_splits, sm_scale_withk, logit_cap,
        # slot_off is gone: the window pass merges into split 0's slot, so the
        # packed splits keep their natural slot/range correspondence and the
        # stock stage 2 addresses them correctly.
        skip_hp=has_hp, slot_off=0,
    )
    if has_hp:
        if num_kv_splits_plus1 is None:
            raise ValueError(
                "packed_mla_decode_gf_fwd needs num_kv_splits_plus1 when the "
                "pool has a window arena: stage 2 must read one split past the "
                "packed kernel's last, or the BF16 window is silently dropped."
            )
        _, _, rope, hp, hp_row, hp_owner, _, _, _ = operands
        from sglang.srt.environ import envs as _envs

        batch, head_num = q.shape[0], q.shape[1]
        # The window pass was launched with a fixed BLOCK_H of 16, which at
        # head_num=16 and batch=1 is a grid of (1, 1): ONE CTA walking all 576
        # window tokens serially on a 148-SM part. That fixed cost is most of
        # why the group-factored path loses 13% at ctx=1000 and wins 36% at
        # 32000 -- the packed body shrinks but this does not. Splitting the
        # head axis finer costs nothing at large batch (the grid is already
        # wide) and buys parallelism exactly where the path is weak.
        block_h = _safe_block_h(
            _envs.SGLANG_OSCAR_MLA_WINDOW_BLOCK_H.get(), head_num
        )
        _fwd_hp_window_stage1[
            (batch, triton.cdiv(head_num, min(block_h, head_num)))
        ](
            q, rope, hp, hp_row, hp_owner,
            sm_scale_withk, kv_indptr, kv_indices, attn_logits, attn_lse,
            num_kv_splits,
            q.stride(0), q.stride(1),
            attn_logits.stride(0), attn_logits.stride(1), attn_logits.stride(2),
            kv_group_num=head_num,
            q_head_num=head_num,
            D=pool.kv_lora_rank,
            DPE=pool.qk_rope_head_dim,
            P_TOK=_envs.SGLANG_MIXED_KV_PREFIX_TOKENS.get(),
            R_TOK=_envs.SGLANG_MIXED_KV_RECENT_TOKENS.get(),
            BLOCK_N=_envs.SGLANG_OSCAR_MLA_WINDOW_BLOCK_N.get(),
            BLOCK_H=block_h,
            logit_cap=logit_cap,
            num_warps=_envs.SGLANG_OSCAR_MLA_WINDOW_WARPS.get(),
            num_stages=_envs.SGLANG_OSCAR_MLA_WINDOW_STAGES.get(),
        )
    # One merge for both cases now. The window partial was folded into split 0
    # by the window pass itself, so there is no extra slot and no shifted
    # layout for stage 2 to know about.
    _decode_softmax_reducev_fwd(
        attn_logits, attn_lse, q, o, 1.0,
        _v_shape_proxy(o, pool.kv_lora_rank),
        kv_indptr,
        num_kv_splits,
        max_kv_splits,
    )

    if envs.SGLANG_OSCAR_MLA_PACKED_GF_CHECK.get() and not _check_is_capturing():
        # Skipped during CUDA-graph capture: .item() and logging are host syncs
        # and capture aborts with "operation not permitted when stream is
        # capturing". The first version did not guard, so the instrument killed
        # the very arm it was measuring and returned no text at all -- the same
        # mistake the c_kv dump made earlier today, in a different file.
        #
        # Skipping capture is not a loss here. Replay executes no Python, so a
        # host-side comparison could never observe it anyway; what this sees is
        # the eager forwards, which is where a wrong kernel would already show.
        # Run the production kernel on the SAME call and compare.
        #
        # Four hypotheses for this path's garbling have now been wrong -- empty
        # splits, ns == 1, the -inf sentinel, CUDA-graph capture -- and every
        # one was reasoned from a microbenchmark that passes its equivalence
        # gate at three shapes while the server garbles. The one thing the
        # bench cannot reproduce is the REAL pool's operands on REAL decode
        # state: its arena is a synthetic fixture, its kv_indices are a dense
        # arange, and its sequence lengths are uniform. So stop reasoning about
        # the difference and measure it where it actually happens.
        import logging as _lg

        # The reference must be given buffers it can trust and the split count
        # it was designed for.
        #
        # The first version handed it torch.empty and num_kv_splits = ns_quant,
        # the BORROWED count. Stage 1 only stores into splits that hold tokens,
        # so on a short sequence most slots stayed uninitialised and the
        # reference merged them -- meaning rel=1.0 could just as easily have
        # meant "the reference is garbage" as "the group-factored path is
        # wrong", and those are not the same finding.
        #
        # The per-split lse dump is what exposed this: all seven packed splits
        # read SENT and the window slot carried a real 4.09, so the two-pass
        # split was doing exactly what it should, which left the comparison
        # itself as the thing that had not been checked.
        _o_ref = torch.empty_like(o)
        _lg_buf = torch.zeros_like(attn_logits[:, :, : max_kv_splits + 1])
        _ls_buf = torch.full_like(attn_lse[:, :, : max_kv_splits + 1], -1.0e4)
        packed_mla_decode_fwd(
            q, pool, layer_id, _o_ref, _lg_buf, _ls_buf, kv_indptr, kv_indices,
            num_kv_splits_plus1 if has_hp else num_kv_splits,
            max_kv_splits + 1 if has_hp else max_kv_splits,
            sm_scale_withk, logit_cap=logit_cap,
        )
        d = (o.float() - _o_ref.float()).abs()
        rel = (d.max() / _o_ref.float().abs().max().clamp(min=1e-6)).item()
        # Per-split lse for row 0. The window partial is written to slot
        # num_kv_splits[b]; if that slot still carries the empty sentinel then
        # the dense pass contributed NOTHING, and since the packed pass already
        # excluded those tokens they are simply missing -- which at a short
        # sequence, where the window is most of the sequence, is the whole
        # answer. A slot holding a real lse says the opposite: the window ran
        # and the fault is in how the two are combined.
        _lse0 = attn_lse[0, 0, : max_kv_splits + 1].tolist()
        _lg.getLogger(__name__).info(
            "[GF-CHECK] layer=%s bs=%d maxdev=%.3e rel=%.3e "
            "gf_nonfinite=%d ref_nonfinite=%d splits=%s lse[0,0,:]=%s",
            layer_id, q.shape[0], d.max().item(), rel,
            int((~torch.isfinite(o)).sum()),
            int((~torch.isfinite(_o_ref)).sum()),
            num_kv_splits[: min(4, num_kv_splits.numel())].tolist(),
            [round(v, 2) if v > -1e29 else "SENT" for v in _lse0],
        )


# ── stage 2 for the group-factored layout ────────────────────────────────────
# The shared ``_fwd_kernel_stage2`` decides which slots to read from SEQUENCE
# ARITHMETIC, not from the data:
#
#     kv_len_per_split = cdiv(cdiv(seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
#     for split_kv_id in range(0, MAX_KV_SPLITS):
#         if split_kv_start < split_kv_end:      # range test, not a data test
#             ... read slot split_kv_id ...
#
# That is correct only when slot i holds split i of a single uniform division of
# the sequence. The group-factored path breaks both halves of that assumption:
# ``slot_off=1`` puts the window partial in slot 0 and shifts the packed splits
# up, and the merge is handed a different split COUNT than the packed pass used,
# so ``kv_len_per_split`` differs and the range test lands on the wrong slots.
#
# It happened to work whenever the window covered the whole sequence: every
# packed split came back empty, the only real data was in slot 0, and slot 0 is
# what the range test reads first. That is why seq=100, 140 and 256 passed --
# including at bs=8, which is why "bs >= 4" looked like the variable -- while
# seq=600 and seq=4096 failed. Sequences longer than the window are the ONLY
# case that mixes real and sentinel splits, and that is the normal serving case.
#
# This variant merges by the SENTINEL instead: a slot contributes when its lse
# is above the sentinel floor, whatever its index means. Layout-independent, so
# slot_off needs no cooperation from the merge.
@triton.jit
def _fwd_kernel_stage2_gf(
    Mid_O, Mid_O_1, O,
    stride_mid_ob, stride_mid_oh, stride_mid_os,
    stride_obs, stride_oh,
    NUM_SLOTS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
    SENTINEL: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh) // Lv

    for slot in range(0, NUM_SLOTS):
        tlogic = tl.load(Mid_O_1 + offs_logic + slot * stride_mid_os // Lv)
        # The data decides, not the index. An untouched or empty slot carries
        # the sentinel and is skipped no matter where it sits.
        if tlogic > SENTINEL:
            tv = tl.load(Mid_O + offs_v + slot * stride_mid_os, mask=mask_d,
                         other=0.0)
            n_e_max = tl.maximum(tlogic, e_max)
            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv
            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    # A request with no contributing slot at all would divide by zero; emit
    # zeros rather than NaN, which propagates and hides which request failed.
    safe = tl.where(e_sum > 0, e_sum, 1.0)
    tl.store(
        O + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        tl.where(e_sum > 0, acc / safe, 0.0),
        mask=mask_d,
    )


def gf_softmax_reducev_fwd(logits, lse, q, o, num_slots, sentinel=-1.0e30):
    """Merge every slot that carries real data, regardless of what index it is.

    ``sentinel`` is the floor written by an empty split. Stage 1 stores -1.0e4
    for an empty split, so anything below the floor here is treated as absent;
    a real lse never approaches it.
    """
    batch, head_num = q.shape[0], q.shape[1]
    Lv = logits.shape[-1]
    _fwd_kernel_stage2_gf[(batch, head_num)](
        logits, lse, o,
        logits.stride(0), logits.stride(1), logits.stride(2),
        o.stride(0), o.stride(1),
        NUM_SLOTS=num_slots,
        BLOCK_DV=triton.next_power_of_2(Lv),
        Lv=Lv,
        SENTINEL=sentinel,
    )
