"""Split-KV decode attention over the unified pool when K (and optionally V)
is product-quantized: the quant tier's stage-1 reconstructs centroids inline
(or scores through a per-query lookup table, ADC), the HP tier and the
stage-2 reduction are the INT2 unified path's."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.QuantKernel.oscar_pq_kv import _nibble_layout
from sglang.srt.environ import envs
from sglang.srt.layers.attention.triton_ops.decode_attention import (
    _MIN_BLOCK_KV,
    _decode_att_m_fwd,
    _decode_grouped_att_m_fwd,
    _unified_stage2,
    tanh,
)


@triton.jit
def _pq_build_lut_kernel(
    Q,
    Codebook,
    Lut,
    HEAD_DIM: tl.constexpr,
    N_SUB: tl.constexpr,
    SUB_DIM: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
):
    batch_q_head = tl.program_id(0)
    sub = tl.program_id(1)
    centroids = tl.arange(0, N_CENTROIDS)
    acc = tl.zeros([N_CENTROIDS], dtype=tl.float32)
    for dim in tl.static_range(SUB_DIM):
        q_value = tl.load(Q + batch_q_head * HEAD_DIM + sub * SUB_DIM + dim).to(tl.float32)
        centroid = tl.load(Codebook + (sub * N_CENTROIDS + centroids) * SUB_DIM + dim).to(
            tl.float32
        )
        acc += q_value * centroid
    tl.store(Lut + (batch_q_head * N_SUB + sub) * N_CENTROIDS + centroids, acc)


def _build_pq_lut(q: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """``lut[b, h, s, c] = q[b, h, s*sub_dim:(s+1)*sub_dim] . codebook[s, c]``."""
    batch, q_heads, head_dim = q.shape
    n_sub, n_centroids, sub_dim = codebook.shape
    assert q.is_contiguous() and codebook.is_contiguous()
    assert head_dim == n_sub * sub_dim
    lut = torch.empty((batch, q_heads, n_sub, n_centroids), dtype=torch.float32, device=q.device)
    _pq_build_lut_kernel[(batch * q_heads, n_sub)](
        q,
        codebook,
        lut,
        HEAD_DIM=head_dim,
        N_SUB=int(n_sub),
        SUB_DIM=int(sub_dim),
        N_CENTROIDS=int(n_centroids),
        num_warps=8,
        num_stages=2,
    )
    return lut


@triton.jit
def _fwd_grouped_kernel_stage1_pq(
    Q,
    K_Codes,
    V_Buffer,
    V_Scales_Zeros,
    K_Codebook,
    K_Lut,
    K_Codes2,
    K_Codebook2,
    K_Lut2,
    V_Codebook,
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_ks,
    stride_lut_b,
    stride_lut_h,
    stride_lut_sub,
    stride_lut_centroid,
    stride_k2bs,
    stride_k2h,
    stride_k2s,
    stride_lut2_b,
    stride_lut2_h,
    stride_lut2_sub,
    stride_lut2_centroid,
    stride_vbs,
    stride_vh,
    stride_vs,
    stride_vszbs,
    stride_vszh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    xai_temperature_len: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    K_SUB_DIM: tl.constexpr,
    K_N_CENTROIDS: tl.constexpr,
    K2_N_CENTROIDS: tl.constexpr,
    V_SUB_DIM: tl.constexpr,
    V_N_CENTROIDS: tl.constexpr,
    V_GROUP_SIZE: tl.constexpr,
    HAS_K_STAGE2: tl.constexpr,
    K2_NIBBLE: tl.constexpr,
    V_IS_PQ: tl.constexpr,
    USE_ADC: tl.constexpr,
):
    """Graph-safe PQ-K attention with optional residual K and PQ V. Follows the
    tier-local indptr; no gathered tensor has a data-dependent host shape, so
    padded CUDA-graph index buffers are safe."""
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (cur_head_id + 1) * VALID_BLOCK_H) & (cur_head < q_head_num)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)

    offs_dk = tl.arange(0, BLOCK_DK)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_dk = offs_dk < Lk
    mask_dv = offs_dv < Lv
    q = tl.load(
        Q + cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dk[None, :],
        mask=mask_h[:, None] & mask_dk[None, :],
        other=0.0,
    )

    batch_kv_start = tl.load(kv_indptr + cur_batch)
    seq_len = tl.load(kv_indptr + cur_batch + 1) - batch_kv_start
    kv_splits = tl.load(num_kv_splits + cur_batch)
    kv_len_per_split = tl.cdiv(tl.cdiv(seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    split_start = kv_len_per_split * split_kv_id
    split_end = tl.minimum(split_start + kv_len_per_split, seq_len)

    if xai_temperature_len > 0:
        offs_qidx = seq_len - 1
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        qtemp = tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale
        xai_temperature_reg = tl.where(offs_qidx > xai_temperature_len, qtemp, 1.0)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_end > split_start:
        k_sub = offs_dk // K_SUB_DIM
        k_sub_off = offs_dk % K_SUB_DIM
        if not V_IS_PQ:
            v_byte_dim: tl.constexpr = Lv // 4
            v_byte_off = offs_dv % v_byte_dim
            v_shift = (offs_dv // v_byte_dim) * 2
            v_group = offs_dv // V_GROUP_SIZE

        for start_n in range(split_start, split_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            valid_n = offs_n < split_end
            kv_loc = tl.load(kv_indices + batch_kv_start + offs_n, mask=valid_n, other=0).to(
                tl.int64
            )

            if USE_ADC:
                qk = tl.zeros([BLOCK_H, BLOCK_N], dtype=tl.float32)
                for sub in tl.static_range(Lk // K_SUB_DIM):
                    code = tl.load(
                        K_Codes + kv_loc * stride_kbs + cur_kv_head * stride_kh + sub * stride_ks,
                        mask=valid_n,
                        other=0,
                    ).to(tl.int32)
                    qk += tl.load(
                        K_Lut
                        + cur_batch * stride_lut_b
                        + cur_head[:, None] * stride_lut_h
                        + sub * stride_lut_sub
                        + code[None, :] * stride_lut_centroid,
                        mask=mask_h[:, None] & valid_n[None, :],
                        other=0.0,
                    )
                    if HAS_K_STAGE2:
                        if K2_NIBBLE:
                            k2_byte: tl.constexpr = sub // 2
                            k2_shift: tl.constexpr = (sub % 2) * 4
                        else:
                            k2_byte: tl.constexpr = sub
                            k2_shift: tl.constexpr = 0
                        code2 = tl.load(
                            K_Codes2
                            + kv_loc * stride_k2bs
                            + cur_kv_head * stride_k2h
                            + k2_byte * stride_k2s,
                            mask=valid_n,
                            other=0,
                        ).to(tl.int32)
                        if K2_NIBBLE:
                            code2 = (code2 >> k2_shift) & 0xF
                        qk += tl.load(
                            K_Lut2
                            + cur_batch * stride_lut2_b
                            + cur_head[:, None] * stride_lut2_h
                            + sub * stride_lut2_sub
                            + code2[None, :] * stride_lut2_centroid,
                            mask=mask_h[:, None] & valid_n[None, :],
                            other=0.0,
                        )
                qk *= sm_scale
            else:
                k_code = tl.load(
                    K_Codes
                    + kv_loc[None, :] * stride_kbs
                    + cur_kv_head * stride_kh
                    + k_sub[:, None] * stride_ks,
                    mask=mask_dk[:, None] & valid_n[None, :],
                    other=0,
                ).to(tl.int64)
                k = tl.load(
                    K_Codebook + (k_sub[:, None] * K_N_CENTROIDS + k_code) * K_SUB_DIM + k_sub_off[:, None],
                    mask=mask_dk[:, None] & valid_n[None, :],
                    other=0.0,
                ).to(q.dtype)
                if HAS_K_STAGE2:
                    if K2_NIBBLE:
                        k2_byte_idx = k_sub // 2
                        k2_shift_t = (k_sub % 2) * 4
                    else:
                        k2_byte_idx = k_sub
                        k2_shift_t = k_sub * 0
                    k_code2 = tl.load(
                        K_Codes2
                        + kv_loc[None, :] * stride_k2bs
                        + cur_kv_head * stride_k2h
                        + k2_byte_idx[:, None] * stride_k2s,
                        mask=mask_dk[:, None] & valid_n[None, :],
                        other=0,
                    ).to(tl.int32)
                    if K2_NIBBLE:
                        k_code2 = (k_code2 >> k2_shift_t[:, None]) & 0xF
                    k_code2 = k_code2.to(tl.int64)
                    k += tl.load(
                        K_Codebook2
                        + (k_sub[:, None] * K2_N_CENTROIDS + k_code2) * K_SUB_DIM
                        + k_sub_off[:, None],
                        mask=mask_dk[:, None] & valid_n[None, :],
                        other=0.0,
                    ).to(q.dtype)
                qk = tl.dot(q, k) * sm_scale
            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)
            if xai_temperature_len > 0:
                qk *= xai_temperature_reg
            qk = tl.where(mask_h[:, None] & valid_n[None, :], qk, float("-inf"))

            next_e_max = tl.maximum(tl.max(qk, axis=1), e_max)
            rescale = tl.exp(e_max - next_e_max)
            p = tl.exp(qk - next_e_max[:, None])
            acc *= rescale[:, None]

            if V_IS_PQ and BLOCK_DV == Lv:
                # One PQ subspace at a time: each code byte is loaded once and
                # no [BLOCK_N, Lv] V tile is materialized.
                v_dim = tl.arange(0, V_SUB_DIM)
                v_sub_ids = tl.arange(0, Lv // V_SUB_DIM)
                for sub in tl.static_range(Lv // V_SUB_DIM):
                    code = tl.load(
                        V_Buffer + kv_loc * stride_vbs + cur_kv_head * stride_vh + sub * stride_vs,
                        mask=valid_n,
                        other=0,
                    ).to(tl.int64)
                    values = tl.load(
                        V_Codebook + (sub * V_N_CENTROIDS + code[:, None]) * V_SUB_DIM + v_dim[None, :],
                        mask=valid_n[:, None],
                        other=0.0,
                    ).to(q.dtype)
                    partial = tl.dot(p.to(values.dtype), values)
                    partial_full = tl.broadcast_to(
                        partial[:, None, :], (BLOCK_H, Lv // V_SUB_DIM, V_SUB_DIM)
                    )
                    sub_mask = v_sub_ids[None, :, None] == sub
                    acc += tl.reshape(tl.where(sub_mask, partial_full, 0.0), (BLOCK_H, BLOCK_DV))
            elif V_IS_PQ:
                v_sub = offs_dv // V_SUB_DIM
                v_sub_off = offs_dv % V_SUB_DIM
                v_code = tl.load(
                    V_Buffer
                    + kv_loc[:, None] * stride_vbs
                    + cur_kv_head * stride_vh
                    + v_sub[None, :] * stride_vs,
                    mask=valid_n[:, None] & mask_dv[None, :],
                    other=0,
                ).to(tl.int64)
                v = tl.load(
                    V_Codebook + (v_sub[None, :] * V_N_CENTROIDS + v_code) * V_SUB_DIM + v_sub_off[None, :],
                    mask=valid_n[:, None] & mask_dv[None, :],
                    other=0.0,
                ).to(q.dtype)
                acc += tl.dot(p.to(v.dtype), v)
            else:
                v_packed = tl.load(
                    V_Buffer
                    + kv_loc[:, None] * stride_vbs
                    + cur_kv_head * stride_vh
                    + v_byte_off[None, :] * stride_vs,
                    mask=valid_n[:, None] & mask_dv[None, :],
                    other=0,
                )
                v_q = ((v_packed >> v_shift[None, :]) & 0x03).to(tl.float32)
                v_sz_base = kv_loc[:, None] * stride_vszbs + cur_kv_head * stride_vszh
                v_scale = tl.load(
                    V_Scales_Zeros + v_sz_base + 2 * v_group[None, :],
                    mask=valid_n[:, None] & mask_dv[None, :],
                    other=1.0,
                ).to(tl.float32)
                v_zero = tl.load(
                    V_Scales_Zeros + v_sz_base + 2 * v_group[None, :] + 1,
                    mask=valid_n[:, None] & mask_dv[None, :],
                    other=0.0,
                ).to(tl.float32)
                v = ((v_q - v_zero) * v_scale).to(q.dtype)
                acc += tl.dot(p.to(v.dtype), v)
            e_sum = e_sum * rescale + tl.sum(p, axis=1)
            e_max = next_e_max

        out_base = (
            cur_batch * stride_mid_ob + cur_head[:, None] * stride_mid_oh + split_kv_id * stride_mid_os
        )
        tl.store(
            Att_Out + out_base + offs_dv[None, :],
            acc / e_sum[:, None],
            mask=mask_h[:, None] & mask_dv[None, :],
        )
        lse_off = (
            cur_batch * stride_mid_ob + cur_head * stride_mid_oh + split_kv_id * stride_mid_os
        ) // Lv
        tl.store(Att_Lse + lse_off, e_max + tl.log(e_sum), mask=mask_h)


def _pick_block_h(kv_group_num: int, *, use_adc: bool, batch: int) -> int:
    configured = envs.SGLANG_OSCAR_PQ_BLOCK_H.get()
    requested = configured if configured > 0 else (1 if use_adc and batch < 4 else 4)
    if requested >= kv_group_num:
        return triton.next_power_of_2(kv_group_num)
    block_h = 1 << (requested.bit_length() - 1)
    while block_h > 1 and kv_group_num % block_h != 0:
        block_h //= 2
    return block_h


def _pick_tile(*, batch: int, large_tile_safe: bool) -> tuple[int, int, int]:
    """(BLOCK_N, num_warps, num_stages). Low batch benefits from a long
    sequence tile; high batch already exposes enough programs and needs the
    low-register BN=32 kernel. Measured on H100 for D=128; env overrides."""
    if large_tile_safe:
        default_block_n, default_warps, default_stages = (256, 8, 2) if batch < 16 else (64, 4, 2)
    else:
        default_block_n, default_warps, default_stages = 32, 4, 2
    configured_block_n = envs.SGLANG_OSCAR_PQ_BLOCK_N.get()
    configured_warps = envs.SGLANG_OSCAR_PQ_NUM_WARPS.get()
    configured_stages = envs.SGLANG_OSCAR_PQ_NUM_STAGES.get()
    requested_block_n = triton.next_power_of_2(
        max(16, configured_block_n if configured_block_n > 0 else default_block_n)
    )
    block_n = min(requested_block_n, 256 if large_tile_safe else 64)
    requested_warps = configured_warps if configured_warps > 0 else default_warps
    num_warps = min(8, triton.next_power_of_2(requested_warps))
    requested_stages = configured_stages if configured_stages > 0 else default_stages
    num_stages = min(max(1, requested_stages), 2 if block_n >= 256 else 3)
    return block_n, num_warps, num_stages


def _decode_grouped_att_m_fwd_pq(
    q,
    k_codes,
    v_buffer,
    v_scales_zeros,
    k_codebook,
    att_out,
    att_lse,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    logit_cap,
    xai_temperature_len=-1,
    k_codes2=None,
    k_codebook2=None,
    v_codebook=None,
):
    """Launch the inline PQ/RVQ quant-tier stage-1 for MHA, GQA or MQA."""
    Lk = int(q.shape[-1])
    Lv = int(att_out.shape[-1])
    k_n_sub, k_n_centroids, k_sub_dim = k_codebook.shape
    assert int(k_n_sub) * int(k_sub_dim) == Lk
    assert k_codes.shape[-1] == k_n_sub

    has_k_stage2 = k_codes2 is not None
    k2_nibble = False
    if has_k_stage2:
        assert k_codebook2 is not None
        assert k_codebook2.shape[0] == k_n_sub and k_codebook2.shape[2] == k_sub_dim
        k_codes2_arg, k_codebook2_arg = k_codes2, k_codebook2
        k2_n_centroids = int(k_codebook2.shape[1])
        k2_nibble = _nibble_layout(int(k_codes2.shape[-1]), int(k_n_sub), k2_n_centroids)
    else:
        k_codes2_arg, k_codebook2_arg = k_codes, k_codebook
        k2_n_centroids = int(k_n_centroids)

    batch, q_head_num = q.shape[0], q.shape[1]
    adc_mode = envs.SGLANG_OSCAR_PQ_USE_ADC.get()
    use_adc = batch < 4 if adc_mode < 0 else adc_mode != 0
    if use_adc:
        k_lut = _build_pq_lut(q, k_codebook)
        k_lut2 = _build_pq_lut(q, k_codebook2_arg) if has_k_stage2 else k_lut
        lut_strides = k_lut.stride()
        lut2_strides = k_lut2.stride()
    else:
        # Pointer/stride placeholders for the compile-time disabled ADC branch.
        k_lut, k_lut2 = k_codebook, k_codebook2_arg
        lut_strides = (0, 0, k_codebook.stride(0), k_codebook.stride(1))
        lut2_strides = (0, 0, k_codebook2_arg.stride(0), k_codebook2_arg.stride(1))

    v_is_pq = v_codebook is not None
    if v_is_pq:
        v_n_sub, v_n_centroids, v_sub_dim = v_codebook.shape
        assert int(v_n_sub) * int(v_sub_dim) == Lv
        assert v_buffer.shape[-1] == v_n_sub
        v_codebook_arg = v_codebook
        v_group_size = Lv
        # A PQ V tier has a zero-width scale arena; its data pointer may be
        # null, so hand the disabled branch a real tensor instead.
        v_scales_zeros = v_codebook
    else:
        assert v_buffer.shape[-1] * 4 == Lv
        v_n_centroids, v_sub_dim = 1, 1
        v_codebook_arg = k_codebook
        v_num_groups = int(v_scales_zeros.shape[-1]) // 2
        assert Lv % v_num_groups == 0
        v_group_size = Lv // v_num_groups

    kv_group_num = q_head_num // k_codes.shape[1]
    block_h = _pick_block_h(kv_group_num, use_adc=use_adc, batch=batch)
    large_tile_safe = max(Lk, Lv) <= 128 and v_is_pq and not has_k_stage2
    block_n, num_warps, num_stages = _pick_tile(batch=batch, large_tile_safe=large_tile_safe)
    grid = (batch, triton.cdiv(q_head_num, min(block_h, kv_group_num)), max_kv_splits)
    _fwd_grouped_kernel_stage1_pq[grid](
        q,
        k_codes,
        v_buffer,
        v_scales_zeros,
        k_codebook,
        k_lut,
        k_codes2_arg,
        k_codebook2_arg,
        k_lut2,
        v_codebook_arg,
        sm_scale,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_codes.stride(0),
        k_codes.stride(1),
        k_codes.stride(2),
        *lut_strides,
        k_codes2_arg.stride(0),
        k_codes2_arg.stride(1),
        k_codes2_arg.stride(2),
        *lut2_strides,
        v_buffer.stride(0),
        v_buffer.stride(1),
        v_buffer.stride(2),
        v_scales_zeros.stride(0),
        v_scales_zeros.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=q_head_num,
        BLOCK_DK=triton.next_power_of_2(Lk),
        BLOCK_DV=triton.next_power_of_2(Lv),
        BLOCK_N=block_n,
        BLOCK_H=block_h,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        Lk=Lk,
        Lv=Lv,
        K_SUB_DIM=int(k_sub_dim),
        K_N_CENTROIDS=int(k_n_centroids),
        K2_N_CENTROIDS=k2_n_centroids,
        V_SUB_DIM=int(v_sub_dim),
        V_N_CENTROIDS=int(v_n_centroids),
        V_GROUP_SIZE=v_group_size,
        HAS_K_STAGE2=has_k_stage2,
        K2_NIBBLE=k2_nibble,
        V_IS_PQ=v_is_pq,
        USE_ADC=use_adc,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def decode_attention_fwd_pq_unified(
    q,
    hp_k_buffer,
    hp_v_buffer,
    quant_k_codes,
    quant_v_buffer,
    quant_v_scales_zeros,
    o,
    hp_kv_indptr,
    hp_kv_indices,
    quant_kv_indptr,
    quant_kv_indices,
    attn_logits,
    attn_lse,
    hp_num_kv_splits,
    quant_num_kv_splits,
    hp_max_kv_splits,
    quant_max_kv_splits,
    sm_scale,
    *,
    k_codebook,
    k_codes2=None,
    k_codebook2=None,
    v_codebook=None,
    logit_cap=0.0,
    sinks=None,
    xai_temperature_len=-1,
):
    """Unified HP + PQ-K (+ INT2 or PQ V) decode attention; same scratch
    contract as ``decode_attention_fwd_int2_unified``."""
    if sinks is not None:
        raise NotImplementedError("Mixed KV windows do not support sink tokens in PQ decode.")
    total_splits = hp_max_kv_splits + quant_max_kv_splits
    assert attn_logits.shape[2] == total_splits
    attn_lse.fill_(float("-inf"))
    hp_logits = attn_logits[:, :, :hp_max_kv_splits, :]
    hp_lse = attn_lse[:, :, :hp_max_kv_splits]
    quant_logits = attn_logits[:, :, hp_max_kv_splits:, :]
    quant_lse = attn_lse[:, :, hp_max_kv_splits:]
    kv_group_num = q.shape[1] // hp_k_buffer.shape[1]

    if hp_kv_indices.numel() > 0:
        hp_stage1 = _decode_att_m_fwd if kv_group_num == 1 else _decode_grouped_att_m_fwd
        hp_stage1(
            q,
            hp_k_buffer,
            hp_v_buffer,
            hp_logits,
            hp_lse,
            hp_kv_indptr,
            hp_kv_indices,
            hp_num_kv_splits,
            hp_max_kv_splits,
            sm_scale,
            logit_cap,
            xai_temperature_len,
        )
    if quant_kv_indices.numel() > 0:
        _decode_grouped_att_m_fwd_pq(
            q,
            quant_k_codes,
            quant_v_buffer,
            quant_v_scales_zeros,
            k_codebook,
            quant_logits,
            quant_lse,
            quant_kv_indptr,
            quant_kv_indices,
            quant_num_kv_splits,
            quant_max_kv_splits,
            sm_scale,
            logit_cap,
            xai_temperature_len,
            k_codes2=k_codes2,
            k_codebook2=k_codebook2,
            v_codebook=v_codebook,
        )
    _unified_stage2(attn_logits, attn_lse, o, total_splits=total_splits)
    return o
