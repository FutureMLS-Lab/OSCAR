"""Real INT2 pack/unpack kernels for the MLA latent (``c_kv``).

The existing MLA path (``mem_cache/mla_int2_kv_pool.py``) is *fake* quant: it
rounds ``c_kv`` and writes the result straight back into the BF16 pool, so it
measures accuracy but saves no memory and costs extra time. These kernels store
the quantized codes for real -- 8//bits codes per byte, plus two fp32 group
parameters -- so the latent occupies `bits` bits/value plus scale overhead.
bits=2 is the default; bits=4 exists because some models' latents are not
INT2-representable at any grouping (measured: Kimi-K3 at 0.28-0.46 rel-err).

Layout, per row of ``group_size`` values:
    codes : uint8[group_size // (8 // bits)]  little-endian BITS-bit fields
    params: fp32[2]                  (scale, zero) meaning depends on the mode

Both quantizer modes of ``_fake_quant_int2_groupwise`` are reproduced exactly:

* uniform  q = round((x - min) / s).clamp(0, 3),  x_deq = q * s + min
           -> (scale, zero) = (s, min),               x_deq = q * scale + zero
* lloyd    q = #{t : z >= t} for the three LM thresholds,
           x_deq = (q - uniform_zero) * uniform_scale
           -> (scale, zero) = (uniform_scale, uniform_zero),
              x_deq = (q - zero) * scale

Known, measured limit: ~0.010 % of codes can differ from the torch reference by
one step when quantizing *raw* (unrotated, BF16-coarse) c_kv. Those are values
whose quotient lands within half a ULP of a .5 rounding boundary -- e.g. a true
quotient of 1.49999997 is exactly 1.5 once rounded to fp32, so torch rounds it
to 2 while a quotient computed 1 ULP lower rounds to 1. Both are defensible;
matching torch exactly would require replicating its instruction sequence. On
the path that actually runs (rotate -> quantize -> unrotate) the effect is
relL2 ~3.6e-07, seven orders below the quantization error itself.

Dequant mirrors whichever form the reference uses -- ``q * scale + min`` for
uniform, ``(q - zero) * scale`` for Lloyd-Max -- because the algebraically equal
alternative rounds differently in fp32 and costs bit-exactness.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Lloyd-Max constants for N(0,1), mirrored from mla_int2_kv_pool so the two
# implementations cannot drift apart silently.
from sglang.srt.mem_cache.mla_int2_kv_pool import (
    _LM_CENTROIDS,
    _LM_RATIO,
    _LM_SPAN,
    _LM_THRESHOLDS,
)


@triton.jit
def _round_half_even(x):
    """torch.round semantics (ties to even), exactly.

    NOT ``floor(x + 0.5)``: for x just below a .5 boundary the fp32 addition
    itself rounds up to the boundary and floor then jumps a whole step. That
    produced off-by-one codes on 1.34 % of real GLM-5.2 c_kv groups -- an error
    of one full quantization step, not an epsilon. ``x - floor(x)`` is exact for
    the [0, 3] range used here, so comparing the fraction is safe.
    """
    f = tl.floor(x)
    frac = x - f
    is_odd = (f - 2.0 * tl.floor(f * 0.5)) == 1.0
    up = (frac > 0.5) | ((frac == 0.5) & is_odd)   # tie -> round to even
    return f + tl.where(up, 1.0, 0.0)


@triton.jit
def _quant_pack_kernel(
    x_ptr, codes_ptr, params_ptr,
    n_groups, group_size: tl.constexpr, LLOYD: tl.constexpr,
    BITS: tl.constexpr, PF: tl.constexpr,
    MASK: tl.constexpr, MAXQ: tl.constexpr,
    T0: tl.constexpr, T1: tl.constexpr, T2: tl.constexpr,
    LM_SPAN3: tl.constexpr, LM_RATIO: tl.constexpr, LM_C0: tl.constexpr,
):
    gid = tl.program_id(0)
    if gid >= n_groups:
        return
    base = gid * group_size

    # pass 1: whole-group statistics
    offs = tl.arange(0, group_size)
    x = tl.load(x_ptr + base + offs).to(tl.float32)
    if LLOYD:
        mean = tl.fdiv(tl.sum(x, axis=0), float(group_size), ieee_rounding=True)
        d = x - mean
        std = tl.sqrt(tl.fdiv(tl.sum(d * d, axis=0), float(group_size), ieee_rounding=True) + 1e-8)
        scale = LM_SPAN3 * LM_RATIO * std
        zero = -LM_C0 / LM_SPAN3 - tl.fdiv(mean, scale, ieee_rounding=True)
    else:
        x_min = tl.min(x, axis=0)
        rng = tl.max(x, axis=0) - x_min
        # true division, not multiply-by-reciprocal: a 1-ULP difference in
        # scale shifts every quotient downstream
        scale = tl.where(tl.abs(rng) > 1e-8, tl.fdiv(rng, MAXQ, ieee_rounding=True), 1.0)
        # store x_min itself: dequant then mirrors the reference's
        # ``q * scale + x_min`` exactly instead of an algebraically equal but
        # differently-rounded ``(q - zero) * scale``
        zero = x_min
        mean = 0.0
        std = 1.0

    # pass 2: four interleaved planes -> one packed byte each
    nb: tl.constexpr = group_size // PF
    ob = tl.arange(0, nb)
    packed = tl.zeros([nb], dtype=tl.int32)
    for j in tl.static_range(PF):
        xj = tl.load(x_ptr + base + PF * ob + j).to(tl.float32)
        if LLOYD:
            zj = tl.fdiv(xj - mean, std, ieee_rounding=True)
            qj = ((zj >= T0).to(tl.float32)
                  + (zj >= T1).to(tl.float32)
                  + (zj >= T2).to(tl.float32))
        else:
            # IEEE round-to-nearest division: Triton's default fast division
            # differs from torch by 1 ULP, which flips the rounding whenever the
            # quotient sits exactly on a .5 boundary. That was 402 codes (0.010 %)
            # off by a full step on real GLM-5.2 c_kv.
            qj = _round_half_even(tl.fdiv(xj - zero, scale, ieee_rounding=True))
            qj = tl.minimum(tl.maximum(qj, 0.0), MAXQ)
        packed |= qj.to(tl.int32) << (BITS * j)

    tl.store(codes_ptr + gid * nb + ob, packed.to(tl.uint8))
    tl.store(params_ptr + gid * 2 + 0, scale)
    tl.store(params_ptr + gid * 2 + 1, zero)


@triton.jit
def _dequant_kernel(
    codes_ptr, params_ptr, out_ptr,
    n_groups, group_size: tl.constexpr, LLOYD: tl.constexpr,
    BITS: tl.constexpr, PF: tl.constexpr,
    MASK: tl.constexpr, MAXQ: tl.constexpr,
):
    gid = tl.program_id(0)
    if gid >= n_groups:
        return
    offs = tl.arange(0, group_size)
    byte = tl.load(codes_ptr + gid * (group_size // PF) + offs // PF).to(tl.int32)
    q = ((byte >> (BITS * (offs % PF))) & MASK).to(tl.float32)
    scale = tl.load(params_ptr + gid * 2 + 0)
    zero = tl.load(params_ptr + gid * 2 + 1)
    if LLOYD:
        val = (q - zero) * scale        # reference: (q - uniform_zero) * uniform_scale
    else:
        val = q * scale + zero          # reference: q * scale + x_min
    tl.store(out_ptr + gid * group_size + offs, val)


def quantize_pack(x: torch.Tensor, group_size: int = 128, lloyd_max: bool = False, bits: int = 2):
    """``[..., D]`` float -> (packed uint8 codes, fp32 group params)."""
    # Lloyd-Max here is a THREE-THRESHOLD codebook, i.e. 2-bit by construction.
    # Running it at any other width would emit codes in 0..3 while dequant reads
    # a wider field -- silently wrong, and silently wasting the extra bits.
    assert bits in (2, 4), f"unsupported bits={bits}"
    assert not (lloyd_max and bits != 2), (
        "lloyd_max is a 2-bit codebook; use uniform quantization at bits=%d" % bits)
    assert x.shape[-1] % group_size == 0, (x.shape, group_size)
    assert group_size % 4 == 0
    flat = x.reshape(-1, group_size).contiguous().to(torch.float32)
    n = flat.shape[0]
    codes = torch.empty((n, group_size // (8 // bits)), dtype=torch.uint8, device=x.device)
    params = torch.empty((n, 2), dtype=torch.float32, device=x.device)
    _quant_pack_kernel[(n,)](
        flat, codes, params, n, group_size=group_size, LLOYD=lloyd_max, BITS=bits, PF=8 // bits,
        MASK=(1 << bits) - 1, MAXQ=float((1 << bits) - 1),
        T0=_LM_THRESHOLDS[0], T1=_LM_THRESHOLDS[1], T2=_LM_THRESHOLDS[2],
        LM_SPAN3=_LM_SPAN / 3.0, LM_RATIO=_LM_RATIO, LM_C0=_LM_CENTROIDS[0],
    )
    return codes, params


def dequantize(codes: torch.Tensor, params: torch.Tensor, shape, group_size: int = 128,
               dtype: torch.dtype = torch.float32, lloyd_max: bool = False,
               bits: int = 2) -> torch.Tensor:
    n = codes.shape[0]
    out = torch.empty((n, group_size), dtype=torch.float32, device=codes.device)
    _dequant_kernel[(n,)](codes, params, out, n, group_size=group_size, LLOYD=lloyd_max, BITS=bits, PF=8 // bits, MASK=(1 << bits) - 1, MAXQ=float((1 << bits) - 1))
    return out.reshape(shape).to(dtype)


def bytes_per_value(group_size: int = 128, bits: int = 2) -> float:
    """Storage cost: 2 bits of codes + two fp32 group params, amortized."""
    return (group_size / (8 // bits) + 8) / group_size


# ── fused, block-tiled variant ───────────────────────────────────────────────
# The kernels above use one program per 128-value group. That is 512 bytes of
# work per program, so at any realistic token count the run time is pure launch
# and scheduling overhead -- measured 0.033 ms whether you hand it 1 token or
# 4096. This variant gives each program GROUPS_PER_BLOCK groups and folds the
# dequant into the same launch, which is what the current wiring needs anyway
# (the codes are unpacked straight back).


@triton.jit
def _fused_quant_dequant_kernel(
    x_ptr, codes_ptr, params_ptr, out_ptr,
    n_groups,
    GS: tl.constexpr, GPB: tl.constexpr, LLOYD: tl.constexpr,
    BITS: tl.constexpr, PF: tl.constexpr,
    MASK: tl.constexpr, MAXQ: tl.constexpr,
    T0: tl.constexpr, T1: tl.constexpr, T2: tl.constexpr,
    LM_SPAN3: tl.constexpr, LM_RATIO: tl.constexpr, LM_C0: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * GPB + tl.arange(0, GPB)
    rmask = rows < n_groups
    cols = tl.arange(0, GS)
    idx = rows[:, None] * GS + cols[None, :]
    x = tl.load(x_ptr + idx, mask=rmask[:, None], other=0.0).to(tl.float32)

    if LLOYD:
        mean = tl.fdiv(tl.sum(x, axis=1), float(GS), ieee_rounding=True)
        d = x - mean[:, None]
        std = tl.sqrt(
            tl.fdiv(tl.sum(d * d, axis=1), float(GS), ieee_rounding=True) + 1e-8
        )
        z = tl.fdiv(d, std[:, None], ieee_rounding=True)
        q = ((z >= T0).to(tl.float32)
             + (z >= T1).to(tl.float32)
             + (z >= T2).to(tl.float32))
        scale = LM_SPAN3 * LM_RATIO * std
        zero = -LM_C0 / LM_SPAN3 - tl.fdiv(mean, scale, ieee_rounding=True)
        deq = (q - zero[:, None]) * scale[:, None]
    else:
        big = tl.full(x.shape, float("inf"), tl.float32)
        x_min = tl.min(tl.where(rmask[:, None], x, big), axis=1)
        x_max = tl.max(tl.where(rmask[:, None], x, -big), axis=1)
        rng = x_max - x_min
        scale = tl.where(
            tl.abs(rng) > 1e-8, tl.fdiv(rng, MAXQ, ieee_rounding=True), 1.0
        )
        zero = x_min
        q = _round_half_even(
            tl.fdiv(x - zero[:, None], scale[:, None], ieee_rounding=True)
        )
        q = tl.minimum(tl.maximum(q, 0.0), MAXQ)
        deq = q * scale[:, None] + zero[:, None]

    tl.store(out_ptr + idx, deq, mask=rmask[:, None])
    tl.store(params_ptr + rows * 2 + 0, scale, mask=rmask)
    tl.store(params_ptr + rows * 2 + 1, zero, mask=rmask)

    # Pack four interleaved lanes into one byte. Triton cannot index the last
    # axis of a reshaped 3-D tile, so re-read each lane strided instead; the
    # data is in L2 by now and this kernel is launch-bound, not bandwidth-bound.
    nb: tl.constexpr = GS // PF
    ob = tl.arange(0, nb)
    packed = tl.zeros([GPB, nb], dtype=tl.int32)
    for j in tl.static_range(PF):
        jdx = rows[:, None] * GS + (PF * ob[None, :] + j)
        xj = tl.load(x_ptr + jdx, mask=rmask[:, None], other=0.0).to(tl.float32)
        if LLOYD:
            zj = tl.fdiv(xj - mean[:, None], std[:, None], ieee_rounding=True)
            qj = ((zj >= T0).to(tl.float32)
                  + (zj >= T1).to(tl.float32)
                  + (zj >= T2).to(tl.float32))
        else:
            qj = _round_half_even(
                tl.fdiv(xj - zero[:, None], scale[:, None], ieee_rounding=True)
            )
            qj = tl.minimum(tl.maximum(qj, 0.0), MAXQ)
        packed |= qj.to(tl.int32) << (BITS * j)
    tl.store(codes_ptr + rows[:, None] * nb + ob[None, :],
             packed.to(tl.uint8), mask=rmask[:, None])


def quantize_dequantize_fused(x, group_size: int = 128, lloyd_max: bool = False, bits: int = 2,
                              groups_per_block: int = 4):
    """One launch: quantize, pack, and write the dequantized values back.

    Returns (codes, params, dequantized) so a caller that still needs BF16 in the
    pool pays a single kernel instead of two plus a round trip through memory.

    groups_per_block=4 measured fastest on H100 (4/8/16 tie at 0.024 ms for 8192
    tokens; 1 costs 0.034, 64 costs 0.030).
    """
    # Lloyd-Max here is a THREE-THRESHOLD codebook, i.e. 2-bit by construction.
    # Running it at any other width would emit codes in 0..3 while dequant reads
    # a wider field -- silently wrong, and silently wasting the extra bits.
    assert bits in (2, 4), f"unsupported bits={bits}"
    assert not (lloyd_max and bits != 2), (
        "lloyd_max is a 2-bit codebook; use uniform quantization at bits=%d" % bits)
    assert x.shape[-1] % group_size == 0 and group_size % 4 == 0
    flat = x.reshape(-1, group_size).contiguous().to(torch.float32)
    n = flat.shape[0]
    codes = torch.empty((n, group_size // (8 // bits)), dtype=torch.uint8, device=x.device)
    params = torch.empty((n, 2), dtype=torch.float32, device=x.device)
    out = torch.empty_like(flat)
    grid = (triton.cdiv(n, groups_per_block),)
    _fused_quant_dequant_kernel[grid](
        flat, codes, params, out, n,
        GS=group_size, GPB=groups_per_block, LLOYD=lloyd_max, BITS=bits, PF=8 // bits,
        MASK=(1 << bits) - 1, MAXQ=float((1 << bits) - 1),
        T0=_LM_THRESHOLDS[0], T1=_LM_THRESHOLDS[1], T2=_LM_THRESHOLDS[2],
        LM_SPAN3=_LM_SPAN / 3.0, LM_RATIO=_LM_RATIO, LM_C0=_LM_CENTROIDS[0],
    )
    return codes, params, out.reshape(x.shape)


# ── scratch reuse ────────────────────────────────────────────────────────────
# The overhead breakdown says the three per-call allocations are 23 % of the
# wrapper cost (0.0052 of 0.023 ms) while the launch itself is 50-62 %. Reusing
# scratch removes that 23 %.
#
# Safe because sglang writes the KV cache from one stream, serially, inside the
# forward pass, and the returned dequantized view is consumed before the next
# write. It is NOT safe to hold onto the returned tensors across calls -- copy
# them if you need to.
_SCRATCH: dict = {}


def _scratch(n_groups: int, group_size: int, device, key: str, bits: int = 2):
    # bits is part of the KEY, not just the shape: the codes buffer is
    # group_size // (8 // bits) wide, so a cache keyed without it would hand a
    # 2-bit buffer to a 4-bit request and silently write half the codes.
    cap, buf = _SCRATCH.get((key, device, group_size, bits), (0, None))
    if cap < n_groups:
        cap = max(n_groups, cap * 2, 1024)
        if key == "codes":
            buf = torch.empty((cap, group_size // (8 // bits)), dtype=torch.uint8, device=device)
        elif key == "params":
            buf = torch.empty((cap, 2), dtype=torch.float32, device=device)
        else:
            buf = torch.empty((cap, group_size), dtype=torch.float32, device=device)
        _SCRATCH[(key, device, group_size, bits)] = (cap, buf)
    return buf[:n_groups]


def quantize_dequantize_reuse(x, group_size: int = 128, lloyd_max: bool = False, bits: int = 2,
                              groups_per_block: int = 4):
    """Same as ``quantize_dequantize_fused`` but on reused scratch buffers.

    Returns views into module-level scratch: valid until the next call.
    """
    # Lloyd-Max here is a THREE-THRESHOLD codebook, i.e. 2-bit by construction.
    # Running it at any other width would emit codes in 0..3 while dequant reads
    # a wider field -- silently wrong, and silently wasting the extra bits.
    assert bits in (2, 4), f"unsupported bits={bits}"
    assert not (lloyd_max and bits != 2), (
        "lloyd_max is a 2-bit codebook; use uniform quantization at bits=%d" % bits)
    assert x.shape[-1] % group_size == 0 and group_size % 4 == 0
    flat = x.reshape(-1, group_size)
    if not flat.is_contiguous() or flat.dtype != torch.float32:
        flat = flat.contiguous().to(torch.float32)
    n = flat.shape[0]
    codes = _scratch(n, group_size, x.device, "codes", bits)
    params = _scratch(n, group_size, x.device, "params", bits)
    out = _scratch(n, group_size, x.device, "out", bits)
    _fused_quant_dequant_kernel[(triton.cdiv(n, groups_per_block),)](
        flat, codes, params, out, n,
        GS=group_size, GPB=groups_per_block, LLOYD=lloyd_max, BITS=bits, PF=8 // bits,
        MASK=(1 << bits) - 1, MAXQ=float((1 << bits) - 1),
        T0=_LM_THRESHOLDS[0], T1=_LM_THRESHOLDS[1], T2=_LM_THRESHOLDS[2],
        LM_SPAN3=_LM_SPAN / 3.0, LM_RATIO=_LM_RATIO, LM_C0=_LM_CENTROIDS[0],
    )
    return codes, params, out.reshape(x.shape)


# ── packed *storage* path ────────────────────────────────────────────────────
# Everything above hands the dequantized values straight back, which is why the
# MLA pool saved no memory. These two kernels are the storage pair: one writes
# codes for a scattered set of pool slots, the other reads a scattered set back.
# The codes live in the *rotated* frame (that is where quantization happens);
# the caller applies R^T to the dequantized block, which is the same arithmetic
# the fake-quant path did inline.


@triton.jit
def _scatter_pack_kernel(
    x_ptr, slots_ptr, codes_ptr, params_ptr,
    n_rows,
    D: tl.constexpr, GS: tl.constexpr, NG: tl.constexpr, LLOYD: tl.constexpr,
    BITS: tl.constexpr, PF: tl.constexpr,
    MASK: tl.constexpr, MAXQ: tl.constexpr,
    T0: tl.constexpr, T1: tl.constexpr, T2: tl.constexpr,
    LM_SPAN3: tl.constexpr, LM_RATIO: tl.constexpr, LM_C0: tl.constexpr,
):
    """Quantize ``x[i, :D]`` groupwise and write it to pool row ``slots[i]``.

    One program per token row (D = kv_lora_rank values = NG groups). Writing
    straight to the destination row is what makes this a storage path: no
    intermediate dense buffer of the whole batch exists at any point.
    """
    pid = tl.program_id(0)
    if pid >= n_rows:
        return
    slot = tl.load(slots_ptr + pid).to(tl.int32)
    # A negative slot is a padded/dummy row. Send it to row 0, which both
    # allocators hold back for exactly this ("writing dummy outputs from padded
    # tokens"), rather than masking the store -- masking is unsafe under graph
    # capture because the predicate is baked in at capture time.
    s = tl.where(slot >= 0, slot, 0).to(tl.int64)

    nb: tl.constexpr = GS // PF
    for g in tl.static_range(NG):
        offs = g * GS + tl.arange(0, GS)
        x = tl.load(x_ptr + pid * D + offs).to(tl.float32)
        if LLOYD:
            mean = tl.fdiv(tl.sum(x, axis=0), float(GS), ieee_rounding=True)
            d = x - mean
            std = tl.sqrt(
                tl.fdiv(tl.sum(d * d, axis=0), float(GS), ieee_rounding=True) + 1e-8
            )
            scale = LM_SPAN3 * LM_RATIO * std
            zero = -LM_C0 / LM_SPAN3 - tl.fdiv(mean, scale, ieee_rounding=True)
        else:
            x_min = tl.min(x, axis=0)
            rng = tl.max(x, axis=0) - x_min
            scale = tl.where(
                tl.abs(rng) > 1e-8, tl.fdiv(rng, MAXQ, ieee_rounding=True), 1.0
            )
            zero = x_min
            mean = 0.0
            std = 1.0

        ob = tl.arange(0, nb)
        packed = tl.zeros([nb], dtype=tl.int32)
        for j in tl.static_range(PF):
            xj = tl.load(x_ptr + pid * D + g * GS + PF * ob + j).to(tl.float32)
            if LLOYD:
                zj = tl.fdiv(xj - mean, std, ieee_rounding=True)
                qj = ((zj >= T0).to(tl.float32)
                      + (zj >= T1).to(tl.float32)
                      + (zj >= T2).to(tl.float32))
            else:
                qj = _round_half_even(
                    tl.fdiv(xj - zero, scale, ieee_rounding=True)
                )
                qj = tl.minimum(tl.maximum(qj, 0.0), MAXQ)
            packed |= qj.to(tl.int32) << (BITS * j)
        tl.store(codes_ptr + s * (D // PF) + g * nb + ob, packed.to(tl.uint8))
        tl.store(params_ptr + s * (2 * NG) + 2 * g + 0, scale)
        tl.store(params_ptr + s * (2 * NG) + 2 * g + 1, zero)


@triton.jit
def _gather_dequant_kernel(
    slots_ptr, codes_ptr, params_ptr, out_ptr,
    n_rows,
    D: tl.constexpr, GS: tl.constexpr, NG: tl.constexpr, LLOYD: tl.constexpr,
    BITS: tl.constexpr, PF: tl.constexpr,
    MASK: tl.constexpr, MAXQ: tl.constexpr,
):
    """``out[i, :D] = dequant(pool_row[slots[i]])`` -- still rotated."""
    pid = tl.program_id(0)
    if pid >= n_rows:
        return
    slot = tl.load(slots_ptr + pid).to(tl.int32)
    s = tl.where(slot >= 0, slot, 0).to(tl.int64)

    offs = tl.arange(0, D)
    gid = offs // GS
    byte = tl.load(codes_ptr + s * (D // PF) + offs // PF).to(tl.int32)
    q = ((byte >> (BITS * (offs % PF))) & MASK).to(tl.float32)
    scale = tl.load(params_ptr + s * (2 * NG) + 2 * gid + 0)
    zero = tl.load(params_ptr + s * (2 * NG) + 2 * gid + 1)
    if LLOYD:
        val = (q - zero) * scale
    else:
        val = q * scale + zero
    tl.store(out_ptr + pid * D + offs, val.to(out_ptr.dtype.element_ty))


@triton.jit
def _assemble_rows_kernel(
    c_ptr, slots_ptr, rope_ptr, hp_ptr, hp_row_ptr, hp_owner_ptr, out_ptr,
    n_rows,
    D: tl.constexpr, ROPE: tl.constexpr, OUT_D: tl.constexpr,
    HAS_HP: tl.constexpr,
):
    """Assemble the ``[c_kv | k_pe]`` row the MLA kernels expect.

    ``c_ptr`` holds the un-rotated dequantized latent. A row that still lives in
    the BF16 window arena overrides it: ``hp_row[slot]`` names the arena row and
    ``hp_owner[row]`` says which slot currently owns it. The owner check is what
    makes a *reused* prefix safe -- the arena is a per-request ring, so a cached
    slot's stale arena pointer must not be followed once the ring has lapped.
    """
    pid = tl.program_id(0)
    if pid >= n_rows:
        return
    slot = tl.load(slots_ptr + pid).to(tl.int32)
    valid = slot >= 0
    s = tl.where(valid, slot, 0).to(tl.int64)

    offs = tl.arange(0, D)
    val = tl.load(c_ptr + pid * D + offs).to(tl.float32)

    if HAS_HP:
        r = tl.load(hp_row_ptr + s).to(tl.int32)
        rr = tl.where(r >= 0, r, 0).to(tl.int64)
        owner = tl.load(hp_owner_ptr + rr).to(tl.int32)
        use_hp = (r >= 0) & (owner == slot) & valid
        hpv = tl.load(hp_ptr + rr * D + offs).to(tl.float32)
        val = tl.where(use_hp, hpv, val)

    val = tl.where(valid, val, 0.0)
    tl.store(out_ptr + pid * OUT_D + offs, val.to(out_ptr.dtype.element_ty))

    ro = tl.arange(0, ROPE)
    rv = tl.load(rope_ptr + s * ROPE + ro).to(tl.float32)
    rv = tl.where(valid, rv, 0.0)
    tl.store(out_ptr + pid * OUT_D + D + ro, rv.to(out_ptr.dtype.element_ty))


def scatter_pack_rows(x, slots, codes_buf, params_buf, group_size, lloyd_max, bits=2):
    """Quantize ``x`` (``[n, D]``, rotated frame) into ``codes/params`` at ``slots``."""
    # Lloyd-Max here is a THREE-THRESHOLD codebook, i.e. 2-bit by construction.
    # Running it at any other width would emit codes in 0..3 while dequant reads
    # a wider field -- silently wrong, and silently wasting the extra bits.
    assert bits in (2, 4), f"unsupported bits={bits}"
    assert not (lloyd_max and bits != 2), (
        "lloyd_max is a 2-bit codebook; use uniform quantization at bits=%d" % bits)
    n, d = x.shape
    assert d % group_size == 0 and group_size % 4 == 0
    if n == 0:
        return
    x = x.contiguous().to(torch.float32)
    _scatter_pack_kernel[(n,)](
        x, slots, codes_buf, params_buf, n,
        D=d, GS=group_size, NG=d // group_size, LLOYD=lloyd_max, BITS=bits, PF=8 // bits,
        MASK=(1 << bits) - 1, MAXQ=float((1 << bits) - 1),
        T0=_LM_THRESHOLDS[0], T1=_LM_THRESHOLDS[1], T2=_LM_THRESHOLDS[2],
        LM_SPAN3=_LM_SPAN / 3.0, LM_RATIO=_LM_RATIO, LM_C0=_LM_CENTROIDS[0],
    )


def gather_dequant_rows(slots, codes_buf, params_buf, out, group_size, lloyd_max, bits=2):
    """``out[i] = dequant(codes[slots[i]])`` in the rotated frame."""
    n, d = out.shape
    if n == 0:
        return out
    _gather_dequant_kernel[(n,)](
        slots, codes_buf, params_buf, out, n,
        D=d, GS=group_size, NG=d // group_size, LLOYD=lloyd_max, BITS=bits, PF=8 // bits,
        MASK=(1 << bits) - 1, MAXQ=float((1 << bits) - 1),
    )
    return out


def assemble_rows(c, slots, rope_buf, hp_buf, hp_row_of_slot, hp_owner_of_row, out):
    """Write ``[c | k_pe]`` into ``out``, letting the BF16 window arena win."""
    n, d = c.shape
    if n == 0:
        return out
    rope_d = rope_buf.shape[-1]
    has_hp = hp_buf is not None
    _assemble_rows_kernel[(n,)](
        c, slots, rope_buf,
        hp_buf if has_hp else c,
        hp_row_of_slot if has_hp else slots,
        hp_owner_of_row if has_hp else slots,
        out, n,
        D=d, ROPE=rope_d, OUT_D=out.shape[-1], HAS_HP=has_hp,
    )
    return out
