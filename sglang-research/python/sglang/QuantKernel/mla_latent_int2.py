"""Real INT2 pack/unpack kernels for the MLA latent (``c_kv``).

The existing MLA path (``mem_cache/mla_int2_kv_pool.py``) is *fake* quant: it
rounds ``c_kv`` and writes the result straight back into the BF16 pool, so it
measures accuracy but saves no memory and costs extra time. These kernels store
the quantized codes for real -- 4 codes per byte, plus two fp32 group
parameters -- so the latent occupies 2 bits/value plus scale overhead.

Layout, per row of ``group_size`` values:
    codes : uint8[group_size // 4]   little-endian 2-bit fields (v0 = bits 0-1)
    params: fp32[2]                  (scale, zero) meaning depends on the mode

Both quantizer modes of ``_fake_quant_int2_groupwise`` are reproduced exactly:

* uniform  q = round((x - min) / s).clamp(0, 3),  x_deq = q * s + min
           -> (scale, zero) = (s, min),               x_deq = q * scale + zero
* lloyd    q = #{t : z >= t} for the three LM thresholds,
           x_deq = (q - uniform_zero) * uniform_scale
           -> (scale, zero) = (uniform_scale, uniform_zero),
              x_deq = (q - zero) * scale

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
        mean = tl.fdiv(tl.sum(x, axis=0), group_size, ieee_rounding=True)
        d = x - mean
        std = tl.sqrt(tl.fdiv(tl.sum(d * d, axis=0), group_size, ieee_rounding=True) + 1e-8)
        scale = LM_SPAN3 * LM_RATIO * std
        zero = -LM_C0 / LM_SPAN3 - tl.fdiv(mean, scale, ieee_rounding=True)
    else:
        x_min = tl.min(x, axis=0)
        rng = tl.max(x, axis=0) - x_min
        # true division, not multiply-by-reciprocal: a 1-ULP difference in
        # scale shifts every quotient downstream
        scale = tl.where(tl.abs(rng) > 1e-8, tl.fdiv(rng, 3.0, ieee_rounding=True), 1.0)
        # store x_min itself: dequant then mirrors the reference's
        # ``q * scale + x_min`` exactly instead of an algebraically equal but
        # differently-rounded ``(q - zero) * scale``
        zero = x_min
        mean = 0.0
        std = 1.0

    # pass 2: four interleaved planes -> one packed byte each
    nb: tl.constexpr = group_size // 4
    ob = tl.arange(0, nb)
    packed = tl.zeros([nb], dtype=tl.int32)
    for j in tl.static_range(4):
        xj = tl.load(x_ptr + base + 4 * ob + j).to(tl.float32)
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
            qj = tl.minimum(tl.maximum(qj, 0.0), 3.0)
        packed |= qj.to(tl.int32) << (2 * j)

    tl.store(codes_ptr + gid * nb + ob, packed.to(tl.uint8))
    tl.store(params_ptr + gid * 2 + 0, scale)
    tl.store(params_ptr + gid * 2 + 1, zero)


@triton.jit
def _dequant_kernel(
    codes_ptr, params_ptr, out_ptr,
    n_groups, group_size: tl.constexpr, LLOYD: tl.constexpr,
):
    gid = tl.program_id(0)
    if gid >= n_groups:
        return
    offs = tl.arange(0, group_size)
    byte = tl.load(codes_ptr + gid * (group_size // 4) + offs // 4).to(tl.int32)
    q = ((byte >> (2 * (offs % 4))) & 0x3).to(tl.float32)
    scale = tl.load(params_ptr + gid * 2 + 0)
    zero = tl.load(params_ptr + gid * 2 + 1)
    if LLOYD:
        val = (q - zero) * scale        # reference: (q - uniform_zero) * uniform_scale
    else:
        val = q * scale + zero          # reference: q * scale + x_min
    tl.store(out_ptr + gid * group_size + offs, val)


def quantize_pack(x: torch.Tensor, group_size: int = 128, lloyd_max: bool = False):
    """``[..., D]`` float -> (packed uint8 codes, fp32 group params)."""
    assert x.shape[-1] % group_size == 0, (x.shape, group_size)
    assert group_size % 4 == 0
    flat = x.reshape(-1, group_size).contiguous().to(torch.float32)
    n = flat.shape[0]
    codes = torch.empty((n, group_size // 4), dtype=torch.uint8, device=x.device)
    params = torch.empty((n, 2), dtype=torch.float32, device=x.device)
    _quant_pack_kernel[(n,)](
        flat, codes, params, n, group_size=group_size, LLOYD=lloyd_max,
        T0=_LM_THRESHOLDS[0], T1=_LM_THRESHOLDS[1], T2=_LM_THRESHOLDS[2],
        LM_SPAN3=_LM_SPAN / 3.0, LM_RATIO=_LM_RATIO, LM_C0=_LM_CENTROIDS[0],
    )
    return codes, params


def dequantize(codes: torch.Tensor, params: torch.Tensor, shape, group_size: int = 128,
               dtype: torch.dtype = torch.float32, lloyd_max: bool = False) -> torch.Tensor:
    n = codes.shape[0]
    out = torch.empty((n, group_size), dtype=torch.float32, device=codes.device)
    _dequant_kernel[(n,)](codes, params, out, n, group_size=group_size, LLOYD=lloyd_max)
    return out.reshape(shape).to(dtype)


def bytes_per_value(group_size: int = 128) -> float:
    """Storage cost: 2 bits of codes + two fp32 group params, amortized."""
    return (group_size / 4 + 8) / group_size
