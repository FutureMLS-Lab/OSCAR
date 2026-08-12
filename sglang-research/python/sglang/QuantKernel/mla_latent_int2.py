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

To keep one dequant kernel for both, the uniform mode stores ``zero`` already
negated (``-min/scale``) so that ``(q - zero) * scale == q * scale + min``.
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
    """torch.round semantics (ties to even) without depending on libdevice,
    whose symbol path moves between Triton versions."""
    r = tl.floor(x + 0.5)
    is_tie = (r - x) == 0.5
    is_odd = (r - 2.0 * tl.floor(r * 0.5)) == 1.0
    return tl.where(is_tie & is_odd, r - 1.0, r)


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
        mean = tl.sum(x, axis=0) / group_size
        d = x - mean
        std = tl.sqrt(tl.sum(d * d, axis=0) / group_size + 1e-8)
        scale = LM_SPAN3 * LM_RATIO * std
        zero = -LM_C0 / LM_SPAN3 - mean / scale
    else:
        x_min = tl.min(x, axis=0)
        rng = tl.max(x, axis=0) - x_min
        scale = tl.where(tl.abs(rng) > 1e-8, rng / 3.0, 1.0)
        # stored pre-negated so dequant is (q - zero) * scale in both modes
        zero = -x_min / scale
        mean = 0.0
        std = 1.0

    # pass 2: four interleaved planes -> one packed byte each
    nb: tl.constexpr = group_size // 4
    ob = tl.arange(0, nb)
    packed = tl.zeros([nb], dtype=tl.int32)
    for j in tl.static_range(4):
        xj = tl.load(x_ptr + base + 4 * ob + j).to(tl.float32)
        if LLOYD:
            zj = (xj - mean) / std
            qj = ((zj >= T0).to(tl.float32)
                  + (zj >= T1).to(tl.float32)
                  + (zj >= T2).to(tl.float32))
        else:
            qj = _round_half_even(xj / scale + zero)
            qj = tl.minimum(tl.maximum(qj, 0.0), 3.0)
        packed |= qj.to(tl.int32) << (2 * j)

    tl.store(codes_ptr + gid * nb + ob, packed.to(tl.uint8))
    tl.store(params_ptr + gid * 2 + 0, scale)
    tl.store(params_ptr + gid * 2 + 1, zero)


@triton.jit
def _dequant_kernel(
    codes_ptr, params_ptr, out_ptr,
    n_groups, group_size: tl.constexpr,
):
    gid = tl.program_id(0)
    if gid >= n_groups:
        return
    offs = tl.arange(0, group_size)
    byte = tl.load(codes_ptr + gid * (group_size // 4) + offs // 4).to(tl.int32)
    q = ((byte >> (2 * (offs % 4))) & 0x3).to(tl.float32)
    scale = tl.load(params_ptr + gid * 2 + 0)
    zero = tl.load(params_ptr + gid * 2 + 1)
    tl.store(out_ptr + gid * group_size + offs, (q - zero) * scale)


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
               dtype: torch.dtype = torch.float32) -> torch.Tensor:
    n = codes.shape[0]
    out = torch.empty((n, group_size), dtype=torch.float32, device=codes.device)
    _dequant_kernel[(n,)](codes, params, out, n, group_size=group_size)
    return out.reshape(shape).to(dtype)


def bytes_per_value(group_size: int = 128) -> float:
    """Storage cost: 2 bits of codes + two fp32 group params, amortized."""
    return (group_size / 4 + 8) / group_size
