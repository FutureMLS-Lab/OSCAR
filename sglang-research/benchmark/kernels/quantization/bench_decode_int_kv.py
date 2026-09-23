"""Microbench for int2 decode attention with groupwise KV scales.

Purpose
-------
Gate the ``GROUPED: tl.constexpr`` specialization in the int2 decode kernels.
We measure the default (``num_groups == 1``) path against the grouped path for
MHA and GQA layouts, and verify output correctness against a dense dequantized
reference before timing.

Each (shape, group) row is followed by a ``+mixed`` row that exercises
``decode_attention_fwd_int2_unified`` on a ``UnifiedInt2HPKVPool`` (equivalent
to enabling ``SGLANG_ENABLE_MIXED_KV_WINDOWS``). BF16 rows from the Triton and
FlashInfer backends are printed at the top of each shape group as reference
baselines.

Two ways to drive a run:

1) CLI sweep (default): pick a single (model, seq_len, batch_size) shape and
   sweep ``--group-sizes`` × {mixed=off, mixed=on}. All BF16 baselines are
   also emitted. See ``--help``.

2) YAML config: ``--config path.yaml`` lets you specify each row independently
   (different seq_len / batch_size / backend / group_size / mixed_kv). Schema:

       defaults:                # optional, applied to all cases
         model: Qwen/Qwen3-8B
         max_kv_splits: 32
         warmup: 5
         repeat: 30
         hp_prefix: 32
         hp_recent: 128
         no_correctness: false
         profiling_mode: event

       cases:
         # `id` is REQUIRED and must be a unique integer (you choose the scheme).
         # `baseline_id` (optional, per-case): another case's id whose mean is
         # used to compute this row's speedup ratio. Must reference a case
         # that appears earlier in the file.
         - id: 1
           seq_len: 81920
           batch_size: 1
           backend: triton_bf16
         - id: 2
           seq_len: 81920
           batch_size: 1
           backend: int2          # int2|triton_bf16|flashinfer_bf16
           group_size: 16         # 0 = whole head_dim (scalar); >0 = group size
           mixed_kv: false
           baseline_id: 1         # speedup ratio printed against case id=1

   Each YAML case becomes one row in the output table; per-case fields
   override ``defaults``.

Group-size convention: ``0`` (or ``1`` for legacy compat) means the whole
head_dim shares a single (scale, zero) pair (the "scalar" path). Any value
greater than 1 enables groupwise quantisation with that group size.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from rich import box as rich_box
from rich.console import Console
from rich.live import Live
from rich.table import Table

try:
    import flashinfer as _flashinfer

    _FLASHINFER_AVAILABLE = True
except ImportError:
    _flashinfer = None  # type: ignore[assignment]
    _FLASHINFER_AVAILABLE = False

from sglang.srt.layers.attention.triton_ops.decode_attention import (
    decode_attention_fwd,
    decode_attention_fwd_int2_unified,
    decode_attention_fwd_quantized,
)
from sglang.srt.mem_cache.kv_quant_kernels import dequantize_kv_int2_triton
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.unified_kv_allocator import UnifiedInt2HPKVAllocator
from sglang.srt.mem_cache.unified_kv_pool import (
    UnifiedInt2HPKVPool,
    compute_page_geometry,
)


DEVICE = "cuda"
DTYPE = torch.bfloat16
HADAMARD_ORDER = 16


@dataclass
class ShapeCase:
    name: str
    batch_size: int
    seq_len: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int


def _shape_from_model(model_name: str, seq_len: int, batch_size: int) -> ShapeCase:
    """Load a HuggingFace model config and derive a ``ShapeCase`` from it.

    Extracts ``num_attention_heads``, ``num_key_value_heads``, and ``head_dim``
    (falling back to ``hidden_size // num_attention_heads`` when the model does
    not expose ``head_dim`` directly).
    """
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except Exception as exc:
        raise SystemExit(f"Cannot load model config for {model_name!r}: {exc}") from exc

    num_q_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_q_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // num_q_heads)
    name = model_name.rstrip("/").split("/")[-1]
    return ShapeCase(
        name=name,
        batch_size=batch_size,
        seq_len=seq_len,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )


def _fwht_last_dim(x: torch.Tensor, order: int) -> torch.Tensor:
    orig_shape = x.shape
    y = x.float().view(*orig_shape[:-1], orig_shape[-1] // order, order)
    for s in range(int(math.log2(order))):
        stride = 1 << s
        for i in range(order):
            partner = i ^ stride
            if i < partner:
                lo = y[..., i].clone()
                hi = y[..., partner].clone()
                y[..., i] = lo + hi
                y[..., partner] = lo - hi
    y = y / math.sqrt(order)
    return y.view(orig_shape).to(x.dtype)


def _dequantize(
    kv_dtype: str,
    quantized: torch.Tensor,
    scales_zeros: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    if kv_dtype == "int2":
        return dequantize_kv_int2_triton(quantized, scales_zeros, head_dim, DTYPE)
    raise ValueError(f"Only int2 is supported, got {kv_dtype}")


def _compute_num_kv_splits(shape: ShapeCase, max_kv_splits: int) -> tuple[torch.Tensor, int]:
    """Compute per-sequence KV split counts with the production heuristic.

    Mirrors ``TritonAttnBackend.get_num_kv_splits`` / ``get_num_kv_splits_triton``:
    each sequence is split across as many SM tiles as can be kept busy, capped
    at *max_kv_splits*.  The effective maximum of the returned tensor is used to
    size ``attn_logits`` / ``attn_lse`` scratch buffers so stage-2 loops are as
    short as possible.

    Returns
    -------
    num_kv_splits : torch.Tensor  [batch_size], int32
    effective_max : int           ``max(num_kv_splits)``, always >= 1
    """
    from sglang.srt.layers.attention.triton_backend import get_num_kv_splits_triton
    from sglang.srt.utils.common import get_device_core_count
    import triton as _triton

    device_core_count = get_device_core_count(torch.cuda.current_device())
    num_seq = shape.batch_size
    seq_lens = torch.full((num_seq,), shape.seq_len, dtype=torch.int32, device=DEVICE)
    num_kv_splits = torch.empty(num_seq, dtype=torch.int32, device=DEVICE)
    schedule_seq = 256 if num_seq < 256 else _triton.next_power_of_2(num_seq)

    get_num_kv_splits_triton[(1,)](
        num_kv_splits,
        seq_lens,
        num_seq,
        1,                    # num_group — 1 for standard (non-speculative) decode
        shape.num_q_heads,
        shape.num_kv_heads,
        max_kv_splits,
        device_core_count,
        MAX_NUM_SEQ=schedule_seq,
    )

    effective_max = max(1, int(num_kv_splits.max().item()))
    return num_kv_splits, effective_max


def _build_case(
    shape: ShapeCase,
    kv_dtype: str,
    group_size: Optional[int],
    max_kv_splits: int = 8,
) -> dict:
    total_tokens = shape.batch_size * shape.seq_len
    pool = MHATokenToKVPool(
        total_tokens + 8,
        page_size=1,
        dtype=kv_dtype,
        head_num=shape.num_kv_heads,
        head_dim=shape.head_dim,
        layer_num=1,
        device=DEVICE,
        enable_memory_saver=False,
        model_dtype=DTYPE,
        kv_cache_quant_group_size=group_size,
    )
    from types import SimpleNamespace

    layer = SimpleNamespace(layer_id=0)
    loc = torch.arange(total_tokens, dtype=torch.int64, device=DEVICE)
    cache_k = torch.randn(
        total_tokens, shape.num_kv_heads, shape.head_dim, dtype=DTYPE, device=DEVICE
    )
    cache_v = torch.randn_like(cache_k)
    pool.set_kv_buffer(layer, loc, cache_k, cache_v)

    kv_indptr = torch.arange(
        0,
        (shape.batch_size + 1) * shape.seq_len,
        shape.seq_len,
        dtype=torch.int32,
        device=DEVICE,
    )
    kv_indices = loc

    q = torch.randn(
        shape.batch_size, shape.num_q_heads, shape.head_dim, dtype=DTYPE, device=DEVICE
    )
    q_for_decode = q
    if kv_dtype == "int2":
        q_for_decode = _fwht_last_dim(q, HADAMARD_ORDER)
    num_kv_splits, max_kv_splits = _compute_num_kv_splits(shape, max_kv_splits)
    attn_logits = torch.empty(
        shape.batch_size,
        shape.num_q_heads,
        max_kv_splits,
        shape.head_dim,
        dtype=torch.float32,
        device=DEVICE,
    )
    attn_lse = torch.empty(
        shape.batch_size,
        shape.num_q_heads,
        max_kv_splits,
        dtype=torch.float32,
        device=DEVICE,
    )
    o = torch.empty_like(q_for_decode)

    return {
        "pool": pool,
        "q_for_decode": q_for_decode,
        "o": o,
        "attn_logits": attn_logits,
        "attn_lse": attn_lse,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "num_kv_splits": num_kv_splits,
        "max_kv_splits": max_kv_splits,
        "sm_scale": 1.0 / math.sqrt(shape.head_dim),
        "loc": loc,
        "kv_dtype": kv_dtype,
        "shape": shape,
    }


def _run_kernel(case: dict) -> torch.Tensor:
    pool = case["pool"]
    decode_attention_fwd_quantized(
        case["q_for_decode"],
        pool.get_raw_key_buffer(0),
        pool.get_raw_value_buffer(0),
        pool.get_key_scales_zeros(0),
        pool.get_value_scales_zeros(0),
        case["o"],
        case["kv_indptr"],
        case["kv_indices"],
        case["attn_logits"],
        case["attn_lse"],
        case["num_kv_splits"],
        case["max_kv_splits"],
        case["sm_scale"],
        case["kv_dtype"],
    )
    return case["o"]


def _reference_output(case: dict) -> torch.Tensor:
    shape: ShapeCase = case["shape"]
    pool = case["pool"]
    dense_k = _dequantize(
        case["kv_dtype"],
        pool.get_raw_key_buffer(0)[case["loc"]],
        pool.get_key_scales_zeros(0)[case["loc"]],
        shape.head_dim,
    )
    dense_v = _dequantize(
        case["kv_dtype"],
        pool.get_raw_value_buffer(0)[case["loc"]],
        pool.get_value_scales_zeros(0)[case["loc"]],
        shape.head_dim,
    )
    ref_o = torch.empty_like(case["q_for_decode"])
    ref_logits = torch.empty_like(case["attn_logits"])
    ref_lse = torch.empty_like(case["attn_lse"])
    decode_attention_fwd(
        case["q_for_decode"],
        dense_k,
        dense_v,
        ref_o,
        case["kv_indptr"],
        case["kv_indices"],
        ref_logits,
        ref_lse,
        case["num_kv_splits"],
        case["max_kv_splits"],
        case["sm_scale"],
        1.0,
        1.0,
    )
    return ref_o


def _cuda_time(fn, warmup: int = 5, repeat: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i in range(repeat):
        fn()
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / repeat


def _cuda_kernel_time(fn, warmup: int = 5, repeat: int = 50) -> float:
    """Per-call pure GPU kernel time (ms) via ``torch.profiler``.

    Measures only device-side kernel duration, excluding host launch overhead
    and Python time. ``activities=[ProfilerActivity.CUDA]`` keeps recorded
    events to CUDA work only; we sum ``self_device_time_total`` (microseconds)
    across all kernel keys and divide by ``repeat`` to get per-``fn()`` ms.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        for _ in range(repeat):
            fn()
    torch.cuda.synchronize()

    total_us = 0.0
    for evt in prof.key_averages():
        # PyTorch >= 2.4 exposes ``self_device_time_total``; older versions use
        # the CUDA-specific alias.
        kernel_us = getattr(
            evt, "self_device_time_total", getattr(evt, "self_cuda_time_total", 0)
        )
        total_us += kernel_us
    return (total_us / 1000.0) / repeat


def _time_fn(fn, mode: str, warmup: int, repeat: int) -> float:
    """Dispatch timing to either CUDA-event wall time or torch-profiler kernel time."""
    if mode == "torch_profiler":
        return _cuda_kernel_time(fn, warmup=warmup, repeat=repeat)
    return _cuda_time(fn, warmup=warmup, repeat=repeat)


def _tolerances(kv_dtype: str) -> Tuple[float, float]:
    # Hadamard + int2 quant introduces larger errors than bf16.
    assert kv_dtype == "int2"
    return 7e-2, 7e-2


# ---------------------------------------------------------------------------
# BF16 reference baselines: plain Triton and FlashInfer
# ---------------------------------------------------------------------------


def _build_bf16_case(shape: ShapeCase, max_kv_splits: int = 8) -> dict:
    """Allocate shared BF16 tensors used by both Triton-bf16 and FlashInfer-bf16."""
    total_tokens = shape.batch_size * shape.seq_len
    k = torch.randn(
        total_tokens, shape.num_kv_heads, shape.head_dim, dtype=DTYPE, device=DEVICE
    )
    v = torch.randn_like(k)
    q = torch.randn(
        shape.batch_size, shape.num_q_heads, shape.head_dim, dtype=DTYPE, device=DEVICE
    )
    # int32 indices satisfy both the Triton kernels (which cast internally) and
    # FlashInfer (which requires int32).
    kv_indptr = torch.arange(
        0,
        (shape.batch_size + 1) * shape.seq_len,
        shape.seq_len,
        dtype=torch.int32,
        device=DEVICE,
    )
    kv_indices = torch.arange(total_tokens, dtype=torch.int32, device=DEVICE)
    num_kv_splits, max_kv_splits = _compute_num_kv_splits(shape, max_kv_splits)
    attn_logits = torch.empty(
        shape.batch_size,
        shape.num_q_heads,
        max_kv_splits,
        shape.head_dim,
        dtype=torch.float32,
        device=DEVICE,
    )
    attn_lse = torch.empty(
        shape.batch_size,
        shape.num_q_heads,
        max_kv_splits,
        dtype=torch.float32,
        device=DEVICE,
    )
    o = torch.empty_like(q)
    return dict(
        q=q,
        k=k,
        v=v,
        o=o,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        attn_logits=attn_logits,
        attn_lse=attn_lse,
        num_kv_splits=num_kv_splits,
        max_kv_splits=max_kv_splits,
        sm_scale=1.0 / math.sqrt(shape.head_dim),
    )


def _run_triton_bf16(case: dict) -> None:
    """Triton native BF16 decode attention — no quantization."""
    decode_attention_fwd(
        case["q"],
        case["k"],
        case["v"],
        case["o"],
        case["kv_indptr"],
        case["kv_indices"],
        case["attn_logits"],
        case["attn_lse"],
        case["num_kv_splits"],
        case["max_kv_splits"],
        case["sm_scale"],
        1.0,
        1.0,
    )


def _build_flashinfer_state(shape: ShapeCase, bf16: dict) -> Optional[dict]:
    """Plan a FlashInfer BatchDecodeWithPagedKVCacheWrapper for BF16 decode.

    Returns None when flashinfer is not installed.
    The returned dict is passed verbatim to ``_run_flashinfer_bf16`` on every
    timing iteration — ``plan()`` is called only once during setup.

    ``use_tensor_cores`` mirrors sglang's production heuristic: enable when
    ``num_q_heads / num_kv_heads >= 4`` (BF16 path).  The non-tensor-core
    dispatch table does not cover all ``num_qo_heads`` values that arise with
    split-KV at large batch×seq_len and would raise "Unsupported group_size".
    """
    if not _FLASHINFER_AVAILABLE:
        return None
    # FlashInfer NHD paged-KV layout: [num_pages, 2, page_size, num_kv_heads, head_dim]
    # page_size=1 → [total_tokens, 2, 1, num_kv_heads, head_dim]
    paged_kv = torch.stack(
        [bf16["k"].unsqueeze(2), bf16["v"].unsqueeze(2)], dim=1
    )
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=DEVICE)
    # Mirror sglang's should_use_tensor_core heuristic for BF16.
    use_tensor_cores = (shape.num_q_heads // shape.num_kv_heads) >= 4
    wrapper = _flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", use_tensor_cores=use_tensor_cores
    )
    last_page_len = torch.ones(shape.batch_size, dtype=torch.int32, device=DEVICE)
    wrapper.plan(
        bf16["kv_indptr"],
        bf16["kv_indices"],
        last_page_len,
        shape.num_q_heads,
        shape.num_kv_heads,
        shape.head_dim,
        1,  # page_size
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
        sm_scale=bf16["sm_scale"],
    )
    return dict(
        wrapper=wrapper,
        q=bf16["q"].contiguous(),
        paged_kv=paged_kv,
        use_tensor_cores=use_tensor_cores,
    )


def _run_flashinfer_bf16(state: dict) -> None:
    """FlashInfer BF16 decode attention."""
    state["wrapper"].run(state["q"], state["paged_kv"])


# ---------------------------------------------------------------------------
# Mixed KV windows: HP prefix/recent tier + quantized middle tier
# ---------------------------------------------------------------------------


def _build_mixed_kv_case(
    shape: ShapeCase,
    kv_dtype: str,
    group_size: Optional[int],
    hp_prefix: int,
    hp_recent: int,
    max_kv_splits: int = 8,
) -> dict:
    """Build a ``UnifiedInt2HPKVPool`` + split index arrays for
    ``decode_attention_fwd_int2_unified``.

    Each sequence of length ``seq_len`` is partitioned as:
      [hp_prefix tokens | quant_count tokens | hp_recent tokens]
    HP indices are local to the HP view (``loc - pool.hp_global_offset``).

    The unified path uses a single combined stage-1 scratch
    (``attn_logits``/``attn_lse`` of shape ``[bs, heads, hp_max + quant_max, v_head_dim]``),
    with per-tier ``num_kv_splits`` populated by ``get_num_kv_splits_triton``.
    LSE is pre-filled to ``-inf`` so the tier-agnostic stage-2 skips unused
    splits.
    """
    from types import SimpleNamespace

    hp_prefix = min(shape.seq_len, hp_prefix)
    hp_recent = min(max(shape.seq_len - hp_prefix, 0), hp_recent)
    quant_count = shape.seq_len - hp_prefix - hp_recent

    # Size the shared arena in physical pages, not token slots. Each page can
    # be assigned to either the HP tier or the quant tier.
    hp_tokens_per_page, quant_tokens_per_page = compute_page_geometry(DTYPE)

    # The new allocator only allocates whole quant pages (size N_Q). Apply
    # the C1 absorb rule (see ``mem_cache/common.py::_alloc_for_extend_mixed``)
    # to round ``quant_count`` down to a multiple of ``N_Q`` and absorb the
    # tail into ``hp_recent``.
    tail = quant_count % quant_tokens_per_page
    if tail:
        quant_count -= tail
        hp_recent += tail
    total_hp_tokens = shape.batch_size * (hp_prefix + hp_recent)
    total_quant_tokens = shape.batch_size * quant_count
    num_pages = (
        (total_hp_tokens + hp_tokens_per_page - 1) // hp_tokens_per_page
        + (total_quant_tokens + quant_tokens_per_page - 1) // quant_tokens_per_page
        + 16
    )

    pool = UnifiedInt2HPKVPool(
        num_pages=num_pages,
        hp_dtype=DTYPE,
        hp_prefix_tokens=hp_prefix,
        hp_recent_tokens=hp_recent,
        dtype=kv_dtype,
        head_num=shape.num_kv_heads,
        head_dim=shape.head_dim,
        layer_num=1,
        device=DEVICE,
        enable_memory_saver=False,
        model_dtype=DTYPE,
        kv_cache_quant_group_size=group_size,
    )
    allocator = UnifiedInt2HPKVAllocator(
        num_pages=pool.num_pages,
        hp_tokens_per_page=pool.N_H,
        quant_tokens_per_page=pool.N_Q,
        dtype=kv_dtype,
        hp_dtype=DTYPE,
        device=DEVICE,
        kvcache=pool,
        need_sort=False,
    )

    layer = SimpleNamespace(layer_id=0)
    all_locs: List[torch.Tensor] = []
    hp_indices_flat: List[torch.Tensor] = []
    quant_indices_flat: List[torch.Tensor] = []
    hp_indptr_list = [0]
    quant_indptr_list = [0]

    for _ in range(shape.batch_size):
        loc_prefix = allocator.alloc_hp(hp_prefix)
        loc_middle = allocator.alloc_quant(quant_count)
        loc_recent = allocator.alloc_hp(hp_recent)
        loc_seq = torch.cat([loc_prefix, loc_middle, loc_recent])
        all_locs.append(loc_seq)
        # HP indices are local to the HP buffer (subtract global offset).
        hp_idx = torch.cat(
            [
                loc_prefix - pool.hp_global_offset,
                loc_recent - pool.hp_global_offset,
            ]
        )
        hp_indices_flat.append(hp_idx)
        quant_indices_flat.append(loc_middle)
        hp_indptr_list.append(hp_indptr_list[-1] + hp_idx.numel())
        quant_indptr_list.append(quant_indptr_list[-1] + loc_middle.numel())

    all_loc = torch.cat(all_locs)
    cache_k = torch.randn(
        shape.batch_size * shape.seq_len,
        shape.num_kv_heads,
        shape.head_dim,
        dtype=DTYPE,
        device=DEVICE,
    )
    cache_v = torch.randn_like(cache_k)
    pool.set_kv_buffer(layer, all_loc, cache_k, cache_v)

    hp_kv_indices = torch.cat(hp_indices_flat).to(torch.int64)
    quant_kv_indices = torch.cat(quant_indices_flat).to(torch.int64)
    hp_kv_indptr = torch.tensor(hp_indptr_list, dtype=torch.int32, device=DEVICE)
    quant_kv_indptr = torch.tensor(quant_indptr_list, dtype=torch.int32, device=DEVICE)

    q = torch.randn(
        shape.batch_size, shape.num_q_heads, shape.head_dim, dtype=DTYPE, device=DEVICE
    )
    q_for_decode = _fwht_last_dim(q, HADAMARD_ORDER) if kv_dtype == "int2" else q

    # Per-tier split counts (derived from per-tier lengths).
    from dataclasses import replace as _dc_replace
    hp_seq_shape    = _dc_replace(shape, seq_len=hp_prefix + hp_recent)
    quant_seq_shape = _dc_replace(shape, seq_len=quant_count)
    hp_num_kv_splits, hp_max_splits       = _compute_num_kv_splits(hp_seq_shape, max_kv_splits)
    quant_num_kv_splits, quant_max_splits = _compute_num_kv_splits(quant_seq_shape, max_kv_splits)

    total_splits = hp_max_splits + quant_max_splits
    # Single combined stage-1 scratch. LSE starts at -inf so the tier-agnostic
    # stage-2 can safely skip unused splits.
    attn_logits = torch.empty(
        shape.batch_size, shape.num_q_heads, total_splits, shape.head_dim,
        dtype=torch.float32, device=DEVICE,
    )
    attn_lse = torch.full(
        (shape.batch_size, shape.num_q_heads, total_splits),
        float("-inf"),
        dtype=torch.float32, device=DEVICE,
    )
    o = torch.empty_like(q_for_decode)

    return dict(
        pool=pool,
        shape=shape,
        req_locs=all_locs,
        q_for_decode=q_for_decode,
        o=o,
        attn_logits=attn_logits,
        attn_lse=attn_lse,
        hp_kv_indptr=hp_kv_indptr,
        hp_kv_indices=hp_kv_indices,
        quant_kv_indptr=quant_kv_indptr,
        quant_kv_indices=quant_kv_indices,
        hp_num_kv_splits=hp_num_kv_splits,
        hp_max_kv_splits=hp_max_splits,
        quant_num_kv_splits=quant_num_kv_splits,
        quant_max_kv_splits=quant_max_splits,
        sm_scale=1.0 / math.sqrt(shape.head_dim),
        kv_dtype=kv_dtype,
        hp_prefix=hp_prefix,
        hp_recent=hp_recent,
        quant_count=quant_count,
    )


def _run_mixed_kernel(case: dict) -> None:
    """Run the unified HP + int2 decode attention (single combined stage-1 +
    tier-agnostic stage-2, no merge_state)."""
    pool = case["pool"]
    decode_attention_fwd_int2_unified(
        case["q_for_decode"],
        pool.get_hp_key_buffer(0),
        pool.get_hp_value_buffer(0),
        pool.get_raw_key_buffer(0),
        pool.get_raw_value_buffer(0),
        pool.get_key_scales_zeros(0),
        pool.get_value_scales_zeros(0),
        case["o"],
        case["hp_kv_indptr"],
        case["hp_kv_indices"],
        case["quant_kv_indptr"],
        case["quant_kv_indices"],
        case["attn_logits"],
        case["attn_lse"],
        case["hp_num_kv_splits"],
        case["quant_num_kv_splits"],
        case["hp_max_kv_splits"],
        case["quant_max_kv_splits"],
        case["sm_scale"],
    )


def _reference_output_mixed(case: dict) -> torch.Tensor:
    """Dense BF16 reference for the mixed-KV kernel.

    HP tokens are read directly from the HP buffer (exact BF16).
    Quantised tokens are dequantised back to BF16 from the quant buffer.
    The two sources are gathered in original sequence order and then passed
    through the standard ``decode_attention_fwd`` kernel.

    For int2 the query is already Hadamard-rotated (``q_for_decode``), and the
    quant keys were also rotated before quantisation, while HP keys are stored
    in their original (un-rotated) form — exactly mirroring what the mixed
    kernel sees.
    """
    pool = case["pool"]
    shape: ShapeCase = case["shape"]
    kv_dtype: str = case["kv_dtype"]
    req_locs: List[torch.Tensor] = case["req_locs"]

    # Dequantise only the quant slots that are part of this benchmark case.
    # The raw quant view is capacity-sized and also aliases pages currently used
    # by the HP tier, so dequantising the whole buffer can touch a huge amount
    # of irrelevant memory.
    quant_kv_indices = case["quant_kv_indices"]
    if quant_kv_indices.numel() > 0:
        dq_k = _dequantize(
            kv_dtype,
            pool.get_raw_key_buffer(0)[quant_kv_indices],
            pool.get_key_scales_zeros(0)[quant_kv_indices],
            shape.head_dim,
        )
        dq_v = _dequantize(
            kv_dtype,
            pool.get_raw_value_buffer(0)[quant_kv_indices],
            pool.get_value_scales_zeros(0)[quant_kv_indices],
            shape.head_dim,
        )
    else:
        dq_k = torch.empty(
            (0, shape.num_kv_heads, shape.head_dim), dtype=DTYPE, device=DEVICE
        )
        dq_v = torch.empty_like(dq_k)
    hp_k = pool.get_hp_key_buffer(0)   # [hp_size+1, H_kv, D]
    hp_v = pool.get_hp_value_buffer(0)

    seq_k: List[torch.Tensor] = []
    seq_v: List[torch.Tensor] = []
    quant_cursor = 0
    for locs in req_locs:
        locs = locs.to(torch.int64)
        hp_mask = locs > pool.hp_global_offset          # True → HP tier
        hp_idx = locs[hp_mask] - pool.hp_global_offset
        quant_len = locs.numel() - hp_idx.numel()

        k_seq = torch.empty(
            (locs.numel(), shape.num_kv_heads, shape.head_dim),
            dtype=DTYPE,
            device=DEVICE,
        )
        v_seq = torch.empty_like(k_seq)
        if hp_idx.numel() > 0:
            k_seq[hp_mask] = hp_k[hp_idx]
            v_seq[hp_mask] = hp_v[hp_idx]
        if quant_len > 0:
            quant_slice = slice(quant_cursor, quant_cursor + quant_len)
            quant_mask = ~hp_mask
            k_seq[quant_mask] = dq_k[quant_slice]
            v_seq[quant_mask] = dq_v[quant_slice]
            quant_cursor += quant_len
        seq_k.append(k_seq)
        seq_v.append(v_seq)
    assert quant_cursor == quant_kv_indices.numel()

    ref_k = torch.cat(seq_k, dim=0)   # [batch * seq_len, H_kv, D]
    ref_v = torch.cat(seq_v, dim=0)

    kv_indptr = torch.arange(
        0, (shape.batch_size + 1) * shape.seq_len, shape.seq_len,
        dtype=torch.int32, device=DEVICE,
    )
    kv_indices = torch.arange(
        shape.batch_size * shape.seq_len, dtype=torch.int64, device=DEVICE
    )
    ref_o = torch.empty_like(case["q_for_decode"])
    ref_logits = torch.empty(
        shape.batch_size, shape.num_q_heads, 1, shape.head_dim,
        dtype=torch.float32, device=DEVICE,
    )
    ref_lse = torch.empty(
        shape.batch_size, shape.num_q_heads, 1, dtype=torch.float32, device=DEVICE,
    )
    one_split = torch.ones(shape.batch_size, dtype=torch.int32, device=DEVICE)
    decode_attention_fwd(
        case["q_for_decode"], ref_k, ref_v, ref_o,
        kv_indptr, kv_indices, ref_logits, ref_lse,
        one_split, 1, case["sm_scale"], 1.0, 1.0,
    )
    return ref_o


def _build_table() -> Table:
    t = Table(
        box=rich_box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold white",
        show_lines=False,
        pad_edge=True,
    )
    # t.add_column("id",                     justify="left",  no_wrap=True, min_width=6)
    t.add_column("shape",    style="bold", justify="left",  no_wrap=True)
    t.add_column("backend",                justify="left",  no_wrap=True, min_width=19)
    t.add_column("group",                  justify="right", no_wrap=True, min_width=7)
    t.add_column("mixed",                  justify="center",no_wrap=True, min_width=5)
    t.add_column("mean µs",  style="cyan", justify="right", no_wrap=True, min_width=18)
    t.add_column("max_err",                justify="right", no_wrap=True, min_width=8)
    return t


def _err_cell(v: float) -> str:
    if math.isnan(v):
        return "[dim]N/A[/dim]"
    if v < 0.01:
        return f"[green]{v:.4f}[/green]"
    if v < 0.05:
        return f"[yellow]{v:.4f}[/yellow]"
    return f"[bold red]{v:.4f}[/bold red]"


def _format_mean(mean_us: float, baseline_us: Optional[float] = None) -> str:
    """Format the mean cell as ``X.X µs`` plus optional ``(Y.YY×)`` speedup.

    speedup = baseline_us / mean_us, so >1× means this row is faster than its
    referenced baseline.  Color: green for ≥1.05×, yellow for 0.95–1.05×,
    red for ≤0.95× (slower).  When ``baseline_us is None`` only the absolute
    time is shown.
    """
    if math.isnan(mean_us):
        return "[dim]N/A[/dim]"
    s = f"{mean_us:7.1f} µs"
    if baseline_us is None or math.isnan(baseline_us) or mean_us <= 0:
        return s
    ratio = baseline_us / mean_us
    if ratio >= 1.05:
        color = "green"
    elif ratio >= 0.95:
        color = "yellow"
    else:
        color = "red"
    return s + f"  [{color}]({ratio:.2f}×)[/{color}]"


def _add_row(
    table: Table,
    shape_name: str,
    backend: str,
    group_str: str,
    mixed_str: str,
    mean_us: float,
    max_abs_err: float,
    row_style: str = "",
    case_id: Optional[int] = None,
    baseline_us: Optional[float] = None,
    baseline_id: Optional[int] = None,
) -> None:
    # id_cell = "[dim]—[/dim]" if case_id is None else str(case_id)
    # if baseline_id is not None:
    #     id_cell += f"\n[dim](vs {baseline_id})[/dim]"
    table.add_row(
        # id_cell,
        shape_name,
        backend,
        group_str,
        mixed_str,
        _format_mean(mean_us, baseline_us),
        _err_cell(max_abs_err),
        style=row_style,
    )


def _normalize_group_size(raw) -> Optional[int]:
    """Parse a group-size value (int or str) into the internal representation.

    Convention:
      * ``0`` (preferred) or ``1`` (legacy) or ``None``/``"scalar"``/``"none"``
        all mean "the whole head_dim is a single group" — i.e., one
        (scale, zero) pair per head.  Internally returned as ``None``.
      * Any int ``> 1`` enables groupwise quant with that group size.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("0", "1", "none", "scalar", ""):
            return None
        return int(s)
    n = int(raw)
    return None if n in (0, 1) else n


_BACKENDS = {"int2", "triton_bf16", "flashinfer_bf16"}


def _run_one_case(
    case: dict,
    console: Console,
    table: Table,
    baseline_us: Optional[float] = None,
    is_baseline: bool = False,  # kept for backward compat; no longer used
) -> float:
    """Build, time, and emit one row for a single benchmark case.

    Returns the measured mean latency in microseconds (or ``nan`` if skipped /
    unsupported), so ``main()`` can stash it in the ``id → mean_us`` map and
    feed it back as ``baseline_us`` for any later case that references this
    one via its ``baseline_id`` field.

    Required case fields:
        seq_len, batch_size, backend
    Optional fields (with defaults):
        id=None, baseline_id=None, model="Qwen/Qwen3-8B", group_size=0,
        mixed_kv=False, max_kv_splits=32, warmup=5, repeat=30, hp_prefix=32,
        hp_recent=128, no_correctness=False, profiling_mode="event", label=None
    """
    backend = case["backend"]
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; choose from {sorted(_BACKENDS)}")

    seq_len    = int(case["seq_len"])
    batch_size = int(case["batch_size"])
    model      = case.get("model", "Qwen/Qwen3-8B")
    max_kv_splits = int(case.get("max_kv_splits", 8))
    warmup     = int(case.get("warmup", 5))
    repeat     = int(case.get("repeat", 30))
    profiling  = case.get("profiling_mode", "event")
    no_check   = bool(case.get("no_correctness", False))
    hp_prefix  = int(case.get("hp_prefix", 32))
    hp_recent  = int(case.get("hp_recent", 128))
    mixed_kv   = bool(case.get("mixed_kv", False))
    gs_raw     = case.get("group_size", 0)
    group_size = _normalize_group_size(gs_raw)
    case_id    = case.get("id")
    baseline_id = case.get("baseline_id")
    label      = case.get("label") or f"{model.split('/')[-1]} bs={batch_size} sl={seq_len}"

    if backend in ("triton_bf16", "flashinfer_bf16") and (mixed_kv or group_size is not None):
        # mixed_kv / group_size only apply to int* backends
        if mixed_kv or group_size is not None:
            console.print(
                f"[yellow]NOTE[/yellow] case {label}/{backend}: "
                f"mixed_kv/group_size ignored for bf16 backend"
            )
        mixed_kv = False
        group_size = None

    shape = _shape_from_model(model, seq_len, batch_size)

    # Drop cases where the requested group_size doesn't divide head_dim.
    if group_size is not None and shape.head_dim % group_size != 0:
        console.print(
            f"[yellow]SKIP[/yellow] {label}/{backend}: "
            f"group_size={group_size} does not divide head_dim={shape.head_dim}"
        )
        return float("nan")

    gs_str = "scalar" if group_size is None else str(group_size)
    is_bf16 = backend in ("triton_bf16", "flashinfer_bf16")
    if is_bf16:
        gs_str = "—"
    mixed_str = "—" if is_bf16 else ("on" if mixed_kv else "off")

    def _emit(backend_label: str, mean_ms: float, max_err: float, style: str = ""):
        mean_us = mean_ms * 1000.0
        _add_row(
            table, label, backend_label, gs_str, mixed_str,
            mean_us, max_err,
            row_style=style, case_id=case_id,
            baseline_us=baseline_us, baseline_id=baseline_id,
        )
        return mean_us

    if backend == "triton_bf16":
        bf16 = _build_bf16_case(shape, max_kv_splits)
        mean_ms = _time_fn(lambda: _run_triton_bf16(bf16), profiling, warmup, repeat)
        return _emit("triton_bf16", mean_ms, float("nan"), style="cyan")

    if backend == "flashinfer_bf16":
        bf16 = _build_bf16_case(shape, max_kv_splits)
        fi_state = _build_flashinfer_state(shape, bf16)
        if fi_state is None:
            id_cell = "[dim]—[/dim]" if case_id is None else str(case_id)
            table.add_row(
                id_cell, label, "flashinfer_bf16",
                "—", "—", "", "[dim](not installed)[/dim]", style="dim",
            )
            return float("nan")
        mean_ms = _time_fn(lambda: _run_flashinfer_bf16(fi_state), profiling, warmup, repeat)
        tc_tag = "+tc" if fi_state["use_tensor_cores"] else ""
        return _emit(f"flashinfer_bf16{tc_tag}", mean_ms, float("nan"), style="cyan")

    # ---- int2 / int4 / int8 ----
    if mixed_kv:
        mixed = _build_mixed_kv_case(
            shape, backend, group_size, hp_prefix, hp_recent, max_kv_splits
        )
        if not no_check:
            _run_mixed_kernel(mixed)
            ref = _reference_output_mixed(mixed)
            atol, rtol = _tolerances(backend)
            if not torch.allclose(mixed["o"], ref, atol=atol, rtol=rtol):
                err = (mixed["o"] - ref).abs().max().item()
                console.print(
                    f"[bold red]WARN[/bold red] {label}/{backend}/group={group_size}/mixed=on: "
                    f"max_abs_err={err:.4f}"
                )
            max_err = float((mixed["o"] - ref).abs().max().item())
        else:
            max_err = float("nan")
        mean_ms = _time_fn(lambda: _run_mixed_kernel(mixed), profiling, warmup, repeat)
        return _emit(backend, mean_ms, max_err, style="yellow")

    case_d = _build_case(shape, backend, group_size, max_kv_splits)
    if not no_check:
        out = _run_kernel(case_d)
        ref = _reference_output(case_d)
        atol, rtol = _tolerances(backend)
        if not torch.allclose(out, ref, atol=atol, rtol=rtol):
            err = (out - ref).abs().max().item()
            console.print(
                f"[bold red]WARN[/bold red] {label}/{backend}/group={group_size}: "
                f"max_abs_err={err:.4f}"
            )
        max_err = float((out - ref).abs().max().item())
    else:
        max_err = float("nan")
    mean_ms = _time_fn(lambda: _run_kernel(case_d), profiling, warmup, repeat)
    return _emit(backend, mean_ms, max_err)


def _load_yaml_cases(path: str) -> List[dict]:
    """Load a YAML config of the form::

        defaults:
          key: value
          ...
        cases:
          - {id: 1, seq_len: 8192,  batch_size: 1, backend: triton_bf16}
          - {id: 2, seq_len: 8192,  batch_size: 1, backend: int2,
             group_size: 0, baseline_id: 1}                           # ratio vs id=1
          - {id: 3, seq_len: 32768, batch_size: 1, backend: triton_bf16}
          - {id: 4, seq_len: 32768, batch_size: 1, backend: int2,
             group_size: 0, baseline_id: 3}                           # ratio vs id=3

    ``id`` is **required** and must be a unique integer (the user picks the
    numbering scheme; the loader does not auto-assign).  ``baseline_id`` is a
    *per-case* field that points at another case's id (which must appear
    earlier in the file) whose mean is used to compute this row's speedup.

    Returns the case list with defaults merged into each case (case fields win).
    """
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict) or "cases" not in cfg:
        raise ValueError(f"YAML config {path} must define a top-level 'cases' list")
    defaults = cfg.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ValueError(f"'defaults' in {path} must be a mapping")
    cases_raw = cfg.get("cases") or []
    if not isinstance(cases_raw, list):
        raise ValueError(f"'cases' in {path} must be a list")

    out: List[dict] = []
    seen_ids: set = set()
    for i, c in enumerate(cases_raw):
        if not isinstance(c, dict):
            raise ValueError(f"case[{i}] must be a mapping, got {type(c).__name__}")
        merged = {**defaults, **c}
        for required in ("id", "backend", "seq_len", "batch_size"):
            if required not in merged:
                raise ValueError(
                    f"case[{i}] missing required field {required!r}; "
                    f"every case must declare id (int), backend, seq_len, batch_size"
                )
        try:
            cid = int(merged["id"])
        except (TypeError, ValueError) as e:
            raise ValueError(f"case[{i}] id must be an integer, got {merged['id']!r}") from e
        if cid in seen_ids:
            raise ValueError(f"case[{i}] duplicate id={cid}")
        seen_ids.add(cid)
        merged["id"] = cid
        # Validate baseline_id (must reference an earlier case).
        if "baseline_id" in merged:
            try:
                bid = int(merged["baseline_id"])
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"case[{i}] (id={cid}) baseline_id must be an integer, "
                    f"got {merged['baseline_id']!r}"
                ) from e
            if bid not in seen_ids or bid == cid:
                raise ValueError(
                    f"case[{i}] (id={cid}) baseline_id={bid} must reference a case that "
                    f"appears earlier in the file; known earlier ids: {sorted(seen_ids - {cid})}"
                )
            merged["baseline_id"] = bid
        out.append(merged)
    return out


def _build_cases_from_cli_args(args) -> List[dict]:
    """Convert legacy CLI args into the same list-of-cases representation.

    Auto-assigns integer ids and tags every non-baseline row with
    ``baseline_id`` pointing at the first BF16 baseline (if present), so the
    speedup column lights up automatically without YAML.
    """
    cases: List[dict] = []
    common = dict(
        model=args.model, seq_len=args.seq_len, batch_size=args.batch_size,
        max_kv_splits=args.max_kv_splits, warmup=args.warmup, repeat=args.repeat,
        hp_prefix=args.hp_prefix, hp_recent=args.hp_recent,
        no_correctness=args.no_correctness, profiling_mode=args.profiling_mode,
    )
    next_id = 1
    baselines: List[str] = args.baselines or []
    cli_baseline_id: Optional[int] = None
    for b in baselines:
        case = {**common, "backend": b, "id": next_id}
        if cli_baseline_id is None:
            cli_baseline_id = next_id
        else:
            # Other baselines (e.g. flashinfer when triton_bf16 is the anchor)
            # should also report a ratio against the anchor.
            case["baseline_id"] = cli_baseline_id
        next_id += 1
        cases.append(case)
    for kv_dtype in args.dtypes:
        for raw_gs in args.group_sizes:
            gs = _normalize_group_size(raw_gs)
            base = {**common, "backend": kv_dtype, "group_size": gs,
                    "mixed_kv": False, "id": next_id}
            if cli_baseline_id is not None:
                base["baseline_id"] = cli_baseline_id
            next_id += 1
            cases.append(base)
            if not args.no_mixed_kv:
                m = {**common, "backend": kv_dtype, "group_size": gs,
                     "mixed_kv": True, "id": next_id}
                if cli_baseline_id is not None:
                    m["baseline_id"] = cli_baseline_id
                next_id += 1
                cases.append(m)
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help=(
            "Path to a YAML config (see module docstring for the schema). "
            "When provided, takes precedence over the CLI sweep flags below "
            "(--model/--seq-len/--batch-size/--dtypes/--group-sizes/...)."
        ),
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-8B",
        metavar="MODEL",
        help="HuggingFace model name or local path; derives num_q_heads, num_kv_heads, head_dim.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=4096,
        metavar="N",
        help="KV sequence length for the model-derived shape.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        metavar="N",
        help="Batch size for the model-derived shape.",
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        default=["int2"],
        choices=["int2"],
    )
    parser.add_argument(
        "--group-sizes",
        nargs="+",
        default=["0", "16", "32", "64"],
        help=(
            "Group sizes to sweep. '0' (preferred) or '1' (legacy) maps to the "
            "scalar path: the whole head_dim shares a single (scale, zero) pair "
            "(num_groups == 1). Any value >1 enables groupwise quantisation "
            "with that group size."
        ),
    )
    parser.add_argument(
        "--max-kv-splits",
        type=int,
        default=8,
        help=(
            "UPPER BOUND on the number of KV splits per sequence. The actual "
            "per-batch split count is decided dynamically at runtime by the "
            "production heuristic (get_num_kv_splits_triton), which considers "
            "device SM count, num_seq, num_head and seq_len. This flag only "
            "caps that dynamic value; it matches "
            "ServerArgs.triton_attention_num_kv_splits=32 and only matters for "
            "low-batch long-context decode where the heuristic wants to split "
            "more aggressively to expose parallelism."
        ),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--profiling-mode",
        choices=["event", "torch_profiler"],
        default="torch_profiler",
        help=(
            "How to measure per-call latency. 'event' uses CUDA events (wall "
            "time including host launch overhead); 'torch_profiler' uses "
            "torch.profiler to sum pure GPU kernel durations only."
        ),
    )
    parser.add_argument(
        "--no-correctness",
        action="store_true",
        help="Skip the dense-reference correctness check (for pure bench runs).",
    )
    parser.add_argument(
        "--baselines",
        nargs="*",
        default=["triton_bf16", "flashinfer_bf16"],
        choices=["triton_bf16", "flashinfer_bf16"],
        help=(
            "BF16 reference baselines to include for comparison. "
            "Pass with no values (--baselines) to disable all baselines."
        ),
    )
    parser.add_argument(
        "--no-mixed-kv",
        action="store_true",
        help="Skip the mixed-KV-windows variant (SGLANG_ENABLE_MIXED_KV_WINDOWS=1).",
    )
    parser.add_argument(
        "--hp-prefix",
        type=int,
        default=32,
        help="HP-tier prefix tokens per sequence (SGLANG_MIXED_KV_PREFIX_TOKENS default=32).",
    )
    parser.add_argument(
        "--hp-recent",
        type=int,
        default=128,
        help="HP-tier recent tokens per sequence (SGLANG_MIXED_KV_RECENT_TOKENS default=128).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required for this benchmark.")

    torch.manual_seed(args.seed)

    # ---- Build the case list (YAML if provided, else legacy CLI sweep) ----
    if args.config is not None:
        cases = _load_yaml_cases(args.config)
        header = f"[bold]config[/bold] {args.config}  ({len(cases)} cases)"
    else:
        cases = _build_cases_from_cli_args(args)
        # Legacy CLI summary header (single shape).
        ms = _shape_from_model(args.model, args.seq_len, args.batch_size)
        header = (
            f"[bold]model[/bold] {args.model}  "
            f"[bold]Q-heads[/bold] {ms.num_q_heads}  "
            f"[bold]KV-heads[/bold] {ms.num_kv_heads}  "
            f"[bold]head_dim[/bold] {ms.head_dim}  "
            f"[bold]seq_len[/bold] {ms.seq_len}  "
            f"[bold]batch_size[/bold] {ms.batch_size}  "
            f"[bold]max_kv_splits[/bold] {args.max_kv_splits} (upper bound; "
            f"actual chosen by get_num_kv_splits_triton at runtime)"
        )

    console = Console(width=140, height=50, force_terminal=True)
    table = _build_table()
    console.print(header)

    # Per-case baselines: look up the referenced case's mean from this map as
    # we iterate (validation in _load_yaml_cases ensures the referent appears
    # earlier in the file so it has already been measured).
    id_to_us: dict = {}
    with Live(table, console=console, refresh_per_second=8, vertical_overflow="visible"):
        for case in cases:
            bid = case.get("baseline_id")
            baseline_us: Optional[float] = id_to_us.get(bid) if bid is not None else None
            mean_us = _run_one_case(
                case, console, table,
                baseline_us=baseline_us,
                is_baseline=False,
            )
            cid = case.get("id")
            if cid is not None and not math.isnan(mean_us):
                id_to_us[cid] = mean_us


if __name__ == "__main__":
    main()
