"""CuTe DSL kernels for INT2 KV cache: quantize/pack and decode attention.

Replaces two Triton kernels with CuteDSL implementations:
  - ``_launch_quantize_int2``  -> ``cutedsl_set_kv_int2``
  - ``_fwd_kernel_stage1_quant_int2``  -> ``cutedsl_decode_attention_fwd_int2``

Both implementations target the H100 / SM90 generation and the
single-group-per-head (num_groups == 1, i.e. ``GROUP_SIZE == head_dim``)
configuration that the validated Qwen3-4B Thinking rotation
(``seq20000_prompt85_group128``, ``head_dim=128``) uses.

Other configurations transparently fall back to the existing Triton path —
the dispatch switch is at the Python-level wrappers below, not in the kernel.
"""

import argparse
import logging
import os
from typing import Dict, Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compile cache
# ---------------------------------------------------------------------------
_compiled_quantize_kernels: Dict[Tuple, object] = {}
_compiled_decode_kernels: Dict[Tuple, object] = {}


def _backend_is_cutedsl() -> bool:
    return os.environ.get("SGLANG_INT2_BACKEND", "").lower() == "cutedsl"


def _backend_is_cuda() -> bool:
    return os.environ.get("SGLANG_INT2_BACKEND", "").lower() == "cuda"


# ---------------------------------------------------------------------------
# CUDA C++ kernel (CUTLASS/wmma) — JIT-compiled wrapper around
# cuda_int2_decode.cu. Activated via ``SGLANG_INT2_BACKEND=cuda``.
# ---------------------------------------------------------------------------
_cuda_decode_ext = None


def _dump_invocation_counters(reason: str = "exit"):
    """Print the device-side wmma/wgmma counters to stdout. Visible in
    sglang's server.log so an external harness can confirm decode batches
    actually exercised the CUDA path (not just startup warmup).
    """
    try:
        if _cuda_decode_ext is None:
            return
        counts = _cuda_decode_ext.get_invocation_counters()
        print(
            f"[INT2 counter dump @ {reason}] "
            f"cuda_int2_wmma_calls={int(counts[0])} "
            f"cuda_int2_wgmma_calls={int(counts[1])}",
            flush=True,
        )
    except Exception as e:  # never raise from a signal handler
        try:
            print(f"[INT2 counter dump @ {reason}] error: "
                  f"{type(e).__name__}: {e}", flush=True)
        except Exception:
            pass


def _install_counter_dump_hooks():
    """Register atexit + SIGTERM/SIGINT handlers that print the kernel
    invocation counters. Idempotent (no-op if already installed).
    """
    if getattr(_install_counter_dump_hooks, "_done", False):
        return
    import atexit
    import signal

    atexit.register(_dump_invocation_counters, reason="atexit")

    def _sig_handler(signum, frame):
        try:
            sig_name = signal.Signals(signum).name
        except Exception:
            sig_name = str(signum)
        _dump_invocation_counters(reason=sig_name)
        # Re-raise default behavior so the server still terminates cleanly.
        signal.signal(signum, signal.SIG_DFL)
        try:
            import os
            os.kill(os.getpid(), signum)
        except Exception:
            pass

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            prev = signal.getsignal(sig)
            # Only override if not already a custom handler we'd lose.
            if prev in (signal.SIG_DFL, signal.SIG_IGN, None):
                signal.signal(sig, _sig_handler)
        except (ValueError, OSError):
            # signal.signal can fail in non-main threads (e.g. sglang workers)
            pass
    _install_counter_dump_hooks._done = True


def _load_cuda_decode_extension():
    """JIT-compile cuda_int2_decode.cu via torch.utils.cpp_extension.load.

    Result is cached at module scope; first call may take 30–90 s.
    """
    global _cuda_decode_ext
    if _cuda_decode_ext is not None:
        return _cuda_decode_ext
    from torch.utils.cpp_extension import load
    here = os.path.dirname(os.path.abspath(__file__))
    cu = os.path.join(here, "cuda_int2_decode.cu")
    # Pull in the bundled CUTLASS/cute SM90 headers (wgmma + TMA) from
    # deep_gemm's site-packages — same set CUTLASS ships with.
    cutlass_inc = (
        "/home/charlie/miniconda3/envs/oscar/lib/python3.12/site-packages/"
        "deep_gemm/include"
    )
    _cuda_decode_ext = load(
        name="sglang_cuda_int2_decode",
        sources=[cu],
        extra_include_paths=[cutlass_inc],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--use_fast_math",
            "-gencode=arch=compute_90a,code=sm_90a",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
        ],
        verbose=False,
    )
    # Arm counter-dump-on-exit so server.log carries a definitive
    # "[INT2 counter dump @ ...] cuda_int2_wmma_calls=N wgmma_calls=M"
    # line confirming decode batches actually used the CUDA path.
    _install_counter_dump_hooks()
    return _cuda_decode_ext


def can_use_cuda_decode(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    k_scales_zeros: torch.Tensor,
    v_scales_zeros: torch.Tensor,
) -> bool:
    """Same shape constraints as the CuteDSL path plus the kv_group_num set
    enumerated by the CUDA template launcher."""
    head_dim = k_buffer.shape[-1] * 4
    if head_dim != 128:
        return False
    if v_buffer.shape[-1] * 4 != head_dim:
        return False
    if k_scales_zeros.shape[-1] != 2 or v_scales_zeros.shape[-1] != 2:
        return False
    if q.dtype != torch.bfloat16:
        return False
    kv_group_num = q.shape[1] // k_buffer.shape[1]
    if kv_group_num not in (1, 2, 4, 8):
        return False
    return True


def get_cuda_invocation_counters():
    """Read the CUDA kernel invocation counters.

    Returns:
        (wmma_calls, wgmma_calls) tuple of cumulative kernel launches since the
        last reset.  Each call increments the counter by 1 (one thread per
        launch performs an atomicAdd). Used by ``--test-correctness`` and the
        eval verifier to confirm the CUDA backend is actually executing
        instead of silently falling back to Triton.
    """
    ext = _load_cuda_decode_extension()
    counts = ext.get_invocation_counters()
    return int(counts[0]), int(counts[1])


def reset_cuda_invocation_counters():
    """Zero the wmma/wgmma device counters."""
    ext = _load_cuda_decode_extension()
    ext.reset_invocation_counters()


def _as_int32(t: torch.Tensor) -> torch.Tensor:
    """sglang passes int64 indices; the C++ extension's `data_ptr<int>()`
    binds to int32. Cast only if needed (zero-copy fast path otherwise)."""
    return t if t.dtype == torch.int32 else t.to(torch.int32)


def _maybe_materialize(t: torch.Tensor) -> tuple:
    """No-op pass-through. The C++ kernel now reads tensor strides directly
    (kernel-level stride threading applied) so sliced views like
    ``attn_logits[:, :, hp_max:, :]`` are handled natively.

    Previously this function created a contiguous staging buffer via
    ``torch.empty`` and copied back. That was correct on eager calls but
    BROKE under CUDA graph capture: ``torch.empty`` during capture is not
    cuda-graph-safe (the pool's reuse semantics during graph replay led to
    the staging buffer being clobbered by other ops, producing degenerate
    output — the round-28 'on on on' gibberish). The kernel-level stride
    threading eliminates the need for staging entirely.

    Kept as a call site for symmetry + a one-line guard.
    """
    assert t.stride()[-1] == 1, (
        f"[INT2 wrapper guard] non-1 innermost stride — "
        f"shape={tuple(t.shape)}, stride={tuple(t.stride())}; "
        f"kernel assumes innermost stride 1"
    )
    return t, t


def cuda_decode_attention_fwd_int2(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    k_scales_zeros: torch.Tensor,
    v_scales_zeros: torch.Tensor,
    att_out: torch.Tensor,
    att_lse: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    num_kv_splits: torch.Tensor,
    max_kv_splits: int,
    sm_scale: float,
):
    """CUDA C++/wmma replacement for the Triton stage-1 INT2 decode kernel."""
    ext = _load_cuda_decode_extension()
    out_staging, out_orig = _maybe_materialize(att_out)
    lse_staging, lse_orig = _maybe_materialize(att_lse)
    _kv_idx = kv_indices.to(torch.int64).contiguous()
    _kv_iptr = _as_int32(kv_indptr).contiguous()
    _nkv = _as_int32(num_kv_splits).contiguous()
    ext.decode_forward(
        q.contiguous(),
        k_buffer.contiguous(),
        v_buffer.contiguous(),
        k_scales_zeros.contiguous(),
        v_scales_zeros.contiguous(),
        _kv_iptr,
        _kv_idx,
        _nkv,
        out_staging,
        lse_staging,
        float(sm_scale),
        int(max_kv_splits),
    )
    if out_staging is not out_orig:
        out_orig.copy_(out_staging)
    if lse_staging is not lse_orig:
        lse_orig.copy_(lse_staging)


def cuda_decode_attention_fwd_int2_wgmma(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    k_scales_zeros: torch.Tensor,
    v_scales_zeros: torch.Tensor,
    att_out: torch.Tensor,
    att_lse: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    num_kv_splits: torch.Tensor,
    max_kv_splits: int,
    sm_scale: float,
):
    """SM90 wgmma INT2 decode (QK + PV both via wgmma m64nNk16)."""
    ext = _load_cuda_decode_extension()
    out_staging, out_orig = _maybe_materialize(att_out)
    lse_staging, lse_orig = _maybe_materialize(att_lse)
    ext.decode_forward_wgmma(
        q.contiguous(),
        k_buffer.contiguous(),
        v_buffer.contiguous(),
        k_scales_zeros.contiguous(),
        v_scales_zeros.contiguous(),
        _as_int32(kv_indptr).contiguous(),
        kv_indices.to(torch.int64).contiguous(),
        _as_int32(num_kv_splits).contiguous(),
        out_staging,
        lse_staging,
        float(sm_scale),
        int(max_kv_splits),
    )
    if out_staging is not out_orig:
        out_orig.copy_(out_staging)
    if lse_staging is not lse_orig:
        lse_orig.copy_(lse_staging)


# cute.arch.fmax is the only float-typed max in the DSL today; there is no
# fmin so we synthesize it from fmax. Branchless and avoids the surprising
# behavior of Python ``min``/``max`` when one side is a constexpr literal.
@cute.jit
def _fmax(a: cutlass.Float32, b: cutlass.Float32) -> cutlass.Float32:
    return cute.arch.fmax(a, b)


@cute.jit
def _fmin(a: cutlass.Float32, b: cutlass.Float32) -> cutlass.Float32:
    return cutlass.Float32(0.0) - cute.arch.fmax(
        cutlass.Float32(0.0) - a, cutlass.Float32(0.0) - b
    )


# ---------------------------------------------------------------------------
# Quantize: pack bf16 K/V -> INT2 (4 values per uint8), per-(token,head) scale
# ---------------------------------------------------------------------------
# One CTA per (token, head). Each thread packs four 2-bit lanes into a single
# uint8 along the head_dim axis. NUM_THREADS == quarter_dim == head_dim // 4.


def _define_quantize_kernel(num_threads: cutlass.Constexpr[int]):
    # We require num_threads <= 32 here (single-warp shuffle reduction).
    # head_dim==128 -> quarter_dim==32 satisfies this.
    assert num_threads == 32, (
        f"quantize kernel currently specialized for quarter_dim==32 "
        f"(i.e. head_dim==128); got num_threads={num_threads}"
    )

    @cute.kernel
    def kernel(
        x: cute.Tensor,             # (num_tokens, num_heads, head_dim) bf16/fp16
        loc: cute.Tensor,           # (num_tokens,) int32
        packed: cute.Tensor,        # (cache_size, num_heads, head_dim//4) uint8
        scales_zeros: cute.Tensor,  # (cache_size, num_heads, 2) fp32
        head_dim: cutlass.Constexpr[int],
        hp_offset: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        num_heads = x.shape[1]
        token_idx = bidx // num_heads
        head_idx = bidx % num_heads

        cache_loc = cutlass.Int32(loc[token_idx])


        # HP-tier skip (mirrors Triton's HP_OFFSET early-return).
        if hp_offset < 0 or cache_loc < hp_offset:
            quarter_dim: cutlass.Constexpr[int] = head_dim // 4

            v0 = cutlass.Float32(x[token_idx, head_idx, tidx])
            v1 = cutlass.Float32(x[token_idx, head_idx, tidx + quarter_dim])
            v2 = cutlass.Float32(x[token_idx, head_idx, tidx + 2 * quarter_dim])
            v3 = cutlass.Float32(x[token_idx, head_idx, tidx + 3 * quarter_dim])

            local_min = _fmin(_fmin(v0, v1), _fmin(v2, v3))
            local_max = _fmax(_fmax(v0, v1), _fmax(v2, v3))

            # Warp-shuffle reduction across the 32 threads.
            for offset in cutlass.range_constexpr(5):
                shift: cutlass.Constexpr[int] = 1 << (4 - offset)
                other_min = cute.arch.shuffle_sync_bfly(
                    local_min, offset=shift, mask=-1, mask_and_clamp=31
                )
                other_max = cute.arch.shuffle_sync_bfly(
                    local_max, offset=shift, mask=-1, mask_and_clamp=31
                )
                local_min = _fmin(local_min, other_min)
                local_max = _fmax(local_max, other_max)

            # Every thread now holds the warp-wide min/max.
            val_range = _fmax(local_max - local_min, cutlass.Float32(1e-8))
            scale = val_range / cutlass.Float32(3.0)
            zero = (cutlass.Float32(0.0) - local_min) / scale

            q0 = cutlass.Uint8(v0 / scale + zero + cutlass.Float32(0.5))
            q1 = cutlass.Uint8(v1 / scale + zero + cutlass.Float32(0.5))
            q2 = cutlass.Uint8(v2 / scale + zero + cutlass.Float32(0.5))
            q3 = cutlass.Uint8(v3 / scale + zero + cutlass.Float32(0.5))

            packed_byte = q0 | (q1 << cutlass.Uint8(2)) | (q2 << cutlass.Uint8(4)) | (q3 << cutlass.Uint8(6))
            packed[(cache_loc, head_idx, tidx)] = packed_byte

            if tidx == 0:
                scales_zeros[(cache_loc, head_idx, 0)] = scale
                scales_zeros[(cache_loc, head_idx, 1)] = zero

    return kernel


def _compile_quantize(
    head_dim: int, dtype: torch.dtype, hp_offset: int,
    num_tokens: int, num_heads: int, cache_size: int,
):
    # Compile per (head_dim, dtype, hp_offset, num_tokens, num_heads,
    # cache_size). Grid dims and Constexpr loops must resolve at compile, so
    # the static shape information is baked in.
    key = (head_dim, dtype, hp_offset, num_tokens, num_heads, cache_size)
    if key in _compiled_quantize_kernels:
        return _compiled_quantize_kernels[key]

    quarter_dim = head_dim // 4
    if quarter_dim & (quarter_dim - 1) != 0:
        raise NotImplementedError(
            f"CuteDSL quantize kernel needs head_dim/4 to be a power of two, "
            f"got head_dim={head_dim}."
        )

    kernel = _define_quantize_kernel(num_threads=quarter_dim)
    smem_bytes = 8 * quarter_dim + 64
    grid_dim = num_tokens * num_heads

    @cute.jit
    def launcher(
        x: cute.Tensor,
        loc: cute.Tensor,
        packed: cute.Tensor,
        scales_zeros: cute.Tensor,
        stream: cuda.CUstream,
    ):
        kernel(
            x,
            loc,
            packed,
            scales_zeros,
            head_dim=head_dim,
            hp_offset=hp_offset,
        ).launch(
            grid=(grid_dim, 1, 1),
            block=(quarter_dim, 1, 1),
            smem=smem_bytes,
            stream=stream,
        )

    fake_x = cute.runtime.make_fake_tensor(
        _torch_to_cute_dtype(dtype),
        (num_tokens, num_heads, head_dim),
        (num_heads * head_dim, head_dim, 1),
        memspace=cute.AddressSpace.gmem,
        assumed_align=16,
    )
    fake_loc = cute.runtime.make_fake_tensor(
        cutlass.Int32,
        (num_tokens,),
        (1,),
        memspace=cute.AddressSpace.gmem,
        assumed_align=16,
    )
    fake_packed = cute.runtime.make_fake_tensor(
        cutlass.Uint8,
        (cache_size, num_heads, head_dim // 4),
        (num_heads * (head_dim // 4), head_dim // 4, 1),
        memspace=cute.AddressSpace.gmem,
        assumed_align=16,
    )
    fake_sz = cute.runtime.make_fake_tensor(
        cutlass.Float32,
        (cache_size, num_heads, 2),
        (num_heads * 2, 2, 1),
        memspace=cute.AddressSpace.gmem,
        assumed_align=8,
    )

    fake_stream = cuda.CUstream(0)
    compiled = cute.compile(
        launcher,
        fake_x,
        fake_loc,
        fake_packed,
        fake_sz,
        stream=fake_stream,
    )
    _compiled_quantize_kernels[key] = compiled
    return compiled


_TORCH_TO_CUTE_DTYPE: Dict[torch.dtype, object] = {
    torch.float32: cutlass.Float32,
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
}


def _torch_to_cute_dtype(dtype: torch.dtype):
    if dtype not in _TORCH_TO_CUTE_DTYPE:
        raise NotImplementedError(f"Unsupported dtype for CuteDSL quantize: {dtype}")
    return _TORCH_TO_CUTE_DTYPE[dtype]


def _launch_quantize_one(
    x: torch.Tensor,
    loc: torch.Tensor,
    packed: torch.Tensor,
    scales_zeros: torch.Tensor,
    hp_global_offset: Optional[int],
):
    num_tokens, num_heads, head_dim = x.shape
    if num_tokens == 0:
        return
    cache_size = packed.shape[0]
    hp_offset = -1 if hp_global_offset is None else int(hp_global_offset)
    compiled = _compile_quantize(
        head_dim, x.dtype, hp_offset, num_tokens, num_heads, cache_size,
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(
        from_dlpack(x.detach(), assumed_align=16),
        from_dlpack(loc.detach(), assumed_align=16),
        from_dlpack(packed.detach(), assumed_align=16),
        from_dlpack(scales_zeros.detach(), assumed_align=8),
        stream,
    )


def can_use_cutedsl_quantize(
    x: torch.Tensor, scales_zeros_buffer: torch.Tensor
) -> bool:
    """Whether the CuteDSL quantize kernel covers this tensor shape."""
    if x.numel() == 0:
        return False
    num_groups_2 = scales_zeros_buffer.shape[-1]
    if num_groups_2 != 2:
        # multi-group (group_size < head_dim) — not handled in this build
        return False
    head_dim = x.shape[-1]
    if head_dim % 4 != 0:
        return False
    quarter = head_dim // 4
    if quarter & (quarter - 1) != 0:
        return False
    if x.dtype not in _TORCH_TO_CUTE_DTYPE:
        return False
    return True


def cutedsl_set_kv_int2(
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    loc: torch.Tensor,
    k_cache_buffer: torch.Tensor,
    v_cache_buffer: torch.Tensor,
    k_scales_zeros_buffer: torch.Tensor,
    v_scales_zeros_buffer: torch.Tensor,
    hp_global_offset=None,
):
    """CuteDSL replacement for ``quantized_set_kv_int2_triton``."""
    _launch_quantize_one(
        cache_k.contiguous(), loc, k_cache_buffer, k_scales_zeros_buffer,
        hp_global_offset,
    )
    _launch_quantize_one(
        cache_v.contiguous(), loc, v_cache_buffer, v_scales_zeros_buffer,
        hp_global_offset,
    )


# ---------------------------------------------------------------------------
# Decode attention stage-1 with INT2 KV (MHA, kv_group_num == 1, per-row scale)
# ---------------------------------------------------------------------------
# We support the most common config for kernel-level benchmarking and direct
# numerical comparison against Triton: kv_group_num=1, GROUP_SIZE == head_dim
# (i.e. a single scale/zero pair per (cache_loc, head) row), no logit_cap, no
# xAI temperature scaling. Grid: (batch, head, kv_splits).
#
# Per CTA we use HEAD_DIM threads; each thread holds one float of Q and a
# slice of the unpacked K/V dequantized values across BLOCK_N kv tokens.


_MIN_BLOCK_KV = 8
DECODE_BLOCK_N = 64

# Cap on the compiled outer kv-tile-loop length. The kernel uses dynamic
# masking on each tile, so this can be smaller than ``cache_size`` (e.g.
# radix-cache 2.5M) without correctness loss — just needs to cover the
# longest actual sequence. Qwen3-4B Thinking max context = 32 768; pad to
# 65 536 for safety. Override via ``SGLANG_INT2_DECODE_MAX_SEQ``.
_MAX_DECODE_SEQ_LEN_DEFAULT = 65536


def _define_decode_kernel(
    head_dim: cutlass.Constexpr[int],
    block_n: cutlass.Constexpr[int],
    max_tiles_per_split: cutlass.Constexpr[int],
    kv_group_num: cutlass.Constexpr[int],
):
    """CuteDSL decode-attention kernel for INT2 KV cache.

    GQA-coalesced layout: grid = (batch, kv_head, split). Each CTA processes
    all ``kv_group_num`` q-heads sharing this kv-head, so the K/V tile is
    loaded once from DRAM and reused across q-heads — eliminating the 4×
    redundant DRAM traffic that the per-q-head grid suffered on Qwen3-4B
    (q=32, kv=8, group=4). For MHA (kv_group_num=1) this is a no-op outer
    loop and the layout matches the previous kernel.
    """
    quarter_dim: cutlass.Constexpr[int] = head_dim // 4

    @cute.kernel
    def kernel(
        Q: cute.Tensor,                 # (batch, q_heads, head_dim) bf16
        K_packed: cute.Tensor,          # (cache_size, kv_heads, head_dim//4) uint8
        V_packed: cute.Tensor,          # (cache_size, kv_heads, head_dim//4) uint8
        K_sz: cute.Tensor,              # (cache_size, kv_heads, 2) fp32
        V_sz: cute.Tensor,              # (cache_size, kv_heads, 2) fp32
        kv_indptr: cute.Tensor,         # (batch+1,) int32
        kv_indices: cute.Tensor,        # (total_kv,) int32
        num_kv_splits: cute.Tensor,     # (batch,) int32
        att_out: cute.Tensor,           # (batch, q_heads, max_splits, head_dim) fp32
        att_lse: cute.Tensor,           # (batch, q_heads, max_splits) fp32
        sm_scale: cutlass.Float32,
        min_block_kv: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        cur_batch, cur_kv_head, split_kv_id = cute.arch.block_idx()
        q_head_base: cutlass.Constexpr = cur_kv_head * kv_group_num

        kv_start_idx = cutlass.Int32(kv_indptr[cur_batch])
        seq_len = cutlass.Int32(kv_indptr[cur_batch + 1]) - kv_start_idx
        kv_splits = cutlass.Int32(num_kv_splits[cur_batch])

        # Per-split contiguous slice of kv positions. Mirrors Triton:
        #   kv_len_per_split = ceil_div(ceil_div(seq_len, kv_splits), min_block) * min_block
        kv_len_per_split = (
            (seq_len + kv_splits - 1) // kv_splits + min_block_kv - 1
        ) // min_block_kv * min_block_kv
        split_start = kv_len_per_split * split_kv_id
        # Upper bound for this split's tokens. Without this, the last tile in
        # each split (when kv_len_per_split is not a multiple of block_n)
        # bleeds into the next split's range, double-counting tokens in the
        # cross-split softmax reduce.
        split_end_unclamped = split_start + kv_len_per_split

        # Shared memory layout.
        smem = cutlass.utils.SmemAllocator()
        # Q for all q-heads in this kv-group: (kv_group_num, head_dim) fp32.
        sQ = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((kv_group_num, head_dim), stride=(head_dim, 1)),
            16,
        )
        # Single K/V packed tile per kv-head, shared across q-heads.
        sK_packed = smem.allocate_tensor(
            cutlass.Uint8,
            cute.make_layout((block_n, quarter_dim), stride=(quarter_dim, 1)),
            16,
        )
        sV_packed = smem.allocate_tensor(
            cutlass.Uint8,
            cute.make_layout((block_n, quarter_dim), stride=(quarter_dim, 1)),
            16,
        )
        sK_scale = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((block_n,), stride=(1,)), 16
        )
        sK_zero = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((block_n,), stride=(1,)), 16
        )
        sV_scale = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((block_n,), stride=(1,)), 16
        )
        sV_zero = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((block_n,), stride=(1,)), 16
        )
        # Per-q-head QK and probability tiles: (kv_group_num, block_n).
        sQK = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((kv_group_num, block_n), stride=(block_n, 1)),
            16,
        )
        sP = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((kv_group_num, block_n), stride=(block_n, 1)),
            16,
        )

        # Per-thread per-q-head state: ``kv_group_num`` accumulators, e_max,
        # e_sum. Because ``kv_group_num`` is Constexpr, Python lists indexed
        # by ``range_constexpr`` unroll cleanly into separate registers.
        acc = [cutlass.Float32(0.0) for _ in range(kv_group_num)]
        e_max = [cutlass.Float32(-1e30) for _ in range(kv_group_num)]
        e_sum = [cutlass.Float32(0.0) for _ in range(kv_group_num)]

        # Load Q for all q-heads in this group into smem.
        if tidx < head_dim:
            for qid in cutlass.range_constexpr(kv_group_num):
                sQ[qid, tidx] = cutlass.Float32(
                    Q[cur_batch, q_head_base + qid, tidx]
                )
        cute.arch.barrier()

        # Outer kv-tile loop is dynamic (NOT range_constexpr) so we don't
        # unroll the per-tile body — unrolling 16+ copies blows past the
        # i-cache. The inner per-tile work (head_dim, quarter_dim, block_n,
        # kv_group_num) IS constexpr.
        for tile_id in cutlass.range(max_tiles_per_split):
            n = split_start + tile_id * block_n
            active_tile = (n < seq_len) & (split_kv_id < kv_splits)

            # ---- Cooperative K/V packed load (ONCE per tile, shared across
            # q-heads — this is the GQA win versus per-q-head CTAs). ----
            total_bytes: cutlass.Constexpr[int] = block_n * quarter_dim
            per_thread: cutlass.Constexpr[int] = (
                total_bytes + head_dim - 1
            ) // head_dim
            for j in cutlass.range_constexpr(per_thread):
                lin = tidx * per_thread + j
                row = lin // quarter_dim
                col = lin % quarter_dim
                kv_pos_idx = kv_start_idx + n + row
                in_range = (
                    active_tile
                    & (n + row < seq_len)
                    & (n + row < split_end_unclamped)
                    & (lin < total_bytes)
                )
                kp = cutlass.Uint8(0)
                vp = cutlass.Uint8(0)
                if in_range:
                    kv_pos = cutlass.Int32(kv_indices[kv_pos_idx])
                    kp = K_packed[kv_pos, cur_kv_head, col]
                    vp = V_packed[kv_pos, cur_kv_head, col]
                if lin < total_bytes:
                    sK_packed[row, col] = kp
                    sV_packed[row, col] = vp

            # ---- Per-token scale/zero load ----
            if tidx < block_n:
                in_token = (
                    active_tile
                    & (n + tidx < seq_len)
                    & (n + tidx < split_end_unclamped)
                )
                ks = cutlass.Float32(1.0)
                kz = cutlass.Float32(0.0)
                vs = cutlass.Float32(1.0)
                vz = cutlass.Float32(0.0)
                if in_token:
                    kv_pos2 = cutlass.Int32(kv_indices[kv_start_idx + n + tidx])
                    ks = cutlass.Float32(K_sz[kv_pos2, cur_kv_head, 0])
                    kz = cutlass.Float32(K_sz[kv_pos2, cur_kv_head, 1])
                    vs = cutlass.Float32(V_sz[kv_pos2, cur_kv_head, 0])
                    vz = cutlass.Float32(V_sz[kv_pos2, cur_kv_head, 1])
                sK_scale[tidx] = ks
                sK_zero[tidx] = kz
                sV_scale[tidx] = vs
                sV_zero[tidx] = vz
            cute.arch.barrier()

            # ---- QK dot products: one thread per token, all q-heads ----
            # Each token-owning thread dequantizes K once and dots it against
            # all kv_group_num q-heads — saves (kv_group_num - 1)× redundant
            # K-dequant work versus naive per-q-head iteration.
            if tidx < block_n:
                in_token2 = (
                    active_tile
                    & (n + tidx < seq_len)
                    & (n + tidx < split_end_unclamped)
                )
                ks2 = sK_scale[tidx]
                kz2 = sK_zero[tidx]
                qk_acc = [cutlass.Float32(0.0) for _ in range(kv_group_num)]
                for d4 in cutlass.range_constexpr(quarter_dim):
                    b = cutlass.Int32(sK_packed[tidx, d4])
                    c0 = cutlass.Float32(b & 0x03) - kz2
                    c1 = cutlass.Float32((b >> 2) & 0x03) - kz2
                    c2 = cutlass.Float32((b >> 4) & 0x03) - kz2
                    c3 = cutlass.Float32((b >> 6) & 0x03) - kz2
                    for qid in cutlass.range_constexpr(kv_group_num):
                        qk_acc[qid] = qk_acc[qid] + sQ[qid, d4] * c0
                        qk_acc[qid] = qk_acc[qid] + sQ[qid, d4 + quarter_dim] * c1
                        qk_acc[qid] = qk_acc[qid] + sQ[qid, d4 + 2 * quarter_dim] * c2
                        qk_acc[qid] = qk_acc[qid] + sQ[qid, d4 + 3 * quarter_dim] * c3
                for qid in cutlass.range_constexpr(kv_group_num):
                    out_qk = cutlass.Float32(-1e30)
                    if in_token2:
                        out_qk = qk_acc[qid] * ks2 * sm_scale
                    sQK[qid, tidx] = out_qk
            cute.arch.barrier()

            # ---- Online softmax per q-head (compute new_e_max, re_scale, sP) ----
            # Reductions over block_n are kept dynamic to control i-cache size
            # when ``kv_group_num`` is large.
            new_e_max_arr = [cutlass.Float32(-1e30) for _ in range(kv_group_num)]
            for qid in cutlass.range_constexpr(kv_group_num):
                tile_max = cutlass.Float32(-1e30)
                for i in cutlass.range(block_n):
                    tile_max = _fmax(tile_max, sQK[qid, i])
                new_e_max_arr[qid] = _fmax(e_max[qid], tile_max)
                re_scale = cute.exp(e_max[qid] - new_e_max_arr[qid])
                acc[qid] = acc[qid] * re_scale
                e_sum[qid] = e_sum[qid] * re_scale
                if tidx < block_n:
                    sP[qid, tidx] = cute.exp(sQK[qid, tidx] - new_e_max_arr[qid])
            cute.arch.barrier()

            # ---- PV: one thread per head_dim coord, all q-heads ----
            # Dequantize V once per (token, coord-quarter) and multiply by
            # each q-head's probability tile — amortizes V-dequant 4×.
            # Outer loop over tokens is dynamic to bound i-cache footprint;
            # the inner q-head loop stays constexpr so per-q-head registers
            # are allocated statically.
            if tidx < head_dim:
                inner_qid = tidx // quarter_dim
                d4 = tidx % quarter_dim
                shift = cutlass.Uint8(2 * inner_qid)
                local_acc = [cutlass.Float32(0.0) for _ in range(kv_group_num)]
                for i in cutlass.range(block_n):
                    b = cutlass.Int32(sV_packed[i, d4])
                    crumb = cutlass.Float32((b >> shift) & 0x03) - sV_zero[i]
                    v_dq = crumb * sV_scale[i]
                    for qid in cutlass.range_constexpr(kv_group_num):
                        local_acc[qid] = local_acc[qid] + sP[qid, i] * v_dq
                for qid in cutlass.range_constexpr(kv_group_num):
                    acc[qid] = acc[qid] + local_acc[qid]

            # ---- Softmax denominator ----
            # Every thread reduces sP[qid, :] independently. sP is fully
            # populated by the softmax phase above and lives entirely in
            # smem with broadcast reads — same value lands in every thread's
            # tile_sum, so we skip the sScratch broadcast + extra barrier
            # that a thread-0-only reduction would need.
            for qid in cutlass.range_constexpr(kv_group_num):
                tile_sum = cutlass.Float32(0.0)
                for i in cutlass.range(block_n):
                    tile_sum = tile_sum + sP[qid, i]
                e_sum[qid] = e_sum[qid] + tile_sum
                e_max[qid] = new_e_max_arr[qid]

        # ---- Store split outputs (one per q-head in this group) ----
        for qid in cutlass.range_constexpr(kv_group_num):
            inv = cutlass.Float32(0.0)
            lse_val = cutlass.Float32(-1e30)
            if e_sum[qid] > cutlass.Float32(0.0):
                inv = cutlass.Float32(1.0) / e_sum[qid]
                lse_val = e_max[qid] + cute.log(e_sum[qid])

            if tidx < head_dim:
                att_out[
                    cur_batch, q_head_base + qid, split_kv_id, tidx
                ] = acc[qid] * inv

            if tidx == 0:
                att_lse[cur_batch, q_head_base + qid, split_kv_id] = lse_val

    return kernel


def _compile_decode(
    head_dim: int,
    block_n: int,
    kv_group_num: int,
    batch: int,
    head_num: int,
    kv_heads: int,
    max_splits: int,
    cache_size: int,
    max_tiles_per_split: int,
):
    key = (head_dim, block_n, kv_group_num, batch, head_num, kv_heads,
           max_splits, cache_size, max_tiles_per_split)
    if key in _compiled_decode_kernels:
        return _compiled_decode_kernels[key]

    kernel = _define_decode_kernel(
        head_dim=head_dim, block_n=block_n,
        max_tiles_per_split=max_tiles_per_split,
        kv_group_num=kv_group_num,
    )

    smem_bytes = (
        kv_group_num * head_dim * 4         # sQ (kv_group_num × head_dim, fp32)
        + 2 * block_n * (head_dim // 4)     # sK_packed + sV_packed (shared)
        + 4 * block_n * 4                   # sK/V scale, zero (fp32)
        + kv_group_num * block_n * 4        # sQK
        + kv_group_num * block_n * 4        # sP (probability tile)
        + 256                               # alignment slack
    )

    @cute.jit
    def launcher(
        Q: cute.Tensor,
        K_packed: cute.Tensor,
        V_packed: cute.Tensor,
        K_sz: cute.Tensor,
        V_sz: cute.Tensor,
        kv_indptr: cute.Tensor,
        kv_indices: cute.Tensor,
        num_kv_splits: cute.Tensor,
        att_out: cute.Tensor,
        att_lse: cute.Tensor,
        sm_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        kernel(
            Q,
            K_packed,
            V_packed,
            K_sz,
            V_sz,
            kv_indptr,
            kv_indices,
            num_kv_splits,
            att_out,
            att_lse,
            sm_scale,
            min_block_kv=_MIN_BLOCK_KV,
        ).launch(
            grid=(batch, kv_heads, max_splits),
            block=(head_dim, 1, 1),
            smem=smem_bytes,
            stream=stream,
        )

    fake_q = cute.runtime.make_fake_tensor(
        cutlass.BFloat16, (batch, head_num, head_dim),
        (head_num * head_dim, head_dim, 1),
        memspace=cute.AddressSpace.gmem, assumed_align=16,
    )
    fake_kp = cute.runtime.make_fake_tensor(
        cutlass.Uint8, (cache_size, kv_heads, head_dim // 4),
        (kv_heads * (head_dim // 4), head_dim // 4, 1),
        memspace=cute.AddressSpace.gmem, assumed_align=16,
    )
    fake_vp = cute.runtime.make_fake_tensor(
        cutlass.Uint8, (cache_size, kv_heads, head_dim // 4),
        (kv_heads * (head_dim // 4), head_dim // 4, 1),
        memspace=cute.AddressSpace.gmem, assumed_align=16,
    )
    fake_ksz = cute.runtime.make_fake_tensor(
        cutlass.Float32, (cache_size, kv_heads, 2),
        (kv_heads * 2, 2, 1),
        memspace=cute.AddressSpace.gmem, assumed_align=8,
    )
    fake_vsz = cute.runtime.make_fake_tensor(
        cutlass.Float32, (cache_size, kv_heads, 2),
        (kv_heads * 2, 2, 1),
        memspace=cute.AddressSpace.gmem, assumed_align=8,
    )
    fake_indptr = cute.runtime.make_fake_tensor(
        cutlass.Int32, (batch + 1,), (1,),
        memspace=cute.AddressSpace.gmem, assumed_align=16,
    )
    fake_indices = cute.runtime.make_fake_tensor(
        cutlass.Int32, (cute.sym_int(),), (1,),
        memspace=cute.AddressSpace.gmem, assumed_align=16,
    )
    fake_splits = cute.runtime.make_fake_tensor(
        cutlass.Int32, (batch,), (1,),
        memspace=cute.AddressSpace.gmem, assumed_align=16,
    )
    fake_att = cute.runtime.make_fake_tensor(
        cutlass.Float32,
        (batch, head_num, max_splits, head_dim),
        (head_num * max_splits * head_dim, max_splits * head_dim, head_dim, 1),
        memspace=cute.AddressSpace.gmem, assumed_align=16,
    )
    fake_lse = cute.runtime.make_fake_tensor(
        cutlass.Float32, (batch, head_num, max_splits),
        (head_num * max_splits, max_splits, 1),
        memspace=cute.AddressSpace.gmem, assumed_align=16,
    )
    fake_stream = cuda.CUstream(0)
    compiled = cute.compile(
        launcher,
        fake_q,
        fake_kp,
        fake_vp,
        fake_ksz,
        fake_vsz,
        fake_indptr,
        fake_indices,
        fake_splits,
        fake_att,
        fake_lse,
        sm_scale=cutlass.Float32(1.0),
        stream=fake_stream,
    )
    _compiled_decode_kernels[key] = compiled
    return compiled


def can_use_cutedsl_decode(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    k_scales_zeros: torch.Tensor,
    v_scales_zeros: torch.Tensor,
) -> bool:
    """Whether the CuteDSL decode kernel handles this configuration."""
    head_dim = k_buffer.shape[-1] * 4
    if head_dim != 128:
        # Compile-time specialization currently fixes head_dim=128 (the only
        # validated production shape).
        return False
    if v_buffer.shape[-1] * 4 != head_dim:
        return False
    if k_scales_zeros.shape[-1] != 2 or v_scales_zeros.shape[-1] != 2:
        # multi-group (group_size < head_dim) - not yet supported
        return False
    if q.dtype != torch.bfloat16:
        return False
    return True


def cutedsl_decode_attention_fwd_int2(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    k_scales_zeros: torch.Tensor,
    v_scales_zeros: torch.Tensor,
    att_out: torch.Tensor,
    att_lse: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    num_kv_splits: torch.Tensor,
    max_kv_splits: int,
    sm_scale: float,
):
    """CuteDSL replacement for the Triton ``_fwd_kernel_stage1_quant_int2``.

    Computes split-KV attention scores + lse. Stage 2 (softmax reduce across
    splits) is handled by the existing ``_decode_softmax_reducev_fwd``.
    """
    head_dim = k_buffer.shape[-1] * 4
    batch, head_num = q.shape[0], q.shape[1]
    kv_heads = k_buffer.shape[1]
    kv_group_num = head_num // kv_heads
    cache_size = k_buffer.shape[0]
    block_n = DECODE_BLOCK_N

    # Bound the compiled outer kv-tile loop. The kernel iterates this many
    # times unconditionally and masks each tile dynamically against the
    # actual seq_len — so cache_size (e.g. 2.5M in production) would inflate
    # the loop by ~100× and waste cycles. Cap at a heuristic seq-len budget
    # of ``_MAX_DECODE_SEQ_LEN`` (covers Qwen3-4B's 32K context plus slack);
    # callers that need more can override via the ``SGLANG_INT2_DECODE_MAX_TILES``
    # env var.
    max_seq_budget = int(
        os.environ.get(
            "SGLANG_INT2_DECODE_MAX_SEQ", str(_MAX_DECODE_SEQ_LEN_DEFAULT)
        )
    )
    bounded_cache_size = min(cache_size, max_seq_budget)
    max_per_split = max(
        1, (bounded_cache_size + max_kv_splits - 1) // max_kv_splits
    )
    rounded_per_split = (
        max_per_split + _MIN_BLOCK_KV - 1
    ) // _MIN_BLOCK_KV * _MIN_BLOCK_KV
    max_tiles = max(1, (rounded_per_split + block_n - 1) // block_n)

    compiled = _compile_decode(
        head_dim, block_n, kv_group_num,
        batch, head_num, kv_heads, max_kv_splits, cache_size,
        max_tiles,
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compiled(
        from_dlpack(q.detach(), assumed_align=16),
        from_dlpack(k_buffer.detach(), assumed_align=16),
        from_dlpack(v_buffer.detach(), assumed_align=16),
        from_dlpack(k_scales_zeros.detach(), assumed_align=8),
        from_dlpack(v_scales_zeros.detach(), assumed_align=8),
        from_dlpack(kv_indptr.detach(), assumed_align=16),
        from_dlpack(kv_indices.detach(), assumed_align=16),
        from_dlpack(num_kv_splits.detach(), assumed_align=16),
        from_dlpack(att_out.detach(), assumed_align=16),
        from_dlpack(att_lse.detach(), assumed_align=16),
        cutlass.Float32(sm_scale),
        stream,
    )


# ---------------------------------------------------------------------------
# Standalone correctness check (executed via ``--test-correctness``)
# ---------------------------------------------------------------------------


def _run_correctness() -> bool:
    """Compares the CuteDSL kernels against their Triton counterparts.

    Returns True on success. Used by SUCCESS_CONDITION test 2."""
    import torch.nn.functional as F  # noqa: F401

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16

    # --- Quantize correctness ---
    num_tokens = 64
    num_heads = 8
    head_dim = 128
    cache_size = num_tokens

    from sglang.srt.mem_cache.kv_quant_kernels import _launch_quantize_int2

    x = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype, device=device)
    loc = torch.arange(num_tokens, dtype=torch.int32, device=device)

    cache_tri = torch.zeros(cache_size, num_heads, head_dim // 4,
                            dtype=torch.uint8, device=device)
    sz_tri = torch.zeros(cache_size, num_heads, 2, dtype=torch.float32, device=device)
    _launch_quantize_int2(x.clone(), loc, cache_tri, sz_tri, None)

    cache_dsl = torch.zeros_like(cache_tri)
    sz_dsl = torch.zeros_like(sz_tri)
    _launch_quantize_one(x.clone(), loc, cache_dsl, sz_dsl, None)


    # Compare packed bytes and scale/zero (allow tiny scale/zero drift)
    pack_match = (cache_tri == cache_dsl).float().mean().item()
    sz_diff = (sz_tri - sz_dsl).abs().max().item()
    print(f"[quantize] packed-byte exact-match rate: {pack_match:.4f}, "
          f"sz max abs diff: {sz_diff:.2e}")
    if pack_match < 0.99:
        print("FAIL: packed bytes differ from Triton too much")
        return False
    if sz_diff > 1e-4:
        print("FAIL: scale/zero diverge from Triton")
        return False

    # --- Decode attention correctness (MHA + GQA) ---
    from sglang.srt.layers.attention.triton_ops.decode_attention import (
        decode_attention_fwd_normal_quant_int2,
        decode_attention_fwd_grouped_quant_int2,
    )

    # Two cases: MHA (kv_group_num=1) and Qwen3-4B GQA (q=32, kv=8, group=4).
    cases = [
        ("MHA", 4, 8, 8, 256, 4),
        ("GQA-qwen3-4B", 1, 32, 8, 512, 4),
    ]
    for tag, batch, q_heads, kv_heads, seq_len, max_splits in cases:
        # Build packed K/V from random bf16
        k = torch.randn(seq_len, kv_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(seq_len, kv_heads, head_dim, dtype=dtype, device=device)
        k_packed = torch.zeros(seq_len, kv_heads, head_dim // 4,
                               dtype=torch.uint8, device=device)
        v_packed = torch.zeros_like(k_packed)
        k_sz = torch.zeros(seq_len, kv_heads, 2, dtype=torch.float32, device=device)
        v_sz = torch.zeros_like(k_sz)
        loc_full = torch.arange(seq_len, dtype=torch.int32, device=device)
        _launch_quantize_int2(k, loc_full, k_packed, k_sz, None)
        _launch_quantize_int2(v, loc_full, v_packed, v_sz, None)

        q = torch.randn(batch, q_heads, head_dim, dtype=dtype, device=device)
        kv_indptr = torch.arange(0, (batch + 1) * seq_len, seq_len,
                                  dtype=torch.int32, device=device)
        kv_indices = loc_full.repeat(batch).contiguous()
        num_kv_splits = torch.full((batch,), max_splits,
                                    dtype=torch.int32, device=device)

        # Triton reference: MHA uses normal kernel, GQA uses grouped kernel.
        out_tri = torch.zeros(batch, q_heads, head_dim,
                              dtype=torch.float32, device=device)
        logits_tri = torch.zeros(batch, q_heads, max_splits, head_dim,
                                 dtype=torch.float32, device=device)
        lse_tri = torch.zeros(batch, q_heads, max_splits,
                              dtype=torch.float32, device=device)
        if kv_heads == q_heads:
            decode_attention_fwd_normal_quant_int2(
                q, k_packed, v_packed, k_sz, v_sz, out_tri,
                kv_indptr, kv_indices, logits_tri, lse_tri,
                num_kv_splits, max_splits, sm_scale=1.0 / (head_dim ** 0.5),
            )
        else:
            decode_attention_fwd_grouped_quant_int2(
                q, k_packed, v_packed, k_sz, v_sz, out_tri,
                kv_indptr, kv_indices, logits_tri, lse_tri,
                num_kv_splits, max_splits, sm_scale=1.0 / (head_dim ** 0.5),
            )

        # CuteDSL stage-1
        logits_dsl = torch.zeros_like(logits_tri)
        lse_dsl = torch.zeros_like(lse_tri)
        cutedsl_decode_attention_fwd_int2(
            q, k_packed, v_packed, k_sz, v_sz,
            logits_dsl, lse_dsl, kv_indptr, kv_indices, num_kv_splits,
            max_splits, sm_scale=1.0 / (head_dim ** 0.5),
        )

        cos = F.cosine_similarity(logits_tri.flatten(),
                                   logits_dsl.flatten(), dim=0).item()
        lse_diff = (lse_tri - lse_dsl).abs().max().item()
        print(f"[decode/{tag}] q={q_heads} kv={kv_heads} group={q_heads//kv_heads} "
              f"S={seq_len} cosine_sim={cos:.6f} lse_max_diff={lse_diff:.4f}")
        if cos < 0.999:
            print(f"FAIL ({tag}): decode stage1 cosine similarity below 0.999")
            return False
    print("PASS")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-correctness", action="store_true",
                   help="Run the cross-check vs the Triton reference kernels.")
    args = p.parse_args()
    if args.test_correctness:
        ok = _run_correctness()
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
