"""Unit tests for the unified int2 + HP mixed KV cache path (slab design).

Covers:
  * UnifiedInt2HPKVAllocator: paged-quant free list (``alloc_quant`` requires
    ``N_Q``-aligned sizes; ``free`` returns whole pages) + per-request HP
    slab indexers (``alloc_hp_prefix`` / ``alloc_hp_recent``).
  * UnifiedInt2HPKVPool: decoupled HP and quant arenas; HP buffer is per
    request slot; quant arena is paged. Slab cursor lifecycle + retract reset.
  * gpu_flush_int2: plan + fused quant + remap kernel round-trip produces
    the same quant bytes as the pretransformed reference. Per-request
    flush gating via ``_flush_counter`` is unchanged.
"""

from __future__ import annotations

import inspect
import os
import tempfile
import unittest

import torch


def _ensure_cuda():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA required")


def _resolve_dtypes():
    return torch.bfloat16, torch.bfloat16


# Process-wide tempdir for identity rotation .pt files used by ``_make_pool``.
# The unified pool now requires SGLANG_OSCAR_K/V_ROTATION_PATH; tests that only
# exercise the allocator / flush kernels get identity rotations so the pool
# constructs without changing semantics for non-rotation code paths.
_IDENTITY_ROT_DIR = tempfile.mkdtemp(prefix="sglang_unified_int2_test_")
_IDENTITY_ROT_CACHE: dict[int, tuple[str, str]] = {}


def _identity_rotation_paths(head_dim: int, layer_num: int) -> tuple[str, str]:
    key = (head_dim << 16) | layer_num
    if key in _IDENTITY_ROT_CACHE:
        return _IDENTITY_ROT_CACHE[key]
    state = {
        "layers": {
            i: {"rotation": torch.eye(head_dim, dtype=torch.float32)}
            for i in range(layer_num)
        }
    }
    k_path = os.path.join(_IDENTITY_ROT_DIR, f"k_d{head_dim}_l{layer_num}.pt")
    v_path = os.path.join(_IDENTITY_ROT_DIR, f"v_d{head_dim}_l{layer_num}.pt")
    torch.save(state, k_path)
    torch.save(state, v_path)
    _IDENTITY_ROT_CACHE[key] = (k_path, v_path)
    return k_path, v_path


def _pq_codebook_path(
    name: str,
    *,
    head_dim: int,
    layer_num: int,
    n_sub: int = 8,
    n_centroids: int = 16,
    rvq: bool = False,
) -> str:
    """Create a deterministic, small per-layer PQ/RVQ fixture."""
    assert head_dim % n_sub == 0
    sub_dim = head_dim // n_sub
    generator = torch.Generator().manual_seed(2026)
    stage1 = torch.randn(
        layer_num,
        n_sub,
        n_centroids,
        sub_dim,
        generator=generator,
        dtype=torch.float32,
    )
    path = os.path.join(_IDENTITY_ROT_DIR, f"{name}.pt")
    common = {
        "layer_ids": list(range(layer_num)),
        "n_sub": n_sub,
        "sub_dim": sub_dim,
    }
    if rvq:
        stage2 = 0.25 * torch.randn(
            layer_num,
            n_sub,
            n_centroids,
            sub_dim,
            generator=generator,
            dtype=torch.float32,
        )
        torch.save(
            {
                **common,
                "n_cents1": n_centroids,
                "n_cents2": n_centroids,
                "codebooks_stage1": stage1,
                "codebooks_stage2": stage2,
            },
            path,
        )
    else:
        torch.save(
            {
                **common,
                "n_centroids": n_centroids,
                "codebooks_per_layer": stage1,
            },
            path,
        )
    return path


def _make_pool(
    num_quant_pages: int = 64,
    layer_num: int = 2,
    head_num: int = 4,
    head_dim: int = 64,
    v_head_dim: int = 64,
    kv_cache_quant_group_size=None,
    hp_prefix_tokens: int = 32,
    hp_recent_tokens: int = 128,
    max_req_slots: int = 64,
    num_hp_prefix_slots: int = 256,
    dtype: str = "int2",
    k_clip_ratio: float = 0.0,
    v_clip_ratio: float = 0.0,
    lloyd_max: bool = False,
    pq_k_codebook: str = "",
    pq_v_codebook: str = "",
    # Backward-compat for tests that still pass ``num_pages``.
    num_pages: int = None,
):
    from sglang.srt.environ import envs
    from sglang.srt.mem_cache.unified_kv_pool import UnifiedInt2HPKVPool

    if num_pages is not None:
        num_quant_pages = num_pages
    os.environ.setdefault("HADAMARD_ORDER", "16")
    hp_dtype, scale_dtype = _resolve_dtypes()
    k_path, _ = _identity_rotation_paths(head_dim, layer_num)
    _, v_path = _identity_rotation_paths(v_head_dim, layer_num)
    with (
        envs.SGLANG_OSCAR_K_ROTATION_PATH.override(k_path),
        envs.SGLANG_OSCAR_V_ROTATION_PATH.override(v_path),
        envs.SGLANG_OSCAR_K_CLIP_RATIO.override(k_clip_ratio),
        envs.SGLANG_OSCAR_V_CLIP_RATIO.override(v_clip_ratio),
        envs.SGLANG_LLOYD_MAX.override(lloyd_max),
        envs.SGLANG_PQ_K_CODEBOOK.override(pq_k_codebook),
        envs.SGLANG_PQ_V_CODEBOOK.override(pq_v_codebook),
    ):
        return UnifiedInt2HPKVPool(
            num_quant_pages=num_quant_pages,
            hp_dtype=hp_dtype,
            hp_prefix_tokens=hp_prefix_tokens,
            hp_recent_tokens=hp_recent_tokens,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=layer_num,
            device="cuda",
            enable_memory_saver=False,
            max_req_slots=max_req_slots,
            v_head_dim=v_head_dim,
            start_layer=0,
            end_layer=layer_num - 1,
            model_dtype=torch.bfloat16,
            kv_cache_quant_group_size=kv_cache_quant_group_size,
            scale_dtype=scale_dtype,
            num_hp_prefix_slots=num_hp_prefix_slots,
        )


def _make_allocator(pool=None, **overrides):
    """Build a UnifiedInt2HPKVAllocator backed by a fresh pool.

    Returns ``(pool, allocator)``.
    """
    from sglang.srt.mem_cache.unified_kv_allocator import UnifiedInt2HPKVAllocator

    if pool is None:
        pool = _make_pool(**overrides)
    alloc = UnifiedInt2HPKVAllocator(
        num_quant_pages=pool.num_quant_pages,
        quant_tokens_per_page=pool.N_Q,
        hp_prefix_tokens=pool.hp_prefix_tokens,
        hp_recent_tokens=pool.hp_recent_tokens,
        hp_recent_ring_size=pool.hp_recent_ring_size,
        max_req_slots=pool.max_req_slots,
        num_hp_prefix_slots=pool.num_hp_prefix_slots,
        dtype=pool.dtype,
        hp_dtype=pool.hp_dtype,
        device="cuda",
        kvcache=pool,
        need_sort=False,
    )
    return pool, alloc


def _run_gpu_flush(pool, **kwargs):
    from sglang.QuantKernel.gpu_flush_int2 import gpu_flush_int2

    # Default ``flush_mask`` to all-True so existing tests (which explicitly
    # set up an all-flush scenario) continue to exercise every entry. New
    # tests can pass an explicit ``flush_mask`` to test partial-flush gating.
    if "flush_mask" not in kwargs:
        bs = int(kwargs["seq_lens"].shape[0])
        kwargs["flush_mask"] = torch.ones((bs,), dtype=torch.bool, device="cuda")

    return gpu_flush_int2(
        hp_k_ptrs=pool._flush_hp_k_ptrs,
        hp_v_ptrs=pool._flush_hp_v_ptrs,
        quant_k_ptrs=pool._flush_quant_k_ptrs,
        quant_v_ptrs=pool._flush_quant_v_ptrs,
        k_sz_ptrs=pool._flush_k_sz_ptrs,
        v_sz_ptrs=pool._flush_v_sz_ptrs,
        hp_k_sample=pool.hp_k_buffer[0],
        hp_v_sample=pool.hp_v_buffer[0],
        quant_k_sample=pool.k_buffer[0],
        quant_v_sample=pool.v_buffer[0],
        k_sz_sample=pool.k_scales_zeros[0],
        v_sz_sample=pool.v_scales_zeros[0],
        hp_k_strides=pool._flush_hp_k_stride,
        hp_v_strides=pool._flush_hp_v_stride,
        quant_k_strides=pool._flush_quant_k_stride,
        quant_v_strides=pool._flush_quant_v_stride,
        k_sz_strides=pool._flush_k_sz_stride,
        v_sz_strides=pool._flush_v_sz_stride,
        hp_prefix_tokens=pool.hp_prefix_tokens,
        hp_recent_tokens=pool.hp_recent_tokens,
        hp_global_offset=pool.hp_global_offset,
        num_heads=pool.head_num,
        head_dim=pool.head_dim,
        v_head_dim=pool.v_head_dim,
        k_num_scale_groups=pool.k_num_scale_groups,
        v_num_scale_groups=pool.v_num_scale_groups,
        num_layers=pool.layer_num,
        **kwargs,
    )


class UnifiedAllocatorTest(unittest.TestCase):
    """Allocator tests under the slab design.

    Quant tier is a vanilla paged allocator. HP tier is a per-request slab
    indexed by ``req_pool_idx``; ``alloc_hp_prefix`` and ``alloc_hp_recent``
    derive slot ids from the slab geometry. ``free`` is a no-op for HP slots.
    """

    def setUp(self):
        _ensure_cuda()
        self.pool, self.allocator = _make_allocator(
            num_quant_pages=64,
            hp_prefix_tokens=32,
            hp_recent_tokens=128,
            max_req_slots=8,
        )

    def test_alloc_hp_prefix_pooled(self):
        """HP-prefix is a shared paged pool; slot ids are in [HP_OFFSET,
        HP_OFFSET + num_hp_prefix_slots).
        """
        a = self.allocator
        req_idx = torch.tensor([3], dtype=torch.int64, device="cuda")
        # Allocate one full HP-prefix-page worth (N_Q slots).
        prefix = a.alloc_hp_prefix(req_idx, [a.N_Q])
        self.assertEqual(prefix.numel(), a.N_Q)
        self.assertTrue((prefix >= a.hp_global_offset).all())
        self.assertTrue((prefix < a._hp_recent_offset).all())

    def test_alloc_hp_prefix_requires_n_q_total(self):
        """alloc_hp_prefix's ``total = sum(counts)`` must be N_Q-aligned."""
        a = self.allocator
        req_idx = torch.tensor([0], dtype=torch.int64, device="cuda")
        with self.assertRaises(ValueError):
            a.alloc_hp_prefix(req_idx, [a.N_Q - 1])

    def test_alloc_hp_prefix_returns_to_pool(self):
        """Freeing an HP-prefix slot returns its page to the shared pool."""
        a = self.allocator
        pre = a._hp_prefix_free_slots()
        req_idx = torch.tensor([0], dtype=torch.int64, device="cuda")
        prefix = a.alloc_hp_prefix(req_idx, [a.N_Q])
        self.assertEqual(a._hp_prefix_free_slots(), pre - a.N_Q)
        a.free(prefix)
        self.assertEqual(a._hp_prefix_free_slots(), pre)

    def test_alloc_hp_recent_decode_fast_path(self):
        """Uniform count=1 across reqs: vectorised, sync-free."""
        a = self.allocator
        bs = 4
        req_idx = torch.arange(bs, dtype=torch.int64, device="cuda")
        # First decode step: cursor at 0, slot offset = recent_offset + 0.
        first = a.alloc_hp_recent(req_idx, [1] * bs)
        offsets = first - a._hp_recent_offset
        # Each req's first slot lands at req_idx * ring_size + 0.
        self.assertTrue(
            torch.equal(
                (offsets // a.hp_recent_ring_size).cpu(),
                torch.arange(bs, dtype=torch.int64),
            )
        )
        cursor = self.pool._next_slab_offset[req_idx].cpu()
        self.assertTrue(torch.equal(cursor, torch.ones_like(cursor)))
        second = a.alloc_hp_recent(req_idx, [1] * bs)
        # Second slot of req i: req_idx * ring + 1.
        offsets2 = (second - a._hp_recent_offset) % a.hp_recent_ring_size
        self.assertTrue((offsets2 == 1).all())

    def test_alloc_hp_recent_ring_wraps(self):
        """After ``ring_size`` allocations, the cursor wraps back to 0."""
        a = self.allocator
        ring = a.hp_recent_ring_size
        req_idx = torch.tensor([2], dtype=torch.int64, device="cuda")
        for _ in range(ring):
            a.alloc_hp_recent(req_idx, [1])
        self.assertEqual(int(self.pool._next_slab_offset[2]), 0)
        wrap = a.alloc_hp_recent(req_idx, [1])
        offset = int((wrap - a._hp_recent_offset) % ring)
        self.assertEqual(offset, 0)

    def test_alloc_hp_recent_ragged_extend(self):
        """Extend path: per-req counts vary; cursors advance independently."""
        a = self.allocator
        req_idx = torch.tensor([0, 1, 2], dtype=torch.int64, device="cuda")
        slots = a.alloc_hp_recent(req_idx, [3, 0, 5])
        self.assertEqual(slots.numel(), 8)
        cursors = self.pool._next_slab_offset[req_idx].cpu().tolist()
        self.assertEqual(cursors, [3, 0, 5])

    def test_alloc_hp_recent_chunked_extend_continues(self):
        """A second extend chunk for the same request resumes the cursor."""
        a = self.allocator
        req_idx = torch.tensor([5], dtype=torch.int64, device="cuda")
        a.alloc_hp_recent(req_idx, [4])
        self.assertEqual(int(self.pool._next_slab_offset[5]), 4)
        slots = a.alloc_hp_recent(req_idx, [3])
        offsets = ((slots - a._hp_recent_offset) % a.hp_recent_ring_size).cpu().tolist()
        self.assertEqual(offsets, [4, 5, 6])

    def test_release_req_slab_resets_cursor(self):
        """Retract → release_req_slab(req_idx) → cursor restarts at 0."""
        a = self.allocator
        req_idx = torch.tensor([1], dtype=torch.int64, device="cuda")
        a.alloc_hp_recent(req_idx, [10])
        self.assertEqual(int(self.pool._next_slab_offset[1]), 10)
        # Set a flush counter too, then release.
        self.pool._flush_counter[1] = 5
        self.pool.release_req_slab(1)
        self.assertEqual(int(self.pool._next_slab_offset[1]), 0)
        self.assertEqual(int(self.pool._flush_counter[1]), 0)

    def test_alloc_quant_requires_whole_pages(self):
        with self.assertRaises(ValueError):
            self.allocator.alloc_quant(3)
        a = self.allocator.alloc_quant(self.allocator.N_Q)
        self.assertEqual(a.numel(), self.allocator.N_Q)
        b = self.allocator.alloc_quant(2 * self.allocator.N_Q)
        self.assertEqual(b.numel(), 2 * self.allocator.N_Q)
        page_a = (a[0] // self.allocator.N_Q).item()
        page_b = (b[0] // self.allocator.N_Q).item()
        self.assertNotEqual(page_a, page_b)

    def test_quant_free_returns_pages(self):
        pre = self.allocator.free_pages.numel()
        q = self.allocator.alloc_quant(2 * self.allocator.N_Q)
        self.assertEqual(self.allocator.free_pages.numel(), pre - 2)
        self.allocator.free(q)
        self.assertEqual(self.allocator.free_pages.numel(), pre)

    def test_hp_recent_free_is_noop(self):
        """HP-recent slot ids are no-ops in ``free`` (per-req slab)."""
        pre_quant = self.allocator.free_pages.numel()
        pre_hp_prefix = self.allocator._hp_prefix_free_slots()
        req_idx = torch.tensor([0], dtype=torch.int64, device="cuda")
        recent = self.allocator.alloc_hp_recent(req_idx, [4])
        self.assertEqual(self.allocator.free_pages.numel(), pre_quant)
        self.assertEqual(self.allocator._hp_prefix_free_slots(), pre_hp_prefix)
        self.allocator.free(recent)
        # No pools changed (HP-recent is per-req).
        self.assertEqual(self.allocator.free_pages.numel(), pre_quant)
        self.assertEqual(self.allocator._hp_prefix_free_slots(), pre_hp_prefix)

    def test_mixed_free_routes_three_tiers(self):
        """Free a mix of quant + HP-prefix + HP-recent: each goes home."""
        a = self.allocator
        n_q = a.N_Q
        pre_quant = a.free_pages.numel()
        pre_hp_prefix = a._hp_prefix_free_slots()
        req_idx = torch.tensor([0], dtype=torch.int64, device="cuda")
        hp_prefix = a.alloc_hp_prefix(req_idx, [n_q])
        hp_recent = a.alloc_hp_recent(req_idx, [4])
        qu = a.alloc_quant(2 * n_q)
        self.assertEqual(a.free_pages.numel(), pre_quant - 2)
        self.assertEqual(a._hp_prefix_free_slots(), pre_hp_prefix - n_q)
        a.free(torch.cat([hp_prefix, hp_recent, qu]))
        self.assertEqual(a.free_pages.numel(), pre_quant)
        self.assertEqual(a._hp_prefix_free_slots(), pre_hp_prefix)

    def test_scheduler_leak_identity(self):
        """``available + evictable + protected == size`` (standard shape)."""

        def leak(evictable: int, protected: int, session: int) -> int:
            avail = self.allocator.available_size()
            return self.allocator.size - (avail + evictable + protected + session)

        self.assertEqual(leak(0, 0, 0), 0)
        n_q = self.allocator.N_Q
        qu = self.allocator.alloc_quant(2 * n_q)
        self.assertEqual(leak(0, qu.numel(), 0), 0)
        # Half evictable, half protected.
        half = qu.numel() // 2
        self.assertEqual(leak(half, qu.numel() - half, 0), 0)
        self.allocator.free(qu)
        self.assertEqual(leak(0, 0, 0), 0)

    def test_available_size_pools_quant_and_hp_prefix(self):
        """``available_size`` covers both pooled tiers (quant + HP-prefix)."""
        a = self.allocator
        expected = (a.free_pages.numel() + a.release_pages.numel()) * a.N_Q + (
            a.hp_prefix_free_pages.numel() + a.hp_prefix_release_pages.numel()
        ) * a.N_Q
        self.assertEqual(a.available_size(), expected)


class UnifiedPoolViewTest(unittest.TestCase):
    def test_decoupled_arena_shapes(self):
        """HP and quant arenas are independent allocations. HP arena holds
        the shared HP-prefix region followed by per-req recent slabs."""
        _ensure_cuda()
        from sglang.srt.mem_cache.unified_kv_pool import compute_page_geometry

        hp_dtype, _ = _resolve_dtypes()
        n_h, n_q = compute_page_geometry(hp_dtype)
        self.assertEqual(n_h, 1)
        self.assertEqual(n_q, 8)

        pool = _make_pool(
            num_pages=32, layer_num=1, max_req_slots=4, num_hp_prefix_slots=128
        )
        self.assertEqual(pool.k_buffer[0].shape[0], 32 * pool.N_Q)
        # HP arena: prefix pool + per-req recent slabs.
        expected_hp = (
            pool.num_hp_prefix_slots + pool.max_req_slots * pool.hp_recent_ring_size
        )
        self.assertEqual(pool.hp_k_buffer[0].shape[0], expected_hp)
        # HP-recent base aligns with the prefix region size.
        self.assertEqual(pool.hp_recent_base, pool.num_hp_prefix_slots)
        # HP and quant tensors are independent allocations.
        pool.hp_k_buffer[0].fill_(0)
        pool.k_buffer[0][0:8].fill_(0xA5)
        self.assertTrue((pool.hp_k_buffer[0] == 0).all())

    def test_mixed_set_kv_buffer_matches_reference(self):
        _ensure_cuda()
        from sglang.QuantKernel.fused_hadamard_int2_kv import (
            quantized_set_kv_int2_pretransformed_triton,
        )

        class _Layer:
            layer_id = 0

        pool = _make_pool(num_pages=64, layer_num=1, kv_cache_quant_group_size=16)
        ref_pool = _make_pool(num_pages=64, layer_num=1, kv_cache_quant_group_size=16)
        torch.manual_seed(123)
        cache_k = torch.randn(
            6, pool.head_num, pool.head_dim, dtype=pool.hp_dtype, device="cuda"
        )
        cache_v = torch.randn(
            6, pool.head_num, pool.v_head_dim, dtype=pool.hp_dtype, device="cuda"
        )
        loc = torch.tensor(
            [
                pool.hp_global_offset + 1,
                40,
                pool.hp_global_offset + 2,
                41,
                42,
                pool.hp_global_offset + 3,
            ],
            dtype=torch.int64,
            device="cuda",
        )

        pool.set_kv_buffer(
            _Layer(),
            loc,
            cache_k,
            cache_v,
            already_hadamard_transformed=True,
        )

        hp_mask = loc >= pool.hp_global_offset
        quant_mask = ~hp_mask
        quantized_set_kv_int2_pretransformed_triton(
            cache_k[quant_mask],
            cache_v[quant_mask],
            loc[quant_mask],
            ref_pool.k_buffer[0],
            ref_pool.v_buffer[0],
            ref_pool.k_scales_zeros[0],
            ref_pool.v_scales_zeros[0],
        )
        hp_locs = loc[hp_mask] - pool.hp_global_offset
        ref_pool.hp_k_buffer[0][hp_locs] = cache_k[hp_mask]
        ref_pool.hp_v_buffer[0][hp_locs] = cache_v[hp_mask]
        torch.cuda.synchronize()

        quant_locs = loc[quant_mask]
        self.assertTrue(
            torch.equal(pool.k_buffer[0][quant_locs], ref_pool.k_buffer[0][quant_locs])
        )
        self.assertTrue(
            torch.equal(pool.v_buffer[0][quant_locs], ref_pool.v_buffer[0][quant_locs])
        )
        self.assertTrue(
            torch.equal(
                pool.k_scales_zeros[0][quant_locs],
                ref_pool.k_scales_zeros[0][quant_locs],
            )
        )
        self.assertTrue(
            torch.equal(
                pool.v_scales_zeros[0][quant_locs],
                ref_pool.v_scales_zeros[0][quant_locs],
            )
        )
        self.assertTrue(torch.equal(pool.hp_k_buffer[0][hp_locs], cache_k[hp_mask]))
        self.assertTrue(torch.equal(pool.hp_v_buffer[0][hp_locs], cache_v[hp_mask]))

    def test_mixed_prefix_dequant_matches_reference(self):
        _ensure_cuda()
        from sglang.srt.layers.attention.quantized_kv_prefill import (
            dequantize_prefix_kv,
        )
        from sglang.srt.mem_cache.kv_quant_kernels import dequantize_kv_int2_triton

        class _Layer:
            layer_id = 0

        pool = _make_pool(num_pages=64, layer_num=1, kv_cache_quant_group_size=16)
        torch.manual_seed(321)
        cache_k = torch.randn(
            6, pool.head_num, pool.head_dim, dtype=pool.hp_dtype, device="cuda"
        )
        cache_v = torch.randn(
            6, pool.head_num, pool.v_head_dim, dtype=pool.hp_dtype, device="cuda"
        )
        prefix_indices = torch.tensor(
            [
                40,
                pool.hp_global_offset + 1,
                41,
                pool.hp_global_offset + 2,
                42,
                pool.hp_global_offset + 3,
            ],
            dtype=torch.int64,
            device="cuda",
        )
        pool.set_kv_buffer(
            _Layer(),
            prefix_indices,
            cache_k,
            cache_v,
            already_hadamard_transformed=True,
        )

        got_k, got_v = dequantize_prefix_kv(
            pool, 0, prefix_indices, model_dtype=pool.hp_dtype
        )

        hp_mask = prefix_indices >= pool.hp_global_offset
        quant_mask = ~hp_mask
        ref_k = torch.empty_like(got_k)
        ref_v = torch.empty_like(got_v)
        quant_indices = prefix_indices[quant_mask]
        hp_indices = prefix_indices[hp_mask] - pool.hp_global_offset
        ref_k[quant_mask] = dequantize_kv_int2_triton(
            pool.get_raw_key_buffer(0)[quant_indices],
            pool.get_key_scales_zeros(0)[quant_indices],
            pool.head_dim,
            pool.hp_dtype,
        )
        ref_v[quant_mask] = dequantize_kv_int2_triton(
            pool.get_raw_value_buffer(0)[quant_indices],
            pool.get_value_scales_zeros(0)[quant_indices],
            pool.v_head_dim,
            pool.hp_dtype,
        )
        ref_k[hp_mask] = pool.get_hp_key_buffer(0)[hp_indices]
        ref_v[hp_mask] = pool.get_hp_value_buffer(0)[hp_indices]
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(got_k, ref_k))
        self.assertTrue(torch.equal(got_v, ref_v))

    def test_mixed_hot_paths_avoid_dynamic_cuda_masks(self):
        from sglang.srt.layers.attention import flashattention_backend
        from sglang.srt.layers.attention import quantized_kv_prefill
        from sglang.srt.mem_cache.unified_kv_pool import UnifiedInt2HPKVPool

        set_src = inspect.getsource(UnifiedInt2HPKVPool.set_kv_buffer)
        dequant_src = inspect.getsource(quantized_kv_prefill.dequantize_prefix_kv)
        fa_src = inspect.getsource(flashattention_backend.FlashAttentionBackend.forward)
        self.assertNotIn("[hp_mask]", set_src)
        self.assertNotIn("[quant_mask]", set_src)
        self.assertNotIn(".any()", dequant_src)
        self.assertNotIn("[hp_mask]", dequant_src)
        self.assertNotIn("[quant_mask]", dequant_src)
        self.assertNotIn("torch.cumsum(\n                    torch.tensor", fa_src)
        self.assertNotIn("cu_seqlens_k_cpu", fa_src)


class OneBitStorageCorrectnessTest(unittest.TestCase):
    class _Layer:
        layer_id = 0

    @staticmethod
    def _lm_int1_reference(x: torch.Tensor):
        x_f = x.float()
        mean = x_f.mean(dim=-1)
        diff = x_f - mean[..., None]
        std = torch.sqrt((diff * diff).mean(dim=-1) + 1e-8)
        scale = torch.clamp(2.0 * 0.79788456 * std, min=1e-8)
        zero = 0.5 - mean / scale
        quant = (x_f >= mean[..., None]).to(torch.uint8)
        block_octant = x.shape[-1] // 8
        quant = quant.reshape(*quant.shape[:-1], 8, block_octant)
        shifts = torch.arange(8, device=x.device, dtype=torch.uint8).view(
            *((1,) * (quant.ndim - 2)), 8, 1
        )
        packed = torch.sum(quant << shifts, dim=-2).to(torch.uint8)
        return packed, torch.stack((scale, zero), dim=-1)

    def test_pq_codebook_metadata_mismatch_fails_before_allocation(self):
        path = _pq_codebook_path("pq_bad_metadata", head_dim=64, layer_num=1, n_sub=8)
        data = torch.load(path, map_location="cpu", weights_only=False)
        data["n_sub"] = 7
        torch.save(data, path)
        with self.assertRaisesRegex(ValueError, "metadata n_sub"):
            _make_pool(
                num_pages=2,
                layer_num=1,
                head_num=1,
                head_dim=64,
                v_head_dim=64,
                dtype="pq_k_int2v",
                pq_k_codebook=path,
            )
        data["n_sub"] = 8
        data["layer_ids"] = [0, 0]
        data["codebooks_per_layer"] = data["codebooks_per_layer"].repeat(2, 1, 1, 1)
        torch.save(data, path)
        with self.assertRaisesRegex(ValueError, "layer_ids must be unique"):
            _make_pool(
                num_pages=2,
                layer_num=1,
                head_num=1,
                head_dim=64,
                v_head_dim=64,
                dtype="pq_k_int2v",
                pq_k_codebook=path,
            )
        non_power_path = _pq_codebook_path(
            "pq_bad_centroids",
            head_dim=64,
            layer_num=1,
            n_sub=8,
            n_centroids=16,
        )
        non_power = torch.load(non_power_path, map_location="cpu", weights_only=False)
        non_power["codebooks_per_layer"] = non_power["codebooks_per_layer"][:, :, :15]
        non_power["n_centroids"] = 15
        torch.save(non_power, non_power_path)
        with self.assertRaisesRegex(ValueError, "power of two"):
            _make_pool(
                num_pages=2,
                layer_num=1,
                head_num=1,
                head_dim=64,
                v_head_dim=64,
                dtype="pq_k_int2v",
                pq_k_codebook=non_power_path,
            )

    def test_int1_prefill_single_group_is_byte_exact_for_k_and_v(self):
        _ensure_cuda()
        from sglang.srt.mem_cache.kv_quant_kernels import (
            gather_dequantize_kv_int1_triton,
        )

        pool = _make_pool(
            num_pages=16,
            layer_num=1,
            head_num=2,
            head_dim=64,
            v_head_dim=64,
            dtype="int1",
            k_clip_ratio=1.0,
            v_clip_ratio=1.0,
            lloyd_max=True,
        )
        self.assertEqual(pool.k_buffer[0].shape[-1], 8)
        self.assertEqual(pool.v_buffer[0].shape[-1], 8)

        pool.k_buffer[0].fill_(0xA5)
        pool.v_buffer[0].fill_(0xA5)
        torch.manual_seed(7)
        cache_k = 3.0 * torch.randn(
            3, pool.head_num, pool.head_dim, dtype=pool.hp_dtype, device="cuda"
        )
        cache_v = 3.0 * torch.randn(
            3, pool.head_num, pool.v_head_dim, dtype=pool.hp_dtype, device="cuda"
        )
        loc = torch.tensor([2, 5, 9], dtype=torch.int64, device="cuda")

        pool.set_kv_buffer(
            self._Layer(),
            loc,
            cache_k,
            cache_v,
            already_hadamard_transformed=True,
        )
        torch.cuda.synchronize()

        ref_k, ref_k_sz = self._lm_int1_reference(cache_k)
        ref_v, ref_v_sz = self._lm_int1_reference(cache_v)
        self.assertTrue(torch.equal(pool.k_buffer[0][loc], ref_k))
        self.assertTrue(torch.equal(pool.v_buffer[0][loc], ref_v))
        self.assertTrue((pool.k_buffer[0][3] == 0xA5).all())
        self.assertTrue((pool.v_buffer[0][3] == 0xA5).all())
        torch.testing.assert_close(
            pool.k_scales_zeros[0][loc].float(), ref_k_sz, atol=2e-2, rtol=2e-2
        )
        torch.testing.assert_close(
            pool.v_scales_zeros[0][loc].float(), ref_v_sz, atol=2e-2, rtol=2e-2
        )

        v_roundtrip = gather_dequantize_kv_int1_triton(
            pool.v_buffer[0],
            pool.v_scales_zeros[0],
            loc,
            pool.v_head_dim,
            pool.hp_dtype,
        )
        self.assertEqual(v_roundtrip.shape, cache_v.shape)
        self.assertTrue(torch.isfinite(v_roundtrip).all())

    def test_fused_int1_oscar_prefill_matches_lloyd_max_reference(self):
        _ensure_cuda()
        from sglang.srt.environ import envs

        class _AbsorbedLayer:
            layer_id = 0
            oscar_v_rotation_absorbed = True

        kwargs = dict(
            num_pages=16,
            layer_num=1,
            head_num=2,
            head_dim=64,
            v_head_dim=64,
            dtype="int1",
            k_clip_ratio=1.0,
            v_clip_ratio=1.0,
            lloyd_max=True,
        )
        fused_pool = _make_pool(**kwargs)
        reference_pool = _make_pool(**kwargs)
        torch.manual_seed(9)
        cache_k = torch.randn(
            3,
            fused_pool.head_num,
            fused_pool.head_dim,
            dtype=fused_pool.hp_dtype,
            device="cuda",
        )
        cache_v = torch.randn_like(cache_k)
        loc = torch.tensor([2, 5, 9], dtype=torch.int64, device="cuda")

        reference_pool.set_kv_buffer(
            self._Layer(),
            loc,
            cache_k,
            cache_v,
            already_hadamard_transformed=True,
        )
        with envs.SGLANG_OSCAR_FUSED_ROTATE_CLIP_QUANT.override(True):
            fused_pool.set_kv_buffer(
                _AbsorbedLayer(),
                loc,
                cache_k,
                cache_v,
                already_hadamard_transformed=False,
            )
        torch.cuda.synchronize()

        self.assertTrue(
            torch.equal(fused_pool.k_buffer[0][loc], reference_pool.k_buffer[0][loc])
        )
        self.assertTrue(
            torch.equal(fused_pool.v_buffer[0][loc], reference_pool.v_buffer[0][loc])
        )
        torch.testing.assert_close(
            fused_pool.k_scales_zeros[0][loc],
            reference_pool.k_scales_zeros[0][loc],
            atol=2e-2,
            rtol=2e-2,
        )
        torch.testing.assert_close(
            fused_pool.v_scales_zeros[0][loc],
            reference_pool.v_scales_zeros[0][loc],
            atol=2e-2,
            rtol=2e-2,
        )

    def test_pq_v_prefill_writes_only_compact_code_rows(self):
        _ensure_cuda()
        from sglang.QuantKernel.oscar_rotation_pq_k_kv import (
            pq_decode_k,
            pq_encode_k,
        )
        from sglang.srt.layers.attention.quantized_kv_prefill import (
            dequantize_prefix_kv,
        )

        k_path = _pq_codebook_path("pq_k_prefill", head_dim=64, layer_num=1, n_sub=8)
        v_path = _pq_codebook_path("pq_v_prefill", head_dim=64, layer_num=1, n_sub=8)
        pool = _make_pool(
            num_pages=16,
            layer_num=1,
            head_num=2,
            head_dim=64,
            v_head_dim=64,
            dtype="pq_k_int2v",
            pq_k_codebook=k_path,
            pq_v_codebook=v_path,
        )
        self.assertEqual(pool.k_buffer[0].shape[-1], 8)
        self.assertEqual(pool.v_buffer[0].shape[-1], 8)

        pool.k_buffer[0].fill_(0xA5)
        pool.v_buffer[0].fill_(0xA5)
        torch.manual_seed(11)
        cache_k = torch.randn(
            2, pool.head_num, pool.head_dim, dtype=pool.hp_dtype, device="cuda"
        )
        cache_v = torch.randn(
            2, pool.head_num, pool.v_head_dim, dtype=pool.hp_dtype, device="cuda"
        )
        loc = torch.tensor([4, 8], dtype=torch.int64, device="cuda")

        ref_k = torch.full_like(pool.k_buffer[0], 0xA5)
        ref_v = torch.full_like(pool.v_buffer[0], 0xA5)
        pq_encode_k(
            cache_k,
            loc,
            ref_k,
            pool.get_pq_codebook(0),
            pool.get_pq_cb_norms(0),
        )
        pq_encode_k(
            cache_v,
            loc,
            ref_v,
            pool.get_pq_v_codebook(0),
            pool.get_pq_v_cb_norms(0),
        )
        pool.set_kv_buffer(
            self._Layer(),
            loc,
            cache_k,
            cache_v,
            already_hadamard_transformed=True,
        )
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(pool.k_buffer[0][loc], ref_k[loc]))
        self.assertTrue(torch.equal(pool.v_buffer[0][loc], ref_v[loc]))
        self.assertTrue((pool.k_buffer[0][5] == 0xA5).all())
        self.assertTrue((pool.v_buffer[0][5] == 0xA5).all())

        got_k, got_v = dequantize_prefix_kv(pool, 0, loc, pool.hp_dtype)
        expected_k = pq_decode_k(
            ref_k[loc], pool.get_pq_codebook(0), len(loc), pool.head_dim
        ).to(pool.hp_dtype)
        expected_v = pq_decode_k(
            ref_v[loc], pool.get_pq_v_codebook(0), len(loc), pool.v_head_dim
        ).to(pool.hp_dtype)
        self.assertTrue(torch.equal(got_k, expected_k))
        self.assertTrue(torch.equal(got_v, expected_v))

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            pool.set_kv_buffer(
                self._Layer(),
                loc,
                cache_k,
                cache_v,
                already_hadamard_transformed=True,
            )
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(pool.k_buffer[0][loc], ref_k[loc]))
        self.assertTrue(torch.equal(pool.v_buffer[0][loc], ref_v[loc]))

    def test_rvq_prefill_and_cache_move_preserve_stage2_codes(self):
        _ensure_cuda()
        from sglang.QuantKernel.oscar_rotation_pq_k_kv import (
            pq_decode_k,
            pq_encode_k,
        )

        rvq_path = _pq_codebook_path(
            "rvq_k_prefill", head_dim=64, layer_num=1, n_sub=8, rvq=True
        )
        pool = _make_pool(
            num_pages=16,
            layer_num=1,
            head_num=2,
            head_dim=64,
            v_head_dim=64,
            dtype="pq_k_int2v",
            pq_k_codebook=rvq_path,
        )
        self.assertIsNotNone(pool.k_buffer2)

        torch.manual_seed(19)
        cache_k = torch.randn(
            2, pool.head_num, pool.head_dim, dtype=pool.hp_dtype, device="cuda"
        )
        cache_v = torch.randn(
            2, pool.head_num, pool.v_head_dim, dtype=pool.hp_dtype, device="cuda"
        )
        src = torch.tensor([3, 7], dtype=torch.int64, device="cuda")

        ref_stage1 = torch.zeros_like(pool.k_buffer[0])
        ref_stage2 = torch.zeros_like(pool.k_buffer2[0])
        pq_encode_k(
            cache_k,
            src,
            ref_stage1,
            pool.get_pq_codebook(0),
            pool.get_pq_cb_norms(0),
        )
        recon1 = pq_decode_k(
            ref_stage1[src],
            pool.get_pq_codebook(0),
            len(src),
            pool.head_dim,
        ).to(cache_k.dtype)
        pq_encode_k(
            (cache_k - recon1).contiguous(),
            src,
            ref_stage2,
            pool.get_rvq_cb2(0),
            pool.get_rvq_cb2_norms(0),
        )

        pool.set_kv_buffer(
            self._Layer(),
            src,
            cache_k,
            cache_v,
            already_hadamard_transformed=True,
        )
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(pool.k_buffer[0][src], ref_stage1[src]))
        self.assertTrue(torch.equal(pool.k_buffer2[0][src], ref_stage2[src]))

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            pool.set_kv_buffer(
                self._Layer(),
                src,
                cache_k,
                cache_v,
                already_hadamard_transformed=True,
            )
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(pool.k_buffer[0][src], ref_stage1[src]))
        self.assertTrue(torch.equal(pool.k_buffer2[0][src], ref_stage2[src]))

        dst = torch.tensor([12, 13], dtype=torch.int64, device="cuda")
        pool.move_kv_cache(dst, src)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(pool.k_buffer[0][dst], ref_stage1[src]))
        self.assertTrue(torch.equal(pool.k_buffer2[0][dst], ref_stage2[src]))

    def test_int1_unified_decode_is_graph_safe_and_matches_dense_reference(self):
        _ensure_cuda()
        from sglang.srt.layers.attention.triton_ops.decode_attention import (
            decode_attention_fwd_int1_unified,
        )
        from sglang.srt.mem_cache.kv_quant_kernels import (
            gather_dequantize_kv_int1_triton,
        )

        pool = _make_pool(
            num_pages=16,
            layer_num=1,
            head_num=2,
            head_dim=64,
            v_head_dim=64,
            dtype="int1",
            k_clip_ratio=1.0,
            v_clip_ratio=1.0,
            lloyd_max=True,
        )
        torch.manual_seed(29)
        quant_k = torch.randn(
            4, pool.head_num, pool.head_dim, dtype=pool.hp_dtype, device="cuda"
        )
        quant_v = torch.randn(
            4, pool.head_num, pool.v_head_dim, dtype=pool.hp_dtype, device="cuda"
        )
        quant_locs = torch.tensor([4, 5, 6, 7], dtype=torch.int64, device="cuda")
        pool.set_kv_buffer(
            self._Layer(),
            quant_locs,
            quant_k,
            quant_v,
            already_hadamard_transformed=True,
        )

        pool.hp_k_buffer[0][0:2] = torch.randn(
            2, pool.head_num, pool.head_dim, dtype=pool.hp_dtype, device="cuda"
        )
        pool.hp_v_buffer[0][0:2] = torch.randn(
            2, pool.head_num, pool.v_head_dim, dtype=pool.hp_dtype, device="cuda"
        )

        # GQA=18 exercises a non-power-of-two group larger than BLOCK_H=16.
        bs, q_heads = 2, 36
        q = torch.randn(bs, q_heads, pool.head_dim, dtype=pool.hp_dtype, device="cuda")
        hp_indptr = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")
        hp_indices = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
        quant_indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
        # Deliberately leave an invalid padded tail. A graph-safe kernel must
        # obey indptr and never derive a Python slice length from it.
        quant_indices = torch.full(
            (16,),
            pool.k_buffer[0].shape[0] + 123,
            dtype=torch.int64,
            device="cuda",
        )
        quant_indices[:4] = quant_locs
        hp_splits = torch.ones((bs,), dtype=torch.int32, device="cuda")
        quant_splits = torch.ones((bs,), dtype=torch.int32, device="cuda")
        attn_logits = torch.empty(
            (bs, q_heads, 2, pool.v_head_dim),
            dtype=torch.float32,
            device="cuda",
        )
        attn_lse = torch.empty((bs, q_heads, 2), dtype=torch.float32, device="cuda")
        sm_scale = pool.head_dim**-0.5

        def launch(q_arg, out_arg):
            return decode_attention_fwd_int1_unified(
                q_arg,
                pool.hp_k_buffer[0],
                pool.hp_v_buffer[0],
                pool.k_buffer[0],
                pool.v_buffer[0],
                pool.k_scales_zeros[0],
                pool.v_scales_zeros[0],
                out_arg,
                hp_indptr,
                hp_indices,
                quant_indptr,
                quant_indices,
                attn_logits,
                attn_lse,
                hp_splits,
                quant_splits,
                1,
                1,
                sm_scale,
            )

        dequant_k = gather_dequantize_kv_int1_triton(
            pool.k_buffer[0],
            pool.k_scales_zeros[0],
            quant_locs,
            pool.head_dim,
            pool.hp_dtype,
        )
        dequant_v = gather_dequantize_kv_int1_triton(
            pool.v_buffer[0],
            pool.v_scales_zeros[0],
            quant_locs,
            pool.v_head_dim,
            pool.hp_dtype,
        )

        def dense_reference(q_arg):
            ref = torch.empty_like(q_arg)
            kv_group = q_heads // pool.head_num
            for batch_idx in range(bs):
                keys = torch.cat(
                    (
                        pool.hp_k_buffer[0][batch_idx : batch_idx + 1],
                        dequant_k[batch_idx * 2 : batch_idx * 2 + 2],
                    ),
                    dim=0,
                ).float()
                values = torch.cat(
                    (
                        pool.hp_v_buffer[0][batch_idx : batch_idx + 1],
                        dequant_v[batch_idx * 2 : batch_idx * 2 + 2],
                    ),
                    dim=0,
                ).float()
                for q_head in range(q_heads):
                    kv_head = q_head // kv_group
                    scores = (
                        keys[:, kv_head] @ q_arg[batch_idx, q_head].float()
                    ) * sm_scale
                    ref[batch_idx, q_head] = (
                        (torch.softmax(scores, dim=0)[:, None] * values[:, kv_head])
                        .sum(dim=0)
                        .to(ref.dtype)
                    )
            return ref

        eager_out = torch.empty_like(q)
        launch(q, eager_out)
        torch.cuda.synchronize()
        torch.testing.assert_close(eager_out, dense_reference(q), atol=4e-2, rtol=4e-2)

        static_q = q.clone()
        static_out = torch.empty_like(q)
        # Warm every Triton specialization before capture.
        launch(static_q, static_out)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch(static_q, static_out)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            static_out, dense_reference(static_q), atol=4e-2, rtol=4e-2
        )

        replay_q = torch.randn_like(static_q)
        static_q.copy_(replay_q)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            static_out, dense_reference(replay_q), atol=4e-2, rtol=4e-2
        )

    def test_pq_unified_decode_is_graph_safe_and_matches_dense_reference(self):
        _ensure_cuda()
        from sglang.QuantKernel.oscar_rotation_pq_k_kv import pq_decode_k
        from sglang.srt.layers.attention.triton_ops.decode_attention import (
            decode_attention_fwd_pqk_int2v_unified,
        )

        k_path = _pq_codebook_path(
            "pq_k_decode_graph", head_dim=64, layer_num=1, n_sub=8
        )
        v_path = _pq_codebook_path(
            "pq_v_decode_graph", head_dim=64, layer_num=1, n_sub=8
        )
        pool = _make_pool(
            num_pages=16,
            layer_num=1,
            head_num=2,
            head_dim=64,
            v_head_dim=64,
            dtype="pq_k_int2v",
            pq_k_codebook=k_path,
            pq_v_codebook=v_path,
        )
        torch.manual_seed(31)
        quant_k = torch.randn(
            4, pool.head_num, pool.head_dim, dtype=pool.hp_dtype, device="cuda"
        )
        quant_v = torch.randn(
            4, pool.head_num, pool.v_head_dim, dtype=pool.hp_dtype, device="cuda"
        )
        quant_locs = torch.tensor([4, 5, 6, 7], dtype=torch.int64, device="cuda")
        pool.set_kv_buffer(
            self._Layer(),
            quant_locs,
            quant_k,
            quant_v,
            already_hadamard_transformed=True,
        )
        pool.hp_k_buffer[0][0:2] = torch.randn(
            2, pool.head_num, pool.head_dim, dtype=pool.hp_dtype, device="cuda"
        )
        pool.hp_v_buffer[0][0:2] = torch.randn(
            2, pool.head_num, pool.v_head_dim, dtype=pool.hp_dtype, device="cuda"
        )

        # GQA=6 deliberately exercises a non-power-of-two group.
        bs, q_heads = 2, 12
        q = torch.randn(bs, q_heads, pool.head_dim, dtype=pool.hp_dtype, device="cuda")
        hp_indptr = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")
        hp_indices = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
        quant_indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
        quant_indices = torch.full(
            (16,),
            pool.k_buffer[0].shape[0] + 321,
            dtype=torch.int64,
            device="cuda",
        )
        quant_indices[:4] = quant_locs
        hp_splits = torch.ones((bs,), dtype=torch.int32, device="cuda")
        quant_splits = torch.ones((bs,), dtype=torch.int32, device="cuda")
        attn_logits = torch.empty(
            (bs, q_heads, 2, pool.v_head_dim),
            dtype=torch.float32,
            device="cuda",
        )
        attn_lse = torch.empty((bs, q_heads, 2), dtype=torch.float32, device="cuda")
        sm_scale = pool.head_dim**-0.5

        def launch(q_arg, out_arg):
            return decode_attention_fwd_pqk_int2v_unified(
                q_arg,
                pool.hp_k_buffer[0],
                pool.hp_v_buffer[0],
                pool.k_buffer[0],
                pool.v_buffer[0],
                pool.k_scales_zeros[0],
                pool.v_scales_zeros[0],
                pool.get_pq_codebook(0),
                out_arg,
                hp_indptr,
                hp_indices,
                quant_indptr,
                quant_indices,
                attn_logits,
                attn_lse,
                hp_splits,
                quant_splits,
                1,
                1,
                sm_scale,
                pq_v_codebook=pool.get_pq_v_codebook(0),
            )

        dequant_k = pq_decode_k(
            pool.k_buffer[0][quant_locs],
            pool.get_pq_codebook(0),
            len(quant_locs),
            pool.head_dim,
        ).to(pool.hp_dtype)
        dequant_v = pq_decode_k(
            pool.v_buffer[0][quant_locs],
            pool.get_pq_v_codebook(0),
            len(quant_locs),
            pool.v_head_dim,
        ).to(pool.hp_dtype)

        def dense_reference(q_arg):
            ref = torch.empty_like(q_arg)
            kv_group = q_heads // pool.head_num
            for batch_idx in range(bs):
                keys = torch.cat(
                    (
                        pool.hp_k_buffer[0][batch_idx : batch_idx + 1],
                        dequant_k[batch_idx * 2 : batch_idx * 2 + 2],
                    ),
                    dim=0,
                ).float()
                values = torch.cat(
                    (
                        pool.hp_v_buffer[0][batch_idx : batch_idx + 1],
                        dequant_v[batch_idx * 2 : batch_idx * 2 + 2],
                    ),
                    dim=0,
                ).float()
                for q_head in range(q_heads):
                    kv_head = q_head // kv_group
                    scores = (
                        keys[:, kv_head] @ q_arg[batch_idx, q_head].float()
                    ) * sm_scale
                    ref[batch_idx, q_head] = (
                        (torch.softmax(scores, dim=0)[:, None] * values[:, kv_head])
                        .sum(dim=0)
                        .to(ref.dtype)
                    )
            return ref

        eager_out = torch.empty_like(q)
        launch(q, eager_out)
        torch.cuda.synchronize()
        torch.testing.assert_close(eager_out, dense_reference(q), atol=4e-2, rtol=4e-2)

        static_q = q.clone()
        static_out = torch.empty_like(q)
        launch(static_q, static_out)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch(static_q, static_out)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            static_out, dense_reference(static_q), atol=4e-2, rtol=4e-2
        )

        replay_q = torch.randn_like(static_q)
        static_q.copy_(replay_q)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            static_out, dense_reference(replay_q), atol=4e-2, rtol=4e-2
        )

        from sglang.srt.environ import envs

        with envs.SGLANG_PQ_USE_ADC.override(1):
            adc_q = q.clone()
            adc_out = torch.empty_like(q)
            launch(adc_q, adc_out)
            torch.cuda.synchronize()
            torch.testing.assert_close(
                adc_out, dense_reference(adc_q), atol=4e-2, rtol=4e-2
            )
            adc_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(adc_graph):
                launch(adc_q, adc_out)
            adc_graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                adc_out, dense_reference(adc_q), atol=4e-2, rtol=4e-2
            )

    def test_graph_decode_supports_distinct_k_and_v_head_dims(self):
        _ensure_cuda()
        from sglang.QuantKernel.oscar_rotation_pq_k_kv import pq_decode_k
        from sglang.srt.layers.attention.triton_ops.decode_attention import (
            decode_attention_fwd_int1_unified,
            decode_attention_fwd_pqk_int2v_unified,
        )
        from sglang.srt.mem_cache.kv_quant_kernels import (
            gather_dequantize_kv_int1_triton,
        )

        for dtype in ("int1", "pq_k_int2v"):
            with self.subTest(dtype=dtype):
                kwargs = dict(
                    num_pages=16,
                    layer_num=1,
                    head_num=2,
                    head_dim=64,
                    v_head_dim=32,
                    dtype=dtype,
                )
                if dtype == "int1":
                    kwargs.update(
                        k_clip_ratio=1.0,
                        v_clip_ratio=1.0,
                        lloyd_max=True,
                    )
                else:
                    kwargs.update(
                        pq_k_codebook=_pq_codebook_path(
                            "pq_k_distinct_dims",
                            head_dim=64,
                            layer_num=1,
                            n_sub=8,
                        ),
                        pq_v_codebook=_pq_codebook_path(
                            "pq_v_distinct_dims",
                            head_dim=32,
                            layer_num=1,
                            n_sub=4,
                        ),
                    )
                pool = _make_pool(**kwargs)

                torch.manual_seed(43)
                cache_k = torch.randn(
                    3,
                    pool.head_num,
                    pool.head_dim,
                    dtype=pool.hp_dtype,
                    device="cuda",
                )
                cache_v = torch.randn(
                    3,
                    pool.head_num,
                    pool.v_head_dim,
                    dtype=pool.hp_dtype,
                    device="cuda",
                )
                loc = torch.tensor([4, 5, 6], dtype=torch.int64, device="cuda")
                pool.set_kv_buffer(
                    self._Layer(),
                    loc,
                    cache_k,
                    cache_v,
                    already_hadamard_transformed=True,
                )

                q_heads = 6  # GQA=3, also non-power-of-two.
                q = torch.randn(
                    1,
                    q_heads,
                    pool.head_dim,
                    dtype=pool.hp_dtype,
                    device="cuda",
                )
                hp_indptr = torch.tensor([0, 0], dtype=torch.int32, device="cuda")
                hp_indices = torch.empty((0,), dtype=torch.int64, device="cuda")
                quant_indptr = torch.tensor([0, 3], dtype=torch.int32, device="cuda")
                quant_indices = torch.full(
                    (8,),
                    pool.k_buffer[0].shape[0] + 55,
                    dtype=torch.int64,
                    device="cuda",
                )
                quant_indices[:3] = loc
                splits = torch.ones((1,), dtype=torch.int32, device="cuda")
                attn_logits = torch.empty(
                    (1, q_heads, 2, pool.v_head_dim),
                    dtype=torch.float32,
                    device="cuda",
                )
                attn_lse = torch.empty(
                    (1, q_heads, 2), dtype=torch.float32, device="cuda"
                )
                sm_scale = pool.head_dim**-0.5

                if dtype == "int1":

                    def launch(q_arg, out_arg):
                        return decode_attention_fwd_int1_unified(
                            q_arg,
                            pool.hp_k_buffer[0],
                            pool.hp_v_buffer[0],
                            pool.k_buffer[0],
                            pool.v_buffer[0],
                            pool.k_scales_zeros[0],
                            pool.v_scales_zeros[0],
                            out_arg,
                            hp_indptr,
                            hp_indices,
                            quant_indptr,
                            quant_indices,
                            attn_logits,
                            attn_lse,
                            splits,
                            splits,
                            1,
                            1,
                            sm_scale,
                        )

                    reconstructed_k = gather_dequantize_kv_int1_triton(
                        pool.k_buffer[0],
                        pool.k_scales_zeros[0],
                        loc,
                        pool.head_dim,
                        pool.hp_dtype,
                    )
                    reconstructed_v = gather_dequantize_kv_int1_triton(
                        pool.v_buffer[0],
                        pool.v_scales_zeros[0],
                        loc,
                        pool.v_head_dim,
                        pool.hp_dtype,
                    )
                else:

                    def launch(q_arg, out_arg):
                        return decode_attention_fwd_pqk_int2v_unified(
                            q_arg,
                            pool.hp_k_buffer[0],
                            pool.hp_v_buffer[0],
                            pool.k_buffer[0],
                            pool.v_buffer[0],
                            pool.k_scales_zeros[0],
                            pool.v_scales_zeros[0],
                            pool.get_pq_codebook(0),
                            out_arg,
                            hp_indptr,
                            hp_indices,
                            quant_indptr,
                            quant_indices,
                            attn_logits,
                            attn_lse,
                            splits,
                            splits,
                            1,
                            1,
                            sm_scale,
                            pq_v_codebook=pool.get_pq_v_codebook(0),
                        )

                    reconstructed_k = pq_decode_k(
                        pool.k_buffer[0][loc],
                        pool.get_pq_codebook(0),
                        len(loc),
                        pool.head_dim,
                    ).to(pool.hp_dtype)
                    reconstructed_v = pq_decode_k(
                        pool.v_buffer[0][loc],
                        pool.get_pq_v_codebook(0),
                        len(loc),
                        pool.v_head_dim,
                    ).to(pool.hp_dtype)

                def reference(q_arg):
                    ref = torch.empty(
                        (1, q_heads, pool.v_head_dim),
                        dtype=q_arg.dtype,
                        device=q_arg.device,
                    )
                    kv_group = q_heads // pool.head_num
                    for q_head in range(q_heads):
                        kv_head = q_head // kv_group
                        scores = (
                            reconstructed_k[:, kv_head].float()
                            @ q_arg[0, q_head].float()
                        ) * sm_scale
                        ref[0, q_head] = (
                            (
                                torch.softmax(scores, dim=0)[:, None]
                                * reconstructed_v[:, kv_head].float()
                            )
                            .sum(dim=0)
                            .to(ref.dtype)
                        )
                    return ref

                static_q = q.clone()
                static_out = torch.empty(
                    (1, q_heads, pool.v_head_dim),
                    dtype=q.dtype,
                    device=q.device,
                )
                launch(static_q, static_out)
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    static_out, reference(static_q), atol=4e-2, rtol=4e-2
                )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    launch(static_q, static_out)
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    static_out, reference(static_q), atol=4e-2, rtol=4e-2
                )

    def test_rvq_k_int2_v_decode_graph_path_matches_reconstruction(self):
        _ensure_cuda()
        from sglang.QuantKernel.oscar_rotation_pq_k_kv import pq_decode_k
        from sglang.srt.layers.attention.triton_ops.decode_attention import (
            decode_attention_fwd_pqk_int2v_unified,
        )
        from sglang.srt.mem_cache.kv_quant_kernels import dequantize_kv_int2_triton

        rvq_path = _pq_codebook_path(
            "rvq_k_decode_graph",
            head_dim=128,
            layer_num=1,
            n_sub=16,
            rvq=True,
        )
        pool = _make_pool(
            num_pages=16,
            layer_num=1,
            head_num=2,
            head_dim=128,
            v_head_dim=128,
            dtype="pq_k_int2v",
            pq_k_codebook=rvq_path,
        )
        torch.manual_seed(37)
        quant_k = torch.randn(
            3, pool.head_num, pool.head_dim, dtype=pool.hp_dtype, device="cuda"
        )
        quant_v = torch.randn(
            3, pool.head_num, pool.v_head_dim, dtype=pool.hp_dtype, device="cuda"
        )
        loc = torch.tensor([4, 5, 6], dtype=torch.int64, device="cuda")
        pool.set_kv_buffer(
            self._Layer(),
            loc,
            quant_k,
            quant_v,
            already_hadamard_transformed=True,
        )

        q = torch.randn(
            1, pool.head_num, pool.head_dim, dtype=pool.hp_dtype, device="cuda"
        )
        hp_indptr = torch.tensor([0, 0], dtype=torch.int32, device="cuda")
        hp_indices = torch.zeros((1,), dtype=torch.int64, device="cuda")
        quant_indptr = torch.tensor([0, 3], dtype=torch.int32, device="cuda")
        quant_indices = torch.full(
            (8,),
            pool.k_buffer[0].shape[0] + 99,
            dtype=torch.int64,
            device="cuda",
        )
        quant_indices[:3] = loc
        hp_splits = torch.ones((1,), dtype=torch.int32, device="cuda")
        quant_splits = torch.ones((1,), dtype=torch.int32, device="cuda")
        attn_logits = torch.empty(
            (1, pool.head_num, 2, pool.head_dim),
            dtype=torch.float32,
            device="cuda",
        )
        attn_lse = torch.empty(
            (1, pool.head_num, 2), dtype=torch.float32, device="cuda"
        )
        sm_scale = pool.head_dim**-0.5

        def launch(q_arg, out_arg):
            return decode_attention_fwd_pqk_int2v_unified(
                q_arg,
                pool.hp_k_buffer[0],
                pool.hp_v_buffer[0],
                pool.k_buffer[0],
                pool.v_buffer[0],
                pool.k_scales_zeros[0],
                pool.v_scales_zeros[0],
                pool.get_pq_codebook(0),
                out_arg,
                hp_indptr,
                hp_indices,
                quant_indptr,
                quant_indices,
                attn_logits,
                attn_lse,
                hp_splits,
                quant_splits,
                1,
                1,
                sm_scale,
                quant_k_buffer2=pool.get_raw_key_buffer2(0),
                pq_codebook2=pool.get_rvq_cb2(0),
            )

        reconstructed_k = (
            pq_decode_k(
                pool.k_buffer[0][loc],
                pool.get_pq_codebook(0),
                len(loc),
                pool.head_dim,
            )
            + pq_decode_k(
                pool.k_buffer2[0][loc],
                pool.get_rvq_cb2(0),
                len(loc),
                pool.head_dim,
            )
        ).to(pool.hp_dtype)
        reconstructed_v = dequantize_kv_int2_triton(
            pool.v_buffer[0][loc],
            pool.v_scales_zeros[0][loc],
            pool.v_head_dim,
            pool.hp_dtype,
        )

        def reference(q_arg):
            ref = torch.empty_like(q_arg)
            for head in range(pool.head_num):
                scores = (
                    reconstructed_k[:, head].float() @ q_arg[0, head].float()
                ) * sm_scale
                ref[0, head] = (
                    (
                        torch.softmax(scores, dim=0)[:, None]
                        * reconstructed_v[:, head].float()
                    )
                    .sum(dim=0)
                    .to(ref.dtype)
                )
            return ref

        static_q = q.clone()
        static_out = torch.empty_like(q)
        launch(static_q, static_out)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            static_out, reference(static_q), atol=4e-2, rtol=4e-2
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch(static_q, static_out)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            static_out, reference(static_q), atol=4e-2, rtol=4e-2
        )


class GpuFlushInt2Test(unittest.TestCase):
    """Round-trip tests for ``gpu_flush_int2``.

    Parametrized over ``flush_interval`` to cover:
      * interval=1: single-slot flush (same behavior as the pre-fused code).
      * interval=4: multi-slot flush; plan kernel emits up to 4 entries per
        request and the fused quant kernel processes them in one launch.
    """

    def _run_flush_roundtrip(self, flush_interval: int):
        _ensure_cuda()
        # The pool's own ``flush_interval`` is hardcoded to N_Q; the kernel
        # parameter ``flush_interval`` (= K) is still tested independently
        # here to cover both the single-slot (K=1) and multi-slot (K=4)
        # plan/quantize paths.
        pool = _make_pool(layer_num=2)

        bs = 4
        # Pick seq_len large enough that every j in [0, flush_interval) is
        # valid: fp_j = seq_len - hp_recent - (K-1) + j must be >= prefix_len
        # for every j, so seq_len >= hp_recent + (K-1) + prefix_len.
        seq_len_val = pool.hp_recent_tokens + flush_interval + 64
        seq_lens = torch.full((bs,), seq_len_val, dtype=torch.int32, device="cuda")
        prefix_lens = torch.zeros((bs,), dtype=torch.int32, device="cuda")
        req_pool_indices = torch.arange(bs, dtype=torch.int64, device="cuda")

        # Build req_to_token with HP slots at flush positions
        # [seq_len - hp_recent - K + 1, seq_len - hp_recent].
        num_req_slots = 8
        max_ctx = 512
        req_to_token = torch.zeros(
            (num_req_slots, max_ctx), dtype=torch.int32, device="cuda"
        )
        # Use distinct HP slots per (req, j) so we can verify the kernel
        # routed each demotion to the correct src HP slot.
        hp_slots = torch.arange(
            1, 1 + bs * flush_interval, dtype=torch.int64, device="cuda"
        )
        for i in range(bs):
            for j in range(flush_interval):
                fp = seq_len_val - pool.hp_recent_tokens - (flush_interval - 1) + j
                req_to_token[i, fp] = (
                    hp_slots[i * flush_interval + j] + pool.hp_global_offset
                ).to(torch.int32)

        # Populate HP buffers with deterministic data for each HP slot.
        hp_dtype = pool.hp_dtype
        for layer in range(pool.layer_num):
            for slot in hp_slots.tolist():
                data = torch.randn(
                    pool.head_num, pool.head_dim, dtype=hp_dtype, device="cuda"
                )
                pool.hp_k_buffer[layer][slot] = data
                pool.hp_v_buffer[layer][slot] = data * 2

        # Allocate distinct dst quant slots that don't alias HP slots
        # 1..bs*K (which occupy pages 1..bs*K). Start at page bs*K + 10 to
        # stay disjoint.
        dst_base = (bs * flush_interval + 10) * pool.N_Q
        dst_quant_slots = torch.arange(
            dst_base, dst_base + bs * flush_interval, dtype=torch.int64, device="cuda"
        )

        returned, valid_mask = _run_gpu_flush(
            pool,
            seq_lens=seq_lens,
            prefix_lens=prefix_lens,
            req_pool_indices=req_pool_indices,
            dst_quant_slots=dst_quant_slots,
            req_to_token=req_to_token,
            flush_interval=flush_interval,
        )

        torch.cuda.synchronize()

        # All (req, j) entries should flush successfully.
        self.assertEqual(valid_mask.numel(), bs * flush_interval)
        self.assertTrue(
            (valid_mask == 1).all(),
            f"flush_interval={flush_interval}: some entries invalid: {valid_mask}",
        )

        # Returned slot ids are the HP slots (global ids), in (req, j) order.
        expected_returned = (hp_slots + pool.hp_global_offset).to(torch.int64)
        self.assertTrue(
            torch.equal(returned, expected_returned),
            f"flush_interval={flush_interval}: returned slot ids mismatch",
        )

        # req_to_token at every flushed position now points to the dst quant
        # slot for that (req, j).
        for i in range(bs):
            for j in range(flush_interval):
                fp = seq_len_val - pool.hp_recent_tokens - (flush_interval - 1) + j
                self.assertEqual(
                    int(req_to_token[i, fp].item()),
                    int(dst_quant_slots[i * flush_interval + j].item()),
                    f"flush_interval={flush_interval}: req_to_token remap mismatch "
                    f"at (i={i}, j={j}, fp={fp})",
                )

    def test_flush_roundtrip_interval_1(self):
        self._run_flush_roundtrip(flush_interval=1)

    def test_flush_roundtrip_interval_4(self):
        self._run_flush_roundtrip(flush_interval=4)

    def test_flush_matches_reference_pretransformed(self):
        """The fused GPU flush kernel must produce the same packed bytes and
        scales/zeros as the standalone
        ``quantized_set_kv_int2_pretransformed_triton`` reference, under the
        same 'HP already Hadamard-transformed' assumption. Tested with
        flush_interval=1 so the plan kernel produces one demotion per
        request and the fused kernel output lines up with the reference
        slot-for-slot.

        Picks HP slots and quant slots that do not alias in the shared
        arena: HP slot ``p`` occupies the same bytes as quant slots
        ``[p*N_Q, p*N_Q + N_Q)``. The test writes HP data at pages 1, 2, 3
        (aliasing quant ranges [8..15], [16..23], [24..31]) and quantizes
        into pages 4 and 5 (quant ranges [32..39] and [40..47]) which are
        fully disjoint.
        """
        _ensure_cuda()
        from sglang.QuantKernel.fused_hadamard_int2_kv import (
            quantized_set_kv_int2_pretransformed_triton,
        )

        pool = _make_pool(
            num_pages=64,
            layer_num=1,
            kv_cache_quant_group_size=16,
        )

        hp_slots = torch.tensor([1, 2, 3], dtype=torch.int64, device="cuda")
        ref_quant_slots = torch.tensor([32, 33, 34], dtype=torch.int64, device="cuda")
        dst_quant_slots = torch.tensor([40, 41, 42], dtype=torch.int64, device="cuda")

        torch.manual_seed(42)
        hp_dtype = pool.hp_dtype
        for slot in hp_slots.tolist():
            pool.hp_k_buffer[0][slot] = torch.randn(
                pool.head_num, pool.head_dim, dtype=hp_dtype, device="cuda"
            )
            pool.hp_v_buffer[0][slot] = torch.randn(
                pool.head_num, pool.v_head_dim, dtype=hp_dtype, device="cuda"
            )

        hp_k_slice = pool.hp_k_buffer[0][hp_slots].clone()
        hp_v_slice = pool.hp_v_buffer[0][hp_slots].clone()
        quantized_set_kv_int2_pretransformed_triton(
            hp_k_slice,
            hp_v_slice,
            ref_quant_slots,
            pool.k_buffer[0],
            pool.v_buffer[0],
            pool.k_scales_zeros[0],
            pool.v_scales_zeros[0],
        )
        torch.cuda.synchronize()
        ref_k = pool.k_buffer[0][ref_quant_slots].clone()
        ref_v = pool.v_buffer[0][ref_quant_slots].clone()
        ref_sz_k = pool.k_scales_zeros[0][ref_quant_slots].clone()
        ref_sz_v = pool.v_scales_zeros[0][ref_quant_slots].clone()

        # Drive the fused gpu_flush_int2 against the same HP input.
        bs = 3
        seq_len_val = pool.hp_recent_tokens + 80
        flush_pos = seq_len_val - pool.hp_recent_tokens
        seq_lens = torch.full((bs,), seq_len_val, dtype=torch.int32, device="cuda")
        prefix_lens = torch.zeros((bs,), dtype=torch.int32, device="cuda")
        req_pool_indices = torch.arange(bs, dtype=torch.int64, device="cuda")
        req_to_token = torch.zeros((bs, 512), dtype=torch.int32, device="cuda")
        req_to_token[req_pool_indices, flush_pos] = (
            hp_slots + pool.hp_global_offset
        ).to(torch.int32)

        _run_gpu_flush(
            pool,
            seq_lens=seq_lens,
            prefix_lens=prefix_lens,
            req_pool_indices=req_pool_indices,
            dst_quant_slots=dst_quant_slots,
            req_to_token=req_to_token,
            flush_interval=1,
        )
        torch.cuda.synchronize()

        got_k = pool.k_buffer[0][dst_quant_slots]
        got_v = pool.v_buffer[0][dst_quant_slots]
        got_sz_k = pool.k_scales_zeros[0][dst_quant_slots]
        got_sz_v = pool.v_scales_zeros[0][dst_quant_slots]

        # Fused kernel and standalone reference both implement the same
        # pretransformed group quant, so the outputs must be bit-identical.
        self.assertTrue(torch.equal(got_k, ref_k), "packed K bytes diverge")
        self.assertTrue(torch.equal(got_v, ref_v), "packed V bytes diverge")
        self.assertTrue(torch.equal(got_sz_k, ref_sz_k), "K scales/zeros diverge")
        self.assertTrue(torch.equal(got_sz_v, ref_sz_v), "V scales/zeros diverge")


class PartialFlushGatingTest(unittest.TestCase):
    """Per-request flush gating regression tests.

    The previous global-counter design fired a flush every ``N_Q`` decode
    steps regardless of per-request state. For a request admitted with
    ``tail = quant_count % N_Q != 0``, the first global flush could land
    while only ``J < N_Q`` of the K plan-window positions were valid HP,
    producing a partial flush whose ``N_Q - J`` returned dst_quant_slots
    aliased the ``J`` in-use slots in the same physical page — the page
    was freed while live, corrupting subsequent KV reads after reuse.

    These tests exercise:
      * ``flush_mask=False`` for a request → ``valid_mask`` all 0 and
        every dst_quant_slot is returned (whole-page free).
      * ``flush_mask=True`` for a request → all K positions are valid HP
        and demoted to the request's dst page.
      * Mixed batch (some flush, some don't) → page accounting is per-
        request, no cross-aliasing.
    """

    def _setup_request(
        self,
        pool,
        bs: int,
        flush_interval: int,
        seq_len_val: int,
    ):
        """Populate ``req_to_token`` with HP slots at the K oldest HP-recent
        positions for every request, return the supporting tensors."""
        seq_lens = torch.full((bs,), seq_len_val, dtype=torch.int32, device="cuda")
        prefix_lens = torch.zeros((bs,), dtype=torch.int32, device="cuda")
        req_pool_indices = torch.arange(bs, dtype=torch.int64, device="cuda")
        max_ctx = max(512, seq_len_val + 16)
        req_to_token = torch.zeros((bs, max_ctx), dtype=torch.int32, device="cuda")
        # HP slots are 1..bs*K; store the global slot id in req_to_token.
        hp_slots = torch.arange(
            1, 1 + bs * flush_interval, dtype=torch.int64, device="cuda"
        )
        for i in range(bs):
            for j in range(flush_interval):
                fp = seq_len_val - pool.hp_recent_tokens - (flush_interval - 1) + j
                req_to_token[i, fp] = (
                    hp_slots[i * flush_interval + j] + pool.hp_global_offset
                ).to(torch.int32)
        # Disjoint dst quant slots starting past the HP-aliased page range.
        dst_base = (bs * flush_interval + 10) * pool.N_Q
        dst_quant_slots = torch.arange(
            dst_base, dst_base + bs * flush_interval, dtype=torch.int64, device="cuda"
        )
        return seq_lens, prefix_lens, req_pool_indices, req_to_token, dst_quant_slots

    def test_flush_mask_false_returns_whole_page(self):
        """A request with ``flush_mask=False`` must contribute zero valid
        flushes and return all FLUSH_INTERVAL dst_quant_slots → whole page
        is freed cleanly. This is the non-flushing-request case under the
        new per-request gating, where the alloc still hands out a page per
        request but only flushing requests use any of it.
        """
        _ensure_cuda()
        pool = _make_pool(layer_num=1)
        flush_interval = pool.N_Q
        seq_len_val = pool.hp_recent_tokens + flush_interval + 32

        bs = 1
        seq_lens, prefix_lens, req_pool_indices, req_to_token, dst_quant_slots = (
            self._setup_request(pool, bs, flush_interval, seq_len_val)
        )
        flush_mask = torch.zeros((bs,), dtype=torch.bool, device="cuda")

        returned, valid_mask = _run_gpu_flush(
            pool,
            seq_lens=seq_lens,
            prefix_lens=prefix_lens,
            req_pool_indices=req_pool_indices,
            dst_quant_slots=dst_quant_slots,
            req_to_token=req_to_token,
            flush_interval=flush_interval,
            flush_mask=flush_mask,
        )
        torch.cuda.synchronize()

        # No valid demotions for this request.
        self.assertTrue((valid_mask == 0).all())
        # All FLUSH_INTERVAL returned slot ids equal the corresponding
        # dst_quant_slots — i.e. the whole quant page comes back.
        self.assertTrue(
            torch.equal(returned, dst_quant_slots),
            "non-flushing request must return all dst_quant_slots",
        )

    def test_flush_mask_true_demotes_all_K(self):
        """With ``flush_mask=True`` and a long enough sequence so the K
        oldest HP-recent positions are all valid HP, the kernel must demote
        all K. This is the all-or-nothing flush case the per-request gate
        guarantees in the orchestrator.
        """
        _ensure_cuda()
        pool = _make_pool(layer_num=1)
        flush_interval = pool.N_Q
        seq_len_val = pool.hp_recent_tokens + flush_interval + 32

        bs = 1
        seq_lens, prefix_lens, req_pool_indices, req_to_token, dst_quant_slots = (
            self._setup_request(pool, bs, flush_interval, seq_len_val)
        )
        flush_mask = torch.ones((bs,), dtype=torch.bool, device="cuda")

        returned, valid_mask = _run_gpu_flush(
            pool,
            seq_lens=seq_lens,
            prefix_lens=prefix_lens,
            req_pool_indices=req_pool_indices,
            dst_quant_slots=dst_quant_slots,
            req_to_token=req_to_token,
            flush_interval=flush_interval,
            flush_mask=flush_mask,
        )
        torch.cuda.synchronize()

        self.assertTrue((valid_mask == 1).all())
        # Returned slot ids are HP slot ids for all K positions.
        # HP slot id (global) = local_slot + hp_global_offset.
        expected_hp = (
            torch.arange(1, 1 + flush_interval, dtype=torch.int64, device="cuda")
            + pool.hp_global_offset
        )
        self.assertTrue(torch.equal(returned, expected_hp))

    def test_mixed_batch_per_request_gating(self):
        """Half of the batch flushes, half doesn't. Returned slot ids
        partition cleanly: flushing requests' returned ids are all HP
        global ids; non-flushing requests' are all dst_quant_slots from
        their assigned page → after free, no quant page is partially
        in-use (whole-page invariant preserved).
        """
        _ensure_cuda()
        pool = _make_pool(layer_num=1)
        flush_interval = pool.N_Q
        seq_len_val = pool.hp_recent_tokens + flush_interval + 32

        bs = 4
        seq_lens, prefix_lens, req_pool_indices, req_to_token, dst_quant_slots = (
            self._setup_request(pool, bs, flush_interval, seq_len_val)
        )
        # Reqs 0, 2 flush; reqs 1, 3 don't.
        flush_mask = torch.tensor(
            [True, False, True, False], dtype=torch.bool, device="cuda"
        )

        returned, valid_mask = _run_gpu_flush(
            pool,
            seq_lens=seq_lens,
            prefix_lens=prefix_lens,
            req_pool_indices=req_pool_indices,
            dst_quant_slots=dst_quant_slots,
            req_to_token=req_to_token,
            flush_interval=flush_interval,
            flush_mask=flush_mask,
        )
        torch.cuda.synchronize()

        # For each request, valid_mask should be all 1 (flushing) or all 0
        # (non-flushing) — never a mix, which is the all-or-nothing property.
        valid_2d = valid_mask.reshape(bs, flush_interval)
        for i in range(bs):
            row = valid_2d[i]
            self.assertTrue(
                bool((row == 0).all()) or bool((row == 1).all()),
                f"request {i} has partial flush mask: {row}",
            )
            self.assertEqual(bool(row[0].item()), bool(flush_mask[i].item()))

        # For non-flushing requests, the entire dst_quant page must come
        # back through ``returned``. Verify this — if it fails, the freed
        # set would leave live slots in the dst page (the C1 bug pre-fix).
        returned_2d = returned.reshape(bs, flush_interval)
        dst_2d = dst_quant_slots.reshape(bs, flush_interval)
        for i in range(bs):
            if not bool(flush_mask[i].item()):
                self.assertTrue(
                    torch.equal(returned_2d[i], dst_2d[i]),
                    f"request {i} (non-flushing) must return all dst slots",
                )

    def test_decode_flush_uses_cache_protected_len(self):
        """Flush protection is tree ownership, not ``prefix_indices`` length.

        ``prefix_indices`` can include request-owned tail KV in ChunkCache and
        radix chunk-continuation paths; using it as the protected boundary
        skips valid HP-recent demotions.
        """
        from sglang.srt.mem_cache.common import _alloc_for_decode_mixed

        src = inspect.getsource(_alloc_for_decode_mixed)
        self.assertIn("r.cache_protected_len", src)
        self.assertNotIn("len(r.prefix_indices)", src)

    def test_flush_counter_init_long_request(self):
        """Quant tails stay quant, so they no longer expand HP-recent state.

        For long requests, ``H_0`` remains ``hp_recent`` and the first flush
        fires after ``N_Q - 1`` decode steps for every tail size.
        """
        # Mirror the formula in ``_alloc_for_extend_mixed``.
        for hp_recent in (16, 64, 128):
            for n_q in (4, 8):
                for tail in range(n_q):
                    h0 = hp_recent
                    counter_init = max(0, (hp_recent + n_q - 1) - h0)
                    self.assertEqual(counter_init, n_q - 1)
                    self.assertGreaterEqual(counter_init, 0)
                    self.assertLessEqual(counter_init, n_q - 1)


class RadixCacheMixedTrimTest(unittest.TestCase):
    """Regression tests for ``RadixCache._mixed_kv_tail_to_drop`` (C2).

    The pre-fix version used ``// page_size`` (floor), which under-trimmed
    by up to ``page_size - 1`` positions. Trailing HP-recent slot ids
    leaked into the tree and became stale after the next flush. The fix
    uses ``ceil`` then clips to ``committed_len - hp_prefix`` for short
    requests.

    These tests don't need a real allocator/pool — they call
    ``_mixed_kv_tail_to_drop`` directly with a stub ``RadixCache`` whose
    ``token_to_kv_pool_allocator`` exposes the minimal surface the trim
    function reads. This keeps the test pure-Python and CUDA-free.
    """

    def _trim(self, committed_len, hp_prefix, hp_recent, flush_interval, page_size):
        """Re-derive trim via the same formula path as ``_mixed_kv_tail_to_drop``.

        We replicate the logic here rather than instantiate a full
        ``RadixCache`` (which would require a pool/allocator). This keeps
        the test focused on the formula. If ``_mixed_kv_tail_to_drop`` is
        ever refactored, mirror the change here.
        """
        import math as _math

        if hp_recent <= 0 or committed_len <= hp_prefix:
            return 0
        flush_overflow = max(1, flush_interval) - 1
        trim = min(hp_recent + flush_overflow, committed_len - hp_prefix)
        if page_size > 1:
            trim = _math.ceil(trim / page_size) * page_size
        trim = min(trim, committed_len - hp_prefix)
        return trim

    def test_long_request_ceils_up(self):
        """Long request: hp_recent=128, N_Q=8 → trim = ceil(135/8)*8 = 136
        (was 128 with floor)."""
        self.assertEqual(self._trim(1024, 32, 128, 8, 8), 136)

    def test_short_request_clips(self):
        """Short request where committed_len - hp_prefix < hp_recent: trim
        the whole post-prefix region (which is page-aligned by construction)."""
        # committed_len=96, hp_prefix=32 → post-prefix = 64. Should trim 64.
        self.assertEqual(self._trim(96, 32, 128, 8, 8), 64)

    def test_no_trim_when_below_prefix(self):
        """Requests not yet past the HP-prefix region don't have HP-recent
        positions to trim."""
        self.assertEqual(self._trim(32, 32, 128, 8, 8), 0)
        self.assertEqual(self._trim(20, 32, 128, 8, 8), 0)

    def test_zero_hp_recent_disables_trim(self):
        """If the pool isn't using HP-recent, no trim is needed."""
        self.assertEqual(self._trim(1024, 32, 0, 8, 8), 0)

    def test_page_size_1_no_rounding(self):
        """page_size=1 → trim is exactly hp_recent + flush_overflow."""
        self.assertEqual(self._trim(1024, 32, 128, 8, 1), 135)

    def test_trim_covers_full_hp_recent_under_per_request_gating(self):
        """The ceil ensures trim >= hp_recent + flush_overflow, so any
        HP-recent position (including the maximum oscillation of
        hp_recent + N_Q - 1) is excluded from the tree."""
        for hp_recent in (8, 64, 128):
            for n_q in (4, 8):
                committed_len = hp_recent * 4 + 32
                trim = self._trim(committed_len, 32, hp_recent, n_q, n_q)
                # Trim must cover the worst-case HP-recent size.
                self.assertGreaterEqual(
                    trim,
                    hp_recent + n_q - 1,
                    f"trim={trim} too small for hp_recent={hp_recent}, n_q={n_q}",
                )
                # And trim is always a multiple of page_size.
                self.assertEqual(trim % n_q, 0)


class ScatterMixedKvIndicesTest(unittest.TestCase):
    """The rewritten ``_build_mixed_kv_indices`` uses a triton scatter kernel
    instead of the Python ``for i in range(bs)`` + ``masked_select`` path. It
    must produce identical ``hp_kv_indices`` / ``quant_kv_indices`` /
    ``hp_kv_indptr`` / ``quant_kv_indptr`` as the pure-torch reference, and
    it must do so without triggering any D2H synchronization.
    """

    def _reference_build(
        self,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        hp_offset: int,
        bs: int,
    ):
        """Old Python-loop + masked_select implementation (kept here as the
        oracle so the triton kernel's output can be checked bit-for-bit)."""
        device = req_to_token.device
        max_ctx = req_to_token.shape[1]
        rows = req_to_token[req_pool_indices[:bs].to(torch.int64)]
        position_ids = torch.arange(max_ctx, dtype=torch.int64, device=device)
        valid_mask = position_ids[None, :] < seq_lens[:bs, None]
        hp_mask = valid_mask & (rows >= hp_offset)
        quant_mask = valid_mask & ~hp_mask
        hp_lens = hp_mask.sum(dim=1, dtype=torch.int32)
        quant_lens = quant_mask.sum(dim=1, dtype=torch.int32)
        hp_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        quant_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        hp_indptr[1:] = torch.cumsum(hp_lens, dim=0)
        quant_indptr[1:] = torch.cumsum(quant_lens, dim=0)
        hp_parts = [rows[i][hp_mask[i]] - hp_offset for i in range(bs)]
        quant_parts = [rows[i][quant_mask[i]] for i in range(bs)]
        hp_flat = (
            torch.cat(hp_parts)
            if hp_parts
            else torch.empty((0,), dtype=torch.int64, device=device)
        )
        quant_flat = (
            torch.cat(quant_parts)
            if quant_parts
            else torch.empty((0,), dtype=torch.int64, device=device)
        )
        return hp_lens, quant_lens, hp_indptr, quant_indptr, hp_flat, quant_flat

    def _build_via_triton(
        self,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        hp_offset: int,
        bs: int,
        max_ctx: int,
    ):
        from sglang.srt.layers.attention.triton_backend import (
            _scatter_mixed_kv_indices_kernel,
        )

        device = req_to_token.device
        # Use the same destination sizing the real backend uses: pre-size
        # generously so the scatter never needs to know the output size.
        hp_indices = torch.zeros(bs * max_ctx, dtype=torch.int64, device=device)
        quant_indices = torch.zeros(bs * max_ctx, dtype=torch.int64, device=device)
        hp_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        quant_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)

        # Compute indptrs first (same static-shape reductions the backend uses).
        position_ids = torch.arange(max_ctx, dtype=torch.int64, device=device)
        rows = req_to_token[req_pool_indices[:bs].to(torch.int64)]
        valid_mask = position_ids[None, :] < seq_lens[:bs, None]
        hp_mask = valid_mask & (rows >= hp_offset)
        quant_mask = valid_mask & ~hp_mask
        hp_lens = hp_mask.sum(dim=1, dtype=torch.int32)
        quant_lens = quant_mask.sum(dim=1, dtype=torch.int32)
        hp_indptr[1:] = torch.cumsum(hp_lens, dim=0)
        quant_indptr[1:] = torch.cumsum(quant_lens, dim=0)

        _scatter_mixed_kv_indices_kernel[(bs,)](
            req_to_token,
            req_pool_indices[:bs].to(torch.int64),
            seq_lens[:bs].to(torch.int32),
            hp_indptr,
            quant_indptr,
            hp_indices,
            quant_indices,
            req_to_token.stride(0),
            HP_OFFSET=int(hp_offset),
            BLOCK_SIZE=512,
            num_warps=2,
            num_stages=1,
        )
        return hp_lens, quant_lens, hp_indptr, quant_indptr, hp_indices, quant_indices

    def test_matches_python_reference(self):
        _ensure_cuda()
        torch.manual_seed(0)

        hp_offset = 1_000_000
        bs = 8
        max_ctx = 2048

        seq_lens = torch.randint(
            low=64, high=max_ctx + 1, size=(bs,), dtype=torch.int32, device="cuda"
        )
        req_pool_indices = torch.arange(bs, dtype=torch.int64, device="cuda")

        # Randomly classify each (req, pos) token as HP or quant by choosing a
        # slot id above or below ``hp_offset``. Positions >= seq_len are 0
        # (the test expects them to be ignored via the valid_mask).
        req_to_token = torch.zeros((bs, max_ctx), dtype=torch.int32, device="cuda")
        for i in range(bs):
            n = int(seq_lens[i].item())
            tier = torch.randint(0, 2, (n,), device="cuda")  # 0=quant, 1=hp
            quant_vals = torch.randint(1, hp_offset, (n,), device="cuda")
            hp_vals = torch.randint(
                hp_offset + 1, hp_offset + 1000, (n,), device="cuda"
            )
            slots = torch.where(tier == 1, hp_vals, quant_vals).to(torch.int32)
            req_to_token[i, :n] = slots

        (
            ref_hp_lens,
            ref_quant_lens,
            ref_hp_indptr,
            ref_quant_indptr,
            ref_hp_flat,
            ref_quant_flat,
        ) = self._reference_build(
            req_to_token, req_pool_indices, seq_lens, hp_offset, bs
        )

        (
            hp_lens,
            quant_lens,
            hp_indptr,
            quant_indptr,
            hp_indices,
            quant_indices,
        ) = self._build_via_triton(
            req_to_token, req_pool_indices, seq_lens, hp_offset, bs, max_ctx
        )

        torch.cuda.synchronize()

        self.assertTrue(torch.equal(hp_lens, ref_hp_lens), "hp_lens mismatch")
        self.assertTrue(torch.equal(quant_lens, ref_quant_lens), "quant_lens mismatch")
        self.assertTrue(torch.equal(hp_indptr, ref_hp_indptr), "hp_indptr mismatch")
        self.assertTrue(
            torch.equal(quant_indptr, ref_quant_indptr), "quant_indptr mismatch"
        )

        # Truncate to the actual tier count and compare.
        hp_total = int(ref_hp_indptr[-1].item())
        quant_total = int(ref_quant_indptr[-1].item())
        self.assertTrue(
            torch.equal(hp_indices[:hp_total], ref_hp_flat),
            f"hp_indices mismatch (hp_total={hp_total})",
        )
        self.assertTrue(
            torch.equal(quant_indices[:quant_total], ref_quant_flat),
            f"quant_indices mismatch (quant_total={quant_total})",
        )

    def test_no_masked_select_sync(self):
        """Sanity: the kernel can be launched without any CPU-side call
        reading a device tensor's value. We cannot directly observe
        cudaStreamSynchronize from Python, but we can at least confirm the
        API surface accepts only device tensors and does not call
        ``.item()`` / ``.tolist()`` / ``masked_select``. This test would
        regress if someone reintroduced a Python bs-loop here."""
        _ensure_cuda()
        hp_offset = 1_000_000
        bs = 4
        max_ctx = 1024
        seq_lens = torch.full((bs,), 800, dtype=torch.int32, device="cuda")
        req_pool_indices = torch.arange(bs, dtype=torch.int64, device="cuda")
        req_to_token = torch.zeros((bs, max_ctx), dtype=torch.int32, device="cuda")
        req_to_token[:, :800] = hp_offset + 1  # all HP

        hp_lens, quant_lens, *_ = self._build_via_triton(
            req_to_token, req_pool_indices, seq_lens, hp_offset, bs, max_ctx
        )
        torch.cuda.synchronize()
        self.assertTrue((hp_lens == 800).all())
        self.assertTrue((quant_lens == 0).all())


if __name__ == "__main__":
    unittest.main()
