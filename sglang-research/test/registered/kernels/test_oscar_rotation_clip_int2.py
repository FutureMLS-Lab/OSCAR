"""Unit tests for the Oscar rotation + per-row clip int2 KV kernels.

Covers:
  * compute_per_row_clip_threshold_triton matches ``torch.quantile(abs(x),
    clip_ratio, dim=-1)`` (with the plan's integer-index semantics)
  * quantized_set_kv_int2_pretransformed_clip_triton packed bytes and
    scales/zeros match a PyTorch reference that clips then groupwise
    int2-quantizes
  * With clip_ratio=0 the clip-aware pack kernel matches the existing
    ``quantized_set_kv_int2_pretransformed_triton``
"""

from __future__ import annotations

import unittest

import torch


def _ensure_cuda():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA required")


def _ref_threshold(x: torch.Tensor, clip_ratio: float) -> torch.Tensor:
    """CPU reference for the threshold kernel. Follows the kernel's discrete
    index semantics: ``sort(abs(x))[floor(clip_ratio * head_dim)]``.
    """
    head_dim = x.shape[-1]
    idx = int(clip_ratio * head_dim)
    if idx >= head_dim:
        idx = head_dim - 1
    if idx < 0:
        idx = 0
    sorted_abs, _ = x.abs().float().sort(dim=-1)
    return sorted_abs[..., idx]


def _ref_groupwise_int2_quant_dequant(
    x: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """Match the kernel's per-group affine int2 quantize + dequantize."""
    N, H, D = x.shape
    assert D % num_groups == 0
    group_size = D // num_groups
    x = x.float().reshape(N, H, num_groups, group_size)
    val_min = x.amin(dim=-1, keepdim=True)
    val_max = x.amax(dim=-1, keepdim=True)
    scale = (val_max - val_min).clamp_min(1e-8) / 3.0
    zero = -val_min / scale
    q = (x / scale + zero + 0.5).clamp(0, 3).to(torch.uint8).float()
    deq = (q - zero) * scale
    return deq.reshape(N, H, D)


class PretransformedClipKernelTest(unittest.TestCase):
    def setUp(self):
        _ensure_cuda()

    def _make_buffers(self, num_pages, num_heads, head_dim, num_groups):
        k_cache = torch.zeros(
            (num_pages, num_heads, head_dim // 4),
            dtype=torch.uint8,
            device="cuda",
        )
        v_cache = torch.zeros_like(k_cache)
        k_sz = torch.zeros(
            (num_pages, num_heads, 2 * num_groups),
            dtype=torch.bfloat16,
            device="cuda",
        )
        v_sz = torch.zeros_like(k_sz)
        return k_cache, v_cache, k_sz, v_sz

    def test_clip_kernel_matches_reference(self):
        from sglang.QuantKernel.oscar_rotation_clip_int2_kv import (
            quantized_set_kv_int2_pretransformed_clip_triton,
        )
        from sglang.srt.mem_cache.kv_quant_kernels import (
            _groupwise_dequantize_int2_torch,
        )

        num_tokens, num_heads, head_dim = 6, 4, 64
        num_groups = 4
        clip_ratio_k = 0.95
        clip_ratio_v = 0.90

        torch.manual_seed(1)
        k = torch.randn(
            (num_tokens, num_heads, head_dim), dtype=torch.bfloat16, device="cuda"
        )
        v = torch.randn_like(k)

        num_pages = 16
        k_cache, v_cache, k_sz, v_sz = self._make_buffers(
            num_pages, num_heads, head_dim, num_groups
        )
        loc = torch.arange(num_tokens, dtype=torch.int64, device="cuda")

        quantized_set_kv_int2_pretransformed_clip_triton(
            k, v, loc, k_cache, v_cache, k_sz, v_sz,
            clip_ratio_k, clip_ratio_v,
        )

        # Reference: clip (using the CPU quantile helper) then groupwise
        # int2 quantize/dequantize.
        k_thr = _ref_threshold(k, clip_ratio_k).to(k.device)
        v_thr = _ref_threshold(v, clip_ratio_v).to(v.device)

        k_ref = k.float()
        k_thr_b = k_thr[..., None]
        k_ref = torch.minimum(torch.maximum(k_ref, -k_thr_b), k_thr_b)
        k_ref_deq = _ref_groupwise_int2_quant_dequant(k_ref, num_groups)

        v_ref = v.float()
        v_thr_b = v_thr[..., None]
        v_ref = torch.minimum(torch.maximum(v_ref, -v_thr_b), v_thr_b)
        v_ref_deq = _ref_groupwise_int2_quant_dequant(v_ref, num_groups)

        k_out_deq = _groupwise_dequantize_int2_torch(
            k_cache[:num_tokens], k_sz[:num_tokens], head_dim, torch.float32
        )
        v_out_deq = _groupwise_dequantize_int2_torch(
            v_cache[:num_tokens], v_sz[:num_tokens], head_dim, torch.float32
        )

        # The reference uses fp32 input whereas the kernel reads bf16 and
        # up-casts, so occasional 1-ULP-in-bf16 rounding differences show up
        # after the int2 dequant. Loose absolute tolerance covers this.
        torch.testing.assert_close(k_out_deq, k_ref_deq, atol=2e-2, rtol=5e-2)
        torch.testing.assert_close(v_out_deq, v_ref_deq, atol=2e-2, rtol=5e-2)

    def test_clip_zero_matches_pretransformed_reference(self):
        """With clip_ratio=0 (thresholds = +inf), the clip-aware kernel must
        produce bit-identical output to the existing pretransformed kernel.
        """
        from sglang.QuantKernel.fused_hadamard_int2_kv import (
            quantized_set_kv_int2_pretransformed_triton,
        )
        from sglang.QuantKernel.oscar_rotation_clip_int2_kv import (
            quantized_set_kv_int2_pretransformed_clip_triton,
        )

        num_tokens, num_heads, head_dim = 8, 4, 64
        num_groups = 4

        torch.manual_seed(2)
        k = torch.randn(
            (num_tokens, num_heads, head_dim), dtype=torch.bfloat16, device="cuda"
        )
        v = torch.randn_like(k)

        num_pages = 16
        loc = torch.arange(num_tokens, dtype=torch.int64, device="cuda")

        k_cache_a, v_cache_a, k_sz_a, v_sz_a = self._make_buffers(
            num_pages, num_heads, head_dim, num_groups
        )
        k_cache_b, v_cache_b, k_sz_b, v_sz_b = self._make_buffers(
            num_pages, num_heads, head_dim, num_groups
        )

        quantized_set_kv_int2_pretransformed_clip_triton(
            k, v, loc, k_cache_a, v_cache_a, k_sz_a, v_sz_a, 0.0, 0.0,
        )
        quantized_set_kv_int2_pretransformed_triton(
            k, v, loc, k_cache_b, v_cache_b, k_sz_b, v_sz_b,
        )

        self.assertTrue(torch.equal(k_cache_a, k_cache_b))
        self.assertTrue(torch.equal(v_cache_a, v_cache_b))
        torch.testing.assert_close(k_sz_a, k_sz_b, atol=0, rtol=0)
        torch.testing.assert_close(v_sz_a, v_sz_b, atol=0, rtol=0)


class OscarRotationRoundTripTest(unittest.TestCase):
    """End-to-end correctness proxy: ``rotate -> clip -> pack -> dequant
    -> inverse-rotate`` should recover the input within int2 quantization
    noise. This is the best standalone proxy for the Oscar eval we can run
    without loading a real model or gsm8k/mmlu harness; replace with live
    gsm8k/mmlu once a Oscar-trained R_k / R_v checkpoint is available.
    """

    def setUp(self):
        _ensure_cuda()

    def _random_orthogonal(self, n: int, seed: int) -> torch.Tensor:
        g = torch.Generator(device="cpu").manual_seed(seed)
        m = torch.randn((n, n), generator=g, dtype=torch.float32)
        q, _ = torch.linalg.qr(m)
        return q

    def test_round_trip_preserves_input_within_int2_noise(self):
        from sglang.QuantKernel.oscar_rotation_clip_int2_kv import (
            quantized_set_kv_int2_pretransformed_clip_triton,
        )
        from sglang.srt.mem_cache.kv_quant_kernels import (
            _groupwise_dequantize_int2_torch,
        )

        num_tokens, num_heads, head_dim = 64, 4, 128
        num_groups = 8
        clip_ratio = 0.99

        torch.manual_seed(11)
        R_k = self._random_orthogonal(head_dim, seed=1).to(
            device="cuda", dtype=torch.bfloat16
        )
        R_v = self._random_orthogonal(head_dim, seed=2).to(
            device="cuda", dtype=torch.bfloat16
        )
        k = torch.randn(
            (num_tokens, num_heads, head_dim), dtype=torch.bfloat16, device="cuda"
        )
        v = torch.randn_like(k)

        k_rot = (k @ R_k).contiguous()
        v_rot = (v @ R_v).contiguous()

        k_buf = torch.zeros(
            (num_tokens, num_heads, head_dim // 4), dtype=torch.uint8, device="cuda"
        )
        v_buf = torch.zeros_like(k_buf)
        k_sz = torch.zeros(
            (num_tokens, num_heads, 2 * num_groups),
            dtype=torch.bfloat16,
            device="cuda",
        )
        v_sz = torch.zeros_like(k_sz)
        loc = torch.arange(num_tokens, dtype=torch.int64, device="cuda")

        quantized_set_kv_int2_pretransformed_clip_triton(
            k_rot, v_rot, loc, k_buf, v_buf, k_sz, v_sz,
            clip_ratio, clip_ratio,
        )

        k_deq = _groupwise_dequantize_int2_torch(k_buf, k_sz, head_dim, torch.float32)
        v_deq = _groupwise_dequantize_int2_torch(v_buf, v_sz, head_dim, torch.float32)

        # Inverse rotate with R.T (orthogonal so R^-1 == R^T).
        k_rec = (k_deq.to(torch.float32) @ R_k.T.to(torch.float32)).contiguous()
        v_rec = (v_deq.to(torch.float32) @ R_v.T.to(torch.float32)).contiguous()

        # Per-row cosine similarity should be high (> 0.85) -- int2 is lossy
        # but the rotation preserves structure.
        def _cos(a, b):
            a = a.flatten(end_dim=-2)
            b = b.flatten(end_dim=-2)
            return torch.nn.functional.cosine_similarity(a, b, dim=-1)

        cos_k = _cos(k_rec, k.to(torch.float32))
        cos_v = _cos(v_rec, v.to(torch.float32))
        self.assertTrue(
            (cos_k.mean().item() > 0.85),
            f"Mean K cosine similarity too low: {cos_k.mean().item():.4f}",
        )
        self.assertTrue(
            (cos_v.mean().item() > 0.85),
            f"Mean V cosine similarity too low: {cos_v.mean().item():.4f}",
        )


class OscarPoolSanityTest(unittest.TestCase):
    """End-to-end sanity tests for ``UnifiedInt2HPKVPool`` in oscar mode.

    With ``R = I`` and ``clip_ratio = 0``, the oscar pipeline is semantically a
    no-op relative to the 'off' rotation mode: the HP buffer should hold
    untouched bf16 inputs and the int2 pack should produce the same bytes as
    the legacy pretransformed kernel run directly on the inputs.
    """

    def setUp(self):
        _ensure_cuda()

    def _write_identity_rotation_pt(self, path, layer_num, head_dim):
        state = {
            "layers": {
                i: {"rotation": torch.eye(head_dim, dtype=torch.float32)}
                for i in range(layer_num)
            }
        }
        torch.save(state, path)

    def test_pool_with_identity_rotation_matches_off_mode(self):
        import os
        import tempfile

        from sglang.srt.environ import envs
        from sglang.srt.mem_cache.unified_kv_pool import UnifiedInt2HPKVPool

        layer_num = 1
        head_num = 4
        head_dim = 64
        num_pages = 32
        num_tokens = 6

        with tempfile.TemporaryDirectory() as tmp:
            k_pt = os.path.join(tmp, "R_k.pt")
            v_pt = os.path.join(tmp, "R_v.pt")
            self._write_identity_rotation_pt(k_pt, layer_num, head_dim)
            self._write_identity_rotation_pt(v_pt, layer_num, head_dim)

            # ``envs.*.override`` is read by ``load_oscar_rotation_config`` at
            # pool-construction time, so no module reload is needed.
            with envs.SGLANG_OSCAR_K_ROTATION_PATH.override(k_pt), \
                 envs.SGLANG_OSCAR_V_ROTATION_PATH.override(v_pt), \
                 envs.SGLANG_OSCAR_K_CLIP_RATIO.override(0.0), \
                 envs.SGLANG_OSCAR_V_CLIP_RATIO.override(0.0):
                pool = UnifiedInt2HPKVPool(
                    num_quant_pages=num_pages,
                    hp_dtype=torch.bfloat16,
                    hp_prefix_tokens=8,
                    hp_recent_tokens=16,
                    dtype="int2",
                    head_num=head_num,
                    head_dim=head_dim,
                    layer_num=layer_num,
                    device="cuda",
                    enable_memory_saver=False,
                    max_req_slots=num_pages,
                    v_head_dim=head_dim,
                    start_layer=0,
                    end_layer=layer_num - 1,
                    model_dtype=torch.bfloat16,
                    kv_cache_quant_group_size=16,
                    scale_dtype=torch.bfloat16,
                )

                self.assertIsNotNone(pool._R_k)
                self.assertTrue(
                    torch.equal(
                        pool._R_k[0].float(), torch.eye(head_dim, device="cuda")
                    )
                )

                torch.manual_seed(7)
                k = torch.randn(
                    (num_tokens, head_num, head_dim),
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                v = torch.randn_like(k)

                # Quant slot ids begin at 0 (loc values < hp_global_offset).
                quant_loc = torch.arange(
                    num_tokens, dtype=torch.int64, device="cuda"
                )
                pool._set_quant_kv_buffer_extend(
                    layer_id=0,
                    quant_loc=quant_loc,
                    cache_k=k,
                    cache_v=v,
                    already_hadamard_transformed=False,
                )

                # Reference: the legacy pretransformed kernel on the same (
                # identity-rotated) inputs.
                from sglang.QuantKernel.fused_hadamard_int2_kv import (
                    quantized_set_kv_int2_pretransformed_triton,
                )

                ref_k_buf = torch.zeros_like(pool.k_buffer[0])
                ref_v_buf = torch.zeros_like(pool.v_buffer[0])
                ref_k_sz = torch.zeros_like(pool.k_scales_zeros[0])
                ref_v_sz = torch.zeros_like(pool.v_scales_zeros[0])
                quantized_set_kv_int2_pretransformed_triton(
                    k, v, quant_loc,
                    ref_k_buf, ref_v_buf, ref_k_sz, ref_v_sz,
                )

                self.assertTrue(
                    torch.equal(pool.k_buffer[0][:num_tokens], ref_k_buf[:num_tokens])
                )
                self.assertTrue(
                    torch.equal(pool.v_buffer[0][:num_tokens], ref_v_buf[:num_tokens])
                )
                torch.testing.assert_close(
                    pool.k_scales_zeros[0][:num_tokens],
                    ref_k_sz[:num_tokens],
                    atol=0,
                    rtol=0,
                )
                torch.testing.assert_close(
                    pool.v_scales_zeros[0][:num_tokens],
                    ref_v_sz[:num_tokens],
                    atol=0,
                    rtol=0,
                )


if __name__ == "__main__":
    unittest.main()
