"""Regression tests for grouped INT2 decode head tiling.

``_decode_grouped_att_m_fwd_quant_int2`` picks its head tile ``BLOCK_H`` from a
batch-size heuristic. The kernel addresses heads as

    VALID_BLOCK_H = min(BLOCK_H, kv_group_num)
    cur_head      = cur_head_id * VALID_BLOCK_H + arange(BLOCK_H)
    cur_kv_head   = cur_head_id // cdiv(kv_group_num, BLOCK_H)

which is only self-consistent when a head block lies wholly inside one KV
group. With ``BLOCK_H=4`` (chosen at batch >= 16) and ``kv_group_num=6``
(MiniMax-M2.7, 48 q heads / 8 KV heads at TP=4), head block 1 covers q heads
4..7 but reports ``cur_kv_head=0``, so q heads 6 and 7 read KV head 0's cache
for the entire INT2 tier. Nothing raises and no NaN appears; on GPQA it cost
~25 pp and looked like a CUDA-graph defect because the padded replay at client
concurrency 15 was what pushed the decode batch to 16.

These tests pin the invariant that the fused kernel matches dense attention at
every batch size, for KV group counts either side of the tile size.
"""

from __future__ import annotations

import unittest

import torch


def _ensure_cuda():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA required")


HEAD_DIM = 128
HP_LEN = 96
QUANT_LEN = 384


def _dequant(packed: torch.Tensor, sz: torch.Tensor) -> torch.Tensor:
    """[L, KVH, D//4] uint8 -> [L, KVH, D] float32, matching the kernel."""
    b = packed.to(torch.int32)
    crumbs = torch.stack([(b >> s) & 0x3 for s in (0, 2, 4, 6)], dim=-1)
    L, kvh = b.shape[0], b.shape[1]
    crumbs = crumbs.to(torch.float32).permute(0, 1, 3, 2).reshape(L, kvh, HEAD_DIM)
    return (crumbs - sz[..., 1:2]) * sz[..., 0:1]


def _reference(q, hp_k, hp_v, qk, qv, ksz, vsz, kv_group_num, sm_scale):
    bs, h = q.shape[0], q.shape[1]
    out = torch.empty(bs, h, HEAD_DIM, dtype=torch.float32, device=q.device)
    for i in range(bs):
        k = torch.cat(
            [hp_k[i * HP_LEN : (i + 1) * HP_LEN].float(),
             _dequant(qk[i * QUANT_LEN : (i + 1) * QUANT_LEN],
                      ksz[i * QUANT_LEN : (i + 1) * QUANT_LEN])]
        )
        v = torch.cat(
            [hp_v[i * HP_LEN : (i + 1) * HP_LEN].float(),
             _dequant(qv[i * QUANT_LEN : (i + 1) * QUANT_LEN],
                      vsz[i * QUANT_LEN : (i + 1) * QUANT_LEN])]
        )
        for head in range(h):
            g = head // kv_group_num
            s = (k[:, g, :] @ q[i, head].float()) * sm_scale
            out[i, head] = torch.softmax(s, 0) @ v[:, g, :]
    return out


class Int2DecodeHeadTilingTest(unittest.TestCase):
    def _run(self, bs: int, num_kv_heads: int, kv_group_num: int, seed: int = 0):
        from sglang.srt.layers.attention.triton_ops.decode_attention import (
            decode_attention_fwd_int2_unified,
        )

        torch.manual_seed(seed)
        dev = "cuda"
        h = num_kv_heads * kv_group_num
        hp_splits = quant_splits = 4
        total = hp_splits + quant_splits
        sm = HEAD_DIM ** -0.5

        hp_k = (torch.randn(bs * HP_LEN, num_kv_heads, HEAD_DIM, device=dev)
                * 0.5).to(torch.bfloat16)
        hp_v = (torch.randn(bs * HP_LEN, num_kv_heads, HEAD_DIM, device=dev)
                * 0.5).to(torch.bfloat16)
        qk = torch.randint(0, 256, (bs * QUANT_LEN, num_kv_heads, HEAD_DIM // 4),
                           dtype=torch.uint8, device=dev)
        qv = torch.randint(0, 256, (bs * QUANT_LEN, num_kv_heads, HEAD_DIM // 4),
                           dtype=torch.uint8, device=dev)
        ksz = torch.stack(
            [torch.rand(bs * QUANT_LEN, num_kv_heads, device=dev) * 0.3 + 0.05,
             torch.rand(bs * QUANT_LEN, num_kv_heads, device=dev) * 2.0], dim=-1
        ).contiguous()
        vsz = torch.stack(
            [torch.rand(bs * QUANT_LEN, num_kv_heads, device=dev) * 0.3 + 0.05,
             torch.rand(bs * QUANT_LEN, num_kv_heads, device=dev) * 2.0], dim=-1
        ).contiguous()

        q = (torch.randn(bs, h, HEAD_DIM, device=dev) * 0.6).to(torch.bfloat16)
        o = torch.empty(bs, h, HEAD_DIM, dtype=torch.bfloat16, device=dev)
        logits = torch.empty(bs, h, total, HEAD_DIM, dtype=torch.float32, device=dev)
        lse = torch.full((bs, h, total), float("-inf"), dtype=torch.float32, device=dev)

        hp_indptr = torch.arange(0, (bs + 1) * HP_LEN, HP_LEN,
                                 dtype=torch.int32, device=dev)
        q_indptr = torch.arange(0, (bs + 1) * QUANT_LEN, QUANT_LEN,
                                dtype=torch.int32, device=dev)
        hp_idx = torch.arange(bs * HP_LEN, dtype=torch.int64, device=dev)
        q_idx = torch.arange(bs * QUANT_LEN, dtype=torch.int64, device=dev)

        decode_attention_fwd_int2_unified(
            q, hp_k, hp_v, qk, qv, ksz, vsz, o,
            hp_indptr, hp_idx, q_indptr, q_idx, logits, lse,
            torch.full((bs,), hp_splits, dtype=torch.int32, device=dev),
            torch.full((bs,), quant_splits, dtype=torch.int32, device=dev),
            hp_splits, quant_splits, sm,
        )
        torch.cuda.synchronize()

        ref = _reference(q, hp_k, hp_v, qk, qv, ksz, vsz, kv_group_num, sm)
        return (o.float() - ref).norm().item() / ref.norm().item()

    def test_matches_dense_reference_across_the_tile_heuristic_boundary(self):
        """The heuristic switches BLOCK_H at batch 4 and again at batch 16.

        Two OPPOSITE regimes have to be covered, because the heuristic picks
        BLOCK_H=4 below kv_group_num 9 and BLOCK_H=8/16 above it:

            kv_group_num 5..7   BLOCK_H 4 at batch >= 16  -> breaks at LARGE batch
            kv_group_num 9..15  BLOCK_H 8 at batch <  16  -> breaks at SMALL batch

        The first regime is MiniMax-M2.7 (48Q/8KV at TP=4 = 6). The second is
        GLM-4.7-FP8 (96Q/8KV = 12), and it is only reachable with **two or more
        KV heads per rank**: with a single KV head every head block maps to
        kv_head 0, which is also the only KV head, so the mis-mapping is masked
        and reads correct data by accident. That is why the 12-group cases below
        use num_kv_heads 2 and 8 rather than 1.
        """
        _ensure_cuda()
        # kv_group_num 5, 6 and 7 are the ones a BLOCK_H of 4 cannot tile;
        # 9..15 are the ones a BLOCK_H of 8 cannot tile; 4 and 8 are controls.
        for num_kv_heads, kv_group_num in ((2, 6), (4, 5), (4, 7),
                                           (2, 12), (8, 12), (2, 9), (2, 15),
                                           (2, 4), (2, 8)):
            for bs in (2, 8, 15, 16, 20):
                with self.subTest(kv_group_num=kv_group_num, bs=bs):
                    rel = self._run(bs, num_kv_heads, kv_group_num)
                    self.assertLess(
                        rel, 5e-3,
                        f"grouped INT2 decode diverged from dense attention at "
                        f"bs={bs}, kv_group_num={kv_group_num}: rel_l2={rel:.3e}",
                    )

    def test_safe_block_h_keeps_head_blocks_inside_one_kv_group(self):
        from sglang.srt.layers.attention.triton_ops.decode_attention import (
            _safe_block_h,
        )

        for kv_group_num in range(1, 33):
            for block_h in (1, 2, 4, 8, 16):
                got = _safe_block_h(block_h, kv_group_num)
                valid = min(got, kv_group_num)
                self.assertEqual(
                    kv_group_num % valid, 0,
                    f"_safe_block_h({block_h}, {kv_group_num}) = {got} still "
                    f"straddles a KV head (VALID_BLOCK_H={valid})",
                )
                self.assertEqual(got & (got - 1), 0, "BLOCK_H must stay a power of 2")
                self.assertGreaterEqual(got, block_h)


if __name__ == "__main__":
    unittest.main()
