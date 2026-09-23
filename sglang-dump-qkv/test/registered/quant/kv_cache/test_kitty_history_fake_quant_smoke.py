import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=12, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=12, suite="stage-b-test-small-1-gpu-amd")


class _DummyLayer:
    def __init__(self, layer_id: int = 0, num_heads: int = 2, head_dim: int = 16):
        self.layer_id = layer_id
        self.tp_q_head_num = num_heads
        self.tp_k_head_num = num_heads
        self.tp_v_head_num = num_heads
        self.qk_head_dim = head_dim
        self.v_head_dim = head_dim
        self.head_dim = head_dim
        self.is_cross_attention = False
        self.k_scale = None
        self.v_scale = None


class TestKittyHistoryFakeQuantSmoke(unittest.TestCase):
    def setUp(self):
        self.device = get_device()
        self.dtype = torch.float16
        self.env_patch = patch.dict(
            os.environ,
            {
                "SGLANG_KITTY_HISTORY_FAKE_QUANT": "1",
                "SGLANG_KITTY_SINK_LENGTH": "1",
                "SGLANG_KITTY_RECENT_WINDOW": "2",
                "SGLANG_KITTY_HISTORY_QUANT_BITS": "2",
                "SGLANG_KITTY_HISTORY_CLIP_RATIO": "0",
            },
            clear=False,
        )
        self.env_patch.start()

    def tearDown(self):
        self.env_patch.stop()

    def _make_pool(self):
        return MHATokenToKVPool(
            size=64,
            page_size=1,
            dtype=self.dtype,
            head_num=2,
            head_dim=16,
            layer_num=1,
            device=self.device,
            enable_memory_saver=False,
        )

    def _make_req_to_token_pool(self):
        return ReqToTokenPool(
            size=2,
            max_context_len=32,
            device=self.device,
            enable_memory_saver=False,
        )

    def _make_forward_batch(self, req_to_token_pool: ReqToTokenPool, seq_len: int):
        return SimpleNamespace(
            req_to_token_pool=req_to_token_pool,
            req_pool_indices=torch.tensor([0], dtype=torch.int32, device=self.device),
            seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=self.device),
        )

    def _make_values(self, num_tokens: int, offset: float):
        total = num_tokens * 2 * 16
        values = torch.arange(total, dtype=torch.float32, device=self.device)
        values = (values / 17.0 + offset).reshape(num_tokens, 2, 16)
        return values.to(self.dtype)

    def test_sink_recent_and_history_aging(self):
        pool = self._make_pool()
        req_to_token_pool = self._make_req_to_token_pool()
        layer = _DummyLayer()
        req_to_token_pool.req_epoch[0] = 1

        loc = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long, device=self.device)
        req_to_token_pool.req_to_token[0, :6] = loc.to(torch.int32)
        cache_k = self._make_values(6, 0.125)
        cache_v = self._make_values(6, 0.625)

        pool.set_kv_buffer(
            layer,
            loc,
            cache_k,
            cache_v,
            forward_batch=self._make_forward_batch(req_to_token_pool, seq_len=6),
        )

        stored_k = pool.get_key_buffer(0)[loc]
        stored_v = pool.get_value_buffer(0)[loc]

        self.assertTrue(torch.equal(stored_k[0], cache_k[0]))
        self.assertTrue(torch.equal(stored_v[0], cache_v[0]))
        self.assertTrue(torch.equal(stored_k[4], cache_k[4]))
        self.assertTrue(torch.equal(stored_k[5], cache_k[5]))
        self.assertTrue(torch.equal(stored_v[4], cache_v[4]))
        self.assertTrue(torch.equal(stored_v[5], cache_v[5]))

        for idx in (1, 2, 3):
            self.assertFalse(torch.equal(stored_k[idx], cache_k[idx]))
            self.assertFalse(torch.equal(stored_v[idx], cache_v[idx]))

        debug_state = pool.get_history_fake_quant_debug_state()
        self.assertEqual(int(debug_state["k_rows"][0]), 3)
        self.assertEqual(int(debug_state["v_rows"][0]), 3)

        pre_aged_k = pool.get_key_buffer(0)[5].clone()
        pre_aged_v = pool.get_value_buffer(0)[5].clone()

        loc_decode_1 = torch.tensor([7], dtype=torch.long, device=self.device)
        req_to_token_pool.req_to_token[0, 6] = 7
        decode_k_1 = self._make_values(1, 1.125)
        decode_v_1 = self._make_values(1, 1.625)
        pool.set_kv_buffer(
            layer,
            loc_decode_1,
            decode_k_1,
            decode_v_1,
            forward_batch=self._make_forward_batch(req_to_token_pool, seq_len=7),
        )

        post_aged_k = pool.get_key_buffer(0)[5].clone()
        post_aged_v = pool.get_value_buffer(0)[5].clone()
        self.assertFalse(torch.equal(pre_aged_k, post_aged_k))
        self.assertFalse(torch.equal(pre_aged_v, post_aged_v))

        debug_state = pool.get_history_fake_quant_debug_state()
        self.assertEqual(int(debug_state["k_rows"][0]), 4)
        self.assertEqual(int(debug_state["v_rows"][0]), 4)

        frozen_k = pool.get_key_buffer(0)[5].clone()
        frozen_v = pool.get_value_buffer(0)[5].clone()

        loc_decode_2 = torch.tensor([8], dtype=torch.long, device=self.device)
        req_to_token_pool.req_to_token[0, 7] = 8
        decode_k_2 = self._make_values(1, 2.125)
        decode_v_2 = self._make_values(1, 2.625)
        pool.set_kv_buffer(
            layer,
            loc_decode_2,
            decode_k_2,
            decode_v_2,
            forward_batch=self._make_forward_batch(req_to_token_pool, seq_len=8),
        )

        self.assertTrue(torch.equal(pool.get_key_buffer(0)[5], frozen_k))
        self.assertTrue(torch.equal(pool.get_value_buffer(0)[5], frozen_v))
        self.assertTrue(torch.equal(pool.get_key_buffer(0)[7], decode_k_1[0]))
        self.assertTrue(torch.equal(pool.get_value_buffer(0)[7], decode_v_1[0]))

        debug_state = pool.get_history_fake_quant_debug_state()
        self.assertEqual(int(debug_state["k_rows"][0]), 5)
        self.assertEqual(int(debug_state["v_rows"][0]), 5)
        self.assertEqual(int(debug_state["watermarks"][0, 0]), 5)

    def test_req_epoch_resets_watermark_on_slot_reuse(self):
        pool = self._make_pool()
        req_to_token_pool = self._make_req_to_token_pool()
        layer = _DummyLayer()

        req_to_token_pool.req_epoch[0] = 1
        first_loc = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long, device=self.device)
        req_to_token_pool.req_to_token[0, :6] = first_loc.to(torch.int32)
        pool.set_kv_buffer(
            layer,
            first_loc,
            self._make_values(6, 0.25),
            self._make_values(6, 0.75),
            forward_batch=self._make_forward_batch(req_to_token_pool, seq_len=6),
        )

        req_to_token_pool.req_epoch[0] = 2
        second_loc = torch.tensor([9, 10, 11, 12, 13], dtype=torch.long, device=self.device)
        req_to_token_pool.req_to_token[0, :5] = second_loc.to(torch.int32)
        second_k = self._make_values(5, 3.25)
        second_v = self._make_values(5, 3.75)
        pool.set_kv_buffer(
            layer,
            second_loc,
            second_k,
            second_v,
            forward_batch=self._make_forward_batch(req_to_token_pool, seq_len=5),
        )

        stored_k = pool.get_key_buffer(0)[second_loc]
        stored_v = pool.get_value_buffer(0)[second_loc]
        self.assertTrue(torch.equal(stored_k[0], second_k[0]))
        self.assertTrue(torch.equal(stored_v[0], second_v[0]))
        self.assertFalse(torch.equal(stored_k[1], second_k[1]))
        self.assertFalse(torch.equal(stored_k[2], second_k[2]))
        self.assertFalse(torch.equal(stored_v[1], second_v[1]))
        self.assertFalse(torch.equal(stored_v[2], second_v[2]))
