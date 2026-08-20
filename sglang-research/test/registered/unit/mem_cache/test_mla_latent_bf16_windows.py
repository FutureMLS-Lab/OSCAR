"""BF16 sink + BF16 recent windows over the MLA/NSA shared latent.

Four things need to hold, and each has already been a real bug in the
neighbouring per-head pool:

1. With both windows at 0 the pool is byte-for-byte the fake-quant pool it
   was before the windows existed.
2. A windowed row is the *input* row, not a re-quantized one, and ``k_pe``
   is never touched in either tier.
3. A row that leaves the recent window is quantized in place -- otherwise
   "recent 256" silently means "the whole generation stays BF16", which
   would read as a huge accuracy win that is really just BF16.
4. A CUDA-graph padded decode replay -- stale ``req_pool_indices``, seq_lens
   forced to the graph fill value -- must not write any live position. This
   is the failure that cost the per-head pool 57 -> 28 on GPQA.

Tiers are read off the data, not off the config, with the same
``mixed_kv_audit.latent_grid_error`` the runtime auditor uses -- so this also
pins the detector down, which matters because the naive version of it is
wrong: the round trip *ends* with a dense un-rotation, so a quantized row
looks like arbitrary floats until you rotate it back.
"""

import os
import unittest

import torch

from sglang.srt.utils import is_cuda, is_hip
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="stage-b-test-1-gpu-small")

KV_LORA_RANK = 512
ROPE_DIM = 64
GROUP = 128
PAGE = 64
LAYERS = 2
POOL_TOKENS = 8192


class _Layer:
    def __init__(self, layer_id: int):
        self.layer_id = layer_id
        self.is_cross_attention = False
        self.oscar_v_rotation_absorbed = False


class _Mode:
    def __init__(self, decode: bool):
        self._decode = decode

    def is_decode_or_idle(self):
        return self._decode


class _ReqToTokenPool:
    def __init__(self, req_to_token):
        self.req_to_token = req_to_token


class _Batch:
    """Only the ForwardBatch fields ``note_forward_batch`` reads."""

    def __init__(
        self,
        decode,
        positions,
        seq_lens,
        req_pool_indices,
        req_to_token,
        extend_seq_lens=None,
        extend_seq_lens_cpu=None,
        extend_prefix_lens_cpu=None,
    ):
        self.forward_mode = _Mode(decode)
        self.positions = positions
        self.seq_lens = seq_lens
        self.req_pool_indices = req_pool_indices
        self.req_to_token_pool = _ReqToTokenPool(req_to_token)
        self.extend_seq_lens = extend_seq_lens
        self.extend_seq_lens_cpu = extend_seq_lens_cpu
        self.extend_prefix_lens_cpu = extend_prefix_lens_cpu


def _grid_err(pool, rows: torch.Tensor, layer_id: int = 0):
    from sglang.srt.mem_cache.mixed_kv_audit import latent_grid_error

    return latent_grid_error(pool, layer_id, rows)


class TestMLALatentBF16Windows(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available() or not (is_cuda() or is_hip()):
            self.skipTest("CUDA is required for the MLA latent window tests.")
        torch.manual_seed(0)

    # -- construction ----------------------------------------------------

    @staticmethod
    def _build_pool():
        from sglang.srt.mem_cache.mla_int2_kv_pool import NSAInt2HPKVPool

        return NSAInt2HPKVPool(
            size=POOL_TOKENS,
            page_size=PAGE,
            dtype=torch.bfloat16,
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=ROPE_DIM,
            layer_num=LAYERS,
            device="cuda",
            kv_cache_dim=KV_LORA_RANK + ROPE_DIM,
            enable_memory_saver=False,
            start_layer=0,
            end_layer=LAYERS - 1,
            index_head_dim=128,
            rotation_path="hadamard",
            group_size=GROUP,
        )

    @classmethod
    def _make_pool(cls, sink: int, recent: int):
        """Build a pool with the windows configured through the real env vars."""
        from sglang.srt.environ import envs

        with envs.SGLANG_MIXED_KV_PREFIX_TOKENS.override(
            sink
        ), envs.SGLANG_MIXED_KV_RECENT_TOKENS.override(recent):
            return cls._build_pool()

    @staticmethod
    def _slots(n: int, first_page: int = 1) -> torch.Tensor:
        """Token slots from page ``first_page`` on -- page 0 is the reserved
        padding sink that neither allocator ever hands out."""
        return torch.arange(
            first_page * PAGE, first_page * PAGE + n, dtype=torch.int64, device="cuda"
        )

    # -- 0. the floor is the default -------------------------------------

    def test_windows_default_to_the_int2_floor(self):
        """No env vars set must still mean sink 64 / recent 256.

        The two vars' own defaults are the per-head pool's older 32/128, and
        a latent pool that inherited those would sit below the floor every
        other INT2 KV config holds to. Opt-in is also not an option: it is
        how this pool ended up 2-bitting the attention sink for every NSA
        model in the first place.
        """
        from sglang.srt.environ import envs
        from sglang.srt.mem_cache.mla_int2_kv_pool import (
            DEFAULT_LATENT_RECENT_TOKENS,
            DEFAULT_LATENT_SINK_TOKENS,
        )

        for name in (
            "SGLANG_MIXED_KV_PREFIX_TOKENS",
            "SGLANG_MIXED_KV_RECENT_TOKENS",
            "SGLANG_ENABLE_MIXED_KV_WINDOWS",
        ):
            os.environ.pop(name, None)
        self.assertFalse(envs.SGLANG_MIXED_KV_PREFIX_TOKENS.is_set())

        pool = self._build_pool()
        self.assertTrue(pool.latent_windows_enabled())
        self.assertEqual(pool.hp_prefix_tokens, DEFAULT_LATENT_SINK_TOKENS)
        self.assertEqual(pool.hp_recent_tokens, DEFAULT_LATENT_RECENT_TOKENS)
        self.assertEqual((pool.hp_prefix_tokens, pool.hp_recent_tokens), (64, 256))

    # -- 1. disabled == the pool as it was -------------------------------

    def test_windows_off_is_bit_identical(self):
        pool = self._make_pool(sink=0, recent=0)
        self.assertFalse(pool.latent_windows_enabled())
        self.assertFalse(pool._latent_windows)

        n = 300
        loc = self._slots(n)
        nope = torch.randn(n, 1, KV_LORA_RANK, dtype=torch.bfloat16, device="cuda")
        rope = torch.randn(n, 1, ROPE_DIM, dtype=torch.bfloat16, device="cuda")

        # note_forward_batch must not even record anything when disabled.
        pool.note_forward_batch(
            _Batch(
                decode=False,
                positions=torch.arange(n, device="cuda"),
                seq_lens=torch.tensor([n], device="cuda"),
                req_pool_indices=torch.tensor([0], device="cuda"),
                req_to_token=torch.zeros(2, 4096, dtype=torch.int32, device="cuda"),
                extend_seq_lens=torch.tensor([n], device="cuda"),
                extend_seq_lens_cpu=[n],
                extend_prefix_lens_cpu=[0],
            )
        )
        self.assertIsNone(pool._fb_window_meta)

        pool.set_mla_kv_buffer(_Layer(0), loc, nope, rope)
        got = pool.get_key_buffer(0)[loc].clone()

        # The pre-window code path, spelled out: every row goes through the
        # INT2 round trip, k_pe goes through untouched.
        want = torch.cat([pool._apply_fake_int2_c_kv(0, nope), rope], dim=-1)
        self.assertTrue(
            torch.equal(got, want),
            "windows-off write differs from quantize-every-token",
        )

        # Same for the concatenated NSA setter, which is the one GLM-5.2
        # actually logs.
        cat = torch.cat([nope, rope], dim=-1)
        pool.set_kv_buffer(_Layer(1), loc, cat, None)
        got2 = pool.get_key_buffer(1)[loc]
        self.assertTrue(
            torch.equal(got2, want), "windows-off set_kv_buffer differs"
        )

    # -- 2 + 3. the windows, and the demote that makes them windows ------

    def test_prefill_tiers_and_decode_demote(self):
        sink, recent = 64, 256
        pool = self._make_pool(sink=sink, recent=recent)
        self.assertTrue(pool.latent_windows_enabled())
        self.assertEqual(pool.hp_prefix_tokens, sink)
        self.assertEqual(pool.hp_recent_tokens, recent)

        L = 1000
        loc = self._slots(L)
        req_to_token = torch.zeros(4, 4096, dtype=torch.int32, device="cuda")
        req_to_token[0, :L] = loc.to(torch.int32)
        nope = torch.randn(L, 1, KV_LORA_RANK, dtype=torch.bfloat16, device="cuda")
        rope = torch.randn(L, 1, ROPE_DIM, dtype=torch.bfloat16, device="cuda")

        pool.note_forward_batch(
            _Batch(
                decode=False,
                positions=torch.arange(L, device="cuda"),
                seq_lens=torch.tensor([L], dtype=torch.int32, device="cuda"),
                req_pool_indices=torch.tensor([0], dtype=torch.int64, device="cuda"),
                req_to_token=req_to_token,
                extend_seq_lens=torch.tensor([L], dtype=torch.int32, device="cuda"),
                extend_seq_lens_cpu=[L],
                extend_prefix_lens_cpu=[0],
            )
        )
        pool.set_mla_kv_buffer(_Layer(0), loc, nope, rope)

        buf = pool.get_key_buffer(0)
        stored = buf[loc]
        # k_pe is never a party to the round trip, in any tier.
        self.assertTrue(
            torch.equal(stored[..., KV_LORA_RANK:], rope), "k_pe was modified"
        )
        # A windowed row is the input row itself.
        self.assertTrue(
            torch.equal(stored[:sink, :, :KV_LORA_RANK], nope[:sink]),
            "sink rows were quantized",
        )
        self.assertTrue(
            torch.equal(stored[L - recent :, :, :KV_LORA_RANK], nope[L - recent :]),
            "recent rows were quantized",
        )
        mid = stored[sink : L - recent]
        self.assertFalse(
            torch.equal(mid[:, :, :KV_LORA_RANK], nope[sink : L - recent]),
            "middle rows were left in BF16",
        )
        # Same tier detector the runtime auditor uses, and the separation it
        # relies on: quantized rows sit on the 4-level grid, bf16 rows do not.
        from sglang.srt.mem_cache.mixed_kv_audit import LATENT_GRID_TOL

        e_mid = _grid_err(pool, mid)
        e_sink = _grid_err(pool, stored[:sink])
        e_recent = _grid_err(pool, stored[L - recent :])
        print(
            f"\n[grid_err] quant max={e_mid.max():.4f}  "
            f"sink min={e_sink.min():.4f}  recent min={e_recent.min():.4f}  "
            f"tol={LATENT_GRID_TOL}"
        )
        self.assertLess(
            float(e_mid.max()),
            LATENT_GRID_TOL,
            "middle band is not on the INT2 grid",
        )
        self.assertGreater(
            float(min(e_sink.min(), e_recent.min())),
            LATENT_GRID_TOL,
            "a BF16 window row looks quantized",
        )

        # Decode: each step must demote exactly position L-1-recent and leave
        # everything newer alone.
        for step in range(4):
            new_pos = L + step
            new_loc = self._slots(1, first_page=40 + step)
            req_to_token[0, new_pos] = new_loc.to(torch.int32)
            seq_lens = torch.tensor(
                [new_pos + 1], dtype=torch.int32, device="cuda"
            )
            pool.note_forward_batch(
                _Batch(
                    decode=True,
                    positions=seq_lens.to(torch.int64) - 1,
                    seq_lens=seq_lens,
                    req_pool_indices=torch.tensor(
                        [0], dtype=torch.int64, device="cuda"
                    ),
                    req_to_token=req_to_token,
                )
            )
            new_nope = torch.randn(
                1, 1, KV_LORA_RANK, dtype=torch.bfloat16, device="cuda"
            )
            new_rope = torch.randn(1, 1, ROPE_DIM, dtype=torch.bfloat16, device="cuda")
            pool.set_mla_kv_buffer(_Layer(0), new_loc, new_nope, new_rope)

            # the freshly written token is inside the recent window
            self.assertTrue(
                torch.equal(buf[new_loc][..., :KV_LORA_RANK], new_nope),
                f"step {step}: decode write was quantized",
            )
            demoted = new_pos - recent
            self.assertLess(
                float(_grid_err(pool, buf[loc[demoted : demoted + 1]]).max()),
                LATENT_GRID_TOL,
                f"step {step}: position {demoted} left the recent window "
                "without being quantized",
            )
            # everything strictly newer than the demote point is still BF16,
            # byte-for-byte the prefill input
            self.assertTrue(
                torch.equal(
                    buf[loc[demoted + 1 : L]][..., :KV_LORA_RANK],
                    nope[demoted + 1 : L],
                ),
                f"step {step}: a position inside the recent window changed",
            )
            self.assertTrue(
                torch.equal(
                    buf[loc[demoted + 1 : L]][..., KV_LORA_RANK:], rope[demoted + 1 : L]
                ),
                f"step {step}: k_pe changed during the demote",
            )

    # -- 4. CUDA-graph padded replay -------------------------------------

    def test_padded_decode_replay_touches_only_slot_zero(self):
        sink, recent = 64, 256
        pool = self._make_pool(sink=sink, recent=recent)

        L = 600
        loc = self._slots(L)
        req_to_token = torch.zeros(4, 4096, dtype=torch.int32, device="cuda")
        req_to_token[0, :L] = loc.to(torch.int32)
        nope = torch.randn(L, 1, KV_LORA_RANK, dtype=torch.bfloat16, device="cuda")
        rope = torch.randn(L, 1, ROPE_DIM, dtype=torch.bfloat16, device="cuda")
        pool.note_forward_batch(
            _Batch(
                decode=False,
                positions=torch.arange(L, device="cuda"),
                seq_lens=torch.tensor([L], dtype=torch.int32, device="cuda"),
                req_pool_indices=torch.tensor([0], dtype=torch.int64, device="cuda"),
                req_to_token=req_to_token,
                extend_seq_lens=torch.tensor([L], dtype=torch.int32, device="cuda"),
                extend_seq_lens_cpu=[L],
                extend_prefix_lens_cpu=[0],
            )
        )
        pool.set_mla_kv_buffer(_Layer(0), loc, nope, rope)
        buf = pool.get_key_buffer(0)
        before = buf.clone()

        # One real request plus three padded graph slots: seq_lens forced to
        # the fill value, req_pool_indices left stale (pointing at the live
        # request), out_cache_loc pointing at the reserved slot 0.
        seq_lens = torch.tensor([L + 1, 1, 1, 1], dtype=torch.int32, device="cuda")
        pool.note_forward_batch(
            _Batch(
                decode=True,
                positions=seq_lens.to(torch.int64) - 1,
                seq_lens=seq_lens,
                req_pool_indices=torch.tensor(
                    [0, 0, 0, 0], dtype=torch.int64, device="cuda"
                ),
                req_to_token=req_to_token,
            )
        )
        new_loc = torch.tensor(
            [40 * PAGE, 0, 0, 0], dtype=torch.int64, device="cuda"
        )
        req_to_token[0, L] = 40 * PAGE
        pool.set_mla_kv_buffer(
            _Layer(0),
            new_loc,
            torch.randn(4, 1, KV_LORA_RANK, dtype=torch.bfloat16, device="cuda"),
            torch.randn(4, 1, ROPE_DIM, dtype=torch.bfloat16, device="cuda"),
        )

        changed = (buf != before).any(dim=-1).any(dim=-1).nonzero().flatten().tolist()
        expected = {0, 40 * PAGE, int(loc[L - recent])}
        self.assertTrue(
            set(changed) <= expected,
            f"padded replay touched unexpected slots: "
            f"{sorted(set(changed) - expected)}",
        )
        self.assertIn(
            int(loc[L - recent]),
            changed,
            "the real request's demote did not happen",
        )

    # -- the reserved sink is really reserved ----------------------------

    def test_allocators_never_hand_out_page_zero(self):
        from sglang.srt.mem_cache.allocator import (
            PagedTokenToKVPoolAllocator,
            TokenToKVPoolAllocator,
        )

        pool = self._make_pool(sink=64, recent=256)
        paged = PagedTokenToKVPoolAllocator(
            POOL_TOKENS, PAGE, torch.bfloat16, "cuda", pool, need_sort=False
        )
        self.assertGreaterEqual(int(paged.free_pages.min()), 1)
        self.assertEqual(int(paged.alloc(PAGE).min()), PAGE)

        flat = TokenToKVPoolAllocator(
            POOL_TOKENS, torch.bfloat16, "cuda", pool, need_sort=False
        )
        self.assertEqual(int(flat.free_pages.min()), 1)


if __name__ == "__main__":
    unittest.main()
