"""Unit tests for pool_configurator.py -- CPU only, no GPU required.

Tests the end-to-end computation: available_bytes -> MemoryPoolConfig,
verifying tokens are correct, constraints are respected, and memory
invariants hold (tokens * per_token_cost <= available_bytes).
"""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


@contextlib.contextmanager
def mock_cpu_env(kv_size=2, tp_size=1):
    """Mock GPU-dependent functions for CPU-only testing."""
    with (
        patch("torch._utils._element_size", return_value=kv_size),
        patch(
            "sglang.srt.model_executor.pool_configurator.get_attention_tp_size",
            return_value=tp_size,
        ),
    ):
        yield


def _make_model_runner(
    *,
    num_kv_heads=4,
    head_dim=64,
    v_head_dim=64,
    num_layers=32,
    use_mla_backend=False,
    is_hybrid_swa=False,
    full_attention_layer_ids=None,
    swa_attention_layer_ids=None,
    swa_num_kv_heads=None,
    swa_head_dim=None,
    swa_v_head_dim=None,
    swa_full_tokens_ratio=0.5,
    page_size=1,
    mambaish_config=None,
    kv_cache_dtype="fake_bf16",
    kv_cache_quant_group_size=None,
):
    """Create a mock ModelRunner with the fields configurators need."""
    mr = MagicMock()

    mr.use_mla_backend = use_mla_backend
    mr.is_draft_worker = False
    mr.num_effective_layers = num_layers
    mr.start_layer = 0
    mr.end_layer = num_layers
    mr.mambaish_config = mambaish_config
    mr.is_hybrid_swa = is_hybrid_swa

    mc = SimpleNamespace()
    mc.head_dim = head_dim
    mc.v_head_dim = v_head_dim
    mc.is_hybrid_swa = is_hybrid_swa
    mc.full_attention_layer_ids = (
        full_attention_layer_ids
        if full_attention_layer_ids is not None
        else list(range(num_layers))
    )
    mc.swa_attention_layer_ids = (
        swa_attention_layer_ids if swa_attention_layer_ids is not None else []
    )
    mc.swa_head_dim = swa_head_dim or head_dim
    mc.swa_v_head_dim = swa_v_head_dim or v_head_dim
    mc.get_num_kv_heads = lambda tp_size: num_kv_heads
    mc.get_swa_num_kv_heads = lambda tp_size: swa_num_kv_heads or num_kv_heads
    mc.hf_config = SimpleNamespace(architectures=["LlamaForCausalLM"])
    mr.model_config = mc

    mr.kv_cache_dtype = kv_cache_dtype

    sa = SimpleNamespace()
    sa.swa_full_tokens_ratio = swa_full_tokens_ratio
    sa.page_size = page_size
    sa.kv_cache_quant_group_size = kv_cache_quant_group_size
    mr.server_args = sa

    spec = MagicMock()
    spec.is_dflash.return_value = False
    spec.is_none.return_value = True
    mr.spec_algorithm = spec

    return mr


KV_SIZE = 2  # bf16


def _int_quant_per_token(k_dim, v_dim, kv_dtype, group_size):
    pack_factor = {"int2": 4, "int4": 2, "int8": 1}[kv_dtype]
    effective_k_group = k_dim if group_size is None else group_size
    effective_v_group = v_dim if group_size is None else group_size
    return (
        k_dim // pack_factor
        + v_dim // pack_factor
        + 8 * (k_dim // effective_k_group + v_dim // effective_v_group)
    )


def _full_per_token(mr):
    mc = mr.model_config
    return mc.get_num_kv_heads(1) * (mc.head_dim + mc.v_head_dim) * KV_SIZE


def _swa_per_token(mr):
    mc = mr.model_config
    return mc.get_swa_num_kv_heads(1) * (mc.swa_head_dim + mc.swa_v_head_dim) * KV_SIZE


def _actual_memory_used(mr, config):
    """Compute actual memory consumed by the pool sizes in config."""
    mc = mr.model_config
    full_pt = _full_per_token(mr)
    swa_pt = _swa_per_token(mr)
    nf = len(mc.full_attention_layer_ids)
    ns = len(mc.swa_attention_layer_ids)

    if mr.is_hybrid_swa:
        full = config.full_max_total_num_tokens or 0
        swa = config.swa_max_total_num_tokens or 0
        return full * full_pt * nf + swa * swa_pt * ns
    else:
        return config.max_total_num_tokens * full_pt * (nf + ns)


class TestDefaultConfigurator(unittest.TestCase):
    """Default (MHA): available_bytes -> tokens, memory invariant holds."""

    def _run(self, available_bytes, page_size=1, **kwargs):
        mr = _make_model_runner(page_size=page_size, **kwargs)
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available_bytes, page_size)
        return mr, cfg, config

    def test_memory_utilization(self):
        """Memory used should be <= available and within 1% of available."""
        available = 10_000_000
        mr, cfg, config = self._run(available)
        used = _actual_memory_used(mr, config)
        self.assertLessEqual(used, available)
        self.assertGreater(used, available * 0.99)

    def test_page_alignment(self):
        available = 10_000_000
        _, _, config = self._run(available, page_size=128)
        self.assertEqual(config.max_total_num_tokens % 128, 0)

    def test_constraint_respected(self):
        """calculate_pool_sizes_from_max_tokens respects the limit."""
        mr, cfg, config = self._run(10_000_000)
        with mock_cpu_env():
            constrained = cfg.calculate_pool_sizes_from_max_tokens(100, page_size=1)
        self.assertEqual(constrained.max_total_num_tokens, 100)

    def test_constraint_page_aligned(self):
        mr, cfg, _ = self._run(10_000_000, page_size=128)
        with mock_cpu_env():
            constrained = cfg.calculate_pool_sizes_from_max_tokens(1000, page_size=128)
        self.assertEqual(constrained.max_total_num_tokens, 896)  # 1000 // 128 * 128

    def test_no_swa_fields(self):
        _, _, config = self._run(10_000_000)
        self.assertIsNone(config.full_max_total_num_tokens)
        self.assertIsNone(config.swa_max_total_num_tokens)

    def test_grouped_int4_uses_k_and_v_metadata_cost(self):
        mr = _make_model_runner(
            kv_cache_dtype="int4",
            head_dim=64,
            v_head_dim=96,
            kv_cache_quant_group_size=32,
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)

        expected_cell_size = (
            mr.model_config.get_num_kv_heads(1)
            * _int_quant_per_token(64, 96, "int4", 32)
            * mr.num_effective_layers
        )
        self.assertEqual(cfg._cell_size, expected_cell_size)

    def test_grouped_int4_rejects_incompatible_head_dim(self):
        mr = _make_model_runner(
            kv_cache_dtype="int4",
            head_dim=96,
            v_head_dim=96,
            kv_cache_quant_group_size=64,
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            with self.assertRaisesRegex(ValueError, "head_dim"):
                create_memory_pool_configurator(mr)

    def test_grouped_asymmetric_kv_bytes(self):
        """Sanity: ``_get_int_kv_bytes_per_head_pair`` sums packed K+V bytes and
        per-group scale/zero overhead for int2. The default scale_dtype_bytes=4
        (fp32) matches the historical pool allocation; scale_dtype_bytes=2 (bf16)
        is opt-in via env. int2 head_dim=64, v_head_dim=96, group_size=32 with
        fp32 scales should be 16+24+2*4*(2+3)=80."""
        from sglang.srt.model_executor.pool_configurator import (
            _get_int_kv_bytes_per_head_pair,
        )

        # Default scale_dtype_bytes=4 (fp32).
        self.assertEqual(
            _get_int_kv_bytes_per_head_pair(64, 96, "int2", 32),
            16 + 24 + 2 * 4 * (2 + 3),
        )
        # No grouping collapses to one scale/zero pair per head.
        self.assertEqual(
            _get_int_kv_bytes_per_head_pair(64, 96, "int2", None),
            16 + 24 + 2 * 4 * (1 + 1),
        )
        # Symmetric K/V.
        self.assertEqual(
            _get_int_kv_bytes_per_head_pair(64, 64, "int2", 32),
            16 + 16 + 2 * 4 * (2 + 2),
        )
        # bf16 scales (scale_dtype_bytes=2): halves the scales/zeros overhead.
        self.assertEqual(
            _get_int_kv_bytes_per_head_pair(64, 64, "int2", 32, 2),
            16 + 16 + 2 * 2 * (2 + 2),
        )
        self.assertEqual(
            _get_int_kv_bytes_per_head_pair(64, 96, "int2", 32, 2),
            16 + 24 + 2 * 2 * (2 + 3),
        )

    def test_int2_cell_size_matches_pool_scale_dtype(self):
        """Invariant: for int2 non-mixed, the configurator's cell_size matches
        the pool's actual per-token byte footprint, including scales/zeros sized
        by SGLANG_MIXED_KV_SCALE_DTYPE. This is the regression fence for the
        pool/cost-model mismatch that over-provisioned max_total_num_tokens."""
        import os
        from unittest.mock import patch as _patch

        from sglang.srt.model_executor.pool_configurator import (
            _get_int_kv_bytes_per_head_pair,
            create_memory_pool_configurator,
        )

        num_kv_heads = 4
        head_dim = 64
        v_head_dim = 96
        num_layers = 8
        group_size = 32

        for scale_name, scale_bytes in (("float32", 4), ("bfloat16", 2)):
            mr = _make_model_runner(
                kv_cache_dtype="int2",
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                v_head_dim=v_head_dim,
                num_layers=num_layers,
                kv_cache_quant_group_size=group_size,
            )
            with mock_cpu_env(), _patch.dict(
                os.environ, {"SGLANG_MIXED_KV_SCALE_DTYPE": scale_name}, clear=False
            ):
                cfg = create_memory_pool_configurator(mr)

            expected_cell_size = (
                num_kv_heads
                * _get_int_kv_bytes_per_head_pair(
                    head_dim, v_head_dim, "int2", group_size, scale_bytes
                )
                * num_layers
            )
            self.assertEqual(
                cfg._cell_size,
                expected_cell_size,
                f"cell_size mismatch for scale_dtype={scale_name}",
            )


class TestHybridSWAConfigurator(unittest.TestCase):
    """Hybrid SWA: full/swa split, ratio, memory invariant."""

    def _make_swa_runner(self, full_layers=16, swa_layers=16, ratio=0.5, page_size=1):
        return _make_model_runner(
            is_hybrid_swa=True,
            full_attention_layer_ids=list(range(full_layers)),
            swa_attention_layer_ids=list(range(full_layers, full_layers + swa_layers)),
            swa_num_kv_heads=4,
            page_size=page_size,
            swa_full_tokens_ratio=ratio,
        )

    def _run(self, available_bytes, **kwargs):
        mr = self._make_swa_runner(**kwargs)
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available_bytes, mr.server_args.page_size)
        return mr, cfg, config

    def test_memory_utilization(self):
        """Memory used should be <= available and within 1% of available."""
        available = 10_000_000
        mr, _, config = self._run(available)
        used = _actual_memory_used(mr, config)
        self.assertLessEqual(used, available)
        self.assertGreater(used, available * 0.99)

    def test_ratio_respected(self):
        """swa_tokens ~= full_tokens * ratio (within page alignment)"""
        available = 10_000_000
        for ratio in [0.25, 0.5, 0.75, 1.0]:
            mr, _, config = self._run(available, ratio=ratio, page_size=1)
            full = config.full_max_total_num_tokens
            swa = config.swa_max_total_num_tokens
            self.assertEqual(swa, int(full * ratio), f"ratio={ratio}")

    def test_ratio_with_page_alignment(self):
        """With page alignment, swa_tokens = align(full_tokens * ratio)"""
        available = 10_000_000
        mr, _, config = self._run(available, ratio=0.5, page_size=128)
        full = config.full_max_total_num_tokens
        swa = config.swa_max_total_num_tokens
        self.assertEqual(full % 128, 0)
        self.assertEqual(swa % 128, 0)
        self.assertEqual(swa, (int(full * 0.5) // 128) * 128)

    def test_max_total_equals_full(self):
        """For hybrid, max_total_num_tokens = full_max_total_num_tokens"""
        _, _, config = self._run(10_000_000)
        self.assertEqual(config.max_total_num_tokens, config.full_max_total_num_tokens)

    def test_constraint_respected(self):
        """full_tokens = constrained value after re-run"""
        mr, cfg, _ = self._run(10_000_000, page_size=1)
        with mock_cpu_env():
            config = cfg.calculate_pool_sizes_from_max_tokens(200, page_size=1)
        self.assertEqual(config.full_max_total_num_tokens, 200)
        self.assertEqual(config.swa_max_total_num_tokens, 100)

    def test_constraint_memory_within_budget(self):
        """After constraint, memory <= original budget (but less than profiled due to constraint)."""
        available = 10_000_000
        mr, cfg, original = self._run(available, page_size=1)
        user_limit = original.full_max_total_num_tokens // 2
        with mock_cpu_env():
            config = cfg.calculate_pool_sizes_from_max_tokens(
                user_limit, mr.server_args.page_size
            )
        used = _actual_memory_used(mr, config)
        self.assertLessEqual(used, available)
        # constrained should use roughly half the memory
        original_used = _actual_memory_used(mr, original)
        self.assertAlmostEqual(used / original_used, 0.5, delta=0.01)

    def test_different_layer_counts(self):
        """Asymmetric full/swa layer counts"""
        available = 10_000_000
        mr, _, config = self._run(available, full_layers=24, swa_layers=8, ratio=0.5)
        used = _actual_memory_used(mr, config)
        self.assertLessEqual(used, available)
        self.assertEqual(
            config.swa_max_total_num_tokens,
            int(config.full_max_total_num_tokens * 0.5),
        )

    def test_grouped_int2_rejects_hybrid_swa(self):
        mr = _make_model_runner(
            is_hybrid_swa=True,
            full_attention_layer_ids=list(range(8)),
            swa_attention_layer_ids=list(range(8, 16)),
            swa_num_kv_heads=4,
            head_dim=64,
            v_head_dim=96,
            swa_head_dim=32,
            swa_v_head_dim=64,
            kv_cache_dtype="int2",
            kv_cache_quant_group_size=32,
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            with self.assertRaisesRegex(ValueError, "hybrid SWA"):
                create_memory_pool_configurator(mr)


class TestAllSWAConfigurator(unittest.TestCase):
    """All-SWA (full_layers=0): special case."""

    def _run(self, available_bytes, ratio=0.5, page_size=1):
        mr = _make_model_runner(
            is_hybrid_swa=True,
            full_attention_layer_ids=[],
            swa_attention_layer_ids=list(range(32)),
            swa_num_kv_heads=4,
            swa_full_tokens_ratio=ratio,
            page_size=page_size,
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available_bytes, page_size)
        return mr, cfg, config

    def test_full_max_is_zero(self):
        _, _, config = self._run(10_000_000)
        self.assertEqual(config.full_max_total_num_tokens, 0)

    def test_max_total_equals_swa(self):
        _, _, config = self._run(10_000_000)
        self.assertEqual(config.max_total_num_tokens, config.swa_max_total_num_tokens)

    def test_memory_utilization(self):
        """Memory used should be <= available and within 1% of available."""
        available = 10_000_000
        mr, _, config = self._run(available)
        swa_pt = _swa_per_token(mr)
        ns = len(mr.model_config.swa_attention_layer_ids)
        used = config.swa_max_total_num_tokens * swa_pt * ns
        self.assertLessEqual(used, available)
        self.assertGreater(used, available * 0.99)

    def test_constraint_respected(self):
        mr, cfg, _ = self._run(10_000_000, page_size=1)
        with mock_cpu_env():
            config = cfg.calculate_pool_sizes_from_max_tokens(500, page_size=1)
        self.assertEqual(config.max_total_num_tokens, 500)
        self.assertEqual(config.swa_max_total_num_tokens, 500)


class TestFactory(unittest.TestCase):
    def test_default_for_non_swa(self):
        mr = _make_model_runner(is_hybrid_swa=False)
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                DefaultPoolConfigurator,
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
        self.assertIsInstance(cfg, DefaultPoolConfigurator)

    def test_swa_for_hybrid(self):
        mr = _make_model_runner(
            is_hybrid_swa=True,
            full_attention_layer_ids=list(range(16)),
            swa_attention_layer_ids=list(range(16, 32)),
            swa_num_kv_heads=4,
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                HybridSWAPoolConfigurator,
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
        self.assertIsInstance(cfg, HybridSWAPoolConfigurator)


if __name__ == "__main__":
    unittest.main()
