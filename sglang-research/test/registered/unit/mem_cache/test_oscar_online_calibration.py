from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.entrypoints.warmup import (
    _load_oscar_calibration_messages,
    _pack_oscar_prompt_batches,
    _render_oscar_prompt_ids,
    _resolve_oscar_calibration_prompts_path,
)
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.mem_cache.memory_pool import (
    determine_oscar_calibration_required,
    ensure_oscar_rotation_paths,
    get_oscar_pair_artifact_paths,
    load_oscar_rotations,
    oscar_calibration_required,
)
from sglang.srt.mem_cache.oscar_calibration import (
    OscarOnlineCalibrator,
    build_hadamard,
)
from sglang.srt.mem_cache.unified_kv_pool import UnifiedInt2HPKVPool


class _FakePool:
    device = "cpu"
    start_layer = 0
    layer_num = 1
    head_num = 2
    head_dim = 4
    v_head_dim = 4


class OscarOnlineMathTest(unittest.TestCase):
    @patch(
        "sglang.srt.mem_cache.oscar_calibration.get_attention_tp_size",
        return_value=1,
    )
    def test_chunked_qqt_sst_matches_direct_reference(self, _):
        torch.manual_seed(3)
        q = torch.randn(5, 4, 4, dtype=torch.float64)
        k = torch.randn(5, 2, 4, dtype=torch.float64)
        v = torch.randn(5, 2, 4, dtype=torch.float64)

        with envs.SGLANG_OSCAR_CALIBRATION_TOKENS.override(5):
            calibrator = OscarOnlineCalibrator(
                pool=_FakePool(),
                total_q_heads=4,
                total_kv_heads=2,
                model_path="test",
            )
        calibrator.start(prompt_sha256="a" * 64)
        calibrator.observe(layer_id=0, q=q[:2], k=k[:2], v=v[:2])
        calibrator.observe(layer_id=0, q=q[2:], k=k[2:], v=v[2:])

        covariance_sums = calibrator._local_covariance_sums()
        q_covariances = []
        v_covariances = []
        for head in range(2):
            q_group = q[:, head * 2 : (head + 1) * 2].reshape(-1, 4)
            q_cov = q_group.T @ q_group / q_group.shape[0]
            q_covariances.append(q_cov)
            energy = torch.einsum("td,de,te->t", k[:, head], q_cov, k[:, head])
            v_covariances.append(
                torch.einsum("t,td,te->de", energy, v[:, head], v[:, head])
                / energy.sum().clamp_min(1e-12)
            )

        torch.testing.assert_close(
            covariance_sums[0, 0], torch.stack(q_covariances).sum(0)
        )
        torch.testing.assert_close(
            covariance_sums[1, 0], torch.stack(v_covariances).sum(0)
        )

    def test_hadamard_is_orthogonal(self):
        h = build_hadamard(8)
        torch.testing.assert_close(h @ h.T, torch.eye(8, dtype=torch.float64))

    @patch(
        "sglang.srt.mem_cache.oscar_calibration.get_attention_tp_size",
        return_value=4,
    )
    def test_replicated_kv_heads_fail_explicitly(self, _):
        pool = SimpleNamespace(
            device="cpu",
            start_layer=0,
            layer_num=1,
            head_num=1,
            head_dim=128,
            v_head_dim=128,
        )
        with (
            envs.SGLANG_OSCAR_CALIBRATION_TOKENS.override(5),
            self.assertRaisesRegex(ValueError, "replicated KV heads"),
        ):
            OscarOnlineCalibrator(
                pool=pool,
                total_q_heads=4,
                total_kv_heads=2,
                model_path="test",
            )

    @patch(
        "sglang.srt.mem_cache.oscar_calibration.get_attention_tp_size",
        return_value=1,
    )
    def test_mismatched_kv_dimensions_fail_explicitly(self, _):
        pool = SimpleNamespace(
            device="cpu",
            start_layer=0,
            layer_num=1,
            head_num=2,
            head_dim=128,
            v_head_dim=64,
        )
        with (
            envs.SGLANG_OSCAR_CALIBRATION_TOKENS.override(5),
            self.assertRaisesRegex(ValueError, "equal K/V head dimensions"),
        ):
            OscarOnlineCalibrator(
                pool=pool,
                total_q_heads=4,
                total_kv_heads=2,
                model_path="test",
            )

    @patch(
        "sglang.srt.mem_cache.oscar_calibration.get_attention_tp_size",
        return_value=1,
    )
    def test_non_power_of_two_head_dim_fails_explicitly(self, _):
        pool = SimpleNamespace(
            device="cpu",
            start_layer=0,
            layer_num=1,
            head_num=2,
            head_dim=96,
            v_head_dim=96,
        )
        with (
            envs.SGLANG_OSCAR_CALIBRATION_TOKENS.override(5),
            self.assertRaisesRegex(ValueError, "power-of-two"),
        ):
            OscarOnlineCalibrator(
                pool=pool,
                total_q_heads=4,
                total_kv_heads=2,
                model_path="test",
            )


class OscarIdentityLifecycleTest(unittest.TestCase):
    def test_default_rotation_paths_are_model_scoped_and_deterministic(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"HF_HOME": tmp}, clear=False),
            envs.SGLANG_OSCAR_K_ROTATION_PATH.override(""),
            envs.SGLANG_OSCAR_V_ROTATION_PATH.override(""),
        ):
            first = ensure_oscar_rotation_paths("Qwen/Qwen3-8B", "v1")
            second = ensure_oscar_rotation_paths("Qwen/Qwen3-8B", "v1")
            self.assertTrue(determine_oscar_calibration_required())
        self.assertEqual(first, second)
        self.assertNotEqual(first[0], first[1])
        self.assertIn("oscar-rotations", first[0])
        self.assertTrue(first[0].endswith("k_rotation_qqt_r_h_pbr.pt"))

    def test_auto_missing_paths_load_identity(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            envs.SGLANG_OSCAR_CALIBRATION_ACTIVE.override(True),
            envs.SGLANG_OSCAR_K_ROTATION_PATH.override(f"{tmp}/k.pt"),
            envs.SGLANG_OSCAR_V_ROTATION_PATH.override(f"{tmp}/v.pt"),
        ):
            rotations = load_oscar_rotations(
                f"{tmp}/k.pt",
                layer_num=2,
                start_layer=3,
                head_dim=4,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        self.assertTrue(rotations.is_contiguous())
        torch.testing.assert_close(rotations, torch.eye(4).repeat(2, 1, 1))

    def test_missing_path_when_calibration_inactive_still_fails(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            envs.SGLANG_OSCAR_CALIBRATION_ACTIVE.override(False),
            self.assertRaises(FileNotFoundError),
        ):
            load_oscar_rotations(
                f"{tmp}/missing.pt",
                layer_num=1,
                start_layer=0,
                head_dim=4,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

    def test_same_kv_destination_is_rejected(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            envs.SGLANG_OSCAR_K_ROTATION_PATH.override(f"{tmp}/same.pt"),
            envs.SGLANG_OSCAR_V_ROTATION_PATH.override(f"{tmp}/same.pt"),
            self.assertRaises(ValueError),
        ):
            determine_oscar_calibration_required()

    def test_transaction_markers_are_pair_specific(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            envs.SGLANG_OSCAR_CALIBRATION_LOCK_DIR.override(f"{tmp}/runtime-locks"),
        ):
            with (
                envs.SGLANG_OSCAR_K_ROTATION_PATH.override(f"{tmp}/k1.pt"),
                envs.SGLANG_OSCAR_V_ROTATION_PATH.override(f"{tmp}/v1.pt"),
            ):
                first_artifacts = get_oscar_pair_artifact_paths()
            with (
                envs.SGLANG_OSCAR_K_ROTATION_PATH.override(f"{tmp}/k2.pt"),
                envs.SGLANG_OSCAR_V_ROTATION_PATH.override(f"{tmp}/v2.pt"),
            ):
                second_artifacts = get_oscar_pair_artifact_paths()
        self.assertNotEqual(first_artifacts["pending"], second_artifacts["pending"])
        self.assertEqual(
            Path(first_artifacts["lock"]).parent,
            Path(tmp) / "runtime-locks",
        )
        self.assertEqual(Path(first_artifacts["pending"]).parent, Path(tmp))

    def test_active_snapshot_does_not_recheck_files(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            envs.SGLANG_OSCAR_CALIBRATION_ACTIVE.override(True),
            envs.SGLANG_OSCAR_K_ROTATION_PATH.override(f"{tmp}/k.pt"),
            envs.SGLANG_OSCAR_V_ROTATION_PATH.override(f"{tmp}/v.pt"),
        ):
            Path(f"{tmp}/k.pt").touch()
            Path(f"{tmp}/v.pt").touch()
            self.assertTrue(oscar_calibration_required())

    def test_pair_update_preserves_addresses_and_is_atomic(self):
        fake = _FakePool()
        fake._R_k = torch.eye(4).repeat(2, 1, 1)
        fake._R_v = torch.eye(4).repeat(2, 1, 1)
        fake._oscar_rotation_version = 0
        fake._oscar_calibration_pending = True
        k_ptr = fake._R_k.data_ptr()
        v_ptr = fake._R_v.data_ptr()
        q, _ = torch.linalg.qr(torch.randn(4, 4))
        k = q.repeat(2, 1, 1)
        v = q.T.repeat(2, 1, 1)

        UnifiedInt2HPKVPool.update_oscar_rotations_(fake, k, v)
        self.assertEqual(fake._R_k.data_ptr(), k_ptr)
        self.assertEqual(fake._R_v.data_ptr(), v_ptr)
        torch.testing.assert_close(fake._R_k, k)
        torch.testing.assert_close(fake._R_v, v)

        old_k = fake._R_k.clone()
        with self.assertRaises(ValueError):
            UnifiedInt2HPKVPool.update_oscar_rotations_(
                fake, torch.eye(4).repeat(2, 1, 1), torch.eye(3)
            )
        torch.testing.assert_close(fake._R_k, old_k)


class OscarPromptInputTest(unittest.TestCase):
    def test_official_gpqa_csv_is_materialized_as_cached_jsonl(self):
        fields = [
            "Question",
            "Correct Answer",
            "Incorrect Answer 1",
            "Incorrect Answer 2",
            "Incorrect Answer 3",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "gpqa_diamond.csv"
            with csv_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "Question": "Question?",
                        "Correct Answer": "Correct",
                        "Incorrect Answer 1": "Wrong 1",
                        "Incorrect Answer 2": "Wrong 2",
                        "Incorrect Answer 3": "Wrong 3",
                    }
                )
            with patch.dict(os.environ, {"HF_HOME": f"{tmp}/hf-cache"}, clear=False):
                messages = _load_oscar_calibration_messages(str(csv_path))
            cached = list(
                (Path(tmp) / "hf-cache" / "oscar-calibration").glob("*.jsonl")
            )
        self.assertEqual(len(messages), 1)
        self.assertEqual(len(cached), 1)
        self.assertEqual(messages[0][0]["role"], "user")

    def test_default_prompt_source_downloads_once_into_local_cache(self):
        csv_source = (
            b"Question,Correct Answer,Incorrect Answer 1,"
            b"Incorrect Answer 2,Incorrect Answer 3\n"
            b"Question?,Correct,Wrong 1,Wrong 2,Wrong 3\n"
        )
        response = MagicMock()
        response.__enter__.return_value.read.return_value = csv_source
        with (
            tempfile.TemporaryDirectory() as tmp,
            envs.SGLANG_OSCAR_CALIBRATION_PROMPTS_PATH.override(""),
            patch.dict(os.environ, {"HF_HOME": f"{tmp}/hf-cache"}, clear=False),
            patch("urllib.request.urlopen", return_value=response) as download,
        ):
            resolved = _resolve_oscar_calibration_prompts_path()
            second = _resolve_oscar_calibration_prompts_path()
        self.assertTrue(resolved.endswith(".jsonl"))
        self.assertEqual(resolved, second)
        download.assert_called_once()

    def test_local_prompt_path_takes_precedence(self):
        with envs.SGLANG_OSCAR_CALIBRATION_PROMPTS_PATH.override(
            "/local/prompts.jsonl"
        ):
            resolved = _resolve_oscar_calibration_prompts_path()
        self.assertEqual(resolved, "/local/prompts.jsonl")

    def test_jsonl_render_keeps_prompts_whole_under_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.jsonl"
            with path.open("w") as handle:
                handle.write(
                    json.dumps({"messages": [{"role": "user", "content": "one"}]})
                    + "\n"
                )
                handle.write(
                    json.dumps({"messages": [{"role": "user", "content": "two"}]})
                    + "\n"
                )

            messages = _load_oscar_calibration_messages(str(path))

        class _Tokenizer:
            def apply_chat_template(self, prompt, **_kwargs):
                return [1, 2, 3, 4]

        class _Manager:
            tokenizer = _Tokenizer()

        ids = _render_oscar_prompt_ids(_Manager(), messages, max_token_budget=6)
        self.assertEqual(ids, [[1, 2, 3, 4]])
        all_ids = _render_oscar_prompt_ids(_Manager(), messages, max_token_budget=10)
        self.assertEqual(all_ids, [[1, 2, 3, 4], [1, 2, 3, 4]])

    def test_batch_packing_respects_page_rounded_chunk_budget(self):
        prompts = [[1] * 5, [2] * 5, [3] * 5]
        batches = _pack_oscar_prompt_batches(
            prompts,
            max_batch_size=3,
            chunked_prefill_size=16,
            page_size=4,
        )
        self.assertEqual([len(batch) for batch in batches], [2, 1])

    def test_batch_extra_keys_are_indexed_and_parallel_expanded(self):
        request = GenerateReqInput(
            input_ids=[[1], [2]],
            sampling_params={"n": 2},
            extra_key=["first", "second"],
        )
        request.normalize_batch_and_arguments()
        self.assertEqual(
            [request[i].extra_key for i in range(4)],
            ["first", "second", "first", "second"],
        )

    def test_batch_extra_key_length_is_validated(self):
        request = GenerateReqInput(
            input_ids=[[1], [2]],
            sampling_params={},
            extra_key=["only-one"],
        )
        with self.assertRaises(ValueError):
            request.normalize_batch_and_arguments()


if __name__ == "__main__":
    unittest.main()
