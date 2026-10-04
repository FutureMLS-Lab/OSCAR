"""CPU checks for startup OSCAR calibration: the exact qqt/sst moments, the
pair-path lifecycle (default destinations, identity fallback, pending
markers), in-place rotation installation, and prompt rendering/packing."""
import csv
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.entrypoints.oscar_startup_calibration import (
    load_oscar_calibration_messages,
    pack_oscar_prompt_batches,
    render_oscar_prompt_ids,
    resolve_oscar_calibration_prompts_path,
)
from sglang.srt.environ import envs
from sglang.srt.mem_cache.memory_pool import load_oscar_rotations
from sglang.srt.mem_cache.oscar_calibration import (
    OscarOnlineCalibrator,
    build_hadamard,
    get_active_oscar_calibrator,
    set_active_oscar_calibrator,
)
from sglang.srt.mem_cache.oscar_rotation_paths import (
    determine_oscar_calibration_required,
    ensure_oscar_rotation_paths,
    get_oscar_pair_artifact_paths,
    oscar_calibration_required,
)
from sglang.srt.mem_cache.unified_kv_pool import UnifiedInt2HPKVPool


def _calibrator(**overrides):
    kwargs = dict(
        layer_ids=[0],
        local_kv_heads=2,
        head_dim=4,
        v_head_dim=4,
        device="cpu",
        total_q_heads=4,
        total_kv_heads=2,
        tp_size=1,
        tp_rank=0,
        model_path="test",
    )
    kwargs.update(overrides)
    return OscarOnlineCalibrator(**kwargs)


def _assert_raises(exc_type, pattern, fn):
    try:
        fn()
    except exc_type as exc:
        assert pattern in str(exc), f"{pattern!r} not in {exc}"
        return
    raise AssertionError(f"expected {exc_type.__name__} matching {pattern!r}")


# -- Math ---------------------------------------------------------------------


def test_chunked_qqt_sst_matches_direct_reference():
    torch.manual_seed(3)
    q = torch.randn(5, 4, 4, dtype=torch.float64)
    k = torch.randn(5, 2, 4, dtype=torch.float64)
    v = torch.randn(5, 2, 4, dtype=torch.float64)
    with envs.SGLANG_OSCAR_CALIBRATION_TOKENS.override(5):
        calibrator = _calibrator()
    calibrator.start(prompt_sha256="a" * 64)
    calibrator.observe(layer_id=0, q=q[:2], k=k[:2], v=v[:2])
    calibrator.observe(layer_id=0, q=q[2:], k=k[2:], v=v[2:])
    assert calibrator.complete
    sums = calibrator.local_covariance_sums()

    q_covs, v_covs = [], []
    for head in range(2):
        q_group = q[:, head * 2 : (head + 1) * 2].reshape(-1, 4)
        q_cov = q_group.T @ q_group / q_group.shape[0]
        q_covs.append(q_cov)
        energy = torch.einsum("td,de,te->t", k[:, head], q_cov, k[:, head])
        v_covs.append(
            torch.einsum("t,td,te->de", energy, v[:, head], v[:, head])
            / energy.sum().clamp_min(1e-12)
        )
    torch.testing.assert_close(sums[0, 0], torch.stack(q_covs).sum(0))
    torch.testing.assert_close(sums[1, 0], torch.stack(v_covs).sum(0))


def test_finalize_produces_orthogonal_rotations_in_layer_order():
    torch.manual_seed(5)
    with envs.SGLANG_OSCAR_CALIBRATION_TOKENS.override(64):
        calibrator = _calibrator(layer_ids=[3, 7], head_dim=8, v_head_dim=8)
    calibrator.start(prompt_sha256="b" * 64)
    for layer_id in (3, 7):
        calibrator.observe(
            layer_id=layer_id,
            q=torch.randn(64, 4, 8),
            k=torch.randn(64, 2, 8),
            v=torch.randn(64, 2, 8),
        )
    result = calibrator.finalize(
        covariance_sums=calibrator.local_covariance_sums(),
        buffers=calibrator.allocate_result_buffers(),
    )
    calibrator.broadcast_result(result)
    assert calibrator.state == "finalized"
    eye = torch.eye(8)
    for stack in (result.k_rotations, result.v_rotations):
        assert stack.shape == (2, 8, 8)
        for i in range(2):
            torch.testing.assert_close(stack[i] @ stack[i].T, eye, atol=1e-4, rtol=0)
    assert list(result.k_state["layers"].keys()) == [3, 7]
    assert result.k_state["objective"] == "qqt_r_h_pbr"
    assert result.v_state["objective"] == "sst_r_h_pbr"
    assert result.generation_id


def test_hadamard_is_orthogonal():
    h = build_hadamard(8)
    torch.testing.assert_close(h @ h.T, torch.eye(8, dtype=torch.float64))


def test_geometry_errors_are_explicit():
    with envs.SGLANG_OSCAR_CALIBRATION_TOKENS.override(5):
        # Fewer KV heads than TP ranks replicate each head over tp/kv ranks;
        # the calibrator accepts that and knows the replication factor.
        replicated = _calibrator(local_kv_heads=1, head_dim=128, v_head_dim=128, tp_size=4, tp_rank=3)
        assert replicated.kv_replication == 2
        _assert_raises(
            ValueError,
            "multiple of the global KV heads",
            lambda: _calibrator(local_kv_heads=1, head_dim=128, v_head_dim=128, tp_size=3, total_kv_heads=2),
        )
        _assert_raises(
            ValueError,
            "equal K/V head dimensions",
            lambda: _calibrator(head_dim=128, v_head_dim=64),
        )
        _assert_raises(
            ValueError, "power-of-two", lambda: _calibrator(head_dim=96, v_head_dim=96)
        )


def test_observe_ignores_unknown_layers_and_stops_at_budget():
    with envs.SGLANG_OSCAR_CALIBRATION_TOKENS.override(3):
        calibrator = _calibrator()
    calibrator.start(prompt_sha256="c" * 64)
    calibrator.observe(layer_id=9, q=torch.randn(2, 4, 4), k=torch.randn(2, 2, 4), v=torch.randn(2, 2, 4))
    assert calibrator.min_captured_tokens() == 0
    calibrator.observe(layer_id=0, q=torch.randn(5, 4, 4), k=torch.randn(5, 2, 4), v=torch.randn(5, 2, 4))
    assert calibrator.min_captured_tokens() == 3
    assert calibrator.complete


# -- Pair lifecycle -----------------------------------------------------------


def test_default_rotation_paths_are_model_scoped_and_deterministic():
    with (
        tempfile.TemporaryDirectory() as tmp,
        patch.dict(os.environ, {"HF_HOME": tmp}, clear=False),
        envs.SGLANG_OSCAR_K_ROTATION_PATH.override(""),
        envs.SGLANG_OSCAR_V_ROTATION_PATH.override(""),
    ):
        first = ensure_oscar_rotation_paths(model_path="Qwen/Qwen3-8B", revision="v1")
        second = ensure_oscar_rotation_paths(model_path="Qwen/Qwen3-8B", revision="v1")
        assert determine_oscar_calibration_required()
    assert first == second
    assert first[0] != first[1]
    assert "oscar-rotations" in first[0]
    assert first[0].endswith("k_rotation_qqt_r_h_pbr.pt")


def test_auto_missing_paths_load_identity_for_contiguous_and_listed_layers():
    with (
        tempfile.TemporaryDirectory() as tmp,
        envs.SGLANG_OSCAR_CALIBRATION_ACTIVE.override(True),
        envs.SGLANG_OSCAR_K_ROTATION_PATH.override(f"{tmp}/k.pt"),
        envs.SGLANG_OSCAR_V_ROTATION_PATH.override(f"{tmp}/v.pt"),
    ):
        rotations = load_oscar_rotations(
            f"{tmp}/k.pt", layer_num=2, start_layer=3, head_dim=4,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        listed = load_oscar_rotations(
            f"{tmp}/k.pt", layer_num=3, start_layer=0, head_dim=4,
            device=torch.device("cpu"), dtype=torch.float32, layer_ids=[3, 7, 11],
        )
        _assert_raises(
            ValueError,
            "one head_dim",
            lambda: load_oscar_rotations(
                f"{tmp}/k.pt", layer_num=2, start_layer=0, head_dim=[4, 8],
                device=torch.device("cpu"), dtype=torch.float32,
            ),
        )
    assert rotations.is_contiguous()
    torch.testing.assert_close(rotations, torch.eye(4).repeat(2, 1, 1))
    torch.testing.assert_close(listed, torch.eye(4).repeat(3, 1, 1))


def test_missing_path_when_calibration_inactive_still_fails():
    with (
        tempfile.TemporaryDirectory() as tmp,
        envs.SGLANG_OSCAR_CALIBRATION_ACTIVE.override(False),
    ):
        try:
            load_oscar_rotations(
                f"{tmp}/missing.pt", layer_num=1, start_layer=0, head_dim=4,
                device=torch.device("cpu"), dtype=torch.float32,
            )
        except FileNotFoundError:
            return
    raise AssertionError("missing checkpoint must fail without calibration")


def test_pending_marker_blocks_loading_a_half_published_pair():
    with (
        tempfile.TemporaryDirectory() as tmp,
        envs.SGLANG_OSCAR_CALIBRATION_ACTIVE.override(False),
        envs.SGLANG_OSCAR_K_ROTATION_PATH.override(f"{tmp}/k.pt"),
        envs.SGLANG_OSCAR_V_ROTATION_PATH.override(f"{tmp}/v.pt"),
        envs.SGLANG_OSCAR_CALIBRATION_LOCK_DIR.override(f"{tmp}/locks"),
    ):
        state = {"layers": {0: {"rotation": torch.eye(4)}}}
        torch.save(state, f"{tmp}/k.pt")
        torch.save(state, f"{tmp}/v.pt")
        assert not determine_oscar_calibration_required()
        Path(get_oscar_pair_artifact_paths()["pending"]).write_text("{}")
        assert determine_oscar_calibration_required()
        _assert_raises(
            RuntimeError,
            "interrupted",
            lambda: load_oscar_rotations(
                f"{tmp}/k.pt", layer_num=1, start_layer=0, head_dim=4,
                device=torch.device("cpu"), dtype=torch.float32,
            ),
        )


def test_same_kv_destination_is_rejected():
    with (
        tempfile.TemporaryDirectory() as tmp,
        envs.SGLANG_OSCAR_K_ROTATION_PATH.override(f"{tmp}/same.pt"),
        envs.SGLANG_OSCAR_V_ROTATION_PATH.override(f"{tmp}/same.pt"),
    ):
        _assert_raises(ValueError, "distinct", determine_oscar_calibration_required)


def test_transaction_markers_are_pair_specific():
    with (
        tempfile.TemporaryDirectory() as tmp,
        envs.SGLANG_OSCAR_CALIBRATION_LOCK_DIR.override(f"{tmp}/runtime-locks"),
    ):
        with (
            envs.SGLANG_OSCAR_K_ROTATION_PATH.override(f"{tmp}/k1.pt"),
            envs.SGLANG_OSCAR_V_ROTATION_PATH.override(f"{tmp}/v1.pt"),
        ):
            first = get_oscar_pair_artifact_paths()
        with (
            envs.SGLANG_OSCAR_K_ROTATION_PATH.override(f"{tmp}/k2.pt"),
            envs.SGLANG_OSCAR_V_ROTATION_PATH.override(f"{tmp}/v2.pt"),
        ):
            second = get_oscar_pair_artifact_paths()
    assert first["pending"] != second["pending"]
    assert Path(first["lock"]).parent == Path(tmp) / "runtime-locks"
    assert Path(first["pending"]).parent == Path(tmp)


def test_active_snapshot_does_not_recheck_files():
    with (
        tempfile.TemporaryDirectory() as tmp,
        envs.SGLANG_OSCAR_CALIBRATION_ACTIVE.override(True),
        envs.SGLANG_OSCAR_K_ROTATION_PATH.override(f"{tmp}/k.pt"),
        envs.SGLANG_OSCAR_V_ROTATION_PATH.override(f"{tmp}/v.pt"),
    ):
        Path(f"{tmp}/k.pt").touch()
        Path(f"{tmp}/v.pt").touch()
        assert oscar_calibration_required()


class _FakePool:
    """Only what update_oscar_rotations_ and the registry touch."""

    def __init__(self):
        self._R_k = torch.eye(4).repeat(2, 1, 1)
        self._R_v = torch.eye(4).repeat(2, 1, 1)
        self._oscar_rotation_version = 0
        self._oscar_calibration_pending = True
        self._oscar_calibrator = None
        self._layer_groups = None


def test_pair_update_preserves_addresses_and_is_atomic():
    fake = _FakePool()
    k_ptr, v_ptr = fake._R_k.data_ptr(), fake._R_v.data_ptr()
    q, _ = torch.linalg.qr(torch.randn(4, 4))
    k, v = q.repeat(2, 1, 1), q.T.repeat(2, 1, 1)

    UnifiedInt2HPKVPool.update_oscar_rotations_(fake, k, v)
    assert fake._R_k.data_ptr() == k_ptr and fake._R_v.data_ptr() == v_ptr
    torch.testing.assert_close(fake._R_k, k)
    torch.testing.assert_close(fake._R_v, v)
    assert fake._oscar_rotation_version == 1 and not fake._oscar_calibration_pending

    old_k = fake._R_k.clone()
    _assert_raises(
        ValueError,
        "does not match",
        lambda: UnifiedInt2HPKVPool.update_oscar_rotations_(
            fake, torch.eye(4).repeat(2, 1, 1), torch.eye(3)
        ),
    )
    torch.testing.assert_close(fake._R_k, old_k)
    _assert_raises(
        ValueError,
        "not orthogonal",
        lambda: UnifiedInt2HPKVPool.update_oscar_rotations_(
            fake, torch.eye(4).repeat(2, 1, 1) * 2, torch.eye(4).repeat(2, 1, 1)
        ),
    )


def test_attach_and_detach_drive_the_process_registry():
    fake = _FakePool()
    with envs.SGLANG_OSCAR_CALIBRATION_TOKENS.override(5):
        calibrator = _calibrator()
    UnifiedInt2HPKVPool.attach_oscar_calibrator(fake, calibrator)
    assert get_active_oscar_calibrator() is calibrator
    assert fake._oscar_calibrator is calibrator
    UnifiedInt2HPKVPool.detach_oscar_calibrator(fake)
    assert get_active_oscar_calibrator() is None
    fake._oscar_calibration_pending = False
    _assert_raises(
        RuntimeError,
        "no checkpoint",
        lambda: UnifiedInt2HPKVPool.attach_oscar_calibrator(fake, calibrator),
    )
    set_active_oscar_calibrator(None)


# -- Prompt source ------------------------------------------------------------

_GPQA_FIELDS = [
    "Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3",
]


def test_official_gpqa_csv_is_materialized_as_cached_jsonl():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "gpqa_diamond.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_GPQA_FIELDS)
            writer.writeheader()
            writer.writerow(dict(zip(_GPQA_FIELDS, ["Question?", "Correct", "W1", "W2", "W3"])))
        with patch.dict(os.environ, {"HF_HOME": f"{tmp}/hf-cache"}, clear=False):
            messages = load_oscar_calibration_messages(str(csv_path))
        cached = list((Path(tmp) / "hf-cache" / "oscar-calibration").glob("*.jsonl"))
    assert len(messages) == 1 and len(cached) == 1
    assert messages[0][0]["role"] == "user"
    assert "A) " in messages[0][0]["content"]


def test_default_prompt_source_downloads_once_into_local_cache():
    csv_source = (
        b"Question,Correct Answer,Incorrect Answer 1,Incorrect Answer 2,Incorrect Answer 3\n"
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
        resolved = resolve_oscar_calibration_prompts_path()
        second = resolve_oscar_calibration_prompts_path()
    assert resolved.endswith(".jsonl") and resolved == second
    download.assert_called_once()


def test_local_prompt_path_takes_precedence():
    with envs.SGLANG_OSCAR_CALIBRATION_PROMPTS_PATH.override("/local/prompts.jsonl"):
        assert resolve_oscar_calibration_prompts_path() == "/local/prompts.jsonl"


def test_jsonl_render_keeps_prompts_whole_under_budget():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "prompts.jsonl"
        with path.open("w") as handle:
            for text in ("one", "two"):
                handle.write(json.dumps({"messages": [{"role": "user", "content": text}]}) + "\n")
        messages = load_oscar_calibration_messages(str(path))

    class _Tokenizer:
        def apply_chat_template(self, prompt, **_kwargs):
            return [1, 2, 3, 4]

    assert render_oscar_prompt_ids(_Tokenizer(), messages, max_token_budget=6) == [[1, 2, 3, 4]]
    assert render_oscar_prompt_ids(_Tokenizer(), messages, max_token_budget=10) == [
        [1, 2, 3, 4], [1, 2, 3, 4],
    ]


def test_batch_packing_respects_page_rounded_chunk_budget():
    batches = pack_oscar_prompt_batches(
        [[1] * 5, [2] * 5, [3] * 5], max_batch_size=3, chunked_prefill_size=16, page_size=4
    )
    assert [len(batch) for batch in batches] == [2, 1]
    _assert_raises(
        ValueError,
        "one page-rounded",
        lambda: pack_oscar_prompt_batches(
            [[1] * 20], max_batch_size=3, chunked_prefill_size=16, page_size=4
        ),
    )


def test_save_rows_writes_every_token_and_the_sampled_queries():
    torch.manual_seed(7)
    with envs.SGLANG_OSCAR_CALIBRATION_TOKENS.override(70):
        calibrator = _calibrator(layer_ids=[2], head_dim=8, v_head_dim=8)
    calibrator.start(prompt_sha256="c" * 64)
    q = torch.randn(70, 4, 8)
    k = torch.randn(70, 2, 8)
    v = torch.randn(70, 2, 8)
    positions = torch.cat([torch.arange(40), torch.arange(30)])
    calibrator.observe(layer_id=2, q=q[:50], k=k[:50], v=v[:50], positions=positions[:50], scaling=0.25)
    calibrator.observe(layer_id=2, q=q[50:], k=k[50:], v=v[50:], positions=positions[50:], scaling=0.25)
    assert calibrator.complete
    with tempfile.TemporaryDirectory() as d:
        path = calibrator.save_rows(Path(d))
        assert os.path.basename(path) == "oscar_rows_rank0.pt"
        payload = torch.load(path)
    layer = payload["layers"][2]
    torch.testing.assert_close(layer["k"], k)
    torch.testing.assert_close(layer["v"], v)
    assert layer["positions"].tolist() == positions.tolist()
    stride = payload["q_sample_stride"]
    assert layer["q_samples"].shape == ((70 + stride - 1) // stride, 4, 8)
    torch.testing.assert_close(layer["q_samples"][1], q[stride])
    assert layer["q_scaling"] == 0.25
    assert payload["q_head_offset"] == 0 and payload["tokens"] == 70 and payload["global_q_heads"] == 4
