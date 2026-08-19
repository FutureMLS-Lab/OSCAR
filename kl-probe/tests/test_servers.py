import pytest

from kl_probe.config import ModelSpec, RunCfg, RuntimeCfg, ServerRuntime
from kl_probe.servers import PATCHES_MOUNT, _compose_override, _env_for


def _runtime() -> RuntimeCfg:
    return RuntimeCfg(
        ref=ServerRuntime(gpus=["0"], port=38090),
        candidates={"cand": ServerRuntime(gpus=["1"], port=38091)},
    )


def test_compose_override_injects_env_and_patch_mounts(tmp_path):
    patch = tmp_path / "excluded-moe-mxfp8.py"
    patch.write_text("print('patched')\n")
    cfg = RunCfg(
        run_name="test",
        reference=ModelSpec(model_path="ref", image="lmsysorg/sglang:latest", name="ref"),
        candidates=[
            ModelSpec(
                model_path="cand",
                image="lmsysorg/sglang:latest",
                name="cand",
                env={"SGLANG_MINIMAX_SPARSE_DECODE_BACKEND": "trtllm", "TORCHDYNAMO_DISABLE": "1"},
                patches=[str(patch)],
            )
        ],
        runtime=_runtime(),
    )
    override = _compose_override(cfg)
    cand = override["services"]["cand"]
    assert cand["environment"] == {
        "SGLANG_MINIMAX_SPARSE_DECODE_BACKEND": "trtllm",
        "TORCHDYNAMO_DISABLE": "1",
    }
    assert cand["volumes"] == [f"{patch}:{PATCHES_MOUNT}/excluded-moe-mxfp8.py:ro"]
    ref = override["services"]["ref"]
    assert "environment" not in ref and "volumes" not in ref
    env = _env_for(cfg)
    assert env["CAND_PATCHES"] == "excluded-moe-mxfp8.py"
    assert env["REF_PATCHES"] == ""


def test_missing_patch_file_is_rejected(tmp_path):
    cfg = RunCfg(
        run_name="test",
        reference=ModelSpec(model_path="ref", image="lmsysorg/sglang:latest", name="ref"),
        candidates=[
            ModelSpec(
                model_path="cand",
                image="lmsysorg/sglang:latest",
                name="cand",
                patches=[str(tmp_path / "nope.py")],
            )
        ],
        runtime=_runtime(),
    )
    with pytest.raises(ValueError, match="loader patch not found"):
        _compose_override(cfg)


def test_compose_override_uses_explicit_gpu_device_ids():
    cfg = RunCfg(
        run_name="test",
        reference=ModelSpec(model_path="ref", image="lmsysorg/sglang:latest", name="ref"),
        candidates=[ModelSpec(model_path="cand", image="lmsysorg/sglang:latest", name="cand")],
        runtime=RuntimeCfg(
            ref=ServerRuntime(gpus=["2", "3"], port=38090),
            candidates={"cand": ServerRuntime(gpus=["4"], port=38091)},
        ),
    )
    override = _compose_override(cfg)
    ref_devices = override["services"]["ref"]["deploy"]["resources"]["reservations"]["devices"]
    cand_devices = override["services"]["cand"]["deploy"]["resources"]["reservations"]["devices"]
    assert ref_devices[0]["device_ids"] == ["2", "3"]
    assert cand_devices[0]["device_ids"] == ["4"]
