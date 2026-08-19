import pytest

from kl_probe.config import Provider, detect_engine, load_config, parse_gpus


def test_detect_engine_distinguishes_providers():
    # the internal tgl/smg builds used for MiniMax M3 (quant-bench, turbo-skills)
    assert (
        detect_engine(
            "598726163780.dkr.ecr.us-west-2.amazonaws.com/tgl:de-minimax-m3-721a615d-x86-jheo"
        )
        is Provider.TGL
    )
    assert (
        detect_engine(
            "598726163780.dkr.ecr.us-west-2.amazonaws.com/smg:smg-minimax-m3-6877b267-tgl-de-minimax-m3-dea4bd032-b200-flashinfer0612-base"
        )
        is Provider.TGL
    )
    assert detect_engine("lmsysorg/sglang:latest") is Provider.SGLANG
    assert detect_engine("vllm/vllm-openai:nightly") is Provider.VLLM
    with pytest.raises(ValueError, match="found: none"):
        detect_engine("nvcr.io/nvidia/tritonserver:latest")
    with pytest.raises(ValueError, match="found: sglang, tgl"):
        detect_engine("example/sglang-tgl:hybrid")  # ambiguous


def test_parse_gpus_accepts_list_and_comma_string():
    assert parse_gpus([2, "3"]) == ["2", "3"]
    assert parse_gpus("4,5") == ["4", "5"]


def test_parse_gpus_rejects_bools_and_duplicates():
    with pytest.raises(ValueError, match="gpus must be"):
        parse_gpus(True)  # YAML `gpus: yes`
    with pytest.raises(ValueError, match="gpus must be"):
        parse_gpus([0, True])
    with pytest.raises(ValueError, match="duplicate"):
        parse_gpus([2, "2"])
    with pytest.raises(ValueError, match="duplicate"):
        parse_gpus("3,3")


def test_load_config_uses_runtime_section(tmp_path):
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(
        """
run_name: test
reference:
  model_path: ref/model
  image: lmsysorg/sglang:latest
candidates:
  - model_path: cand/model
    image: lmsysorg/sglang:latest
runtime:
  ref:
    gpus: [0]
    port: 38090
  candidates:
    cand:
      gpus: [1]
      port: 38091
"""
    )
    cfg = load_config(cfg_path)
    assert cfg.runtime.ref.gpus == ["0"]
    assert cfg.runtime.cand.url == "http://127.0.0.1:38091"


def test_model_env_is_coerced_and_validated(tmp_path):
    template = """
run_name: test
reference:
  model_path: ref/model
  image: lmsysorg/sglang:latest
candidates:
  - model_path: cand/model
    image: lmsysorg/sglang:latest
    env:
      {env}
runtime:
  ref: {{gpus: [0], port: 38090}}
  candidates:
    cand: {{gpus: [1], port: 38091}}
"""
    cfg_path = tmp_path / "run.yaml"

    cfg_path.write_text(template.format(env="TORCHDYNAMO_DISABLE: 1"))
    cfg = load_config(cfg_path)
    assert cfg.candidate.env == {"TORCHDYNAMO_DISABLE": "1"}  # int coerced to str
    assert cfg.reference.env == {}

    cfg_path.write_text(template.format(env='"BAD-NAME": x'))
    with pytest.raises(ValueError, match="invalid env var name"):
        load_config(cfg_path)

    cfg_path.write_text(template.format(env="SOME_FLAG: yes"))  # YAML bool trap
    with pytest.raises(ValueError, match="must be a string or number"):
        load_config(cfg_path)


def test_bad_scalar_types_are_rejected(tmp_path):
    template = """
run_name: test
reference:
  model_path: ref/model
  image: lmsysorg/sglang:latest
  tp: {tp}
candidates:
  - model_path: cand/model
    image: lmsysorg/sglang:latest
runtime:
  ref:
    gpus: [0]
    port: {port}
  candidates:
    cand:
      gpus: [1]
      port: 38091
"""
    cfg_path = tmp_path / "run.yaml"

    cfg_path.write_text(template.format(tp='"2"', port=38090))
    with pytest.raises(ValueError, match="tp must be a positive integer"):
        load_config(cfg_path)

    cfg_path.write_text(template.format(tp=1, port='"38090x"'))
    with pytest.raises(ValueError, match="port must be a port number"):
        load_config(cfg_path)


def test_degenerate_pipeline_scalars_are_rejected(tmp_path):
    template = """
run_name: test
reference: {{model_path: ref/model, image: lmsysorg/sglang:latest}}
candidates:
  - {{model_path: cand/model, image: lmsysorg/sglang:latest}}
scoring:
  concurrency: {concurrency}
generation:
  top_p: {top_p}
runtime:
  ref: {{gpus: [0], port: 38090}}
  candidates:
    cand: {{gpus: [1], port: 38091}}
"""
    cfg_path = tmp_path / "run.yaml"

    cfg_path.write_text(template.format(concurrency=0, top_p=0.95))
    with pytest.raises(ValueError, match="concurrency must be an integer >= 1"):
        load_config(cfg_path)  # Semaphore(0) would hang every request

    cfg_path.write_text(template.format(concurrency=8, top_p=0.0))
    with pytest.raises(ValueError, match="top_p must be in"):
        load_config(cfg_path)


def test_model_placement_keys_are_rejected(tmp_path):
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(
        """
run_name: test
reference:
  model_path: ref/model
  image: lmsysorg/sglang:latest
  gpus: "0"
candidates:
  - model_path: cand/model
    image: lmsysorg/sglang:latest
runtime:
  ref:
    gpus: [0]
    port: 38090
  candidates:
    cand:
      gpus: [1]
      port: 38091
"""
    )
    with pytest.raises(ValueError, match="unknown key.*gpus"):
        load_config(cfg_path)
