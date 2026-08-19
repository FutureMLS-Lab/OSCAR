import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kl_probe.cli import _reuse_scores
from kl_probe.config import load_config

_CONFIG = """
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


def _cfg(tmp_path):
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(_CONFIG)
    cfg = load_config(cfg_path)
    # cmd_run resolves "auto" to an int before reuse; mimic that here.
    cfg.scoring.top_logprobs_num = 40
    cfg.output_dir = str(tmp_path / "runs")
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _make_src(tmp_path, *, generations=True, ref_scores=True, cand_scores=(), metrics=None):
    """A prior run dir. generations.jsonl is required for reuse; ref/candidate
    score parquets are optional (reused only when present)."""
    src = tmp_path / "prior"
    src.mkdir(exist_ok=True)
    if generations:
        (src / "generations.jsonl").write_text('{"pid": 0}\n')
    if ref_scores:
        pq.write_table(pa.table({"pid": [0]}), src / "scores_ref.parquet")
    for name in cand_scores:
        pq.write_table(pa.table({"pid": [0]}), src / f"scores_{name}.parquet")
    if metrics is not None:
        (src / "metrics.json").write_text(json.dumps(metrics))
    return src


def test_reuse_copies_generations_and_ref(tmp_path):
    cfg = _cfg(tmp_path)
    reused = _reuse_scores(cfg, _make_src(tmp_path))
    assert "ref" in reused
    assert (cfg.run_dir / "generations.jsonl").read_text() == '{"pid": 0}\n'
    assert (cfg.run_dir / "scores_ref.parquet").is_file()


def test_reuse_rejects_missing_generations(tmp_path):
    cfg = _cfg(tmp_path)
    src = tmp_path / "empty"
    src.mkdir()
    with pytest.raises(SystemExit, match="generations"):
        _reuse_scores(cfg, src)


def test_reuse_without_ref_scores_skips_ref(tmp_path):
    cfg = _cfg(tmp_path)
    reused = _reuse_scores(cfg, _make_src(tmp_path, ref_scores=False))
    assert "ref" not in reused
    assert (cfg.run_dir / "generations.jsonl").is_file()
    assert not (cfg.run_dir / "scores_ref.parquet").is_file()


def test_reuse_rejects_reference_mismatch(tmp_path):
    cfg = _cfg(tmp_path)
    src = _make_src(tmp_path, metrics={"reference": "other/model", "top_logprobs_num": 40})
    with pytest.raises(SystemExit, match="source reference"):
        _reuse_scores(cfg, src)


def test_reuse_rejects_topk_mismatch(tmp_path):
    cfg = _cfg(tmp_path)
    src = _make_src(tmp_path, metrics={"reference": "ref/model", "top_logprobs_num": 20})
    with pytest.raises(SystemExit, match="top_logprobs_num"):
        _reuse_scores(cfg, src)


def test_reuse_reuses_matching_candidate(tmp_path):
    cfg = _cfg(tmp_path)
    src = _make_src(
        tmp_path,
        cand_scores=("cand",),
        metrics={
            "reference": "ref/model",
            "top_logprobs_num": 40,
            "candidates": {"cand": {"model_path": "cand/model"}},
        },
    )
    reused = _reuse_scores(cfg, src)
    assert reused == {"ref", "cand"}
    assert (cfg.run_dir / "scores_cand.parquet").is_file()


def test_reuse_skips_candidate_model_mismatch(tmp_path):
    cfg = _cfg(tmp_path)
    src = _make_src(
        tmp_path,
        cand_scores=("cand",),
        metrics={
            "reference": "ref/model",
            "top_logprobs_num": 40,
            "candidates": {"cand": {"model_path": "different/model"}},
        },
    )
    reused = _reuse_scores(cfg, src)
    assert reused == {"ref"}
    assert not (cfg.run_dir / "scores_cand.parquet").is_file()


def test_reuse_candidate_without_metrics_is_reused(tmp_path):
    # No metrics.json (e.g. a crashed prior run): reuse present score parquets,
    # trusting the dir, since model_path can't be verified.
    cfg = _cfg(tmp_path)
    reused = _reuse_scores(cfg, _make_src(tmp_path, cand_scores=("cand",)))
    assert reused == {"ref", "cand"}
    assert (cfg.run_dir / "scores_cand.parquet").is_file()
