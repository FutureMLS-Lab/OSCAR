import asyncio
import json

import pytest

import kl_probe.score as score_mod
from kl_probe.config import ModelSpec, RunCfg, RuntimeCfg, ServerRuntime
from kl_probe.engine import PositionScore
from kl_probe.score import run_score


def _cfg() -> RunCfg:
    return RunCfg(
        run_name="test",
        reference=ModelSpec(model_path="ref", image="lmsysorg/sglang:latest", name="ref"),
        candidates=[ModelSpec(model_path="cand", image="lmsysorg/sglang:latest", name="cand")],
        runtime=RuntimeCfg(
            ref=ServerRuntime(gpus=["0"], port=38090),
            candidates={"cand": ServerRuntime(gpus=["1"], port=38091)},
        ),
    )


class _FakeClient:
    """Scores every generated position as its own token id, optionally shifted
    to simulate a client parsing bug."""

    def __init__(self, shift: int = 0, drop_last: bool = False):
        self.shift = shift
        self.drop_last = drop_last

    async def score(self, input_ids, prompt_len, top_k):
        ids = input_ids[prompt_len:]
        if self.drop_last:
            ids = ids[:-1]
        return [
            PositionScore(token_id=t + self.shift, logprob=-0.5, top_ids=[t], top_logprobs=[-0.5])
            for t in ids
        ]

    async def close(self):
        pass


def _run(tmp_path, monkeypatch, client) -> int:
    monkeypatch.setattr(score_mod, "make_client", lambda *a, **kw: client)
    gens = tmp_path / "generations.jsonl"
    gens.write_text(json.dumps({"pid": 0, "prompt_ids": [1, 2], "output_ids": [10, 11, 12]}) + "\n")
    cfg = _cfg()
    return asyncio.run(
        run_score(cfg, cfg.reference, cfg.runtime.ref, gens, tmp_path / "scores.parquet")
    )


def test_run_score_writes_aligned_rows(tmp_path, monkeypatch):
    # all positions from index 1 are scored: full=[1,2,10,11,12] -> 4 scored
    assert _run(tmp_path, monkeypatch, _FakeClient()) == 4
    assert (tmp_path / "scores.parquet").exists()
    assert not list(tmp_path.glob("*.tmp"))  # atomic rename left no temp file


def test_run_score_rejects_shifted_token_ids(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="don't match the sequence"):
        _run(tmp_path, monkeypatch, _FakeClient(shift=1))


def test_run_score_rejects_position_count_mismatch(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="scored 3 positions for 4"):
        _run(tmp_path, monkeypatch, _FakeClient(drop_last=True))
