import pytest

from kl_probe.config import detect_engine
from kl_probe.vllm_client import _parse_prompt_logprob_entry, _parse_token_id


def test_detect_engine():
    assert detect_engine("lmsysorg/sglang:latest") == "sglang"
    assert detect_engine("lmsysorg/sglang@sha256:abc") == "sglang"
    assert detect_engine("vllm/vllm-openai:latest") == "vllm"
    assert detect_engine("vllm/vllm-openai:nightly") == "vllm"
    with pytest.raises(ValueError):
        detect_engine("nvcr.io/nvidia/tritonserver:latest")  # neither


def test_parse_token_id():
    assert _parse_token_id("token_id:123") == 123
    with pytest.raises(RuntimeError):
        _parse_token_id("hello")  # plain text = flag missing on the request


def _entry(*items):
    """items: (token_id, logprob, rank) -> vllm prompt_logprobs entry (JSON
    string keys, like a parsed response)."""
    return {
        str(tid): {"logprob": lp, "rank": rank, "decoded_token": "x"} for tid, lp, rank in items
    }


def test_entry_basic():
    entry = _entry((10, -0.1, 1), (20, -2.0, 2), (30, -3.0, 3))
    ps = _parse_prompt_logprob_entry(entry, expected_token_id=20, top_k=3, pos=0)
    assert ps.token_id == 20
    assert ps.logprob == pytest.approx(-2.0)
    assert ps.top_ids == [10, 20, 30]  # rank order, not dict order
    assert ps.top_logprobs == pytest.approx([-0.1, -2.0, -3.0])


def test_entry_actual_token_beyond_top_k():
    # actual token (99) is rank 7 — included by vllm as an extra entry, but it
    # must not displace the true top-3 in the top-k list
    entry = _entry((10, -0.1, 1), (20, -2.0, 2), (30, -3.0, 3), (99, -9.0, 7))
    ps = _parse_prompt_logprob_entry(entry, expected_token_id=99, top_k=3, pos=5)
    assert ps.logprob == pytest.approx(-9.0)  # exact, from the extra entry
    assert ps.top_ids == [10, 20, 30]
    assert 99 not in ps.top_ids


def test_entry_alignment_mismatch_raises():
    entry = _entry((10, -0.1, 1))
    with pytest.raises(RuntimeError, match="alignment"):
        _parse_prompt_logprob_entry(entry, expected_token_id=42, top_k=1, pos=3)


def test_entry_short_top_k_raises():
    entry = _entry((10, -0.1, 1), (20, -2.0, 2))
    with pytest.raises(RuntimeError, match="max-logprobs"):
        _parse_prompt_logprob_entry(entry, expected_token_id=10, top_k=5, pos=0)
