import math

import numpy as np
import pytest

from kl_probe.metrics import (
    overlap_at,
    truncated_kl,
)


def lp(*probs):
    return [math.log(p) for p in probs]


def test_kl_identical_distributions_is_zero():
    ids = [1, 2, 3]
    logprobs = lp(0.7, 0.2, 0.1)
    kl, tail = truncated_kl(ids, logprobs, ids, logprobs)
    assert kl == pytest.approx(0.0, abs=1e-12)
    assert tail == pytest.approx(0.0, abs=1e-9)


def test_kl_analytic_two_point():
    # p = (0.75, 0.25), q = (0.5, 0.5) over the same support
    kl, tail = truncated_kl([1, 2], lp(0.75, 0.25), [1, 2], lp(0.5, 0.5))
    expected = 0.75 * math.log(0.75 / 0.5) + 0.25 * math.log(0.25 / 0.5)
    assert kl == pytest.approx(expected)
    assert tail == pytest.approx(0.0, abs=1e-9)


def _renorm_kl(p, q):
    zp, zq = sum(p), sum(q)
    return sum((pi / zp) * math.log((pi / zp) / (qi / zq)) for pi, qi in zip(p, q))


def test_kl_missing_token_clamps_to_cand_floor():
    # token 3 in ref's support is missing from cand's list; cand's floor is 0.3
    kl, _ = truncated_kl([1, 3], lp(0.6, 0.3), [1, 2], lp(0.5, 0.3))
    assert kl == pytest.approx(_renorm_kl([0.6, 0.3], [0.5, 0.3]))


def test_kl_exact_cand_overrides_floor():
    exact = {3: math.log(0.01)}
    kl, _ = truncated_kl([1, 3], lp(0.6, 0.3), [1, 2], lp(0.5, 0.3), exact_cand=exact)
    assert kl == pytest.approx(_renorm_kl([0.6, 0.3], [0.5, 0.01]))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_kl_rejects_non_finite_top_logprobs(bad):
    with pytest.raises(ValueError, match="non-finite"):
        truncated_kl([1], [0.0], [1], [bad])


def test_kl_rejects_non_finite_exact_cand():
    with pytest.raises(ValueError, match="exact_cand.*non-finite"):
        truncated_kl([1], [0.0], [1], [0.0], exact_cand={1: float("nan")})


def test_kl_never_negative():
    rng = np.random.default_rng(1)
    for _ in range(200):
        p = rng.dirichlet(np.ones(8))
        q = rng.dirichlet(np.ones(8))
        p_order = np.argsort(p)[::-1][:5]
        q_order = np.argsort(q)[::-1][:5]
        kl, _ = truncated_kl(
            p_order.tolist(),
            np.log(p[p_order]).tolist(),
            q_order.tolist(),
            np.log(q[q_order]).tolist(),
        )
        assert kl >= 0.0


def test_kl_tail_mass_reported():
    _, tail = truncated_kl([1], lp(0.8), [1], lp(0.9))
    assert tail == pytest.approx(0.2)


def test_overlap_at():
    assert overlap_at([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 5) == 1.0
    assert overlap_at([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], 5) == 1.0  # set overlap, order-free
    assert overlap_at([1, 2], [3, 4], 2) == 0.0
    assert overlap_at([1, 2, 3], [1, 9, 10], 1) == 1.0
    assert overlap_at([1, 2, 3], [2, 1, 9], 3) == pytest.approx(2 / 3)


def test_aggregate_rejects_empty_table():
    import pyarrow as pa
    import pytest

    from kl_probe.metrics import TOKENS_SCHEMA, aggregate

    with pytest.raises(RuntimeError, match="no compared positions"):
        aggregate(pa.Table.from_pylist([], schema=TOKENS_SCHEMA))


def test_aggregate_is_bucket_only():
    import pyarrow as pa

    from kl_probe.metrics import TOKENS_SCHEMA, aggregate

    rows = [
        {
            "pid": 0,
            "pos": pos,
            "token_id": 5,
            "kl": 0.1 * (pos + 1),
            "top1_agree": True,
            "overlap5": 1.0,
            "overlap_k": 1.0,
            "delta_logprob": 0.0,
            "ref_top1_rank_in_cand": 0,
            "ref_tail_mass": 0.0,
            "ref_top1": 5,
            "cand_top1": 5,
        }
        # positions 0..1 -> bucket 0, positions 4096..4097 -> bucket 1
        for pos in (0, 1, 4096, 4097)
    ]
    out = aggregate(pa.Table.from_pylist(rows, schema=TOKENS_SCHEMA), bucket_size=4096)
    # no global block — buckets are the whole report
    assert set(out) == {"bucket_size", "buckets"}
    assert [b["bucket"] for b in out["buckets"]] == [0, 1]
    b0 = out["buckets"][0]
    assert b0["pos_range"] == "0-4095"
    assert b0["num_tokens"] == 2 and b0["num_sequences"] == 1
    assert "kl" in b0 and "top1_agreement" in b0
