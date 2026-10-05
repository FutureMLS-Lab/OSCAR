"""CPU checks for the NOVA-KV reproduction pieces: the equal-volume column
dealing keeps q.k exact and balances the groups' volume, the novakv variant
fits from calibration moments with centering and a query-side matrix, and the
codebook trainer turns calibration rows into per-(layer, head) codebooks the
pool's loader accepts."""
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import fit_oscar2_variants as fit  # noqa: E402
import train_pq_codebooks as trainer  # noqa: E402

from sglang.srt.mem_cache.oscar_pq_codebooks import read_pq_codebook_header  # noqa: E402


def _spd(n: int, gen: torch.Generator, spread: float) -> torch.Tensor:
    a = torch.randn(n, n, generator=gen, dtype=torch.float64)
    q, _ = torch.linalg.qr(a)
    scales = torch.logspace(0, spread, n, dtype=torch.float64)
    return q @ torch.diag(scales) @ q.T


def test_equal_volume_dealing_balances_groups_and_is_a_permutation():
    variances = torch.logspace(0, 3, 16, dtype=torch.float64)
    perm = fit.equal_volume_permutation(variances, 4)
    assert sorted(perm.tolist()) == list(range(16))
    dealt = variances[perm].reshape(4, 4).log().sum(1)
    contiguous = variances.reshape(4, 4).log().sum(1)
    # round-robin dealing (i mod L, as published) shrinks the spread by about the group count
    assert dealt.max() - dealt.min() < 0.5 * (contiguous.max() - contiguous.min())


def test_novakv_basis_keeps_logits_exact_after_dealing():
    gen = torch.Generator().manual_seed(1)
    m_q = _spd(16, gen, 2.0)
    s_k = _spd(16, gen, 1.5)
    r, q, vals = fit.metric_basis(fit.unit_mean_eig(m_q), s_k, None)
    perm = fit.equal_volume_permutation(vals, 4)
    r, q = r[:, perm], q[:, perm]
    x = torch.randn(5, 16, generator=gen, dtype=torch.float64)
    y = torch.randn(7, 16, generator=gen, dtype=torch.float64)
    torch.testing.assert_close((x @ q) @ (y @ r).T, x @ y.T, atol=1e-9, rtol=1e-9)


def _moments(gen: torch.Generator, *, layers=(0, 3), heads=2, hd=16):
    out = {}
    for lid in layers:
        m_q = torch.stack([_spd(hd, gen, 1.5) for _ in range(heads)])
        mu = torch.randn(heads, hd, generator=gen, dtype=torch.float64)
        s_k = torch.stack([_spd(hd, gen, 1.0) for _ in range(heads)])
        n = 500.0
        out[lid] = {"M_q": m_q, "k_sum": mu * n, "M_k": (s_k + mu[:, :, None] * mu[:, None, :]) * n,
                    "S_v": torch.stack([_spd(hd, gen, 1.0) for _ in range(heads)]), "rho": None, "count": n}
    return {"model_path": "tiny", "model_revision": None, "prompt_sha256": "0" * 64, "tokens": 500,
            "global_kv_heads": heads, "global_q_heads": 2 * heads, "head_dim": hd, "v_head_dim": hd, "layers": out}


def test_novakv_variant_fits_with_centering_and_query_matrix():
    gen = torch.Generator().manual_seed(2)
    mom = _moments(gen)
    k_state, v_state = fit.fit_variant("novakv", mom, shared=False, k_base="novakv", o_proj=None, vq_group=4)
    fit.self_check(k_state, v_state)
    assert k_state["transform"] == {"key": "novakv", "centered": True, "orthogonal": False, "equal_volume_group": 4}
    entry = k_state["layers"][3]
    assert entry["rotation"].shape == (2, 16, 16) and entry["q_rotation"].shape == (2, 16, 16)
    torch.testing.assert_close(entry["k_mean"].double(), mom["layers"][3]["k_sum"] / 500.0)
    assert v_state["transform"]["orthogonal"]


def test_trainer_builds_per_head_codebooks_from_calibration_rows():
    gen = torch.Generator().manual_seed(3)
    mom = _moments(gen, layers=(0, 1), heads=2, hd=8)
    k_state, _ = fit.fit_variant("novakv", mom, shared=False, k_base="novakv", o_proj=None, vq_group=4)
    tokens = 96
    rows = {"format_version": 1, "model_path": "tiny", "model_revision": None, "prompt_sha256": "0" * 64,
            "tokens": tokens, "tp_rank": 0, "tp_size": 1, "q_head_offset": 0, "local_kv_heads": 2, "global_kv_heads": 2,
            "global_q_heads": 4, "head_dim": 8, "v_head_dim": 8, "q_sample_stride": 32,
            "layers": {lid: {"k": torch.randn(tokens, 2, 8, generator=gen).to(torch.bfloat16),
                             "v": torch.randn(tokens, 2, 8, generator=gen).to(torch.bfloat16),
                             "q_samples": torch.zeros(3, 4, 8), "positions": torch.arange(tokens, dtype=torch.int32),
                             "q_scaling": 0.35, "count": tokens} for lid in (0, 1)}}
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "rows"))
        torch.save(rows, os.path.join(d, "rows", "oscar_rows_rank0.pt"))
        torch.save(k_state, os.path.join(d, "k_rotation_oscar2_novakv.pt"))
        out = os.path.join(d, "k_pq.pt")
        rc = trainer.main(["--rows-dir", os.path.join(d, "rows"), "--rotation", os.path.join(d, "k_rotation_oscar2_novakv.pt"),
                           "--tensor", "k", "--per-head", "--sub-dim", "4", "--centroids", "4", "--iters", "3",
                           "--device", "cpu", "--out", out])
        assert rc == 0
        data = torch.load(out)
    assert data["per_head"] and data["kv_heads"] == 2
    assert tuple(data["codebooks_per_layer"].shape) == (2, 2, 2, 4, 4)
    assert data["rotation_convention"] == "centered_row_times_checkpoint_rotation"
    header = read_pq_codebook_header(data, expected_head_dim=8, label="t")
    assert header.kind == "per_head" and header.kv_heads == 2 and header.n_sub == 2 and header.sub_dim == 4
