"""The joint latent transform: fitter algebra, the packed pool's frame helpers
and the companion loader agree that q'.k' == q.(k - mean) and that the output
un-rotation inverts the write transform."""
import os
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import fit_mla_joint_latent as fit  # noqa: E402
from sglang.srt.mem_cache.mla_int2_kv_pool import _load_latent_companions, _load_or_make_rotations  # noqa: E402
from sglang.srt.mem_cache.mla_packed_kv_pool import (  # noqa: E402
    latent_keys_to_frame,
    latent_output_from_frame,
    latent_query_to_frame,
)


def _fake_dump(*, rank=64, heads=4, rope=16, tokens=600, seqs=3, stride=16, seed=0):
    gen = torch.Generator().manual_seed(seed)
    per = tokens // seqs
    positions = torch.cat([torch.arange(per) for _ in range(seqs)])
    c = torch.randn(tokens, rank, generator=gen, dtype=torch.float64) @ torch.diag(torch.exp(torch.rand(rank, generator=gen, dtype=torch.float64) * 2)) + 0.5
    k_pe = torch.randn(tokens, rope, generator=gen, dtype=torch.float64)
    q_rows = torch.arange(0, tokens, stride)
    q_nope = torch.randn(q_rows.shape[0], heads, rank, generator=gen, dtype=torch.float64)
    q_pe = torch.randn(q_rows.shape[0], heads, rope, generator=gen, dtype=torch.float64)
    return {"c_kv": c, "k_pe": k_pe, "positions": positions, "q_rows": q_rows, "q_nope": q_nope, "q_pe": q_pe}


@pytest.mark.parametrize("variant", ["cov", "key", "value", "joint"])
def test_fitter_algebra(variant):
    rank, heads, v_dim, hidden = 64, 4, 8, 32
    d = _fake_dump(rank=rank, heads=heads)
    stats = fit.latent_statistics(d, scaling=1.0 / 8)
    gen = torch.Generator().manual_seed(1)
    w_uv = torch.randn(heads, v_dim, rank, generator=gen, dtype=torch.float64)
    w_o = torch.randn(heads, v_dim, hidden, generator=gen, dtype=torch.float64)
    m_v = fit.value_metric(w_uv, w_o, stats["rho"])
    assert torch.allclose(m_v, m_v.T) and torch.linalg.eigvalsh(m_v).min() > -1e-9
    entry = fit.fit_layer(stats, variant=variant, lam=1.0, group=16, m_v=m_v)
    fit.self_check(entry)
    r = entry["rotation"].to(torch.float64)
    assert ("q_rotation" in entry) == (variant != "cov")
    if variant == "cov":
        assert torch.allclose(r @ r.T, torch.eye(rank, dtype=torch.float64), atol=1e-5)
    # the transformed centered latent has a flat diagonal within every group
    cc = d["c_kv"] - entry["mean"].to(torch.float64)
    cov = (cc @ r).T @ (cc @ r) / cc.shape[0]
    diag = torch.diag(cov).reshape(-1, 16)
    assert ((diag.max(1).values - diag.min(1).values) / diag.mean(1)).max() < 1e-6


def test_frame_helpers_and_loader():
    rank = 32
    gen = torch.Generator().manual_seed(2)
    a = torch.randn(rank, rank, generator=gen, dtype=torch.float64)
    r = a @ a.T / rank + torch.eye(rank, dtype=torch.float64)  # SPD, non-orthogonal
    q = torch.linalg.inv(r).T
    mean = torch.randn(rank, generator=gen, dtype=torch.float64)
    keys = torch.randn(9, rank, generator=gen, dtype=torch.float64)
    queries = torch.randn(5, rank, generator=gen, dtype=torch.float64)
    kf = latent_keys_to_frame(keys, r, mean)
    qf = latent_query_to_frame(queries, q)
    assert torch.allclose(qf @ kf.T, queries @ (keys - mean).T, atol=1e-9)
    assert torch.allclose(latent_output_from_frame(kf, q, mean), keys, atol=1e-9)
    # the orthogonal case: q == r, mean None, is the old x @ R / x @ R^T pair
    o, _ = torch.linalg.qr(a)
    assert torch.allclose(latent_output_from_frame(latent_keys_to_frame(keys, o, None), o, None), keys, atol=1e-9)

    d = tempfile.mkdtemp(prefix="latent_rot_")
    torch.save({"rotation": r.float(), "q_rotation": q.float(), "mean": mean.float()}, os.path.join(d, "layer_0.pt"))
    torch.save(o.float(), os.path.join(d, "layer_1.pt"))  # plain orthogonal tensor
    rots = _load_or_make_rotations(d, layer_num=2, start_layer=0, kv_lora_rank=rank, device="cpu", dtype=torch.float32)
    qs, means = _load_latent_companions(d, layer_num=2, start_layer=0, kv_lora_rank=rank, device="cpu", dtype=torch.float32)
    assert torch.allclose(rots[0], r.float()) and torch.allclose(rots[1], o.float())
    assert set(qs) == {0} and set(means) == {0}
    assert torch.allclose(qs[0], q.float()) and torch.allclose(means[0], mean.float())


def test_fp8_block_dequant():
    gen = torch.Generator().manual_seed(3)
    w = torch.randn(256, 384, generator=gen, dtype=torch.float64)
    s = torch.rand(2, 3, generator=gen, dtype=torch.float64) + 0.5
    full = fit.dequantize_block_fp8(w, s, (128, 128))
    assert torch.allclose(full[130, 200], w[130, 200] * s[1, 1]) and torch.allclose(full[5, 300], w[5, 300] * s[0, 2])
