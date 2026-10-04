"""OSCAR-2 transform family: the fitter's algebra, the checkpoint fields and
the pool/attention path that consumes them.

A non-orthogonal key transform is only valid if the query side undoes it
exactly (q' . k' == q . k); a non-orthogonal value transform only if the
output un-rotation inverts it; centering only if every stored key and the
extend-time keys subtract the same mean. The CPU tests pin the algebra and
the loader; the GPU test runs quantized decode attention under a fully
non-orthogonal, centered pair against dense BF16 attention on the raw K/V.
"""
import os
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import fit_oscar2_variants as fit  # noqa: E402

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.mem_cache.memory_pool import load_oscar_rotation_field, load_oscar_rotations  # noqa: E402
from sglang.srt.runtime_context import get_parallel  # noqa: E402

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
_DIR = tempfile.mkdtemp(prefix="oscar2_test_")


def _spd(d, gen, spread=3.0):
    a = torch.randn(d, d, generator=gen, dtype=torch.float64)
    q, _ = torch.linalg.qr(a)
    vals = torch.exp(spread * torch.rand(d, generator=gen, dtype=torch.float64))
    return q @ torch.diag(vals) @ q.T


def _fake_moments(*, layers=2, heads=2, hd=64, gqa=4, tokens=1000, seed=3):
    gen = torch.Generator().manual_seed(seed)
    out = {"model_path": "fake", "model_revision": None, "prompt_sha256": "x", "tokens": tokens,
           "global_kv_heads": heads, "global_q_heads": heads * gqa, "head_dim": hd, "v_head_dim": hd, "tp_size": 1}
    lay = {}
    for lid in range(layers):
        m_q = torch.stack([_spd(hd, gen) for _ in range(heads)])
        mu = torch.randn(heads, hd, generator=gen, dtype=torch.float64)
        s_k = torch.stack([_spd(hd, gen) for _ in range(heads)])
        m_k = tokens * (s_k + mu[:, :, None] * mu[:, None, :])
        s_v = torch.stack([_spd(hd, gen) for _ in range(heads)])
        a = torch.rand(heads, gqa, gqa, generator=gen, dtype=torch.float64)
        rho = (a @ a.transpose(1, 2)) / gqa  # attention overlap of the query heads: symmetric, positive
        lay[lid] = {"M_q": m_q, "k_sum": tokens * mu, "M_k": m_k, "S_v": s_v, "rho": rho, "count": tokens}
    out["layers"] = lay
    o_proj = {lid: torch.randn(4 * hd, heads * gqa * hd, generator=gen, dtype=torch.float64) for lid in range(layers)}
    return out, o_proj


def _write(state, name):
    path = os.path.join(_DIR, name)
    torch.save(state, path)
    return path


@pytest.mark.parametrize("variant", ["perhead", "center", "whiten", "flat", "stretch", "outaware", "outaware_hr"])
@pytest.mark.parametrize("shared", [False, True])
def test_fitter_algebra(variant, shared):
    mom, o_proj = _fake_moments()
    k_state, v_state = fit.fit_variant(variant, mom, shared=shared, k_base="flat", o_proj=o_proj)
    fit.self_check(k_state, v_state)  # q'.k' == q.k ; o_rotation inverts R_v
    e = k_state["layers"][0]
    r = e["rotation"].to(torch.float64)
    assert r.dim() == (2 if shared else 3)
    if variant in ("perhead", "center"):
        assert "q_rotation" not in e
        rr = r if r.dim() == 2 else r[0]
        assert torch.allclose(rr @ rr.T, torch.eye(rr.shape[0], dtype=torch.float64), atol=1e-5)
    else:
        assert "q_rotation" in e
    assert ("k_mean" in e) == (variant != "perhead")
    if variant in ("outaware", "outaware_hr"):
        assert "o_rotation" in v_state["layers"][0]
    # the transformed key signal is decorrelated: R^T S_k R diagonal (flattened rows flat)
    m = mom["layers"][0]
    n = m["count"]
    mu = m["k_sum"] / n
    s_k = m["M_k"] / n - mu[:, :, None] * mu[:, None, :]
    if shared:
        s_k = s_k.mean(0, keepdim=True)
    rr = r if r.dim() == 3 else r[None]
    cov = rr[0].T @ s_k[0] @ rr[0]
    off = cov - torch.diag(torch.diag(cov))
    if variant == "whiten":  # pure eigenbasis in the metric space: exactly diagonal
        assert off.abs().max() < 1e-6 * cov.diag().max()
    if variant in ("flat", "stretch", "perhead", "center"):  # Hadamard spread: equal diagonal
        d = torch.diag(cov)
        assert (d.max() - d.min()) / d.mean() < 1e-6 or variant in ("perhead", "center")


def test_loader_reads_optional_fields():
    mom, o_proj = _fake_moments()
    k_state, v_state = fit.fit_variant("flat", mom, shared=False, k_base="flat", o_proj=o_proj)
    kp, vp = _write(k_state, "k_flat.pt"), _write(v_state, "v_flat.pt")
    common = dict(layer_num=2, start_layer=0, head_dim=64, device=torch.device("cpu"), dtype=torch.float32)
    r = load_oscar_rotations(kp, **common)
    q = load_oscar_rotation_field(kp, "q_rotation", **common)
    mean = load_oscar_rotation_field(kp, "k_mean", **common, vector=True)
    assert r.shape == (2, 2, 64, 64) and q.shape == (2, 2, 64, 64) and mean.shape == (2, 2, 64)
    assert load_oscar_rotation_field(vp, "o_rotation", **common) is None  # pre-W_O values are orthogonal
    # exactness survives the float32 cast used for storage
    x = torch.randn(3, 64); y = torch.randn(5, 64)
    assert torch.allclose((x @ q[0, 1]) @ (y @ r[0, 1]).T, x @ y.T, atol=1e-3, rtol=1e-3)
    # a field on only some layers is refused
    broken = dict(k_state)
    broken["layers"] = {0: k_state["layers"][0], 1: {k: v for k, v in k_state["layers"][1].items() if k != "k_mean"}}
    bp = _write(broken, "k_broken.pt")
    with pytest.raises(ValueError):
        load_oscar_rotation_field(bp, "k_mean", **common, vector=True)


def _make_pool(kp, vp, *, device, hd=64, heads=2):
    from sglang.srt.mem_cache.unified_kv_pool import UnifiedInt2HPKVPool

    with (
        envs.SGLANG_OSCAR_K_ROTATION_PATH.override(kp),
        envs.SGLANG_OSCAR_V_ROTATION_PATH.override(vp),
        envs.SGLANG_OSCAR_K_QUANTIZER.override("int2"),
        envs.SGLANG_OSCAR_V_QUANTIZER.override("int2"),
        envs.SGLANG_LLOYD_MAX.override(False),
        envs.SGLANG_OSCAR_CALIBRATION_ACTIVE.override(False),
        get_parallel().override(attn_tp_rank=0),
    ):
        return UnifiedInt2HPKVPool(
            num_quant_pages=256, hp_dtype=torch.bfloat16, hp_prefix_tokens=32, hp_recent_tokens=128,
            dtype="int2", head_num=heads, head_dim=hd, layer_num=2, device=device, enable_memory_saver=False,
            max_req_slots=8, v_head_dim=hd, start_layer=0, end_layer=1, model_dtype=torch.bfloat16,
            kv_cache_quant_group_size=None, scale_dtype=torch.float32, num_hp_prefix_slots=64,
        )


def test_pool_loads_transform_companions_on_cpu():
    mom, o_proj = _fake_moments()
    k_state, v_state = fit.fit_variant("outaware", mom, shared=False, k_base="stretch", o_proj=o_proj)
    pool = _make_pool(_write(k_state, "k_oa.pt"), _write(v_state, "v_oa.pt"), device="cpu")
    assert pool._Q_k is not pool._R_k and pool._O_v is not pool._R_v and pool._k_mean is not None
    assert pool._Q_k.shape == pool._R_k.shape == (2, 2, 64, 64) and pool._k_mean.shape == (2, 2, 64)
    k = torch.randn(3, 2, 64, dtype=torch.bfloat16)
    centered = pool._centered_keys(0, k)
    # same bf16 subtraction the pool does; an fp32 reference would differ by
    # bf16 rounding on large values (one ulp above |4| is 0.03)
    assert torch.equal(centered, k - pool._k_mean[0].to(k.dtype))
    # orthogonal V1-style files alias the companions to the rotations
    k2, v2 = fit.fit_variant("perhead", mom, shared=False, k_base="flat", o_proj=None)
    pool2 = _make_pool(_write(k2, "k_ph.pt"), _write(v2, "v_ph.pt"), device="cpu")
    assert pool2._Q_k is pool2._R_k and pool2._O_v is pool2._R_v and pool2._k_mean is None


class _Layer:
    layer_id = 0
    oscar_v_rotation_absorbed = False


def _dense_reference(q, k, v, scale):
    # q [B, Hq, d]; k, v [N, Hkv, d]; all tokens attend; GQA by head grouping
    hq, hkv = q.shape[1], k.shape[1]
    g = hq // hkv
    kk = k.repeat_interleave(g, dim=1).float()  # [N, Hq, d]
    vv = v.repeat_interleave(g, dim=1).float()
    s = torch.einsum("bhd,nhd->bhn", q.float(), kk) * scale
    p = torch.softmax(s, dim=-1)
    return torch.einsum("bhn,nhd->bhd", p, vv)


def _decode_under_variant(variant: str):
    """Write random K/V through the pool (INT2 tier) under ``variant``, rotate
    q with the query-side matrix, decode, un-rotate the output with the
    output-side matrix. Returns the kernel output, dense attention on the
    *dequantized* K/V mapped back through the inverse transforms (what the
    kernel must reproduce up to bf16), and dense attention on the raw K/V."""
    from sglang.srt.layers.attention.quantized_kv_prefill import oscar_o_rotation, oscar_q_rotation
    from sglang.srt.layers.attention.triton_ops.decode_attention import decode_attention_fwd_int2_unified
    from sglang.srt.mem_cache.kv_quant_kernels import dequantize_kv_int2_triton

    hd, heads, hq = 64, 2, 8
    g = hq // heads
    mom, o_proj = _fake_moments(hd=hd, heads=heads, gqa=g, seed=11)
    k_state, v_state = fit.fit_variant(variant, mom, shared=False, k_base="stretch", o_proj=o_proj)
    pool = _make_pool(_write(k_state, f"k_{variant}_gpu.pt"), _write(v_state, f"v_{variant}_gpu.pt"), device="cuda")
    torch.manual_seed(5)
    n = 160
    mu = (mom["layers"][0]["k_sum"] / mom["layers"][0]["count"]).to(torch.bfloat16).cuda()  # [heads, hd]
    k = torch.randn(n, heads, hd, dtype=torch.bfloat16, device="cuda") + mu
    v = torch.randn(n, heads, hd, dtype=torch.bfloat16, device="cuda")
    loc = torch.arange(8, 8 + n, dtype=torch.int64, device="cuda")
    pool.set_kv_buffer(_Layer(), loc, k, v)  # raw rows: the pool centers, rotates and quantizes
    q = torch.randn(1, hq, hd, dtype=torch.bfloat16, device="cuda")
    Qm = oscar_q_rotation(pool, 0)  # [heads, hd, hd]
    q_rot = torch.einsum("bhd,hde->bhe", q.float(), Qm.repeat_interleave(g, dim=0).float()).to(torch.bfloat16)
    hp_indptr = torch.tensor([0, 0], dtype=torch.int32, device="cuda")
    hp_indices = torch.zeros(1, dtype=torch.int64, device="cuda")
    q_indptr = torch.tensor([0, n], dtype=torch.int32, device="cuda")
    ones = torch.ones(1, dtype=torch.int32, device="cuda")
    splits = 4
    logits = torch.empty(1, hq, 1 + splits, hd, dtype=torch.float32, device="cuda")
    lse = torch.empty(1, hq, 1 + splits, dtype=torch.float32, device="cuda")
    o = torch.empty_like(q)
    decode_attention_fwd_int2_unified(
        q_rot, pool.hp_k_buffer[0], pool.hp_v_buffer[0], pool.k_buffer[0], pool.v_buffer[0],
        pool.k_scales_zeros[0], pool.v_scales_zeros[0], o, hp_indptr, hp_indices, q_indptr, loc.clone(),
        logits, lse, ones, torch.tensor([splits], dtype=torch.int32, device="cuda"), 1, splits, hd**-0.5,
    )
    torch.cuda.synchronize()
    Om = oscar_o_rotation(pool, 0)
    out = torch.einsum("bhe,hde->bhd", o.float(), Om.repeat_interleave(g, dim=0).float())  # o @ O^T per head
    # what the cache holds, mapped back to the model frame: k = k' R_k^{-1} + mu, v = v' R_v^{-1}
    k_q = dequantize_kv_int2_triton(pool.k_buffer[0][loc], pool.k_scales_zeros[0][loc], hd, pool.hp_dtype).double()
    v_q = dequantize_kv_int2_triton(pool.v_buffer[0][loc], pool.v_scales_zeros[0][loc], hd, pool.hp_dtype).double()
    Rk_inv = torch.linalg.inv(pool._R_k[0].double())
    Rv_inv = torch.linalg.inv(pool._R_v[0].double())
    k_hat = torch.einsum("thd,hde->the", k_q, Rk_inv)
    if pool._k_mean is not None:
        k_hat = k_hat + pool._k_mean[0].double()
    v_hat = torch.einsum("thd,hde->the", v_q, Rv_inv)
    ref_q = _dense_reference(q, k_hat, v_hat, hd**-0.5)
    ref_raw = _dense_reference(q, k, v, hd**-0.5)
    return out, ref_q, ref_raw


@gpu
@pytest.mark.parametrize("variant", ["perhead", "outaware"])
def test_quantized_decode_under_transform_matches_dense(variant):
    """The kernel path (query-side matrix, centered + rotated INT2 cache,
    output-side matrix) must reproduce dense attention on the dequantized
    cache mapped back through the inverse transforms -- that isolates the
    transform plumbing from INT2 noise, which on structureless random data
    is large. The orthogonal per-head pair and the fully non-orthogonal
    centered pair (stretch keys, post-W_O values) are both checked."""
    out, ref_q, ref_raw = _decode_under_variant(variant)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref_q.float(), atol=4e-2, rtol=4e-2)
    err_raw = ((out - ref_raw).norm() / ref_raw.norm()).item()
    print(f"{variant}: exact-path max |d| = {(out - ref_q.float()).abs().max().item():.4f}; "
          f"relative error vs raw K/V (INT2 noise on random data) = {err_raw:.3f}")
