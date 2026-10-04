"""CPU checks for the OptR-style rotation trainer: the quantizer reproduces the
write kernel's levels, the causal/sequence mask follows position restarts, and
a tiny end-to-end run writes an orthogonal per-head pair that scores no worse
than its init on the training objective."""
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import train_optr_rotations as optr  # noqa: E402


def test_uniform_levels_match_the_replay_arithmetic():
    torch.manual_seed(0)
    x = torch.randn(6, 16, dtype=torch.float64)
    y = optr.int2_dequant(x, clip=0.0, lloyd_max=False, group=16)
    mn, mx = x.min(1).values, x.max(1).values
    scale = (mx - mn) / 3.0
    zero = -mn / scale
    q = torch.clamp(torch.floor(x / scale[:, None] + zero[:, None] + 0.5), 0, 3)
    torch.testing.assert_close(y, (q - zero[:, None]) * scale[:, None])
    for row in range(6):
        assert len(torch.unique(y[row])) <= 4
    grouped = optr.int2_dequant(x, clip=0.0, lloyd_max=False, group=8)
    torch.testing.assert_close(grouped[:, :8], optr.int2_dequant(x[:, :8], clip=0.0, lloyd_max=False, group=8))


def test_clip_removes_the_row_extreme_and_lloyd_max_matches_the_kernel_formula():
    x = torch.tensor([[0.1, -0.2, 0.3, -0.1, 0.05, 0.15, -0.25, 9.0]], dtype=torch.float64)
    clipped = optr.int2_dequant(x, clip=0.8, lloyd_max=False, group=8)
    assert clipped.abs().max() <= 0.3 + 1e-9  # the 9.0 outlier is clamped to the 0.8-quantile (0.3)
    torch.manual_seed(1)
    z = torch.randn(4, 32, dtype=torch.float64)
    lm = optr.int2_dequant(z, clip=0.0, lloyd_max=True, group=32)
    mean = z.mean(1, keepdim=True)
    diff = z - mean
    std = (diff.pow(2).mean(1, keepdim=True) + 1e-8).sqrt()
    zs = diff / std
    q = (zs >= -0.9810652732849121).double() + (zs >= 0.0).double() + (zs >= 0.9810652732849121).double()
    step = 2 * 1.5095585584640503 / 3.0
    scale = step * 1.16 * std
    zero = 1.5095585584640503 / step - mean / scale
    torch.testing.assert_close(lm, (q - zero) * scale)
    for row in range(4):
        assert len(torch.unique(lm[row])) <= 4


def test_mask_restarts_with_positions():
    positions = torch.tensor([0, 1, 2, 3, 0, 1, 2], dtype=torch.int32)
    mask = optr.causal_mask(positions, torch.tensor([2, 5]))
    assert mask[0].tolist() == [True, True, True, False, False, False, False]
    assert mask[1].tolist() == [False, False, False, False, True, True, False]


def _orthogonal(n: int, gen: torch.Generator) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(n, n, generator=gen))
    return q


def test_end_to_end_tiny_training_writes_an_orthogonal_pair_no_worse_than_init():
    gen = torch.Generator().manual_seed(3)
    heads, q_heads, hd, hidden, tokens, stride = 2, 4, 8, 12, 96, 4
    positions = torch.cat([torch.arange(32), torch.arange(32), torch.arange(32)]).to(torch.int32)
    layers = {}
    for lid in (1, 4):
        layers[lid] = {
            "k": torch.randn(tokens, heads, hd, generator=gen).to(torch.bfloat16) * 3 + 0.5,
            "v": torch.randn(tokens, heads, hd, generator=gen).to(torch.bfloat16),
            "q_samples": torch.randn(tokens // stride, q_heads, hd, generator=gen),
            "positions": positions,
            "q_scaling": hd ** -0.5,
            "count": tokens,
        }
    rows = {"format_version": 1, "model_path": "tiny", "model_revision": None, "prompt_sha256": "0" * 64, "tokens": tokens,
            "tp_rank": 0, "tp_size": 1, "q_head_offset": 0, "local_kv_heads": heads, "global_kv_heads": heads,
            "global_q_heads": q_heads, "head_dim": hd, "v_head_dim": hd, "q_sample_stride": stride, "layers": layers}
    init_layers = {lid: {"layer_id": lid, "rotation": torch.stack([_orthogonal(hd, gen) for _ in range(heads)]),
                         "eigenvalues": torch.ones(heads, hd), "k_mean": torch.randn(heads, hd, generator=gen) * 0.5}
                   for lid in (1, 4)}
    k_init = {"format_version": 3, "source_grouping": "head", "objective": "oscar2_center", "layers": init_layers}
    v_init = {"format_version": 3, "source_grouping": "head", "objective": "oscar2_v_pre_wo",
              "layers": {lid: {"layer_id": lid, "rotation": e["rotation"].clone(), "eigenvalues": None} for lid, e in init_layers.items()}}
    w_o = {lid: torch.randn(hidden, q_heads * hd, generator=gen) for lid in (1, 4)}
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "rows"))
        os.makedirs(os.path.join(d, "init"))
        torch.save(rows, os.path.join(d, "rows", "oscar_rows_rank0.pt"))
        torch.save(k_init, os.path.join(d, "init", "k_rotation_oscar2_center.pt"))
        torch.save(v_init, os.path.join(d, "init", "v_rotation_oscar2_center.pt"))
        torch.save(w_o, os.path.join(d, "w_o.pt"))
        out = os.path.join(d, "optr")
        rc = optr.main(["--rows-dir", os.path.join(d, "rows"), "--init-dir", os.path.join(d, "init"), "--out", out,
                        "--o-proj-file", os.path.join(d, "w_o.pt"), "--steps", "4", "--lr", "0.01", "--device", "cpu"])
        assert rc == 0
        k_state = torch.load(os.path.join(out, "k_rotation_oscar2_optr.pt"))
        v_state = torch.load(os.path.join(out, "v_rotation_oscar2_optr.pt"))
        import json

        summary = json.load(open(os.path.join(out, "training.json")))
    assert sorted(k_state["layers"]) == [1, 4] and k_state["transform"]["orthogonal"]
    for state in (k_state, v_state):
        for e in state["layers"].values():
            r = e["rotation"]
            assert r.shape == (heads, hd, hd)
            torch.testing.assert_close(r @ r.transpose(-1, -2), torch.eye(hd).expand(heads, hd, hd), atol=1e-4, rtol=0)
    assert k_state["layers"][1]["k_mean"].shape == (heads, hd)
    for s in summary["layers"].values():
        assert s["k_rel_final"] <= s["k_rel_init"] + 1e-9
        assert s["v_rel_final"] <= s["v_rel_init"] + 1e-9
        assert 0.0 < s["k_rel_init"] < 10.0
