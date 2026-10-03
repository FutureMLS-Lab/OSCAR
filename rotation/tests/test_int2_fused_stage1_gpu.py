"""The fused HP+INT2 stage-1 must reproduce the two-launch mixed decode.

``decode_attention_fwd_int2_unified`` can run the HP window and the INT2 tier
as one grid (``SGLANG_OSCAR_FUSED_STAGE1``) or as two launches. The fused
kernel's two tier bodies are copies of the standalone kernels, so the partial
softmax states it writes must be bit-identical, not merely close: a wrong
split offset or tier switch would still be "close" on short contexts.
Also checks the fused path captures and replays inside a CUDA graph with
padded (out-of-arena) index tails, which is how the server runs it.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_pq_kv_gpu import _Layer, _make_pool  # noqa: E402

from sglang.srt.environ import envs  # noqa: E402

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _setup(*, head_dim, kv_heads, q_heads, hp_per_req, quant_per_req, bs, hp_max, quant_max):
    from sglang.srt.layers.attention.triton_ops.decode_attention import (
        decode_attention_fwd_int2_unified,
    )

    pool = _make_pool(
        head_dim=head_dim,
        v_head_dim=head_dim,
        head_num=kv_heads,
        num_quant_pages=2048,
    )
    torch.manual_seed(7)
    dev = "cuda"
    n_quant = bs * quant_per_req
    quant_locs = torch.arange(8, 8 + n_quant, dtype=torch.int64, device=dev)
    pool.set_kv_buffer(
        _Layer(),
        quant_locs,
        torch.randn(n_quant, kv_heads, head_dim, dtype=pool.hp_dtype, device=dev),
        torch.randn(n_quant, kv_heads, head_dim, dtype=pool.hp_dtype, device=dev),
        already_hadamard_transformed=True,
    )
    n_hp = bs * hp_per_req
    pool.hp_k_buffer[0][0:n_hp] = torch.randn(n_hp, kv_heads, head_dim, dtype=pool.hp_dtype, device=dev)
    pool.hp_v_buffer[0][0:n_hp] = torch.randn(n_hp, kv_heads, head_dim, dtype=pool.hp_dtype, device=dev)

    hp_indptr = torch.arange(0, bs + 1, dtype=torch.int32, device=dev) * hp_per_req
    hp_indices = torch.arange(0, n_hp, dtype=torch.int64, device=dev)
    quant_indptr = torch.arange(0, bs + 1, dtype=torch.int32, device=dev) * quant_per_req
    # graph-style padded buffer: the tail points far outside the arena
    quant_indices = torch.full((n_quant + 64,), pool.k_buffer[0].shape[0] + 4321, dtype=torch.int64, device=dev)
    quant_indices[:n_quant] = quant_locs
    hp_splits = torch.full((bs,), hp_max, dtype=torch.int32, device=dev)
    # per-request adaptive counts below the cap, like get_num_kv_splits produces
    quant_splits = torch.tensor([max(1, quant_max - i) for i in range(bs)], dtype=torch.int32, device=dev)
    total = hp_max + quant_max
    attn_logits = torch.empty((bs, q_heads, total, head_dim), dtype=torch.float32, device=dev)
    attn_lse = torch.empty((bs, q_heads, total), dtype=torch.float32, device=dev)
    sm_scale = head_dim**-0.5

    def launch(q_arg, out_arg):
        return decode_attention_fwd_int2_unified(
            q_arg,
            pool.hp_k_buffer[0],
            pool.hp_v_buffer[0],
            pool.k_buffer[0],
            pool.v_buffer[0],
            pool.k_scales_zeros[0],
            pool.v_scales_zeros[0],
            out_arg,
            hp_indptr,
            hp_indices,
            quant_indptr,
            quant_indices,
            attn_logits,
            attn_lse,
            hp_splits,
            quant_splits,
            hp_max,
            quant_max,
            sm_scale,
        )

    q = torch.randn(bs, q_heads, head_dim, dtype=pool.hp_dtype, device=dev)
    return pool, q, launch, attn_logits, attn_lse


def _run(launch, q, attn_logits, attn_lse, fused):
    out = torch.empty_like(q)
    with envs.SGLANG_OSCAR_FUSED_STAGE1.override(fused):
        launch(q, out)
    torch.cuda.synchronize()
    active = attn_lse > float("-inf")
    return out, attn_lse.clone(), attn_logits.clone(), active


@gpu
@pytest.mark.parametrize(
    "kv_heads,q_heads,quant_per_req,quant_max",
    [
        (2, 16, 300, 4),  # GQA 8, four active INT2 splits (96-token chunks)
        (2, 12, 40, 4),  # GQA 6 (non-power-of-two head block), mostly empty splits
        (4, 32, 700, 32),  # the production cap: 32 INT2 splits
    ],
)
def test_fused_stage1_matches_two_launch(kv_heads, q_heads, quant_per_req, quant_max):
    pool, q, launch, attn_logits, attn_lse = _setup(
        head_dim=128,
        kv_heads=kv_heads,
        q_heads=q_heads,
        hp_per_req=70,  # 70 tokens over 2 splits of >=32: both HP splits active
        quant_per_req=quant_per_req,
        bs=3,
        hp_max=2,
        quant_max=quant_max,
    )
    out_ref, lse_ref, logits_ref, active_ref = _run(launch, q, attn_logits, attn_lse, fused=False)
    out_fused, lse_fused, logits_fused, active_fused = _run(launch, q, attn_logits, attn_lse, fused=True)
    assert torch.equal(active_ref, active_fused), "the fused grid wrote a different set of splits"
    assert active_ref[:, :, :2].all(), "both HP splits should be active with 70 HP tokens"
    assert active_ref[:, :, 2].all(), "the first INT2 split should be active"
    assert torch.equal(lse_ref[active_ref], lse_fused[active_fused]), (
        f"stage-1 LSE differs: max |d| = {(lse_ref[active_ref] - lse_fused[active_fused]).abs().max().item()}"
    )
    assert torch.equal(logits_ref[active_ref], logits_fused[active_fused]), (
        f"stage-1 partial outputs differ: max |d| = {(logits_ref[active_ref] - logits_fused[active_fused]).abs().max().item()}"
    )
    assert torch.equal(out_ref, out_fused)
    assert torch.isfinite(out_fused).all()


@gpu
def test_fused_stage1_is_cuda_graph_safe():
    pool, q, launch, attn_logits, attn_lse = _setup(
        head_dim=128, kv_heads=2, q_heads=16, hp_per_req=70, quant_per_req=300, bs=3, hp_max=2, quant_max=8
    )
    ref, *_ = _run(launch, q, attn_logits, attn_lse, fused=False)
    with envs.SGLANG_OSCAR_FUSED_STAGE1.override(True):
        static_q = q.clone()
        static_out = torch.empty_like(q)
        launch(static_q, static_out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch(static_q, static_out)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(static_out, ref)
        q2 = torch.randn_like(q)
        static_q.copy_(q2)
        graph.replay()
        torch.cuda.synchronize()
        replayed = static_out.clone()
    ref2, *_ = _run(launch, q2, attn_logits, attn_lse, fused=False)
    assert torch.equal(replayed, ref2)
