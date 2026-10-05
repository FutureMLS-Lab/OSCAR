"""Product-quantized K/V tiers of the unified pool: codebook validation (CPU),
and on a GPU the prefill writes (PQ V compact rows, RVQ stage-2 codes,
cache moves), the graph-safe unified decode against a dense reference (both
ADC and reconstruct paths), and the decode-time flush."""
import os
import tempfile

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.unified_kv_pool import UnifiedInt2HPKVPool
from sglang.srt.runtime_context import get_parallel

_DIR = tempfile.mkdtemp(prefix="oscar_pq_test_")
_ROT_CACHE = {}


class _Layer:
    layer_id = 0
    oscar_v_rotation_absorbed = False


def _identity_rotation_paths(head_dim: int, v_head_dim: int, layer_num: int):
    key = (head_dim, v_head_dim, layer_num)
    if key not in _ROT_CACHE:
        paths = []
        for tag, dim in (("k", head_dim), ("v", v_head_dim)):
            path = os.path.join(_DIR, f"{tag}_d{dim}_l{layer_num}.pt")
            torch.save(
                {"layers": {i: {"rotation": torch.eye(dim)} for i in range(layer_num)}}, path
            )
            paths.append(path)
        _ROT_CACHE[key] = tuple(paths)
    return _ROT_CACHE[key]


def _codebook_path(name, *, head_dim, layer_num, n_sub=8, n_centroids=16, rvq=False):
    sub_dim = head_dim // n_sub
    gen = torch.Generator().manual_seed(2026)
    stage1 = torch.randn(layer_num, n_sub, n_centroids, sub_dim, generator=gen)
    path = os.path.join(_DIR, f"{name}.pt")
    common = {"layer_ids": list(range(layer_num)), "n_sub": n_sub, "sub_dim": sub_dim}
    if rvq:
        stage2 = 0.25 * torch.randn(layer_num, n_sub, n_centroids, sub_dim, generator=gen)
        torch.save({**common, "codebooks_stage1": stage1, "codebooks_stage2": stage2}, path)
    else:
        torch.save({**common, "n_centroids": n_centroids, "codebooks_per_layer": stage1}, path)
    return path


def _make_pool(
    *,
    k_codebook="",
    v_codebook="",
    head_dim=64,
    v_head_dim=64,
    layer_num=1,
    head_num=2,
    num_quant_pages=16,
    device="cuda",
    lloyd_max=False,
):
    k_rot, v_rot = _identity_rotation_paths(head_dim, v_head_dim, layer_num)
    with (
        envs.SGLANG_OSCAR_K_ROTATION_PATH.override(k_rot),
        envs.SGLANG_OSCAR_V_ROTATION_PATH.override(v_rot),
        envs.SGLANG_OSCAR_K_QUANTIZER.override("pq" if k_codebook else "int2"),
        envs.SGLANG_OSCAR_V_QUANTIZER.override("pq" if v_codebook else "int2"),
        envs.SGLANG_OSCAR_PQ_K_CODEBOOK.override(k_codebook),
        envs.SGLANG_OSCAR_PQ_V_CODEBOOK.override(v_codebook),
        envs.SGLANG_LLOYD_MAX.override(lloyd_max),
        envs.SGLANG_OSCAR_CALIBRATION_ACTIVE.override(False),
        get_parallel().override(attn_tp_rank=0),
    ):
        return UnifiedInt2HPKVPool(
            num_quant_pages=num_quant_pages,
            hp_dtype=torch.bfloat16,
            hp_prefix_tokens=32,
            hp_recent_tokens=128,
            dtype="int2",
            head_num=head_num,
            head_dim=head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=False,
            max_req_slots=8,
            v_head_dim=v_head_dim,
            start_layer=0,
            end_layer=layer_num - 1,
            model_dtype=torch.bfloat16,
            kv_cache_quant_group_size=None,
            scale_dtype=torch.float32,
            num_hp_prefix_slots=64,
        )


def _write_quant(pool, loc, k, v):
    pool.set_kv_buffer(_Layer(), loc, k, v, already_hadamard_transformed=True)


# -- CPU ----------------------------------------------------------------------


def test_codebook_metadata_mismatch_fails_before_allocation():
    path = _codebook_path("pq_bad_metadata", head_dim=64, layer_num=1)
    data = torch.load(path, map_location="cpu", weights_only=False)
    data["n_sub"] = 7
    torch.save(data, path)
    with pytest.raises(ValueError, match="metadata n_sub"):
        _make_pool(k_codebook=path, device="cpu")
    data["n_sub"] = 8
    data["layer_ids"] = [0, 0]
    data["codebooks_per_layer"] = data["codebooks_per_layer"].repeat(2, 1, 1, 1)
    torch.save(data, path)
    with pytest.raises(ValueError, match="layer_ids must be unique"):
        _make_pool(k_codebook=path, device="cpu")
    bad = _codebook_path("pq_bad_centroids", head_dim=64, layer_num=1)
    data = torch.load(bad, map_location="cpu", weights_only=False)
    data["codebooks_per_layer"] = data["codebooks_per_layer"][:, :, :15]
    data["n_centroids"] = 15
    torch.save(data, bad)
    with pytest.raises(ValueError, match="power of two"):
        _make_pool(k_codebook=bad, device="cpu")
    with pytest.raises(ValueError, match="requires SGLANG_OSCAR_K_QUANTIZER=pq"):
        _make_pool(v_codebook=_codebook_path("pq_v_only", head_dim=64, layer_num=1), device="cpu")


def test_pq_pool_geometry_on_cpu():
    k_path = _codebook_path("pq_k_geom", head_dim=64, layer_num=2, rvq=True)
    v_path = _codebook_path("pq_v_geom", head_dim=64, layer_num=2)
    pool = _make_pool(k_codebook=k_path, v_codebook=v_path, layer_num=2, device="cpu")
    assert pool.k_quantizer == "pq" and pool.v_quantizer == "pq"
    assert pool.k_buffer[0].shape[-1] == 8 and pool.v_buffer[0].shape[-1] == 8
    # 16-centroid residual codes pack two per byte: 8 sub-vectors -> 4 bytes
    assert pool.k_buffer2 is not None and pool.k_buffer2[1].shape[-1] == 4
    assert pool.k_scales_zeros[0].shape[-1] == 0 and pool.v_scales_zeros[0].shape[-1] == 0
    assert pool.pq_k_set.code_bytes == 12 and pool.pq_v_set.code_bytes == 8
    assert pool.pq_k_codebook2(1) is not None and pool.pq_v_codebook(1) is not None
    assert "k_buffer2" in pool.get_raw_kv_buffer(1)


# -- GPU ----------------------------------------------------------------------

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@gpu
def test_pq_v_prefill_writes_only_compact_code_rows():
    from sglang.QuantKernel.oscar_pq_kv import pq_decode_rows, pq_encode
    from sglang.srt.layers.attention.quantized_kv_prefill import dequantize_prefix_kv

    k_path = _codebook_path("pq_k_prefill", head_dim=64, layer_num=1)
    v_path = _codebook_path("pq_v_prefill", head_dim=64, layer_num=1)
    pool = _make_pool(k_codebook=k_path, v_codebook=v_path)
    pool.k_buffer[0].fill_(0xA5)
    pool.v_buffer[0].fill_(0xA5)
    torch.manual_seed(11)
    cache_k = torch.randn(2, pool.head_num, pool.head_dim, dtype=pool.hp_dtype, device="cuda")
    cache_v = torch.randn(2, pool.head_num, pool.v_head_dim, dtype=pool.hp_dtype, device="cuda")
    loc = torch.tensor([4, 8], dtype=torch.int64, device="cuda")

    ref_k = torch.full_like(pool.k_buffer[0], 0xA5)
    ref_v = torch.full_like(pool.v_buffer[0], 0xA5)
    pq_encode(cache_k, loc, ref_k, pool.pq_k_set.codebooks[0], pool.pq_k_set.norms[0])
    pq_encode(cache_v, loc, ref_v, pool.pq_v_set.codebooks[0], pool.pq_v_set.norms[0])
    _write_quant(pool, loc, cache_k, cache_v)
    torch.cuda.synchronize()
    assert torch.equal(pool.k_buffer[0][loc], ref_k[loc])
    assert torch.equal(pool.v_buffer[0][loc], ref_v[loc])
    assert (pool.k_buffer[0][5] == 0xA5).all() and (pool.v_buffer[0][5] == 0xA5).all()

    got_k, got_v = dequantize_prefix_kv(pool, 0, loc, pool.hp_dtype)
    exp_k = pq_decode_rows(ref_k[loc], pool.pq_k_set.codebooks[0], head_dim=pool.head_dim)
    exp_v = pq_decode_rows(ref_v[loc], pool.pq_v_set.codebooks[0], head_dim=pool.v_head_dim)
    assert torch.equal(got_k, exp_k.to(pool.hp_dtype))
    assert torch.equal(got_v, exp_v.to(pool.hp_dtype))

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _write_quant(pool, loc, cache_k, cache_v)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(pool.k_buffer[0][loc], ref_k[loc])
    assert torch.equal(pool.v_buffer[0][loc], ref_v[loc])


@gpu
def test_rvq_prefill_and_cache_move_preserve_stage2_codes():
    from sglang.QuantKernel.oscar_pq_kv import pq_decode_rows, pq_encode

    pool = _make_pool(k_codebook=_codebook_path("rvq_k_prefill", head_dim=64, layer_num=1, rvq=True))
    assert pool.k_buffer2 is not None
    torch.manual_seed(19)
    cache_k = torch.randn(2, pool.head_num, pool.head_dim, dtype=pool.hp_dtype, device="cuda")
    cache_v = torch.randn(2, pool.head_num, pool.v_head_dim, dtype=pool.hp_dtype, device="cuda")
    src = torch.tensor([3, 7], dtype=torch.int64, device="cuda")
    book = pool.pq_k_set
    ref1 = torch.zeros_like(pool.k_buffer[0])
    ref2 = torch.zeros_like(pool.k_buffer2[0])
    pq_encode(cache_k, src, ref1, book.codebooks[0], book.norms[0])
    recon1 = pq_decode_rows(ref1[src], book.codebooks[0], head_dim=pool.head_dim).to(cache_k.dtype)
    pq_encode((cache_k - recon1).contiguous(), src, ref2, book.stage2_codebooks[0], book.stage2_norms[0])

    _write_quant(pool, src, cache_k, cache_v)
    torch.cuda.synchronize()
    assert torch.equal(pool.k_buffer[0][src], ref1[src])
    assert torch.equal(pool.k_buffer2[0][src], ref2[src])

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _write_quant(pool, src, cache_k, cache_v)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(pool.k_buffer[0][src], ref1[src])
    assert torch.equal(pool.k_buffer2[0][src], ref2[src])

    dst = torch.tensor([12, 13], dtype=torch.int64, device="cuda")
    pool.move_kv_cache(dst, src)
    torch.cuda.synchronize()
    assert torch.equal(pool.k_buffer[0][dst], ref1[src])
    assert torch.equal(pool.k_buffer2[0][dst], ref2[src])


def _dense_reference(q, keys_per_batch, values_per_batch, sm_scale, kv_group):
    bs, q_heads, _ = q.shape
    ref = torch.empty_like(q)
    for b in range(bs):
        keys = keys_per_batch[b].float()
        values = values_per_batch[b].float()
        for h in range(q_heads):
            kv_head = h // kv_group
            scores = (keys[:, kv_head] @ q[b, h].float()) * sm_scale
            ref[b, h] = (torch.softmax(scores, dim=0)[:, None] * values[:, kv_head]).sum(0).to(ref.dtype)
    return ref


@gpu
def test_pq_unified_decode_is_graph_safe_and_matches_dense_reference():
    from sglang.QuantKernel.oscar_pq_kv import pq_decode_rows
    from sglang.srt.layers.attention.triton_ops.decode_attention_pq import (
        decode_attention_fwd_pq_unified,
    )

    pool = _make_pool(
        k_codebook=_codebook_path("pq_k_decode", head_dim=64, layer_num=1),
        v_codebook=_codebook_path("pq_v_decode", head_dim=64, layer_num=1),
    )
    torch.manual_seed(31)
    quant_k = torch.randn(4, pool.head_num, pool.head_dim, dtype=pool.hp_dtype, device="cuda")
    quant_v = torch.randn(4, pool.head_num, pool.v_head_dim, dtype=pool.hp_dtype, device="cuda")
    quant_locs = torch.tensor([4, 5, 6, 7], dtype=torch.int64, device="cuda")
    _write_quant(pool, quant_locs, quant_k, quant_v)
    pool.hp_k_buffer[0][0:2] = torch.randn(2, pool.head_num, pool.head_dim, dtype=pool.hp_dtype, device="cuda")
    pool.hp_v_buffer[0][0:2] = torch.randn(2, pool.head_num, pool.v_head_dim, dtype=pool.hp_dtype, device="cuda")

    bs, q_heads = 2, 12  # GQA 6: a non-power-of-two group
    q = torch.randn(bs, q_heads, pool.head_dim, dtype=pool.hp_dtype, device="cuda")
    hp_indptr = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")
    hp_indices = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
    quant_indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
    # padded graph buffer: entries past indptr point far outside the arena
    quant_indices = torch.full((16,), pool.k_buffer[0].shape[0] + 321, dtype=torch.int64, device="cuda")
    quant_indices[:4] = quant_locs
    ones = torch.ones((bs,), dtype=torch.int32, device="cuda")
    attn_logits = torch.empty((bs, q_heads, 2, pool.v_head_dim), dtype=torch.float32, device="cuda")
    attn_lse = torch.empty((bs, q_heads, 2), dtype=torch.float32, device="cuda")
    sm_scale = pool.head_dim**-0.5

    def launch(q_arg, out_arg):
        return decode_attention_fwd_pq_unified(
            q_arg, pool.hp_k_buffer[0], pool.hp_v_buffer[0], pool.k_buffer[0], pool.v_buffer[0],
            pool.v_scales_zeros[0], out_arg, hp_indptr, hp_indices, quant_indptr, quant_indices,
            attn_logits, attn_lse, ones, ones, 1, 1, sm_scale,
            k_codebook=pool.pq_k_codebook(0), v_codebook=pool.pq_v_codebook(0),
        )

    dq_k = pq_decode_rows(pool.k_buffer[0][quant_locs], pool.pq_k_codebook(0), head_dim=pool.head_dim).to(pool.hp_dtype)
    dq_v = pq_decode_rows(pool.v_buffer[0][quant_locs], pool.pq_v_codebook(0), head_dim=pool.v_head_dim).to(pool.hp_dtype)
    keys = [torch.cat((pool.hp_k_buffer[0][b : b + 1], dq_k[b * 2 : b * 2 + 2])) for b in range(bs)]
    values = [torch.cat((pool.hp_v_buffer[0][b : b + 1], dq_v[b * 2 : b * 2 + 2])) for b in range(bs)]
    kv_group = q_heads // pool.head_num

    def check(q_arg, out_arg):
        torch.cuda.synchronize()
        torch.testing.assert_close(
            out_arg, _dense_reference(q_arg, keys, values, sm_scale, kv_group), atol=4e-2, rtol=4e-2
        )

    for adc in (0, 1):
        with envs.SGLANG_OSCAR_PQ_USE_ADC.override(adc):
            static_q = q.clone()
            static_out = torch.empty_like(q)
            launch(static_q, static_out)
            check(static_q, static_out)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch(static_q, static_out)
            graph.replay()
            check(static_q, static_out)
            static_q.copy_(torch.randn_like(static_q))
            graph.replay()
            check(static_q, static_out)


@gpu
def test_rvq_k_int2_v_decode_matches_reconstruction():
    from sglang.QuantKernel.oscar_pq_kv import pq_decode_rows
    from sglang.srt.layers.attention.triton_ops.decode_attention_pq import (
        decode_attention_fwd_pq_unified,
    )
    from sglang.srt.mem_cache.kv_quant_kernels import dequantize_kv_int2_triton

    pool = _make_pool(
        k_codebook=_codebook_path("rvq_k_decode", head_dim=128, layer_num=1, n_sub=16, rvq=True),
        head_dim=128,
        v_head_dim=128,
    )
    torch.manual_seed(37)
    quant_k = torch.randn(3, pool.head_num, pool.head_dim, dtype=pool.hp_dtype, device="cuda")
    quant_v = torch.randn(3, pool.head_num, pool.v_head_dim, dtype=pool.hp_dtype, device="cuda")
    loc = torch.tensor([4, 5, 6], dtype=torch.int64, device="cuda")
    _write_quant(pool, loc, quant_k, quant_v)

    q = torch.randn(1, pool.head_num, pool.head_dim, dtype=pool.hp_dtype, device="cuda")
    hp_indptr = torch.tensor([0, 0], dtype=torch.int32, device="cuda")
    hp_indices = torch.zeros((1,), dtype=torch.int64, device="cuda")
    quant_indptr = torch.tensor([0, 3], dtype=torch.int32, device="cuda")
    quant_indices = torch.full((8,), pool.k_buffer[0].shape[0] + 99, dtype=torch.int64, device="cuda")
    quant_indices[:3] = loc
    ones = torch.ones((1,), dtype=torch.int32, device="cuda")
    attn_logits = torch.empty((1, pool.head_num, 2, pool.head_dim), dtype=torch.float32, device="cuda")
    attn_lse = torch.empty((1, pool.head_num, 2), dtype=torch.float32, device="cuda")
    sm_scale = pool.head_dim**-0.5

    def launch(q_arg, out_arg):
        return decode_attention_fwd_pq_unified(
            q_arg, pool.hp_k_buffer[0], pool.hp_v_buffer[0], pool.k_buffer[0], pool.v_buffer[0],
            pool.v_scales_zeros[0], out_arg, hp_indptr, hp_indices, quant_indptr, quant_indices,
            attn_logits, attn_lse, ones, ones, 1, 1, sm_scale,
            k_codebook=pool.pq_k_codebook(0), k_codes2=pool.get_raw_key_buffer2(0),
            k_codebook2=pool.pq_k_codebook2(0),
        )

    rk = (
        pq_decode_rows(pool.k_buffer[0][loc], pool.pq_k_codebook(0), head_dim=pool.head_dim)
        + pq_decode_rows(pool.k_buffer2[0][loc], pool.pq_k_codebook2(0), head_dim=pool.head_dim)
    ).to(pool.hp_dtype)
    rv = dequantize_kv_int2_triton(pool.v_buffer[0][loc], pool.v_scales_zeros[0][loc], pool.v_head_dim, pool.hp_dtype)
    static_q = q.clone()
    static_out = torch.empty_like(q)
    launch(static_q, static_out)
    torch.cuda.synchronize()
    torch.testing.assert_close(static_out, _dense_reference(static_q, [rk], [rv], sm_scale, 1), atol=4e-2, rtol=4e-2)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(static_q, static_out)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(static_out, _dense_reference(static_q, [rk], [rv], sm_scale, 1), atol=4e-2, rtol=4e-2)


@gpu
def test_pq_flush_demotes_hp_rows_and_remaps():
    from sglang.QuantKernel.gpu_flush_int2 import FlushPlan
    from sglang.QuantKernel.gpu_flush_pq import gpu_flush_pq_apply
    from sglang.QuantKernel.oscar_pq_kv import pq_encode
    from sglang.QuantKernel.oscar_rotation_clip_int2_kv import _launch_single_clip_int2

    pool = _make_pool(k_codebook=_codebook_path("pq_k_flush", head_dim=64, layer_num=2, rvq=True), layer_num=2)
    n = pool.N_Q
    torch.manual_seed(41)
    hp_rows = torch.arange(3, 3 + n, device="cuda", dtype=torch.int64)
    for l in range(2):
        pool.hp_k_buffer[l][hp_rows] = torch.randn(n, pool.head_num, pool.head_dim, dtype=pool.hp_dtype, device="cuda")
        pool.hp_v_buffer[l][hp_rows] = torch.randn(n, pool.head_num, pool.v_head_dim, dtype=pool.hp_dtype, device="cuda")
    dst = torch.arange(2 * n, 3 * n, device="cuda", dtype=torch.int64)
    flush_pos = torch.arange(10, 10 + n, device="cuda", dtype=torch.int32)
    plan = FlushPlan(
        returned_slot_ids=dst.clone(),
        src_hp_slot=hp_rows.clone(),
        flush_pos=flush_pos,
        valid_mask=torch.ones(n, dtype=torch.int8, device="cuda"),
        dst_quant_slots=dst,
        bs=1,
        flush_interval=n,
    )
    req_to_token = torch.zeros((2, 64), dtype=torch.int32, device="cuda")
    rpi = torch.tensor([1], dtype=torch.int64, device="cuda")
    gpu_flush_pq_apply(plan, req_pool_indices=rpi, req_to_token=req_to_token, kv_pool=pool)
    torch.cuda.synchronize()
    assert torch.equal(req_to_token[1, 10 : 10 + n], dst.to(torch.int32))
    for l in range(2):
        book = pool.pq_k_set
        ref1 = torch.zeros_like(pool.k_buffer[l])
        pq_encode(pool.hp_k_buffer[l][hp_rows], dst, ref1, book.codebooks[l], book.norms[l])
        assert torch.equal(pool.k_buffer[l][dst], ref1[dst])
        assert (pool.k_buffer2[l][dst] != 0).any()
        ref_v = torch.zeros_like(pool.v_buffer[l])
        ref_sz = torch.zeros_like(pool.v_scales_zeros[l])
        _launch_single_clip_int2(pool.hp_v_buffer[l][hp_rows], dst, ref_v, ref_sz, 0.0, hp_global_offset=None, lloyd_max=False)
        assert torch.equal(pool.v_buffer[l][dst], ref_v[dst])
        assert torch.equal(pool.v_scales_zeros[l][dst], ref_sz[dst])


@gpu
@pytest.mark.parametrize("variant", ["kpq", "krvq", "kpq-vpq"])
def test_pq_prefix_dequant_matches_reconstruction(variant):
    """Chunked prefill reads the PQ prefix through ``dequantize_prefix_kv``; a
    long-context recall probe depends on exactly this path, so it is checked
    row by row against the codebook reconstruction (and the HP rows against
    the HP buffer) on a mixed HP/PQ prefix in interleaved order."""
    from sglang.QuantKernel.oscar_pq_kv import pq_decode_rows
    from sglang.srt.layers.attention.quantized_kv_prefill import dequantize_prefix_kv
    from sglang.srt.mem_cache.kv_quant_kernels import dequantize_kv_int2_triton

    rvq = variant == "krvq"
    pool = _make_pool(
        k_codebook=_codebook_path(f"prefix_{variant}_k", head_dim=128, layer_num=1, n_sub=16, rvq=rvq),
        v_codebook=_codebook_path(f"prefix_{variant}_v", head_dim=128, layer_num=1, n_sub=16)
        if variant == "kpq-vpq"
        else "",
        head_dim=128,
        v_head_dim=128,
    )
    torch.manual_seed(41)
    n_quant, n_hp = 6, 3
    loc = torch.arange(4, 4 + n_quant, dtype=torch.int64, device="cuda")
    _write_quant(
        pool,
        loc,
        torch.randn(n_quant, pool.head_num, pool.head_dim, dtype=pool.hp_dtype, device="cuda"),
        torch.randn(n_quant, pool.head_num, pool.v_head_dim, dtype=pool.hp_dtype, device="cuda"),
    )
    pool.hp_k_buffer[0][0:n_hp] = torch.randn(n_hp, pool.head_num, pool.head_dim, dtype=pool.hp_dtype, device="cuda")
    pool.hp_v_buffer[0][0:n_hp] = torch.randn(n_hp, pool.head_num, pool.v_head_dim, dtype=pool.hp_dtype, device="cuda")
    hp_off = pool.hp_global_offset
    # interleaved: hp0 q4 q5 hp1 q6 q7 hp2 q8 q9
    prefix = torch.tensor([hp_off, 4, 5, hp_off + 1, 6, 7, hp_off + 2, 8, 9], dtype=torch.int64, device="cuda")
    is_hp = prefix >= hp_off

    k, v = dequantize_prefix_kv(pool, 0, prefix, torch.bfloat16)
    torch.cuda.synchronize()
    assert k.shape == (9, pool.head_num, 128) and v.shape == (9, pool.head_num, 128)

    rk = pq_decode_rows(pool.k_buffer[0][loc], pool.pq_k_codebook(0), head_dim=128).float()
    if rvq:
        rk = rk + pq_decode_rows(pool.k_buffer2[0][loc], pool.pq_k_codebook2(0), head_dim=128).float()
    if variant == "kpq-vpq":
        rv = pq_decode_rows(pool.v_buffer[0][loc], pool.pq_v_codebook(0), head_dim=128).float()
    else:
        rv = dequantize_kv_int2_triton(pool.v_buffer[0][loc], pool.v_scales_zeros[0][loc], 128, pool.hp_dtype).float()
    exp_k = torch.empty_like(k, dtype=torch.float32)
    exp_v = torch.empty_like(v, dtype=torch.float32)
    exp_k[is_hp] = pool.hp_k_buffer[0][prefix[is_hp] - hp_off].float()
    exp_v[is_hp] = pool.hp_v_buffer[0][prefix[is_hp] - hp_off].float()
    exp_k[~is_hp] = rk[prefix[~is_hp] - 4]
    exp_v[~is_hp] = rv[prefix[~is_hp] - 4]
    torch.testing.assert_close(k.float(), exp_k, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(v.float(), exp_v, atol=2e-2, rtol=2e-2)
    # the quant rows must be real reconstructions, not zeros or the HP rows
    assert exp_k[~is_hp].abs().mean() > 0.05
    assert not torch.allclose(k[~is_hp].float(), exp_k[is_hp][:1].expand_as(k[~is_hp]).float())


# -- Per-head codebooks -------------------------------------------------------


def _per_head_codebook_path(name, *, head_dim, layer_num, kv_heads, n_sub=8, n_centroids=16, replicate=False):
    """A ``[layers, kv_heads, n_sub, C, d]`` codebook file; ``replicate`` gives
    every head the per-layer codebook ``_codebook_path`` would have written
    (same generator), so a per-head pool must reproduce the shared one."""
    sub_dim = head_dim // n_sub
    gen = torch.Generator().manual_seed(2026)
    shared = torch.randn(layer_num, n_sub, n_centroids, sub_dim, generator=gen)
    if replicate:
        books = shared[:, None].expand(layer_num, kv_heads, n_sub, n_centroids, sub_dim).clone()
    else:
        books = torch.randn(layer_num, kv_heads, n_sub, n_centroids, sub_dim, generator=gen)
    path = os.path.join(_DIR, f"{name}.pt")
    torch.save(
        {"layer_ids": list(range(layer_num)), "n_sub": n_sub, "sub_dim": sub_dim, "n_centroids": n_centroids,
         "per_head": True, "kv_heads": kv_heads, "codebooks_per_layer": books},
        path,
    )
    return path


def test_per_head_codebook_header_and_rank_slice_on_cpu():
    from sglang.srt.mem_cache.oscar_pq_codebooks import load_pq_codebook_set, read_pq_codebook_header

    path = _per_head_codebook_path("ph_header", head_dim=64, layer_num=2, kv_heads=4)
    data = torch.load(path)
    header = read_pq_codebook_header(data, expected_head_dim=64, label="t")
    assert header.kind == "per_head" and header.kv_heads == 4 and header.code_bytes == 8
    full = load_pq_codebook_set(path, head_dim=64, layer_ids=[1, 0], device="cpu", label="t")
    assert full.codebooks[0].shape == (4, 8, 16, 8) and full.norms[0].shape == (4, 8, 16)
    torch.testing.assert_close(full.codebooks[0].float(), data["codebooks_per_layer"][1].to(torch.float16).float())
    part = load_pq_codebook_set(path, head_dim=64, layer_ids=[0], device="cpu", label="t", head_slice=(2, 2))
    assert part.codebooks[0].shape == (2, 8, 16, 8)
    torch.testing.assert_close(part.codebooks[0].float(), data["codebooks_per_layer"][0, 2:4].to(torch.float16).float())
    # the pool keeps this rank's heads of a per-head file (rank 1 of two)
    with get_parallel().override(attn_tp_rank=1):
        k_rot, v_rot = _identity_rotation_paths(64, 64, 1)
        with (
            envs.SGLANG_OSCAR_K_ROTATION_PATH.override(k_rot),
            envs.SGLANG_OSCAR_V_ROTATION_PATH.override(v_rot),
            envs.SGLANG_OSCAR_K_QUANTIZER.override("pq"),
            envs.SGLANG_OSCAR_V_QUANTIZER.override("int2"),
            envs.SGLANG_OSCAR_PQ_K_CODEBOOK.override(_per_head_codebook_path("ph_rank", head_dim=64, layer_num=1, kv_heads=4)),
            envs.SGLANG_OSCAR_PQ_V_CODEBOOK.override(""),
            envs.SGLANG_OSCAR_CALIBRATION_ACTIVE.override(False),
        ):
            pool = UnifiedInt2HPKVPool(
                num_quant_pages=4, hp_dtype=torch.bfloat16, hp_prefix_tokens=32, hp_recent_tokens=128,
                dtype="int2", head_num=2, head_dim=64, layer_num=1, device="cpu", enable_memory_saver=False,
                max_req_slots=8, v_head_dim=64, start_layer=0, end_layer=0, model_dtype=torch.bfloat16,
                kv_cache_quant_group_size=None, scale_dtype=torch.float32, num_hp_prefix_slots=64,
            )
    books = torch.load(os.path.join(_DIR, "ph_rank.pt"))["codebooks_per_layer"]
    assert pool.pq_k_codebook(0).shape == (2, 8, 16, 8)
    torch.testing.assert_close(pool.pq_k_codebook(0).float(), books[0, 2:4].to(torch.float16).float())


@gpu
def test_per_head_codebooks_replicated_match_the_shared_pool():
    """A per-head file whose heads all hold the per-layer codebook must give the
    same codes, the same decode (ADC and reconstruct) and the same prefix
    dequantization as the shared file: the head stride is the only difference."""
    from sglang.srt.layers.attention.quantized_kv_prefill import dequantize_prefix_kv
    from sglang.srt.layers.attention.triton_ops.decode_attention_pq import decode_attention_fwd_pq_unified

    pools = {
        "shared": _make_pool(k_codebook=_codebook_path("eq_k", head_dim=64, layer_num=1), v_codebook=_codebook_path("eq_v", head_dim=64, layer_num=1)),
        "per_head": _make_pool(
            k_codebook=_per_head_codebook_path("eq_k_ph", head_dim=64, layer_num=1, kv_heads=2, replicate=True),
            v_codebook=_per_head_codebook_path("eq_v_ph", head_dim=64, layer_num=1, kv_heads=2, replicate=True),
        ),
    }
    assert pools["per_head"].pq_k_codebook(0).dim() == 4 and pools["shared"].pq_k_codebook(0).dim() == 3
    torch.manual_seed(7)
    k = torch.randn(4, 2, 64, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(4, 2, 64, dtype=torch.bfloat16, device="cuda")
    locs = torch.tensor([4, 5, 6, 7], dtype=torch.int64, device="cuda")
    hp_k = torch.randn(2, 2, 64, dtype=torch.bfloat16, device="cuda")
    hp_v = torch.randn(2, 2, 64, dtype=torch.bfloat16, device="cuda")
    for pool in pools.values():
        _write_quant(pool, locs, k, v)
        pool.hp_k_buffer[0][0:2] = hp_k
        pool.hp_v_buffer[0][0:2] = hp_v
    torch.cuda.synchronize()
    assert torch.equal(pools["shared"].k_buffer[0][locs], pools["per_head"].k_buffer[0][locs])
    assert torch.equal(pools["shared"].v_buffer[0][locs], pools["per_head"].v_buffer[0][locs])

    bs, q_heads = 2, 12
    q = torch.randn(bs, q_heads, 64, dtype=torch.bfloat16, device="cuda")
    hp_indptr = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")
    hp_indices = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
    quant_indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
    quant_indices = torch.full((16,), 10_000, dtype=torch.int64, device="cuda")
    quant_indices[:4] = locs
    ones = torch.ones((bs,), dtype=torch.int32, device="cuda")
    attn_logits = torch.empty((bs, q_heads, 2, 64), dtype=torch.float32, device="cuda")
    attn_lse = torch.empty((bs, q_heads, 2), dtype=torch.float32, device="cuda")
    for adc in (0, 1):
        outs = {}
        with envs.SGLANG_OSCAR_PQ_USE_ADC.override(adc):
            for name, pool in pools.items():
                out = torch.empty_like(q)
                decode_attention_fwd_pq_unified(
                    q, pool.hp_k_buffer[0], pool.hp_v_buffer[0], pool.k_buffer[0], pool.v_buffer[0],
                    pool.v_scales_zeros[0], out, hp_indptr, hp_indices, quant_indptr, quant_indices,
                    attn_logits, attn_lse, ones, ones, 1, 1, 64**-0.5,
                    k_codebook=pool.pq_k_codebook(0), v_codebook=pool.pq_v_codebook(0),
                )
                torch.cuda.synchronize()
                outs[name] = out.clone()
        torch.testing.assert_close(outs["per_head"], outs["shared"], atol=1e-6, rtol=0)
    hp_off = pools["shared"].hp_global_offset
    prefix = torch.tensor([hp_off, 4, 5, hp_off + 1, 6, 7], dtype=torch.int64, device="cuda")
    ks, vs = dequantize_prefix_kv(pools["shared"], 0, prefix, torch.bfloat16)
    kp, vp = dequantize_prefix_kv(pools["per_head"], 0, prefix, torch.bfloat16)
    torch.cuda.synchronize()
    assert torch.equal(ks, kp) and torch.equal(vs, vp)


@gpu
def test_per_head_codebooks_encode_each_head_with_its_own_book():
    from sglang.QuantKernel.oscar_pq_kv import pq_decode_rows, pq_encode_decode_reference

    pool = _make_pool(k_codebook=_per_head_codebook_path("own_k_ph", head_dim=64, layer_num=1, kv_heads=2))
    torch.manual_seed(11)
    k = torch.randn(6, 2, 64, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(6, 2, 64, dtype=torch.bfloat16, device="cuda")
    locs = torch.arange(4, 10, dtype=torch.int64, device="cuda")
    _write_quant(pool, locs, k, v)
    torch.cuda.synchronize()
    book = pool.pq_k_codebook(0)
    assert book.shape == (2, 8, 16, 8)
    recon = pq_decode_rows(pool.k_buffer[0][locs], book, head_dim=64).float().cpu()
    for h in range(2):
        ref = pq_encode_decode_reference(k[:, h].float().cpu(), book[h].float().cpu())
        torch.testing.assert_close(recon[:, h], ref, atol=2e-2, rtol=2e-2)
    # the two heads' books differ, so a head decoded with the other head's book does not match
    other = pq_encode_decode_reference(k[:, 0].float().cpu(), book[1].float().cpu())
    assert not torch.allclose(recon[:, 0], other, atol=2e-2, rtol=2e-2)
