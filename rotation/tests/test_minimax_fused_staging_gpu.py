"""GPU equivalence of the fused MiniMax decode staging against the reference
path (decode_block_rows + dequantize_prefix_kv + fill_decode_fake_table):
same K/V rows bit for bit, same fake table. Needs a CUDA device."""
import types

import pytest
import torch

from sglang.srt.layers.attention.minimax_sparse_staging import (
    decode_block_rows,
    fill_decode_fake_table,
    stage_decode_blocks_fused,
)
from sglang.srt.layers.attention.quantized_kv_prefill import dequantize_prefix_kv

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("bs,topk,block_size,heads,head_dim,v_head_dim,groups", [
    (3, 16, 128, 1, 128, 128, 1),
    (2, 8, 64, 2, 128, 128, 2),
    (1, 4, 32, 1, 96, 64, 1),
])
def test_fused_staging_matches_reference(bs, topk, block_size, heads, head_dim, v_head_dim, groups):
    torch.manual_seed(0)
    dev = "cuda"
    n_quant, n_hp, max_ctx = 4096, 512, 3000
    hp_off = n_quant
    quant_k = torch.randint(0, 256, (n_quant, heads, head_dim // 4), dtype=torch.uint8, device=dev)
    quant_v = torch.randint(0, 256, (n_quant, heads, v_head_dim // 4), dtype=torch.uint8, device=dev)
    sz_k = torch.randn(n_quant, heads, 2 * groups, device=dev, dtype=torch.float32)
    sz_v = torch.randn(n_quant, heads, 2 * groups, device=dev, dtype=torch.float32)
    hp_k = torch.randn(n_hp, heads, head_dim, device=dev, dtype=torch.bfloat16)
    hp_v = torch.randn(n_hp, heads, v_head_dim, device=dev, dtype=torch.bfloat16)
    # a page table mixing quant and window slots, plus requests of different length
    req_to_token = torch.randint(0, n_quant + n_hp, (8, max_ctx), dtype=torch.int32, device=dev)
    req_pool_indices = torch.randperm(8, device=dev)[:bs].to(torch.int64)
    seq_lens = torch.tensor([max_ctx - 7, 1500, 700][:bs], device=dev, dtype=torch.int32)
    n_blocks = (max_ctx + block_size - 1) // block_size
    # real top-k indices are distinct per request (a duplicate would make two
    # staged rows claim the same positions, and both paths then depend on
    # scatter order); sample without replacement, keep some -1 holes
    topk_blk = torch.stack([torch.randperm(n_blocks, device=dev)[:topk] for _ in range(bs)]).to(torch.int32)
    topk_blk[0, 0] = -1  # at least one dead entry
    topk_blk[-1, -3:] = -1
    ar_block = torch.arange(block_size, device=dev, dtype=torch.int64)
    pool = types.SimpleNamespace(
        dtype="int2", head_num=heads, head_dim=head_dim, v_head_dim=v_head_dim,
        hp_global_offset=hp_off, mixed_kv_enabled=lambda: True,
        get_raw_key_buffer=lambda l: quant_k, get_raw_value_buffer=lambda l: quant_v,
        get_key_scales_zeros=lambda l: sz_k, get_value_scales_zeros=lambda l: sz_v,
        get_hp_key_buffer=lambda l: hp_k, get_hp_value_buffer=lambda l: hp_v,
    )
    n_rows = bs * topk * block_size
    dump_col = max_ctx
    # reference
    slots, pos, valid = decode_block_rows(topk_blk=topk_blk, req_to_token=req_to_token,
                                          req_pool_indices=req_pool_indices, seq_lens=seq_lens,
                                          block_size=block_size, ar_block=ar_block)
    k_ref = torch.empty(n_rows, heads, head_dim, device=dev, dtype=torch.bfloat16)
    v_ref = torch.empty(n_rows, heads, v_head_dim, device=dev, dtype=torch.bfloat16)
    dequantize_prefix_kv(kv_pool=pool, layer_id=0, prefix_indices=slots.reshape(-1),
                         model_dtype=torch.bfloat16, out_k=k_ref, out_v=v_ref)
    fake_ref = torch.full((bs, max_ctx + 1), -7, dtype=torch.int32, device=dev)
    rows = torch.arange(n_rows, device=dev, dtype=torch.int32).view(bs, topk * block_size)
    fill_decode_fake_table(fake=fake_ref, pos=pos, valid=valid, rows=rows, dump_col=dump_col)
    # fused
    k_f = torch.empty_like(k_ref); v_f = torch.empty_like(v_ref)
    fake_f = torch.full((bs, max_ctx + 1), -7, dtype=torch.int32, device=dev)
    stage_decode_blocks_fused(
        topk_blk=topk_blk, req_to_token=req_to_token, req_pool_indices=req_pool_indices, seq_lens=seq_lens,
        block_size=block_size, quant_k=quant_k, scales_zeros_k=sz_k, hp_k=hp_k,
        quant_v=quant_v, scales_zeros_v=sz_v, hp_v=hp_v, hp_global_offset=hp_off,
        out_k=k_f, out_v=v_f, fake=fake_f, dump_col=dump_col)
    torch.cuda.synchronize()
    assert torch.equal(k_f, k_ref), (k_f.float() - k_ref.float()).abs().max()
    assert torch.equal(v_f, v_ref), (v_f.float() - v_ref.float()).abs().max()
    # live entries must match exactly; the dump column receives dead rows in
    # unspecified order in both paths, so compare it as a set
    live = fake_ref[:, :max_ctx]
    assert torch.equal(fake_f[:, :max_ctx], live)
    assert (fake_f[:, dump_col] != -7).tolist() == (fake_ref[:, dump_col] != -7).tolist()
