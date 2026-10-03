"""GPU equivalence of the two one-launch replacements on the mixed-KV decode
path: the MiniMax side-cache follow (vs follow_flush) and the HP-recent
decode allocation (vs the tensor sequence). Needs CUDA."""
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.minimax_int2_kv_pool import follow_flush, follow_flush_fused
from sglang.srt.mem_cache.unified_kv_allocator import _hp_recent_alloc_kernel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.int8, torch.uint8])
def test_follow_flush_fused_matches_reference(dtype):
    torch.manual_seed(0)
    dev = "cuda"
    layers, n_quant, n_hp, dim = 5, 512, 128, 96
    hp_off = n_quant
    base = torch.randn(layers, n_quant + n_hp, 1, dim, device=dev)
    cache_ref = base.to(dtype) if dtype == torch.bfloat16 else (base * 10).to(dtype)
    cache_f = cache_ref.clone()
    n_rows = 24
    valid = (torch.rand(n_rows, device=dev) > 0.3).to(torch.int8)
    valid[:2] = 0
    src = torch.randperm(n_hp, device=dev)[:n_rows].to(torch.int64)
    src = torch.where(valid.bool(), src, torch.full_like(src, -1))
    dst = torch.randperm(n_quant, device=dev)[:n_rows].to(torch.int64)
    plan = SimpleNamespace(valid_mask=valid, src_hp_slot=src, dst_quant_slots=dst)
    follow_flush(cache_ref, plan, hp_off)
    follow_flush_fused(cache_f, plan, hp_off)
    torch.cuda.synchronize()
    assert torch.equal(cache_f.view(torch.uint8), cache_ref.view(torch.uint8))


def test_hp_recent_alloc_kernel_matches_tensor_path():
    torch.manual_seed(0)
    dev = "cuda"
    n_req, ring, offset, bs = 40, 7, 10_000, 9
    cursors_ref = torch.randint(0, ring, (n_req,), dtype=torch.int32, device=dev)
    cursors_k = cursors_ref.clone()
    rpi = torch.randperm(n_req, device=dev)[:bs].to(torch.int64)
    # reference: the tensor sequence of alloc_hp_recent's decode path
    bases = rpi * ring + offset
    old = cursors_ref[rpi].to(torch.int64)
    slots_ref = (bases + old).contiguous()
    cursors_ref[rpi] = ((old + 1) % ring).to(torch.int32)
    out = torch.empty((bs,), dtype=torch.int64, device=dev)
    _hp_recent_alloc_kernel[(bs,)](rpi, cursors_k, out, bs, ring, offset, num_warps=1)
    torch.cuda.synchronize()
    assert torch.equal(out, slots_ref)
    assert torch.equal(cursors_k, cursors_ref)
