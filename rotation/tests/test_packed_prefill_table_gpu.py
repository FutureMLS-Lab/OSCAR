"""GPU: the one-launch prefill table remap equals the tensor expression it
replaces (bit for bit, int32), holes kept, over a non-contiguous table too."""
import pytest
import torch

from sglang.srt.layers.attention.nsa.packed_staging import remap_prefill_table

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _reference(pt, slot_to_ragged):
    pt = pt.to(torch.int32)
    valid = pt >= 0
    safe = torch.where(valid, pt, torch.zeros_like(pt))
    return torch.where(valid, slot_to_ragged[safe.to(torch.int64)].to(torch.int32), pt)


@pytest.mark.parametrize("num_q,topk", [(3, 2048), (17, 256), (1, 2048)])
def test_remap_matches_reference(num_q, topk):
    torch.manual_seed(0)
    dev = "cuda"
    n_slots = 70000
    slot_to_ragged = torch.randint(0, 65536, (n_slots,), dtype=torch.int32, device=dev)
    pt = torch.randint(-1, n_slots, (num_q, topk), dtype=torch.int32, device=dev)
    pt[0, :5] = -1
    out = remap_prefill_table(pt, slot_to_ragged)
    torch.cuda.synchronize()
    assert out.dtype == torch.int32 and out.shape == pt.shape
    assert torch.equal(out, _reference(pt, slot_to_ragged))
    # a strided view (every other row of a wider table) must give the same answer
    wide = torch.randint(-1, n_slots, (num_q * 2, topk), dtype=torch.int32, device=dev)
    view = wide[::2]
    assert torch.equal(remap_prefill_table(view, slot_to_ragged), _reference(view, slot_to_ragged))
    # int64 input (the fused top-k path hands over int64 indices on some builds)
    assert torch.equal(remap_prefill_table(pt.to(torch.int64), slot_to_ragged), _reference(pt, slot_to_ragged))
