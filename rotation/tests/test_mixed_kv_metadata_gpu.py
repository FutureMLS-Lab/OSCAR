"""The parallel mixed-KV metadata build must reproduce the serial one exactly.

Decode classifies every token slot of every request as HP (slot >= HP offset)
or INT2 and scatters the two tier-local index lists plus their indptrs. The
parallel build (one program per 512-token block) replaces the per-request
serial scan; its layout has to be bit-identical because the attention kernels
read these buffers by position. Random layouts, ragged lengths, requests
shorter and longer than a block, and the sliding-window start offsets.
"""
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
HP_OFFSET = 100_000


def _stand_in(max_ctx, slots, device="cuda"):
    gen = torch.Generator(device="cpu").manual_seed(5)
    rtt = torch.randint(0, 50_000, (slots, max_ctx), generator=gen, dtype=torch.int32)
    hp = torch.rand(slots, max_ctx, generator=gen) < 0.2
    rtt[hp] = rtt[hp] + HP_OFFSET
    return SimpleNamespace(req_to_token=rtt.to(device), mixed_hp_global_offset=HP_OFFSET, device=device)


def _build(fn, be, req_pool_indices, seq_lens, bs, start_pos):
    dev = be.device
    total = int(seq_lens[:bs].sum().item()) + 7
    hp_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=dev)
    q_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=dev)
    hp_idx = torch.full((total,), -1, dtype=torch.int64, device=dev)
    q_idx = torch.full((total,), -1, dtype=torch.int64, device=dev)
    fn(be, req_pool_indices, seq_lens, hp_indptr, hp_idx, q_indptr, q_idx, bs, start_pos=start_pos)
    torch.cuda.synchronize()
    return hp_indptr, q_indptr, hp_idx, q_idx


@gpu
@pytest.mark.parametrize("with_start", [False, True])
def test_parallel_build_matches_serial(with_start):
    be = _stand_in(max_ctx=6000, slots=8)
    seq_lens = torch.tensor([5000, 17, 2049, 512, 1], dtype=torch.int64, device="cuda")
    bs = seq_lens.numel()
    req_pool_indices = torch.tensor([3, 0, 7, 5, 1], dtype=torch.int64, device="cuda")
    start_pos = torch.clamp(seq_lens - 1500, min=0).to(torch.int32) if with_start else None
    ser = _build(TritonAttnBackend._build_mixed_kv_indices_serial, be, req_pool_indices, seq_lens, bs, start_pos)
    par = _build(TritonAttnBackend._build_mixed_kv_indices_fast, be, req_pool_indices, seq_lens, bs, start_pos)
    for name, a, b in zip(("hp_indptr", "quant_indptr", "hp_indices", "quant_indices"), ser, par):
        assert torch.equal(a, b), f"{name} differs: serial {a[:12].tolist()} parallel {b[:12].tolist()}"
    # sanity on the content itself: HP indices are HP-local and the quant ones are raw slots
    n_hp = int(ser[0][-1].item()); n_q = int(ser[1][-1].item())
    scanned = (seq_lens - (start_pos.to(torch.int64) if with_start else 0)).sum().item()
    assert n_hp + n_q == scanned
    assert n_hp > 0 and n_q > 0
    assert (ser[2][:n_hp] >= 0).all() and (ser[2][:n_hp] < 50_000).all()
    assert (ser[3][:n_q] < HP_OFFSET).all() and (ser[3][:n_q] >= 0).all()
    assert (ser[2][n_hp:] == -1).all() and (ser[3][n_q:] == -1).all(), "wrote past the tier length"


@gpu
def test_parallel_build_bs1_long_context():
    be = _stand_in(max_ctx=70_000, slots=2)
    seq_lens = torch.tensor([65_536 + 300], dtype=torch.int64, device="cuda")
    rpi = torch.tensor([1], dtype=torch.int64, device="cuda")
    ser = _build(TritonAttnBackend._build_mixed_kv_indices_serial, be, rpi, seq_lens, 1, None)
    par = _build(TritonAttnBackend._build_mixed_kv_indices_fast, be, rpi, seq_lens, 1, None)
    for a, b in zip(ser, par):
        assert torch.equal(a, b)
