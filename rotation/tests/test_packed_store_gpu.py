"""GPU equivalence of the one-launch packed-latent decode store against the
Python write path of ``_packed_store`` (scatter_pack_rows, the rope index_put,
decode_ring_rows and apply_window_writes): identical codes, params, rope rows,
arena rows and slot<->arena bookkeeping. Needs CUDA."""
import pytest
import torch

from sglang.QuantKernel.mla_latent_int2 import scatter_pack_rows, store_decode_rows
from sglang.srt.mem_cache.mla_packed_kv_pool import apply_window_writes, decode_ring_rows

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _state(n_slots, n_hp, R, rope, gs, bits, dev):
    g = torch.Generator(device=dev).manual_seed(1)
    codes = torch.randint(0, 256, (n_slots, R // (8 // bits)), dtype=torch.uint8, device=dev, generator=g)
    params = torch.randn(n_slots, 2 * (R // gs), device=dev, dtype=torch.float32, generator=g)
    rope_buf = torch.randn(n_slots, rope, device=dev, dtype=torch.bfloat16, generator=g)
    hp_c = torch.randn(n_hp, R, device=dev, dtype=torch.bfloat16, generator=g)
    hp_row = torch.randint(-1, n_hp, (n_slots,), dtype=torch.int32, device=dev, generator=g)
    hp_owner = torch.randint(-1, n_slots, (n_hp,), dtype=torch.int32, device=dev, generator=g)
    return codes, params, rope_buf, hp_c, hp_row, hp_owner


@pytest.mark.parametrize("lloyd", [False, True])
@pytest.mark.parametrize("bs", [1, 6])
def test_fused_store_matches_python_path(lloyd, bs):
    torch.manual_seed(0)
    dev = "cuda"
    R, rope, gs, bits = 512, 64, 128, 2
    P, W, max_reqs = 64, 512, 8
    per_req = P + W
    n_slots, n_hp = 4096, 1 + max_reqs * per_req
    # rows: sink position, recent position, a middle (not kept) position, a
    # request past the arena, a padded row writing slot 0, a wrapped ring row
    seq_lens = torch.tensor([40, 3000, 700, 3000, 3000, 2000][:bs], dtype=torch.int64, device=dev)
    req_idx = torch.tensor([0, 1, 2, 9, 3, 4][:bs], dtype=torch.int64, device=dev)
    loc = torch.tensor([17, 2500, 999, 1200, 0, 3333][:bs], dtype=torch.int64, device=dev)
    c_rot = torch.randn(bs, R, device=dev, dtype=torch.float32)
    k_pe = torch.randn(bs, rope, device=dev, dtype=torch.bfloat16)

    ref = _state(n_slots, n_hp, R, rope, gs, bits, dev)
    got = tuple(t.clone() for t in ref)

    # reference: the Python sequence of _packed_store
    codes, params, rope_buf, hp_c, hp_row, hp_owner = ref
    scatter_pack_rows(c_rot, loc.to(torch.int32), codes, params, gs, lloyd, bits)
    rope_buf[loc] = k_pe
    ring, keep = decode_ring_rows(seq_lens, req_idx, loc, P, W, per_req, max_reqs)
    apply_window_writes(hp_c, hp_owner, hp_row, ring, keep, loc, c_rot, torch.bfloat16)

    # fused
    codes2, params2, rope2, hp_c2, hp_row2, hp_owner2 = got
    store_decode_rows(c_rot, k_pe, loc, seq_lens, req_idx, codes2, params2, rope2, hp_c2, hp_row2, hp_owner2,
                      sink=P, recent=W, per_req_hp=per_req, max_reqs=max_reqs,
                      group_size=gs, lloyd_max=lloyd, bits=bits)
    torch.cuda.synchronize()
    assert torch.equal(codes2, codes)
    assert torch.equal(params2, params)
    assert torch.equal(rope2, rope_buf)
    assert torch.equal(hp_row2, hp_row)
    # row 0 of the arena takes every not-kept row in both paths; with several
    # such rows its final value depends on scatter order, so compare the rest
    # exactly and row 0 only when at most one row aimed at it
    assert torch.equal(hp_c2[1:], hp_c[1:])
    assert torch.equal(hp_owner2[1:], hp_owner[1:])
    if int((ring == 0).sum()) <= 1:
        assert torch.equal(hp_c2[0], hp_c[0]) and bool(hp_owner2[0] == hp_owner[0])
    # and the placement itself did what the windows promise
    pos = seq_lens - 1
    want_keep = ((pos < P) | (pos >= seq_lens - W)) & (req_idx < max_reqs) & (loc > 0)
    assert torch.equal(keep, want_keep)
