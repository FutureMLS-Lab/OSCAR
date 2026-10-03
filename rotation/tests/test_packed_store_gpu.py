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


@pytest.mark.parametrize("lloyd", [False, True])
def test_fused_extend_store_matches_python_path(lloyd):
    """Extend (prefill) write: three requests, per-token positions covering the
    sink, the middle and the recent tail, one request past the arena, one
    padded row. Arena row 0 is excluded: the Python path scatters every
    not-kept row onto it, the kernel leaves it alone (it is never read)."""
    from sglang.QuantKernel.mla_latent_int2 import store_extend_rows
    from sglang.srt.mem_cache.mla_packed_kv_pool import ring_rows_from_positions

    torch.manual_seed(1)
    dev = "cuda"
    R, rope, gs, bits = 512, 64, 128, 2
    P, W, max_reqs = 64, 512, 8
    per_req = P + W
    n_slots, n_hp = 8192, 1 + max_reqs * per_req
    # request 0: 700 tokens of a 700-token sequence (sink + middle + recent);
    # request 1: 40 tokens at the tail of a 3000-token sequence (recent);
    # request 9: past the arena, 30 tokens (never kept); plus one padded row (loc 0)
    seq_lens = torch.tensor([700, 3000, 3000], dtype=torch.int64, device=dev)
    ext = torch.tensor([700, 40, 30], dtype=torch.int64, device=dev)
    req = torch.tensor([0, 1, 9], dtype=torch.int64, device=dev)
    positions = torch.cat([torch.arange(0, 700), torch.arange(2960, 3000), torch.arange(2970, 3000)]).to(dev)
    n = positions.numel()
    loc = torch.randperm(n_slots - 1, device=dev)[:n].to(torch.int64) + 1
    loc[5] = 0  # padded row
    seq_e = torch.repeat_interleave(seq_lens, ext, output_size=n)
    req_e = torch.repeat_interleave(req, ext, output_size=n)
    c_rot = torch.randn(n, R, device=dev, dtype=torch.float32)
    k_pe = torch.randn(n, rope, device=dev, dtype=torch.bfloat16)

    ref = _state(n_slots, n_hp, R, rope, gs, bits, dev)
    ref[5][0] = -1  # arena row 0 is the dummy and starts unowned, as in the pool
    got = tuple(t.clone() for t in ref)
    codes, params, rope_buf, hp_c, hp_row, hp_owner = ref
    scatter_pack_rows(c_rot, loc.to(torch.int32), codes, params, gs, lloyd, bits)
    rope_buf[loc] = k_pe
    ring, keep = ring_rows_from_positions(positions, seq_e, req_e, loc, P, W, per_req, max_reqs)
    apply_window_writes(hp_c, hp_owner, hp_row, ring, keep, loc, c_rot, torch.bfloat16)

    codes2, params2, rope2, hp_c2, hp_row2, hp_owner2 = got
    store_extend_rows(c_rot, k_pe, loc, positions, seq_e, req_e, codes2, params2, rope2, hp_c2, hp_row2, hp_owner2,
                      sink=P, recent=W, per_req_hp=per_req, max_reqs=max_reqs,
                      group_size=gs, lloyd_max=lloyd, bits=bits)
    torch.cuda.synchronize()
    assert torch.equal(codes2, codes) and torch.equal(params2, params) and torch.equal(rope2, rope_buf)
    assert torch.equal(hp_row2, hp_row)
    assert torch.equal(hp_c2[1:], hp_c[1:]) and torch.equal(hp_owner2[1:], hp_owner[1:])
    assert int(hp_owner2[0]) == -1 and int(hp_owner[0]) == -1
    # the placement itself: sink and the last W positions kept, nothing else
    want_keep = ((positions < P) | (positions >= seq_e - W)) & (req_e < max_reqs) & (loc > 0)
    assert torch.equal(keep, want_keep)
    # req0: 64 sink (minus the padded row at position 5) + 512 recent; req1: 40; req9: none
    assert int(keep.sum()) == 63 + 512 + 40
