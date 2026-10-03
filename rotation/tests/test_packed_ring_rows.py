"""CPU: the packed pool's window placement (_ring_rows) for both forward
shapes, through the production method on a stub pool. The hook that feeds
it never fired on the upstream tree, so neither branch had run in months."""
import torch

from sglang.srt.mem_cache.mla_packed_kv_pool import (
    _PackedLatentMixin,
    decode_ring_rows,
    ring_rows_from_positions,
)

P, W, MAX_REQS = 64, 512, 8
PER_REQ = P + W


def _pool():
    p = _PackedLatentMixin.__new__(_PackedLatentMixin)
    p._latent_windows = True
    p._win_p, p._win_r = P, W
    p._per_req_hp, p._max_reqs = PER_REQ, MAX_REQS
    p._fb_window_meta = None
    p._window_fallback_logged = False
    p._window_meta_logged = False
    return p


def _expected(pos, seq, req, loc):
    keep = ((pos < P) | (pos >= seq - W)) & (req < MAX_REQS) & (loc > 0)
    ring = 1 + req * PER_REQ + torch.where(pos < P, pos, P + (pos - P).clamp(min=0) % W)
    return torch.where(keep, ring, torch.zeros_like(ring)), keep


def test_decode_branch_places_sink_recent_and_drops_the_rest():
    p = _pool()
    seq = torch.tensor([40, 3000, 700, 3000, 3000], dtype=torch.int64)   # sink / recent / recent / overflow req / padded
    req = torch.tensor([0, 1, 2, 9, 3], dtype=torch.int64)
    loc = torch.tensor([17, 2500, 999, 1200, 0], dtype=torch.int64)
    p._fb_window_meta = {"is_decode": True, "seq_lens": seq, "req_pool_indices": req}
    p._write_loc = loc
    ring, keep = p._ring_rows(5)
    want_ring, want_keep = _expected(seq - 1, seq, req, loc)
    assert torch.equal(ring, want_ring) and torch.equal(keep, want_keep)
    assert keep.tolist() == [True, True, True, False, False]
    assert torch.equal(ring, decode_ring_rows(seq, req, loc, P, W, PER_REQ, MAX_REQS)[0])


def test_extend_branch_expands_per_token_positions():
    p = _pool()
    # request 0: 5 new tokens at the tail of a 1000-token sequence (recent);
    # request 1: 3 tokens at 67..69 of a 70-token sequence (recent, past sink);
    # request 2: 4 tokens at 100..103 of a 1000-token sequence (middle: 2-bit)
    seq_lens = torch.tensor([1000, 70, 1000], dtype=torch.int64)
    ext = torch.tensor([5, 3, 4], dtype=torch.int64)
    req = torch.tensor([4, 1, 2], dtype=torch.int64)
    positions = torch.tensor(list(range(995, 1000)) + list(range(67, 70)) + list(range(100, 104)), dtype=torch.int64)
    loc = torch.arange(1, 13, dtype=torch.int64) * 11
    p._fb_window_meta = {"is_decode": False, "seq_lens": seq_lens, "req_pool_indices": req,
                         "positions": positions, "extend_seq_lens": ext}
    p._write_loc = loc
    ring, keep = p._ring_rows(12)
    seq_e = torch.repeat_interleave(seq_lens, ext)
    req_e = torch.repeat_interleave(req, ext)
    want_ring, want_keep = _expected(positions, seq_e, req_e, loc)
    assert torch.equal(ring, want_ring) and torch.equal(keep, want_keep)
    assert keep.tolist() == [True] * 8 + [False] * 4
    # recent rows land in the request's own ring, in order
    assert ring[:5].tolist() == [1 + 4 * PER_REQ + P + (q - P) % W for q in range(995, 1000)]


def test_missing_metadata_is_loud_and_safe(caplog):
    p = _pool()
    p._write_loc = torch.tensor([5], dtype=torch.int64)
    ring, keep = p._ring_rows(1)
    assert ring is None and keep is None
    assert p._window_fallback_logged is True


def test_shape_mismatch_falls_back_for_the_forward():
    p = _pool()
    p._fb_window_meta = {"is_decode": True, "seq_lens": torch.tensor([10, 20]), "req_pool_indices": torch.tensor([0, 1])}
    p._write_loc = torch.tensor([5], dtype=torch.int64)
    assert p._ring_rows(1) == (None, None)
    assert p._fb_window_meta.get("fallback") is True
    assert p._decode_window_args(1) is None
