"""CPU equivalence tests for the per-head rotation plumbing (no GPU needed)."""
import sys, torch
sys.path.insert(0, "python")
from sglang.srt.mem_cache.unified_kv_pool import _rotate_heads
from sglang.srt.layers.attention.quantized_kv_prefill import _apply_oscar_rotation

torch.manual_seed(0)
T, H, D, G = 7, 4, 8, 2            # tokens, kv heads, head_dim, kv_group_num
x = torch.randn(T, H, D)
Rs = torch.linalg.qr(torch.randn(D, D))[0]                     # shared [D,D]
Rp = torch.stack([torch.linalg.qr(torch.randn(D, D))[0] for _ in range(H)])  # [H,D,D]

# 1) a per-head stack of the SAME matrix must equal the shared path
Rp_same = Rs.unsqueeze(0).repeat(H, 1, 1)
assert torch.allclose(_rotate_heads(x, Rs), _rotate_heads(x, Rp_same), atol=1e-6)
print("ok 1: per-head(identical matrices) == shared")

# 2) per-head must rotate each head with its own matrix
ref = torch.stack([x[:, h, :] @ Rp[h] for h in range(H)], dim=1)
assert torch.allclose(_rotate_heads(x, Rp), ref, atol=1e-6)
print("ok 2: per-head matches per-head reference")

# 3) GQA: q has H*G heads; each kv head's matrix serves G consecutive q heads
q = torch.randn(T, H * G, D)
got = _apply_oscar_rotation(q, Rp, G)
ref_q = torch.stack([q[:, i, :] @ Rp[i // G] for i in range(H * G)], dim=1)
assert torch.allclose(got, ref_q, atol=1e-6)
print("ok 3: GQA q rotation uses kv-head mapping (q_head // kv_group_num)")

# 4) round trip: rotate then inverse-rotate returns the input (per-head)
rot = _rotate_heads(x, Rp)
inv = torch.einsum("thd,hed->the", rot, Rp)     # same expression the decode inverse uses
assert torch.allclose(inv, x, atol=1e-5), (inv - x).abs().max()
print("ok 4: per-head rotate -> inverse round trip is identity")

# 5) outputs must be CONTIGUOUS -- the store path does k.view(-1, row_dim) and
#    a non-contiguous rotation result makes CUDA-graph capture fail
assert _rotate_heads(x, Rp).is_contiguous(), "per-head rotate must be contiguous"
assert _rotate_heads(x, Rs).is_contiguous(), "shared rotate must be contiguous"
assert _apply_oscar_rotation(q, Rp, G).is_contiguous()
print("ok 5: rotation outputs are contiguous (view-safe)")

# 6) shared path unchanged (V1 regression)
assert torch.allclose(_apply_oscar_rotation(x, Rs), (x @ Rs), atol=1e-6)
print("ok 6: V1 shared path bit-unchanged")
# 7) TP sharding: a rank owning 2 of 4 KV heads must get its own slice
from sglang.srt.mem_cache.unified_kv_pool import _shard_rotation_heads
R4 = torch.stack([torch.full((D, D), float(h)) for h in range(4)])     # [4,D,D]
# per-layer list: each entry is one layer's [H,D,D] -> head axis unambiguous
for rank, expect in ((0, [0.0, 1.0]), (1, [2.0, 3.0])):
    sl = _shard_rotation_heads([R4, R4], local_head_num=2, tp_rank=rank)[0]
    assert sl.shape == (2, D, D) and [float(sl[i][0, 0]) for i in (0, 1)] == expect, (rank, sl.shape)
# 4D [L,H,D,D] -> head axis unambiguous
stacked = R4.unsqueeze(0).repeat(3, 1, 1, 1)
assert _shard_rotation_heads(stacked, 2, 1).shape == (3, 2, D, D)
assert _shard_rotation_heads(Rs, 2, 1) is Rs, "shared rotation must pass through"
# A bare 3D tensor is ambiguous: V1 holds [L,D,D] stacked per-layer shared
# rotations, which look exactly like one layer's [H,D,D]. It must pass through
# untouched -- slicing it as heads broke every V1 model (Qwen3-8B: 36 layers,
# 8 kv heads -> "36 KV heads is not a multiple of this rank's 8").
v1_stacked = torch.stack([torch.full((D, D), float(l)) for l in range(36)])
assert _shard_rotation_heads(v1_stacked, 8, 0) is v1_stacked, "V1 [L,hd,hd] must pass through"
print("ok 7: per-head rotations are sharded by TP rank (shared passes through)")
print("ALL PASS")


# Replicated KV heads: 2 heads over TP 4 -> ranks 0,1 hold head 0, ranks 2,3 hold head 1.
R2 = torch.stack([torch.eye(D) * (h + 1) for h in range(2)])
for rank, head in ((0, 0), (1, 0), (2, 1), (3, 1)):
    got = _shard_rotation_heads([R2], local_head_num=1, tp_rank=rank, tp_size=4)[0]
    assert got.shape == (1, D, D) and float(got[0, 0, 0]) == head + 1, (rank, float(got[0, 0, 0]))
try:
    _shard_rotation_heads([R2], local_head_num=1, tp_rank=3)  # no tp_size: cannot place rank 3
    raise AssertionError("expected a ValueError without tp_size")
except ValueError:
    pass
print("replicated-head sharding OK")


# Two-geometry pools (gemma4_unified): per-layer head counts. A sliding layer with 8 heads next to a
# full-attention layer with 1 head must keep its 8 under TP=1 and split 4/4 under TP=2, while the
# 1-head layer is replicated over both ranks. (v109 sliced every layer by the pool-level 1 -> einsum death.)
from sglang.srt.mem_cache.unified_kv_pool import _shard_head_vectors
R8 = torch.stack([torch.full((D, D), float(h)) for h in range(8)]); R1 = torch.stack([torch.eye(D) * 7.0])
got = _shard_rotation_heads([R8, R1], local_head_num=[8, 1], tp_rank=0, tp_size=1)
assert got[0] is R8 and got[1] is R1, "TP=1 ragged list must pass through untouched"
for rank, first in ((0, 0.0), (1, 4.0)):
    got = _shard_rotation_heads([R8, R1], local_head_num=[4, 1], tp_rank=rank, tp_size=2)
    assert got[0].shape == (4, D, D) and float(got[0][0, 0, 0]) == first, (rank, got[0].shape)
    assert got[1].shape == (1, D, D) and float(got[1][0, 0, 0]) == 7.0, rank
m8 = torch.stack([torch.full((D,), float(h)) for h in range(8)]); m1 = torch.full((1, D), 7.0)
got = _shard_head_vectors([m8, m1], local_head_num=[4, 1], tp_rank=1, tp_size=2)
assert got[0].shape == (4, D) and float(got[0][0, 0]) == 4.0 and got[1].shape == (1, D) and float(got[1][0, 0]) == 7.0
try:
    _shard_rotation_heads([R8, R1], local_head_num=[8], tp_rank=0); raise AssertionError("expected a length mismatch error")
except ValueError:
    pass
print("ragged per-layer head counts OK")
