#!/usr/bin/env python3
"""Equivalence tests for the packed-INT2 MLA latent storage and read path.

Three things have to hold before any end-to-end number is worth reading:

1. **pack/unpack round trip == the torch fake-quant reference.** The packed
   codes must dequantize to exactly what ``_fake_quant_int2_groupwise`` returns,
   because that function *is* the published 80.30 arm's arithmetic. Anything
   else and the packed arm is measuring a different method.
2. **materialize == fake-quant, windows included.** A row inside the BF16
   window arena must come back bit-exact; a row outside it must come back as
   the dequantized code. This is the contract the attention kernels rely on.
3. **the fused decode kernel == attention over the materialized rows.** The
   inline dequant in ``mla_packed_decode`` is a second implementation of (2);
   if it disagrees with the reference read, the score is not the pool's.

Run on one GPU:  python3 test_packed_latent.py
"""
from __future__ import annotations

import sys

import torch

from sglang.QuantKernel.mla_latent_int2 import (
    assemble_rows,
    gather_dequant_rows,
    scatter_pack_rows,
)
from sglang.srt.mem_cache.mla_int2_kv_pool import _fake_quant_int2_groupwise

R = 512
ROPE = 64
GS = 128
NG = R // GS

FAILURES: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILURES.append(name)


def test_pack_roundtrip(lloyd: bool) -> None:
    torch.manual_seed(0)
    n_slots, n = 4096, 777
    x = torch.randn(n, R, device="cuda", dtype=torch.bfloat16).float()
    slots = torch.randperm(n_slots, device="cuda")[:n].to(torch.int32)

    codes = torch.zeros((n_slots, R // 4), dtype=torch.uint8, device="cuda")
    params = torch.zeros((n_slots, NG * 2), dtype=torch.float32, device="cuda")
    scatter_pack_rows(x, slots, codes, params, GS, lloyd)

    out = torch.empty((n, R), dtype=torch.bfloat16, device="cuda")
    gather_dequant_rows(slots, codes, params, out, GS, lloyd)

    ref = _fake_quant_int2_groupwise(x, GS, lloyd).to(torch.bfloat16)
    d = (out.float() - ref.float()).abs()
    # The criterion is per-code, not per-element-magnitude. A handful of codes
    # can differ from torch by exactly ONE quantization step: the quotient lands
    # within half a ULP of a .5 boundary, torch's instruction sequence rounds it
    # up and the kernel's rounds it down (or the reverse). Both are defensible
    # roundings of the same number, and the same tie already exists in the
    # shipped fused kernel. What must NOT happen is a difference of more than
    # one step, which would mean the codes or the group params are wrong.
    step = (ref.float().reshape(-1, GS).amax(-1)
            - ref.float().reshape(-1, GS).amin(-1)) / 3.0
    step = step.clamp(min=1e-6).repeat_interleave(GS).reshape(d.shape)
    over = (d > step * 1.01).sum().item()
    off = (d > step * 0.01).float().mean().item()
    rl2 = (d.pow(2).sum().sqrt() / ref.float().pow(2).sum().sqrt()).item()
    check(
        f"pack/dequant == torch fake-quant (lloyd={lloyd})",
        over == 0 and off < 5e-4,
        f"codes>1 step={over} codes off by 1={off*100:.4f}% relL2={rl2:.2e}",
    )

    # Scatter must land on the right row: a permuted write read back in the
    # same permutation is not a test of addressing, a *second* permutation is.
    perm = torch.randperm(n, device="cuda")
    out2 = torch.empty((n, R), dtype=torch.bfloat16, device="cuda")
    gather_dequant_rows(slots[perm], codes, params, out2, GS, lloyd)
    check(
        f"scatter/gather addressing (lloyd={lloyd})",
        torch.equal(out2, out[perm]),
        "",
    )


def test_materialize_windows() -> None:
    torch.manual_seed(1)
    n_slots, n = 2048, 512
    x = torch.randn(n, R, device="cuda").float()
    pe = torch.randn(n, ROPE, device="cuda").to(torch.bfloat16)
    slots = torch.arange(n, device="cuda", dtype=torch.int32)

    codes = torch.zeros((n_slots, R // 4), dtype=torch.uint8, device="cuda")
    params = torch.zeros((n_slots, NG * 2), dtype=torch.float32, device="cuda")
    rope = torch.zeros((n_slots, ROPE), dtype=torch.bfloat16, device="cuda")
    scatter_pack_rows(x, slots, codes, params, GS, False)
    rope[slots.long()] = pe

    n_hp = 129
    hp = torch.zeros((n_hp, R), dtype=torch.bfloat16, device="cuda")
    hp_row = torch.full((n_slots,), -1, dtype=torch.int32, device="cuda")
    hp_owner = torch.full((n_hp,), -1, dtype=torch.int32, device="cuda")

    # Half the first 128 slots are in the window; of those, half have had their
    # ring row re-issued to someone else (the prefix-cache-hit case) and must
    # therefore fall back to the packed tier rather than read a stranger's row.
    win = torch.arange(0, 128, device="cuda")
    ring = 1 + torch.arange(128, device="cuda")
    hp[ring] = x[win].to(torch.bfloat16)
    hp_row[win] = ring.to(torch.int32)
    hp_owner[ring] = win.to(torch.int32)
    stolen = win[::2]
    hp_owner[hp_row[stolen].long()] = 9999  # re-issued to another slot

    c_rot = torch.empty((n, R), dtype=torch.bfloat16, device="cuda")
    gather_dequant_rows(slots, codes, params, c_rot, GS, False)
    out = torch.empty((n, R + ROPE), dtype=torch.bfloat16, device="cuda")
    assemble_rows(c_rot, slots, rope, hp, hp_row, hp_owner, out)

    deq = _fake_quant_int2_groupwise(x, GS, False).to(torch.bfloat16)
    want = deq.clone()
    kept = win[1::2]                     # still owned -> BF16 exact
    want[kept] = x[kept].to(torch.bfloat16)

    d = (out[:, :R].float() - want.float()).abs()
    check(
        "materialize: window rows exact, stolen rows fall back, rest dequantized",
        d.max().item() < 5e-3 * want.float().abs().max().item(),
        f"max_abs={d.max():.3e}",
    )
    check("materialize: k_pe passthrough", torch.equal(out[:, R:], pe), "")

    # -1 is the padded/invalid slot the graph runner produces; it must read as
    # zeros, not as row 0's contents.
    neg = torch.full((8,), -1, dtype=torch.int32, device="cuda")
    c2 = torch.empty((8, R), dtype=torch.bfloat16, device="cuda")
    gather_dequant_rows(neg, codes, params, c2, GS, False)
    o2 = torch.empty((8, R + ROPE), dtype=torch.bfloat16, device="cuda")
    assemble_rows(c2, neg, rope, hp, hp_row, hp_owner, o2)
    check("materialize: slot -1 reads as zeros", bool((o2 == 0).all()), "")


class _FakePool:
    def __init__(self, ops, kv_lora_rank):
        self._ops = ops
        self.kv_lora_rank = kv_lora_rank

    def packed_read_operands(self, layer_id):
        return self._ops


def test_decode_kernel() -> None:
    from sglang.srt.layers.attention.triton_ops.mla_packed_decode import (
        packed_mla_decode_stage1,
    )
    from sglang.srt.layers.attention.triton_ops.decode_attention import (
        _decode_softmax_reducev_fwd,
    )

    torch.manual_seed(2)
    bs, heads, seq = 3, 16, 300
    n_slots = bs * seq + 16

    x = torch.randn(n_slots, R, device="cuda").float()
    pe = torch.randn(n_slots, ROPE, device="cuda").to(torch.bfloat16)
    slots_all = torch.arange(n_slots, device="cuda", dtype=torch.int32)
    codes = torch.zeros((n_slots, R // 4), dtype=torch.uint8, device="cuda")
    params = torch.zeros((n_slots, NG * 2), dtype=torch.float32, device="cuda")
    rope = torch.zeros((n_slots, ROPE), dtype=torch.bfloat16, device="cuda")
    scatter_pack_rows(x, slots_all, codes, params, GS, False)
    rope[:] = pe

    n_hp = 1 + bs * 64
    hp = torch.zeros((n_hp, R), dtype=torch.bfloat16, device="cuda")
    hp_row = torch.full((n_slots,), -1, dtype=torch.int32, device="cuda")
    hp_owner = torch.full((n_hp,), -1, dtype=torch.int32, device="cuda")
    for b in range(bs):
        base = b * seq
        w = torch.arange(base, base + 64, device="cuda")
        ring = 1 + b * 64 + torch.arange(64, device="cuda")
        hp[ring] = x[w].to(torch.bfloat16)
        hp_row[w] = ring.to(torch.int32)
        hp_owner[ring] = w.to(torch.int32)

    ops = (codes, params, rope, hp, hp_row, hp_owner, GS, False)
    q = torch.randn(bs, heads, R + ROPE, device="cuda", dtype=torch.bfloat16)
    kv_indptr = torch.arange(0, (bs + 1) * seq, seq, device="cuda", dtype=torch.int32)
    kv_indices = torch.arange(bs * seq, device="cuda", dtype=torch.int32)
    max_splits = 8
    num_splits = torch.full((bs,), max_splits, device="cuda", dtype=torch.int32)

    logits = torch.empty(bs, heads, max_splits, R, device="cuda", dtype=torch.float32)
    lse = torch.empty(bs, heads, max_splits, device="cuda", dtype=torch.float32)
    o = torch.empty(bs, heads, R, device="cuda", dtype=torch.bfloat16)
    sm = 1.0 / (R + ROPE) ** 0.5
    packed_mla_decode_stage1(
        q, ops, logits, lse, kv_indptr, kv_indices, num_splits, max_splits, sm, 0.0
    )
    _decode_softmax_reducev_fwd(
        logits, lse, q, o, 1.0,
        torch.empty((0, 1, R), dtype=o.dtype, device=o.device),
        kv_indptr, num_splits, max_splits,
    )

    # Reference: materialize the same rows and run dense attention in fp32.
    c_rot = torch.empty((bs * seq, R), dtype=torch.bfloat16, device="cuda")
    gather_dequant_rows(kv_indices, codes, params, c_rot, GS, False)
    rows = torch.empty((bs * seq, R + ROPE), dtype=torch.bfloat16, device="cuda")
    assemble_rows(c_rot, kv_indices, rope, hp, hp_row, hp_owner, rows)
    rows = rows.view(bs, seq, R + ROPE).float()
    qf = q.float()
    scores = torch.einsum("bhd,bsd->bhs", qf, rows) * sm
    p = torch.softmax(scores, dim=-1)
    ref = torch.einsum("bhs,bsd->bhd", p, rows[:, :, :R])

    d = (o.float() - ref).abs()
    rel = (d.max() / ref.abs().max()).item()
    check(
        "packed decode kernel == attention over materialized rows",
        rel < 2e-2,
        f"max_abs={d.max():.3e} rel={rel:.3e}",
    )


def test_size_arithmetic() -> None:
    from sglang.srt.mem_cache.mla_packed_kv_pool import packed_latent_bytes_per_token

    b = packed_latent_bytes_per_token(512, 64, 128)
    bf16 = (512 + 64) * 2
    check("packed bytes/token/layer", b == 288, f"{b} B vs BF16 {bf16} B ({bf16/b:.2f}x)")
    # GLM-5.2 on B200: what the pool should grow to at the reference arm's budget.
    layers, idx = 78, 128 + 128 // 128 * 4
    bf16_cell = (bf16 + idx) * layers
    packed_cell = (b + idx) * layers
    avail = 645056 * bf16_cell
    print(
        f"       at the reference arm's KV budget ({avail/2**30:.1f} GiB): "
        f"BF16 645,056 tokens -> packed {avail//packed_cell:,} "
        f"({bf16_cell/packed_cell:.2f}x)"
    )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("needs a GPU")
        sys.exit(1)
    test_size_arithmetic()
    for lm in (False, True):
        test_pack_roundtrip(lm)
    test_materialize_windows()
    test_decode_kernel()
    print()
    if FAILURES:
        print("FAILED:", ", ".join(FAILURES))
        sys.exit(1)
    print("ALL PASS")
