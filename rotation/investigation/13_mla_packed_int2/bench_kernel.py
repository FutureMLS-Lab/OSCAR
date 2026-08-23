#!/usr/bin/env python3
"""Time the packed MLA decode kernel against the BF16 one, and sweep its tiling.

The end-to-end finding this exists to attack: the packed arm sustains 121 tok/s
against fake-quant's 198 and BF16's 221, at bs=16 and ~341K KV tokens where the
working set is 7.7 GB/step packed against 30.6 GB BF16. That is not
bandwidth-bound, so the loss is in the kernel, and a microbenchmark answers it
for one GPU-minute instead of an 8-GPU GLM run.

Two things are measured:

* **the honest baseline** -- the same shapes through
  ``_decode_grouped_att_m_fwd`` reading a real BF16 latent cache. Ratio against
  that is the number that matters; a packed kernel that is 2x faster than a
  badly-configured packed kernel is still losing.
* **the tile/warp/stage grid**. BLOCK_N 16 / 8 warps / 1 stage was chosen to
  avoid register spills, not measured, and reverting blindly trades one
  bottleneck for another -- so sweep and report the whole grid, including the
  configs that fail to compile or run out of shared memory.
"""
from __future__ import annotations

import itertools
import sys

import torch

from sglang.QuantKernel.mla_latent_int2 import scatter_pack_rows
from sglang.srt.layers.attention.triton_ops.decode_attention import (
    _decode_grouped_att_m_fwd,
)
from sglang.srt.layers.attention.triton_ops.mla_packed_decode import (
    packed_mla_decode_stage1,
)

R, ROPE, GS = 512, 64, 128
NG = R // GS


def build(bs: int, heads: int, seq: int, windows: int = 576):
    dev = "cuda"
    n_slots = bs * seq + 64
    x = torch.randn(n_slots, R, device=dev)
    pe = torch.randn(n_slots, ROPE, device=dev, dtype=torch.bfloat16)
    slots = torch.arange(n_slots, device=dev, dtype=torch.int32)

    codes = torch.zeros((n_slots, R // 4), dtype=torch.uint8, device=dev)
    params = torch.zeros((n_slots, NG * 2), dtype=torch.float32, device=dev)
    rope = pe.clone()
    scatter_pack_rows(x, slots, codes, params, GS, False)

    # A realistic window arena: sink + recent per request, i.e. the fraction of
    # blocks that must pay the arena probe. Benchmarking with an empty arena
    # would measure a kernel nobody runs.
    n_hp = 1 + bs * windows
    hp = torch.zeros((n_hp, R), dtype=torch.bfloat16, device=dev)
    hp_row = torch.full((n_slots,), -1, dtype=torch.int32, device=dev)
    hp_owner = torch.full((n_hp,), -1, dtype=torch.int32, device=dev)
    for b in range(bs):
        base = b * seq
        idx = torch.cat([
            torch.arange(base, base + 64, device=dev),
            torch.arange(base + seq - (windows - 64), base + seq, device=dev),
        ])
        ring = 1 + b * windows + torch.arange(idx.numel(), device=dev)
        hp[ring] = x[idx].to(torch.bfloat16)
        hp_row[idx] = ring.to(torch.int32)
        hp_owner[ring] = idx.to(torch.int32)

    ops = (codes, params, rope, hp, hp_row, hp_owner, GS, False)

    # BF16 baseline cache in the layout the stock kernel expects.
    kv = torch.empty((n_slots, 1, R + ROPE), dtype=torch.bfloat16, device=dev)
    kv[:, 0, :R] = x.to(torch.bfloat16)
    kv[:, 0, R:] = pe

    q = torch.randn(bs, heads, R + ROPE, device=dev, dtype=torch.bfloat16)
    kv_indptr = torch.arange(0, (bs + 1) * seq, seq, device=dev, dtype=torch.int32)
    kv_indices = torch.arange(bs * seq, device=dev, dtype=torch.int32)
    max_splits = 8
    num_splits = torch.full((bs,), max_splits, device=dev, dtype=torch.int32)
    logits = torch.empty(bs, heads, max_splits, R, device=dev, dtype=torch.float32)
    lse = torch.empty(bs, heads, max_splits, device=dev, dtype=torch.float32)
    return ops, kv, q, kv_indptr, kv_indices, num_splits, max_splits, logits, lse


def timeit(fn, iters: int = 30, warmup: int = 6) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def main() -> None:
    bs = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    heads = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    seq = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    print(f"shape: bs={bs} heads={heads} seq={seq} (KV {bs*seq:,} tokens, "
          f"BF16 {bs*seq*(R+ROPE)*2/2**30:.2f} GiB, packed "
          f"{bs*seq*288/2**30:.2f} GiB per layer-step)\n")

    ops, kv, q, kv_indptr, kv_indices, num_splits, max_splits, logits, lse = build(
        bs, heads, seq
    )
    sm = 1.0 / (R + ROPE) ** 0.5

    base = timeit(lambda: _decode_grouped_att_m_fwd(
        q, kv, kv[..., :R], logits, lse, kv_indptr, kv_indices,
        num_splits, max_splits, sm, 0.0,
    ))
    print(f"BF16 stage-1 baseline: {base:.3f} ms\n")

    print(f"{'BLOCK_N':>7} {'warps':>5} {'stages':>6} {'ms':>9} {'vs BF16':>8}  note")
    rows = []
    for bn, w, st in itertools.product((16, 32, 64), (4, 8), (1, 2, 3)):
        try:
            ms = timeit(lambda: packed_mla_decode_stage1(
                q, ops, logits, lse, kv_indptr, kv_indices,
                num_splits, max_splits, sm, 0.0,
                block_n=bn, num_warps=w, num_stages=st,
            ))
            rows.append((ms, bn, w, st))
            print(f"{bn:>7} {w:>5} {st:>6} {ms:>9.3f} {base/ms:>7.2f}x")
        except Exception as e:  # noqa: BLE001
            # Report, do not skip: "this config does not compile" is part of the
            # grid, and silently dropping it makes the winner look unconstrained.
            msg = str(e).splitlines()[0][:70]
            print(f"{bn:>7} {w:>5} {st:>6} {'-':>9} {'-':>8}  {type(e).__name__}: {msg}")

    # Ablation: the same kernel with HAS_HP off. The window-arena probe loads two
    # [BLOCK_N] int32 vectors and then, behind a block-level tl.max, may load a
    # [BLOCK_N, D] fp32 tile. Even when that branch is almost never taken the
    # compiler must reserve registers for it, so the guard can cost occupancy
    # everywhere to save bandwidth in the 2% of blocks that contain a window
    # row. This says whether that trade is paying.
    ops_nohp = ops[:3] + (None, None, None) + ops[6:]
    try:
        ms_nohp = timeit(lambda: packed_mla_decode_stage1(
            q, ops_nohp, logits, lse, kv_indptr, kv_indices,
            num_splits, max_splits, sm, 0.0,
            block_n=16, num_warps=8, num_stages=1,
        ))
        cur16 = [r[0] for r in rows if r[1:] == (16, 8, 1)]
        print(f"\nablation, window arena OFF (16/8/1): {ms_nohp:.3f} ms "
              f"({base/ms_nohp:.2f}x BF16)"
              + (f"; arena probe costs {cur16[0]/ms_nohp:.2f}x" if cur16 else ""))
    except Exception as e:  # noqa: BLE001
        print(f"\nablation failed: {type(e).__name__}: {e}")

    # Load amplification, which is the arithmetic behind the ratio above. Per KV
    # block the dequant path moves codes as [BLOCK_N, D] uint8 (every byte read
    # four times) plus scale and zero as [BLOCK_N, D] fp32, where the unique
    # information is [BLOCK_N, D/4] bytes and 2 x [BLOCK_N] floats.
    for bn in (16, 32):
        # codes on the [BLOCK_N, D] pattern (each byte read 4x) + scale and
        # zero as [BLOCK_N] vectors, one pair per group.
        moved = bn * R * 1 + bn * NG * 2 * 4
        unique = bn * (R // 4) + bn * NG * 2 * 4
        bf16 = bn * (R + ROPE) * 2
        print(f"  BLOCK_N={bn}: dequant path moves {moved/1024:.1f} KiB/block for "
              f"{unique/1024:.1f} KiB of information; BF16 moves {bf16/1024:.1f} KiB "
              f"-> {moved/bf16:.1f}x the traffic of the kernel it must beat")

    if rows:
        rows.sort()
        ms, bn, w, st = rows[0]
        print(f"\nbest: BLOCK_N={bn} warps={w} stages={st} -> {ms:.3f} ms "
              f"({base/ms:.2f}x BF16); current default is BLOCK_N=16 warps=8 stages=1")
        cur = [r for r in rows if r[1:] == (16, 8, 1)]
        if cur:
            print(f"speedup over the current default: {cur[0][0]/ms:.2f}x")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("needs a GPU")
        sys.exit(1)
    main()
