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

    # Register-side expansion of the group scale/zero. Same arithmetic, same
    # tiles, same dtypes -- only the addressing changes, so the outputs must
    # match bit for bit and any time difference is address computation and load
    # issue alone. Correctness is asserted before the timing is believed.
    outs = {}
    for tag, pb in (("gid", False), ("bcast", True)):
        logits.zero_()
        lse.zero_()
        packed_mla_decode_stage1(
            q, ops, logits, lse, kv_indptr, kv_indices,
            num_splits, max_splits, sm, 0.0,
            block_n=16, num_warps=8, num_stages=1, param_bcast=pb,
        )
        torch.cuda.synchronize()
        outs[tag] = (logits.clone(), lse.clone())
    d_o = (outs["gid"][0] - outs["bcast"][0]).abs().max().item()
    d_l = (outs["gid"][1] - outs["bcast"][1]).abs().max().item()
    identical = d_o == 0.0 and d_l == 0.0
    print(f"\nparam broadcast: max|dout|={d_o:.3e} max|dlse|={d_l:.3e} "
          f"-> {'BIT-IDENTICAL' if identical else 'DIFFERS -- do not trust the timing'}")
    t_gid = timeit(lambda: packed_mla_decode_stage1(
        q, ops, logits, lse, kv_indptr, kv_indices, num_splits, max_splits,
        sm, 0.0, block_n=16, num_warps=8, num_stages=1, param_bcast=False,
    ))
    t_bc = timeit(lambda: packed_mla_decode_stage1(
        q, ops, logits, lse, kv_indptr, kv_indices, num_splits, max_splits,
        sm, 0.0, block_n=16, num_warps=8, num_stages=1, param_bcast=True,
    ))
    print(f"  gid-indexed  {t_gid:.3f} ms ({base/t_gid:.2f}x BF16)")
    print(f"  broadcast    {t_bc:.3f} ms ({base/t_bc:.2f}x BF16)  "
          f"-> {t_gid/t_bc:.2f}x over gid-indexed")

    # What the dequant itself costs. Tiling, traffic and addressing have all been
    # measured and all failed to move this kernel, so the remaining candidate is
    # the per-element arithmetic: the dequant touches BLOCK_N x D elements while
    # the two dots do BLOCK_H x BLOCK_N x D MACs on tensor cores, so the dequant
    # is amortized over BLOCK_H query heads only. At BLOCK_H=16 an ALU op costs
    # ~28 tensor-core MAC-equivalents, which predicts dequant ~5x the dot cost --
    # close enough to the measured 5.5x to be worth testing directly.
    #
    # ABLATE=1 and 2 produce WRONG NUMBERS on purpose. They exist to attribute
    # time, and the pool can never reach them.
    for lvl, what in ((1, "no dequant at all (codes straight to bf16)"),
                      (2, "scale/zero loaded but not applied")):
        try:
            t_ab = timeit(lambda: packed_mla_decode_stage1(
                q, ops, logits, lse, kv_indptr, kv_indices, num_splits,
                max_splits, sm, 0.0, block_n=16, num_warps=8, num_stages=1,
                ablate=lvl,
            ))
            print(f"  ABLATE={lvl} {t_ab:.3f} ms ({base/t_ab:.2f}x BF16) "
                  f"-> {t_gid/t_ab:.2f}x over the real path   [{what}]")
        except Exception as e:  # noqa: BLE001
            print(f"  ABLATE={lvl} failed: {type(e).__name__}: {e}")

    # The transpose. ``tl.dot(q, tl.trans(cvq))`` converts a [BLOCK_N, D] tile's
    # layout through shared memory; reading the codes a second time in the
    # layout the dot wants removes that at the cost of a second dequant, which
    # the ABLATE rows above just measured at zero. This is the only structural
    # difference left between this kernel and the BF16 one it must beat: the
    # BF16 kernel reads K and V separately and never transposes.
    #
    # Not expected to be bit-identical -- same values, but tl.dot over a
    # differently-laid-out operand can reduce in a different order -- so this
    # checks a tolerance rather than equality.
    logits.zero_(); lse.zero_()
    packed_mla_decode_stage1(
        q, ops, logits, lse, kv_indptr, kv_indices, num_splits, max_splits,
        sm, 0.0, block_n=16, num_warps=8, num_stages=1, dual_load=True)
    torch.cuda.synchronize()
    o_dl, l_dl = logits.clone(), lse.clone()
    ref_o, ref_l = outs["gid"]
    den = ref_o.abs().max().clamp(min=1e-6)
    rel = ((o_dl - ref_o).abs().max() / den).item()
    dl_ok = rel < 5e-3
    print(f"\ndual-load: rel max|dout| = {rel:.3e}, max|dlse| = "
          f"{(l_dl - ref_l).abs().max().item():.3e} -> "
          f"{'AGREES' if dl_ok else 'DIFFERS -- do not trust the timing'}")
    try:
        t_dl = timeit(lambda: packed_mla_decode_stage1(
            q, ops, logits, lse, kv_indptr, kv_indices, num_splits, max_splits,
            sm, 0.0, block_n=16, num_warps=8, num_stages=1, dual_load=True,
        ))
        print(f"  transpose    {t_gid:.3f} ms ({base/t_gid:.2f}x BF16)")
        print(f"  dual load    {t_dl:.3f} ms ({base/t_dl:.2f}x BF16)  "
              f"-> {t_gid/t_dl:.2f}x over the transpose")
        # BLOCK_N was tuned against a kernel that transposed; if the transpose
        # was the constraint, the tile optimum moves with it.
        for bn in (32, 64):
            try:
                t = timeit(lambda: packed_mla_decode_stage1(
                    q, ops, logits, lse, kv_indptr, kv_indices, num_splits,
                    max_splits, sm, 0.0, block_n=bn, num_warps=8,
                    num_stages=1, dual_load=True,
                ))
                print(f"  dual load BLOCK_N={bn} {t:.3f} ms ({base/t:.2f}x BF16)")
            except Exception as e:  # noqa: BLE001
                print(f"  dual load BLOCK_N={bn} - {type(e).__name__}: "
                      f"{str(e).splitlines()[0][:60]}")
    except Exception as e:  # noqa: BLE001
        print(f"  dual load failed: {type(e).__name__}: {e}")

    # Head amortization. MLA shares one KV head across every query head, so
    # BLOCK_H is free to grow -- and growing it is the only lever that changes
    # the dequant-to-dot ratio rather than the constant in front of it. The cap
    # is the [BLOCK_H, D] fp32 accumulator: at BLOCK_H=64, D=512 that is 128 KiB.
    print(f"\nhead amortization (BLOCK_H, heads={heads}):")
    for bh in (16, 32, 64, 128):
        if bh > heads:
            continue
        try:
            t_bh = timeit(lambda: packed_mla_decode_stage1(
                q, ops, logits, lse, kv_indptr, kv_indices, num_splits,
                max_splits, sm, 0.0, block_n=16, num_warps=8, num_stages=1,
                block_h=bh,
            ))
            print(f"  BLOCK_H={bh:>3} {t_bh:.3f} ms ({base/t_bh:.2f}x BF16) "
                  f"-> {t_gid/t_bh:.2f}x over BLOCK_H=16")
        except Exception as e:  # noqa: BLE001
            msg = str(e).splitlines()[0][:70]
            print(f"  BLOCK_H={bh:>3} - {type(e).__name__}: {msg}")

    # Traffic accounting, kept because it was REFUTED and the refutation is the
    # finding. Narrowing the bytes actually made the kernel 1.3x slower, so the
    # amplification below is L1-served and is not the bound; what the broadcast
    # arm above removes is the 128x-redundant *address computation* for the same
    # bytes, which is a different cost.
    for bn in (16, 32):
        moved = bn * R * 1 + bn * R * 4 * 2
        unique = bn * (R // 4) + bn * 2 * 4
        bf16 = bn * (R + ROPE) * 2
        print(f"  BLOCK_N={bn}: dequant path moves {moved/1024:.1f} KiB/block for "
              f"{unique/1024:.1f} KiB of information; BF16 moves {bf16/1024:.1f} KiB "
              f"-> {moved/bf16:.1f}x the traffic of the kernel it must beat "
              f"(refuted as the bound)")

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
