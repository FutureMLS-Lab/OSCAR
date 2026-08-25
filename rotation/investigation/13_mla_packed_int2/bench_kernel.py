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
import os
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




def _err(e) -> str:
    """Full first lines of an exception, not just line one.

    A Triton CompilationError's first line is only "at LINE:COL:" -- the actual
    message is on the lines after it. Truncating to splitlines()[0] printed a
    location with no reason and cost a whole job round-trip to re-discover.
    """
    return str(e)[:600].replace("\n", " | ")

def build(bs: int, heads: int, seq: int, windows: int = 576,
          max_splits: int = 8):
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
        # Clamp both ranges into THIS request. Unclamped, seq < windows - 64
        # makes the tail start negative, and torch wraps negative indices, so
        # the arena for a short sequence was built out of other requests' slots
        # -- a malformed fixture that would be read as a kernel failure. The
        # prefix and tail are also clipped against each other so they cannot
        # overlap and assign one slot two ring rows.
        n_pre = min(64, seq)
        tail_lo = max(seq - (windows - 64), n_pre)
        idx = torch.cat([
            torch.arange(base, base + n_pre, device=dev),
            torch.arange(base + tail_lo, base + seq, device=dev),
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

    SPLITS = int(os.environ.get("BENCH_SPLITS", "16"))
    ops, kv, q, kv_indptr, kv_indices, num_splits, max_splits, logits, lse = build(
        bs, heads, seq, max_splits=SPLITS
    )
    print(f"kv splits = {SPLITS} (grid {bs}x{SPLITS} programs). The split sweep "
          f"below shows BF16 is 1.5x faster at 16 than at 8, so every ratio in "
          f"this file is against the baseline's OWN best setting, not its worst.")
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
            msg = _err(e)
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
                      f"{_err(e)}")
    except Exception as e:  # noqa: BLE001
        print(f"  dual load failed: {type(e).__name__}: {e}")

    # SPLIT SWEEP. grid = (bs, head_blocks, splits); at bs=16 with one head
    # block and 8 splits that is 128 programs on ~148 SMs. Occupancy was never
    # a tested lever here, and it is the one knob that changes how many programs
    # exist rather than what each one does. Both arms take the same split count.
    print("\n=== kv-split sweep (grid occupancy) ===")
    print(f"{'splits':>7} {'programs':>9} {'BF16 ms':>9} {'packed ms':>10} {'ratio':>7}")
    for sp in (8, 16, 32, 64):
        try:
            o2, kv2, q2, ind2, idx2, ns2, ms2, lg2, ls2 = build(
                bs, heads, seq, max_splits=sp)
            tb = timeit(lambda: _decode_grouped_att_m_fwd(
                q2, kv2, kv2[..., :R], lg2, ls2, ind2, idx2, ns2, ms2, sm, 0.0))
            tp = timeit(lambda: packed_mla_decode_stage1(
                q2, o2, lg2, ls2, ind2, idx2, ns2, ms2, sm, 0.0,
                block_n=16, num_warps=8, num_stages=1))
            print(f"{sp:>7} {bs*sp:>9} {tb:>9.3f} {tp:>10.3f} {tb/tp:>6.2f}x")
            del o2, kv2, q2, lg2, ls2
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"{sp:>7} {'-':>9}  {type(e).__name__}: {_err(e)}")

    # The group-factored kernel: the only variant that changes WHAT reaches
    # tl.dot (a directly-loaded code tile) rather than how a computed tile is
    # built. Compared with the arena off, because it has no arena path.
    from sglang.srt.layers.attention.triton_ops.mla_packed_decode import (
        packed_mla_decode_stage1_gf,
    )
    try:
        logits.zero_(); lse.zero_()
        packed_mla_decode_stage1(
            q, ops_nohp, logits, lse, kv_indptr, kv_indices, num_splits,
            max_splits, sm, 0.0, block_n=16, num_warps=8, num_stages=1)
        torch.cuda.synchronize()
        ref_o2, ref_l2 = logits.clone(), lse.clone()
        logits.zero_(); lse.zero_()
        packed_mla_decode_stage1_gf(
            q, ops_nohp, logits, lse, kv_indptr, kv_indices, num_splits,
            max_splits, sm, 0.0, block_n=16, num_warps=8, num_stages=1)
        torch.cuda.synchronize()
        den2 = ref_o2.abs().max().clamp(min=1e-6)
        rel2 = ((logits - ref_o2).abs().max() / den2).item()
        dl2 = (lse - ref_l2).abs().max().item()
        # Not bit-identical by construction: the factored form reassociates the
        # sum, so this is a tolerance on a different summation order.
        ok2 = rel2 < 2e-2
        print(f"\ngroup-factored (arena OFF): rel max|dout| = {rel2:.3e}, "
              f"max|dlse| = {dl2:.3e} -> "
              f"{'AGREES' if ok2 else 'DIFFERS -- timing below is meaningless'}")
        t_base2 = timeit(lambda: packed_mla_decode_stage1(
            q, ops_nohp, logits, lse, kv_indptr, kv_indices, num_splits,
            max_splits, sm, 0.0, block_n=16, num_warps=8, num_stages=1))
        print(f"  computed-tile  {t_base2:.3f} ms ({base/t_base2:.2f}x BF16)")
        # BLOCK_N was still improving monotonically at 64 (1.34x/1.86x/2.49x over
        # the computed tile), which is the prediction that separates this variant
        # from the six refuted ones: a loaded tile streams, so the shared-memory
        # ceiling that pinned the old kernel at 16 does not apply. Push until it
        # stops paying, and sweep warps at each tile -- the optimum moved with
        # batch on the old kernel (warps=4 was 1.27x at bs=32) and was never
        # re-checked at a larger tile.
        # BLOCK_H, indicated by the counters rather than guessed: the compiled
        # kernel sits at regs/thread=255, the hard PTX maximum, with 0 spills --
        # so it fits only just, and occupancy is capped at 65536/(255*32) ~= 8
        # warps/SM. The four [BLOCK_H, GS] fp32 accumulators are what eat them:
        # at BLOCK_H=16 that is 16*128*4 floats = 64 registers per thread of
        # accumulator alone. Halving BLOCK_H halves that AND doubles the grid,
        # which is the other thing that is short (128 programs on ~148 SMs).
        # Every arm reports the resources of the kernel it ACTUALLY compiled.
        # The first BLOCK_H sweep returned byte-identical times for 8 and 16
        # because the launcher overwrote the argument with a hardcoded 16 on the
        # first line of its body -- a knob that silently does not move produces a
        # clean-looking table of a single configuration measured twice. regs and
        # smem must differ when BLOCK_H differs, so printing them makes that
        # failure loud instead of invisible.
        seen_res = {}
        for bn, w, bh in itertools.product((32, 64), (4, 8), (8, 16)):
            try:
                k_gf = [None]

                def _run():
                    k_gf[0] = packed_mla_decode_stage1_gf(
                        q, ops_nohp, logits, lse, kv_indptr, kv_indices,
                        num_splits, max_splits, sm, 0.0, block_n=bn,
                        num_warps=w, num_stages=3, block_h=bh)

                t_gf = timeit(_run)
                kk = k_gf[0]
                regs = getattr(kk, "n_regs", None)
                shared = getattr(getattr(kk, "metadata", None), "shared", None)
                seen_res.setdefault((bn, w), {})[bh] = (regs, shared)
                # BLOCK_N is the whole point: if the computed tile was what
                # capped it at 16, removing the tile should let it rise.
                print(f"  factored BLOCK_N={bn:>3} warps={w:>2} BLOCK_H={bh:>2} {t_gf:.3f} ms "
                      f"({base/t_gf:.2f}x BF16) -> {t_base2/t_gf:.2f}x over "
                      f"the computed tile  [regs={regs} smem={shared}]")
            except Exception as e:  # noqa: BLE001
                print(f"  factored BLOCK_N={bn:>3} warps={w:>2} BLOCK_H={bh:>2} - "
                      f"{type(e).__name__}: {_err(e)}")
        for (bn, w), byh in seen_res.items():
            if len(byh) > 1 and len(set(byh.values())) == 1:
                print(f"  !! BLOCK_H DID NOT REACH THE KERNEL at {bn}/{w}: "
                      f"every BLOCK_H compiled to {next(iter(byh.values()))} "
                      f"-- the sweep measured one config twice, ignore its times")

        # WIDE_LOAD: load each code byte once instead of once per 2-bit field.
        # The BLOCK_H sweep priced the code path at ~80% of this kernel (halving
        # BLOCK_H doubles code load+unpack, holds dot work fixed, and cost
        # 1.82x), and the narrow form issues four loads per byte. Correctness is
        # checked BEFORE the timing: a variant that unpacks to a different index
        # order would be quietly wrong and still fast.
        print("\nWIDE_LOAD (one load per code byte, not one per 2-bit field):")
        for bn, w in itertools.product((32, 64), (4, 8)):
            try:
                on = torch.zeros_like(logits)
                ln = torch.zeros_like(lse)
                packed_mla_decode_stage1_gf(
                    q, ops_nohp, on, ln, kv_indptr, kv_indices, num_splits,
                    max_splits, sm, 0.0, block_n=bn, num_warps=w, num_stages=3,
                    wide_load=False)
                ow = torch.zeros_like(logits)
                lw = torch.zeros_like(lse)
                kw = packed_mla_decode_stage1_gf(
                    q, ops_nohp, ow, lw, kv_indptr, kv_indices, num_splits,
                    max_splits, sm, 0.0, block_n=bn, num_warps=w, num_stages=3,
                    wide_load=True)
                fin = torch.isfinite(on) & torch.isfinite(ow)
                dev = (on[fin] - ow[fin]).abs().max().item() if fin.any() else float("nan")
                ok = dev < 1e-3
                # wide_load=False explicitly: it now DEFAULTS to True, so
                # omitting it here would time the wide kernel, label it
                # "narrow", and report 1.00x -- a false refutation of the
                # change, manufactured by the change itself.
                t_n = timeit(lambda: packed_mla_decode_stage1_gf(
                    q, ops_nohp, logits, lse, kv_indptr, kv_indices, num_splits,
                    max_splits, sm, 0.0, block_n=bn, num_warps=w, num_stages=3,
                    wide_load=False))
                t_w = timeit(lambda: packed_mla_decode_stage1_gf(
                    q, ops_nohp, logits, lse, kv_indptr, kv_indices, num_splits,
                    max_splits, sm, 0.0, block_n=bn, num_warps=w, num_stages=3,
                    wide_load=True))
                regs = getattr(kw, "n_regs", None)
                shared = getattr(getattr(kw, "metadata", None), "shared", None)
                print(f"  {bn:>3}/{w} narrow {t_n:.3f} ms  wide {t_w:.3f} ms  "
                      f"{t_n/t_w:.2f}x  maxdev={dev:.2e} "
                      f"{'OK' if ok else '<-- WRONG, ignore the speed'}"
                      f"  [regs={regs} smem={shared}]")
            except Exception as e:  # noqa: BLE001
                print(f"  {bn:>3}/{w} wide - {type(e).__name__}: "
                      f"{_err(e)}")

        # SKIP_HP: what arena-awareness costs the fast kernel.
        #
        # The group-factored kernel is currently benchmark-only because it has
        # no window arena, so its speed does not ship. The computed-tile kernel
        # OVERRIDES arena rows via a full-width [BLOCK_N, D] select, and that
        # branch costs 1.54x across the whole kernel even though ~2% of blocks
        # take it -- the compiler reserves its registers everywhere. Folding
        # that same shape in here would hand back most of the factoring win,
        # because an arena value is arbitrary bf16 and cannot be written as a
        # code plus a per-group scale.
        #
        # Excluding instead of overriding costs two int lookups and a mask, and
        # lets a separate dense BF16 pass own those tokens as an extra split
        # that stage 2 already merges. This arm prices the exclusion. If it is
        # cheap the design is viable; if it is not, the arena has to be handled
        # somewhere other than inside this loop.
        print("\nSKIP_HP (exclude arena tokens; cost of arena-awareness):")
        for bn, w in ((32, 4), (64, 8)):
            try:
                t_plain = timeit(lambda: packed_mla_decode_stage1_gf(
                    q, ops_nohp, logits, lse, kv_indptr, kv_indices, num_splits,
                    max_splits, sm, 0.0, block_n=bn, num_warps=w, num_stages=3))
                t_skip = timeit(lambda: packed_mla_decode_stage1_gf(
                    q, ops, logits, lse, kv_indptr, kv_indices, num_splits,
                    max_splits, sm, 0.0, block_n=bn, num_warps=w, num_stages=3,
                    skip_hp=True))
                print(f"  {bn:>3}/{w} no-arena {t_plain:.3f} ms  "
                      f"arena-aware {t_skip:.3f} ms  "
                      f"cost {t_skip/t_plain:.2f}x  (override path was 1.54x)")
            except Exception as e:  # noqa: BLE001
                print(f"  {bn:>3}/{w} skip_hp - {type(e).__name__}: "
                      f"{_err(e)}")
    except Exception as e:  # noqa: BLE001
        print(f"\ngroup-factored failed: {type(e).__name__}: "
              f"{_err(e)}")

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
            msg = _err(e)
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


def two_pass_equivalence(bs=8, heads=16, seq=4096, windows=576, splits=8):
    """Does exclude-plus-dense-window equal the production override kernel?

    This is the gate on shipping the group-factored kernel, and it is a
    correctness question, not a speed one: a split that drops tokens or counts
    them twice is fast and wrong, and at 2 bits the difference hides easily
    inside quantization noise. Compared after stage 2, on the final output,
    because that is what the model consumes.
    """
    import torch as _t
    from sglang.srt.layers.attention.triton_ops.decode_attention import (
        _decode_softmax_reducev_fwd,
    )
    from sglang.srt.layers.attention.triton_ops.mla_packed_decode import (
        _fwd_hp_window_stage1,
        _safe_block_h,
        _v_shape_proxy,
        packed_mla_decode_stage1,
        packed_mla_decode_stage1_gf,
    )
    import triton as _tr

    ops, kv, q, indptr, idx, ns, ms, lg, ls = build(bs, heads, seq,
                                                    windows=windows,
                                                    max_splits=splits)
    sm = 1.0 / (R + 64) ** 0.5
    P_TOK, R_TOK = 64, windows - 64

    o_ref = _t.zeros((bs, heads, R), dtype=q.dtype, device=q.device)
    packed_mla_decode_stage1(q, ops, lg, ls, indptr, idx, ns, ms, sm, 0.0)
    _decode_softmax_reducev_fwd(lg, ls, q, o_ref, 1.0,
                                _v_shape_proxy(o_ref, R), indptr, ns, ms)

    # One spare split for the window partial.
    lg2 = _t.zeros((bs, heads, ms + 1, R), dtype=_t.float32, device=q.device)
    ls2 = _t.zeros((bs, heads, ms + 1), dtype=_t.float32, device=q.device)
    o_new = _t.zeros_like(o_ref)
    packed_mla_decode_stage1_gf(q, ops, lg2, ls2, indptr, idx, ns, ms, sm, 0.0,
                                skip_hp=True)
    bh = _safe_block_h(16, heads)
    _fwd_hp_window_stage1[(bs, _tr.cdiv(heads, min(bh, heads)))](
        q, ops[2], ops[3], ops[4], ops[5],
        sm, indptr, idx, lg2, ls2, ns,
        q.stride(0), q.stride(1),
        lg2.stride(0), lg2.stride(1), lg2.stride(2),
        kv_group_num=heads, q_head_num=heads, D=R, DPE=64,
        P_TOK=P_TOK, R_TOK=R_TOK, BLOCK_N=32, BLOCK_H=bh, logit_cap=0.0,
        num_warps=4, num_stages=2,
    )
    # Localise the fault before stage 2 can smear it. Two guesses have already
    # been wrong here, so report WHICH tensor and WHICH split goes non-finite
    # rather than reasoning about it: the packed splits and the window split
    # are written by different kernels, and stage 2 mixes them, so a NaN in the
    # merged output cannot say which side produced it.
    for nm, t in (("stage1 att_out", lg2), ("stage1 att_lse", ls2)):
        bad = ~_t.isfinite(t)
        if bad.any():
            per = [int(bad[:, :, k].sum()) for k in range(t.shape[2])]
            print(f"  [diag] {nm}: {int(bad.sum())} non-finite; per split {per} "
                  f"(splits 0..{ms - 1} = packed, split {ms} = window)")
        else:
            print(f"  [diag] {nm}: all finite")
    # An all-excluded split legitimately has lse = -inf; that is not a fault,
    # but stage 2 must not then see EVERY split as -inf for a row.
    finite_lse = _t.isfinite(ls2).any(dim=2)
    if not bool(finite_lse.all()):
        print(f"  [diag] {int((~finite_lse).sum())} (batch, head) rows have "
              f"NO finite split at all -- stage 2 will compute -inf minus -inf")

    _decode_softmax_reducev_fwd(lg2, ls2, q, o_new, 1.0,
                                _v_shape_proxy(o_new, R), indptr, ns + 1, ms + 1)

    # Report NaN as its own verdict. A NaN makes max() NaN and every comparison
    # False, so a bare threshold test silently reads as MISMATCH and hides
    # which of the two failure modes happened -- wrong values, or no values.
    n_ref = int((~_t.isfinite(o_ref)).sum())
    n_new = int((~_t.isfinite(o_new)).sum())
    if n_ref or n_new:
        print(f"\ntwo-pass equivalence: NON-FINITE outputs "
              f"(ref {n_ref}, new {n_new} of {o_ref.numel()}) -- "
              f"this is an empty-split or masking fault, not a value mismatch")
    d = (o_ref.float() - o_new.float()).abs()
    rel = (d.max() / o_ref.float().abs().max().clamp(min=1e-6)).item()
    print(f"\ntwo-pass equivalence vs the production override kernel "
          f"(bs={bs} seq={seq} window={windows}):")
    print(f"  max|dref-dnew| = {d.max().item():.3e}   rel = {rel:.3e}   "
          f"{'MATCH' if rel < 2e-2 else 'MISMATCH -- do not ship'}")
    # A split that silently dropped the window would still look close, because
    # the packed tier holds the same tokens at 2 bits. So check that the window
    # actually contributed: zeroing the arena must CHANGE the answer.
    ops_zero = (ops[0], ops[1], ops[2], _t.zeros_like(ops[3]), ops[4], ops[5],
                ops[6], ops[7])
    o_zero = _t.zeros_like(o_ref)
    packed_mla_decode_stage1(q, ops_zero, lg, ls, indptr, idx, ns, ms, sm, 0.0)
    _decode_softmax_reducev_fwd(lg, ls, q, o_zero, 1.0,
                                _v_shape_proxy(o_zero, R), indptr, ns, ms)
    sens = ((o_ref.float() - o_zero.float()).abs().max()
            / o_ref.float().abs().max().clamp(min=1e-6)).item()
    print(f"  arena sensitivity (zeroing it moves the reference) = {sens:.3e} "
          f"{'ok, the window is load-bearing' if sens > 1e-3 else 'WARNING: the window barely matters here, so MATCH proves little'}")


def _run_equivalence():
    # Several regimes, because one regime is how the split-borrowing bug got
    # through. The gate ran at seq=4096 with 8 splits, where num_kv_splits is
    # never 1, so it never exercised the case where the window slot collides
    # with the last packed slot -- and a live server on 55-token prompts hit it
    # immediately. Short sequences are also where the window covers MOST of the
    # KV, so a dropped window is maximally visible there and nearly invisible
    # at 20000 tokens.
    for kw in (
        dict(bs=8, heads=16, seq=4096, windows=576, splits=8),
        dict(bs=8, heads=16, seq=256, windows=576, splits=1),
        dict(bs=4, heads=16, seq=600, windows=576, splits=2),
    ):
        try:
            two_pass_equivalence(**kw)
        except Exception as e:  # noqa: BLE001
            print(f"\ntwo-pass equivalence {kw} FAILED to run: "
                  f"{type(e).__name__}: {_err(e)}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("needs a GPU")
        sys.exit(1)
    main()
    _run_equivalence()
