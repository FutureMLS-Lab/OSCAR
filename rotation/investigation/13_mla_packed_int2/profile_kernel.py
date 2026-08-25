#!/usr/bin/env python3
"""Profile the packed MLA decode kernel instead of guessing at its knobs.

Six hypotheses were refuted and three knobs were swept (tile, warps, splits)
before anyone looked at a counter. The sweeps did produce the group-factored
2.5x, but they cannot say *why* BLOCK_N=128 dies or whether the remaining 3.0x
against BF16 is memory, issue, occupancy or synchronisation -- and those want
different fixes.

Collected per kernel:

* ``sm__throughput`` / ``dram__throughput`` -- is it compute- or memory-bound at
  all, or neither (which would mean latency-bound)
* ``smsp__warp_issue_stalled_*`` -- *why* warps are not issuing.
  ``long_scoreboard`` is a global-memory dependency, ``mio_throttle`` is
  LSU/shared-memory pressure, ``barrier`` is sync, ``short_scoreboard`` is
  usually MIO/smem latency.
* ``launch__occupancy_limit_*`` -- what caps occupancy: registers, shared
  memory, or the block size. BLOCK_N=128 failed with 280 KB requested, so shared
  memory is a hard ceiling somewhere; this says whether it also binds at 64.
* achieved vs theoretical occupancy.

Also sweeps ``num_stages``, which is Triton's software pipelining -- the
overlap knob. It was measured on the *old* computed-tile kernel (2 and 3 were
far worse) and never on the factored one, where the per-group code loads are
exactly what a pipeline would prefetch. Carrying the old result over would be
assuming the conclusion.
"""
from __future__ import annotations

import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_kernel import R, build, timeit  # noqa: E402

from sglang.srt.layers.attention.triton_ops.decode_attention import (  # noqa: E402
    _decode_grouped_att_m_fwd,
)
from sglang.srt.layers.attention.triton_ops.mla_packed_decode import (  # noqa: E402
    packed_mla_decode_stage1,
    packed_mla_decode_stage1_gf,
)

METRICS = ",".join([
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_wait_per_warp_active.pct",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__shared_mem_per_block_static",
    "launch__registers_per_thread",
])


def stage_sweep(bs, heads, seq, splits):
    """num_stages is the overlap knob; it has never been tried on this kernel."""
    ops, kv, q, indptr, idx, ns, ms, lg, ls = build(bs, heads, seq,
                                                    max_splits=splits)
    ops_nohp = ops[:3] + (None, None, None) + ops[6:]
    sm = 1.0 / (R + 64) ** 0.5
    base = timeit(lambda: _decode_grouped_att_m_fwd(
        q, kv, kv[..., :R], lg, ls, indptr, idx, ns, ms, sm, 0.0))
    print(f"\n=== num_stages sweep (software pipelining / overlap) ===")
    print(f"BF16 baseline {base:.3f} ms at splits={splits}")
    print(f"{'BLOCK_N':>8} {'warps':>6} {'stages':>7} {'ms':>9} {'vs BF16':>8}")
    best = None
    for bn in (32, 64):
        for w in (4, 8):
            for st in (1, 2, 3, 4):
                try:
                    t = timeit(lambda: packed_mla_decode_stage1_gf(
                        q, ops_nohp, lg, ls, indptr, idx, ns, ms, sm, 0.0,
                        block_n=bn, num_warps=w, num_stages=st))
                    print(f"{bn:>8} {w:>6} {st:>7} {t:>9.3f} {base/t:>7.2f}x")
                    if best is None or t < best[0]:
                        best = (t, bn, w, st)
                except Exception as e:  # noqa: BLE001
                    msg = str(e).splitlines()[0][:52]
                    print(f"{bn:>8} {w:>6} {st:>7} {'-':>9}  {type(e).__name__}: {msg}")
    if best:
        print(f"best: {best[1]}/{best[2]}/stages={best[3]} -> {best[0]:.3f} ms "
              f"({base/best[0]:.2f}x BF16)")
    return best


def triton_resources(tag: str, jit_fn):
    """Registers / spills / shared memory straight off the compiled kernel.

    ncu needs GPU performance-counter permission (ERR_NVGPUCTRPERM in an
    unprivileged pod), but the two numbers that explain the BLOCK_N ceiling do
    not need counters at all -- Triton records them on the compiled kernel.
    Shared memory per block against the SM's budget says whether smem is the
    occupancy limiter; a non-zero spill count says the register allocator gave
    up, which is invisible in wall-clock until it is the whole problem.
    """
    try:
        cache = getattr(jit_fn, "cache", None)
        if not cache:
            print(f"  [{tag}] no compiled-kernel cache to read")
            return
        rows = []
        for dev, entries in cache.items():
            for key, k in entries.items():
                shared = getattr(getattr(k, "metadata", None), "shared", None)
                rows.append((
                    getattr(k, "n_regs", None),
                    getattr(k, "n_spills", None),
                    shared,
                ))
        seen = set()
        for regs, spills, shared in rows:
            sig = (regs, spills, shared)
            if sig in seen:
                continue
            seen.add(sig)
            note = ""
            if spills:
                note += "  <-- REGISTER SPILLS"
            if shared and shared > 100_000:
                note += "  <-- smem is the occupancy limiter"
            print(f"  [{tag}] regs/thread={regs} spills={spills} "
                  f"shared/block={shared}{note}")
    except Exception as e:  # noqa: BLE001
        print(f"  [{tag}] resource read failed: {type(e).__name__}: {e}")


def ncu_profile(bs, heads, seq, splits, block_n, warps, stages):
    """Re-exec one kernel launch under ncu and print the counters."""
    ncu = None
    for cand in ("ncu", "/usr/local/cuda/bin/ncu", "/opt/nvidia/nsight-compute/ncu"):
        if subprocess.run(["bash", "-lc", f"command -v {cand}"],
                          capture_output=True).returncode == 0:
            ncu = cand
            break
    if ncu is None:
        print("\nncu not present in this image; skipping counters. The stage "
              "sweep above still stands on its own.")
        return
    child = f"""
import os, sys
sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})
import torch
from bench_kernel import R, build
from sglang.srt.layers.attention.triton_ops.mla_packed_decode import (
    packed_mla_decode_stage1, packed_mla_decode_stage1_gf)
ops, kv, q, indptr, idx, ns, ms, lg, ls = build({bs}, {heads}, {seq}, max_splits={splits})
ops_nohp = ops[:3] + (None, None, None) + ops[6:]
sm = 1.0 / (R + 64) ** 0.5
for _ in range(3):
    packed_mla_decode_stage1_gf(q, ops_nohp, lg, ls, indptr, idx, ns, ms, sm, 0.0,
                                block_n={block_n}, num_warps={warps}, num_stages={stages})
torch.cuda.synchronize()
"""
    with open("/tmp/_ncu_child.py", "w") as f:
        f.write(child)
    cmd = [ncu, "--metrics", METRICS, "--target-processes", "all",
           "--kernel-name", "regex:_fwd_packed_mla_stage1_gf",
           "--launch-count", "1", "--csv",
           sys.executable, "/tmp/_ncu_child.py"]
    print(f"\n=== ncu counters (BLOCK_N={block_n} warps={warps} stages={stages}) ===")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    out = (r.stdout or "") + (r.stderr or "")
    keep = [l for l in out.splitlines()
            if any(k in l for k in ("throughput", "stalled", "occupancy_limit",
                                    "warps_active", "shared_mem", "registers_per"))]
    print("\n".join(keep[:40]) if keep else out[-3000:])


def main():
    bs = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    heads = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    seq = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    splits = int(os.environ.get("BENCH_SPLITS", "16"))
    print(f"shape bs={bs} heads={heads} seq={seq} splits={splits}")
    best = stage_sweep(bs, heads, seq, splits)
    bn, w, st = (best[1], best[2], best[3]) if best else (64, 8, 1)
    print("\n=== compiled-kernel resources (no counters needed) ===")
    ops, kv, q, indptr, idx, ns, ms, lg, ls = build(bs, heads, seq,
                                                    max_splits=splits)
    ops_nohp = ops[:3] + (None, None, None) + ops[6:]
    sm = 1.0 / (R + 64) ** 0.5
    for tag, tbn, tw, tst in (("best", bn, w, st), ("BLOCK_N=64", 64, 8, 1)):
        try:
            k = packed_mla_decode_stage1_gf(
                q, ops_nohp, lg, ls, indptr, idx, ns, ms, sm, 0.0,
                block_n=tbn, num_warps=tw, num_stages=tst)
            shared = getattr(getattr(k, "metadata", None), "shared", None)
            regs = getattr(k, "n_regs", None)
            spills = getattr(k, "n_spills", None)
            note = ""
            if spills:
                note += "  <-- REGISTER SPILLS"
            # B200 has 228 KB of shared memory per SM; anything close to that
            # means one block per SM, i.e. smem is the occupancy limiter.
            if shared and shared > 114_000:
                note += "  <-- >half the SM's smem: 1 block/SM"
            print(f"  [{tag} {tbn}/{tw}/st{tst}] regs/thread={regs} "
                  f"spills={spills} shared/block={shared}{note}")
        except Exception as e:  # noqa: BLE001
            print(f"  [{tag}] {type(e).__name__}: {str(e).splitlines()[0][:70]}")
    ncu_profile(bs, heads, seq, splits, bn, w, st)


if __name__ == "__main__":
    main()
