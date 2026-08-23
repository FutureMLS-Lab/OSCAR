#!/usr/bin/env python3
"""Verify the head-tiling fix numerically, in BOTH broken regimes, with a
negative control.

The defect: ``_fwd_grouped_kernel_stage1`` and its INT2 twin compute

    VALID_BLOCK_H = min(BLOCK_H, kv_group_num)
    cur_head      = cur_head_id * VALID_BLOCK_H + arange(BLOCK_H)   # flat
    cur_kv_head   = cur_head_id // cdiv(kv_group_num, BLOCK_H)      # from block

so the two agree only when a head block lies wholly inside one KV group, i.e.
``BLOCK_H >= kv_group_num or kv_group_num % BLOCK_H == 0``. The batch heuristic
picks BLOCK_H from the batch size and therefore breaks in two OPPOSITE regimes:

    kv_group_num  5..7   BLOCK_H 4 at batch >= 16  -> broken at LARGE batch
    kv_group_num  9..15  BLOCK_H 8 at batch <  16  -> broken at SMALL batch

MiniMax-M2.7 is 48Q/8KV at TP=4 = kv_group_num 6 (the first regime, the one that
cost it GPQA 79.3 -> 53.5). GLM-4.7-FP8 is 96Q/8KV = 12 (the second regime).
Only the first was ever re-measured, hence this probe.

Why the negative control matters: on this project a check that reads "clean" has
repeatedly turned out not to discriminate (the mixed-KV auditor's damage counter
reads 0 even on known-broken code at 85% padded replay). So every geometry is run
twice -- once with the fix live, once with ``_safe_block_h`` monkeypatched to the
identity, which reproduces exactly the pre-fix BLOCK_H. The control MUST fail at
the previously-broken batch sizes. If it does not, this probe proves nothing and
says so.

Run on 1 GPU. TREE must point at a tree that HAS the fix.
"""
import argparse
import os
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("--tree", default=os.environ.get("TREE", "/tmp/tree"))
p.add_argument("--probe", default=None,
               help="path to kernel_reference_probe.py (default: alongside TREE)")
args = p.parse_args()

PROBE = args.probe or os.path.join(
    args.tree, "rotation", "investigation", "10_m27_graphbs",
    "kernel_reference_probe.py")

# (label, q_heads, kv_heads, kv_group_num, batch sizes to test, regime note)
GEOMETRIES = [
    ("M2.7  TP4  12Q/2KV", 12, 2, 6,
     "1,2,4,8,12,15,16,17,20,24,32", "broken pre-fix at batch >= 16"),
    # The mirror regime needs kv_heads >= 2 per rank to be OBSERVABLE. With one
    # KV head per rank every head block maps to kv_head 0, which is also the
    # only KV head, so the mis-mapping is masked and reads correct data by
    # accident. GLM-4.7-FP8 (96Q/8KV, group 12) is therefore harmless at TP=8
    # (12Q/1KV per rank) but damaging at TP<=4. Measured: the 12Q/1KV control
    # showed no defect at all, which is why this list carries both.
    ("GLM-4.7 TP4 24Q/2KV", 24, 2, 12,
     "1,2,4,8,12,15,16,24,32", "broken pre-fix at batch < 16"),
    ("GLM-4.7 TP1 96Q/8KV", 96, 8, 12,
     "1,4,8,12,15,16,24", "broken pre-fix at batch < 16"),
    ("GLM-4.7 TP8 12Q/1KV", 12, 1, 12,
     "1,4,8,12,15,16,24,32", "masked: only one KV head per rank"),
    # clean controls: every other model in the sweep is a power-of-two group
    ("Qwen3-8B TP2 16Q/4KV", 16, 4, 4,
     "1,8,15,16,32", "never broken (power of two)"),
    ("Qwen3-32B TP4 16Q/2KV", 16, 2, 8,
     "1,8,15,16,32", "never broken (power of two)"),
]

NEUTER = """
import sglang.srt.layers.attention.triton_ops.decode_attention as _da
_orig = _da._safe_block_h
_da._safe_block_h = lambda block_h, kv_group_num: block_h
print("[control] _safe_block_h NEUTERED -> pre-fix BLOCK_H restored", flush=True)
"""


def run(q, kv, bs_list, neuter):
    """Run kernel_reference_probe.py in a subprocess, optionally pre-neutered."""
    src = open(PROBE).read()
    if neuter:
        # insert the monkeypatch right after the probe imports the kernel
        anchor = "from sglang.srt.layers.attention.triton_ops.decode_attention import (  # noqa: E402\n    decode_attention_fwd_int2_unified,\n)"
        if anchor not in src:
            return None, "could not find import anchor to inject the control"
        src = src.replace(anchor, anchor + "\n" + NEUTER)
    tmp = "/tmp/_probe_run.py"
    open(tmp, "w").write(src)
    cmd = [sys.executable, tmp, "--tree", args.tree,
           "--heads", str(q), "--kv-heads", str(kv), "--batch-sizes", bs_list]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout + r.stderr, None


def parse(out):
    """Pull (bs, rel_l2, verdict) rows out of the probe's table."""
    rows = []
    for line in (out or "").splitlines():
        f = line.split()
        if len(f) >= 4 and f[0].isdigit() and ".." in line:
            try:
                bs = int(f[0])
                rel = float(f[-2]) if f[-1] in ("OK",) else float(
                    [x for x in f if "e-" in x or "e+" in x][-1])
                rows.append((bs, rel, "OK" if "OK" in line else "WRONG"))
            except (ValueError, IndexError):
                continue
    return rows


print(f"[verify] tree={args.tree}")
print(f"[verify] probe={PROBE}")
print(f"[verify] fix present in tree: ", end="")
dec = os.path.join(args.tree, "sglang-research", "python", "sglang", "srt",
                   "layers", "attention", "triton_ops", "decode_attention.py")
nhits = open(dec).read().count("_safe_block_h") if os.path.isfile(dec) else 0
print(f"{nhits} occurrences of _safe_block_h" +
      ("  (OK)" if nhits else "  *** NO FIX IN THIS TREE ***"))
if not nhits:
    sys.exit("refusing to run: TREE has no fix, there is nothing to verify")

verdicts = {}
for label, q, kv, g, bs_list, note in GEOMETRIES:
    print("\n" + "=" * 78)
    print(f"{label}   kv_group_num={g}   [{note}]")
    print("=" * 78)
    for arm, neuter in (("FIXED  ", False), ("CONTROL", True)):
        out, err = run(q, kv, bs_list, neuter)
        if err:
            print(f"  {arm}: SKIPPED ({err})")
            continue
        rows = parse(out)
        if not rows:
            print(f"  {arm}: NO ROWS PARSED -- probe output follows")
            print("  " + "\n  ".join((out or "").splitlines()[-25:]))
            continue
        worst = max(rows, key=lambda r: r[1])
        bad = [r for r in rows if r[2] == "WRONG"]
        print(f"  {arm}: " + "  ".join(f"bs{b}={r:.1e}" for b, r, _ in rows))
        print(f"           worst rel_l2 {worst[1]:.2e} at bs={worst[0]}, "
              f"{len(bad)} of {len(rows)} batch sizes WRONG"
              + (f" -> {[b for b, _, _ in bad]}" if bad else ""))
        verdicts[(label, arm.strip())] = (worst[1], [b for b, _, _ in bad])

print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
print(f"{'geometry':<24}{'arm':<9}{'worst rel_l2':>14}  broken batch sizes")
ok = True
for (label, arm), (worst, bad) in verdicts.items():
    print(f"{label:<24}{arm:<9}{worst:>14.2e}  {bad if bad else '-'}")
for label, q, kv, g, bs_list, note in GEOMETRIES:
    f = verdicts.get((label, "FIXED"))
    c = verdicts.get((label, "CONTROL"))
    if not f or not c:
        continue
    affected = note.startswith("broken")
    if note.startswith("masked"):
        if c[1]:
            print(f"NOTE  {label}: control DID break at {c[1]} -- the "
                  f"single-KV-head masking argument is wrong, revisit it")
            ok = False
        else:
            print(f"PASS  {label}: control clean too, confirming the defect is "
                  f"MASKED when there is one KV head per rank (every block maps "
                  f"to kv_head 0, which is the only one)")
        continue
    if f[1]:
        print(f"FAIL  {label}: fixed tree still wrong at {f[1]}")
        ok = False
    if affected and not c[1]:
        print(f"INCONCLUSIVE  {label}: negative control did NOT reproduce the "
              f"defect, so 'clean on the fixed tree' proves nothing here")
        ok = False
    if affected and c[1]:
        print(f"PASS  {label}: control breaks at {c[1]} (worst {c[0]:.1e}), "
              f"fixed tree clean everywhere (worst {f[0]:.1e}) -> probe "
              f"discriminates AND the fix holds")
    if not affected and not f[1] and not c[1]:
        print(f"PASS  {label}: clean in both arms, as expected for a "
              f"power-of-two kv_group_num")
print("\nOVERALL:", "PASS" if ok else "NOT PROVEN")
