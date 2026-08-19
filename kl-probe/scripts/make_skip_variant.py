#!/usr/bin/env python3
"""Build MXFP8-skip placement variants of MiniMax-M3 as zero-byte symlink farms.

See docs/mxfp8-skip-variants-plan.md. The two donor checkpoints are layer-atomic:

    model-00001..00031   all non-expert tensors (identical in every variant)
    model-00032..00088   exactly ONE MoE layer's 128x3 routed experts each,
                         shard number = layer + 29

so a variant is a *file selection*, not a build: 31 shared shards + 57 expert
shards, each taken from the NVFP4 donor or from the MXFP8 expert bank, plus three
generated JSON files. No shared shard holds expert tensors for a swapped layer, so
every tensor exists exactly once in exactly one format — which is what makes this
correct under a loader whose index filters *files*, not tensors.

The only real write is the one-time expert bank (57 x 6.96 GiB), extracted from the
uniform-MXFP8 0602 checkpoint. Those bytes cannot be cloned: safetensors forbids
gaps in the data section and no donor (shard, layer) expert run is 4096-aligned.

Usage
-----
    # one time, ~397 GiB, ~10-15 min  (resumable; skips layers already present)
    python scripts/make_skip_variant.py extract-bank

    # prove the extractor against a known-good artifact (needs bank layers 3,4,5)
    python scripts/make_skip_variant.py verify-bank

    # one arm, or every arm in the ARMS table
    python scripts/make_skip_variant.py build --arm mxfp8skip11-late
    python scripts/make_skip_variant.py build --layers 49,50,51,52,53,54,55,56,57,58,59 \
        --out /data/huggingface/MiniMax-M3-NVFP4-mxfp8skip11-late
    python scripts/make_skip_variant.py build-all [--dry-run]

    # MANDATORY before serving: a missing tensor loads as torch.empty garbage
    # with no error, i.e. it shows up as a plausible-but-wrong KL, not a crash.
    python scripts/make_skip_variant.py validate /data/huggingface/MiniMax-M3-NVFP4-mxfp8skip11-late

    python scripts/make_skip_variant.py selftest      # no I/O beyond headers
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------- checkpoints

DONOR = Path("/data/huggingface/MiniMax-M3-NVFP4")  # NVFP4 on all 57 MoE layers
MXFP8_SRC = Path("/scratch/tonyzhang/models/Minimax-M3-0602")  # uniform MXFP8
BANK = Path("/data/huggingface/_m3-mxfp8-expert-bank")
ALT3X6 = Path("/scratch/huggingface/MiniMax-M3-NVFP4-alt3x6-0708")  # known-good
OUT_ROOT = Path("/data/huggingface")

MOE_LAYERS = range(3, 60)  # 0-2 are dense MLP
N_EXPERTS = 128
PROJ = ("w1", "w2", "w3")
SHARD_OFFSET = 29  # expert shard number = layer + SHARD_OFFSET
NUM_SHARDS = 88
LP = "language_model.model.layers"

# alt3x6's 21-layer period-9 skip set — the reference artifact for selftests.
ALT3X6_LAYERS = [3, 4, 5, 12, 13, 14, 21, 22, 23, 30, 31, 32, 39, 40, 41, 48, 49, 50, 57, 58, 59]

# Every arm in configs/minimax-m3-skip11.yaml (K=11) and configs/minimax-m3-skip14.yaml
# (K=14). Single source of truth: validate() cross-checks a built dir against these,
# and selftest() checks each entry's length against the K in its own name.
ARMS: dict[str, list[int]] = {
    "mxfp8skip11-blockstart": [3, 4, 5, 21, 22, 39, 40, 48, 49, 57, 58],
    "mxfp8skip11-blockshift": [4, 13, 14, 22, 23, 40, 41, 49, 50, 57, 58],
    "mxfp8skip11-random": [4, 14, 17, 27, 30, 32, 38, 45, 50, 55, 59],
    "mxfp8skip11-edge": [3, 4, 5, 6, 7, 54, 55, 56, 57, 58, 59],
    "mxfp8skip11-spread": [3, 9, 14, 20, 25, 31, 37, 42, 48, 53, 59],
    "mxfp8skip11-early": list(range(3, 14)),
    "mxfp8skip11-late": list(range(49, 60)),
    "mxfp8skip11-tail8": [3, 4, 5] + list(range(52, 60)),
    "mxfp8skip11-mid": list(range(26, 37)),
    # --- pinned-end + uniform-lattice family (2026-07-27) ----------------------
    # Each arm pins the tail layer 59 plus EXACTLY ONE shallow layer (3 xor 4,
    # never both) and spends the remaining budget on a single-step interior comb.
    # Knobs: which shallow layer, whether 58 also gets a slot, and the comb's
    # (step, phase). Designed as matched contrasts, not independent draws:
    #   pin3t58 vs pin4t58   10/11 layers identical -> 3-vs-4 alone (also the
    #                        cheapest run-to-run replicate available)
    #   pin3s6  vs pin3s6lo  same pins + step 6, comb phase shifted -3
    #   pin3s6  vs pin4t58   uniform +1 on all ten comb layers (second replicate)
    #   pin3s6  vs pin3t58   one comb slot traded for layer 58
    #   pin4s5   no coverage between 4 and 14, buying a dense step-5 lattice
    "mxfp8skip11-pin3s6": [3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 59],
    "mxfp8skip11-pin3s6lo": [3, 6, 12, 18, 24, 30, 36, 42, 48, 54, 59],
    "mxfp8skip11-pin4s5": [4, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59],
    "mxfp8skip11-pin3t58": [3, 10, 16, 22, 28, 34, 40, 46, 52, 58, 59],
    "mxfp8skip11-pin4t58": [4, 10, 16, 22, 28, 34, 40, 46, 52, 58, 59],
    # --- K=14 exploration (2026-07-27), configs/minimax-m3-skip14.yaml ---------
    # Only the two shapes that won at K=11 are carried forward: `mid` (contiguous
    # middle block, best K=11 arm at 0.02681) and `spread` (uniform lattice).
    # The additive null at K=14 is 0.02670, which mid11 already matched with three
    # fewer layers — so these ask whether good placement compounds with budget.
    #
    # MID: how to spend three more layers than mid11 (26-36), and whether the
    # block's contiguity mattered or only its depth band.
    #   mid vs middeep      same width, pushed deeper: is "middle" really middle?
    #   mid vs midsplit2    same 14 layers, same region, centre knocked out ->
    #                       the with/without-split contrast
    #   midtail3 vs midhead3  mid11 kept intact + 3 layers at the tail vs at the
    #                       head. Prices the NVFP4-report tail rule against the
    #                       llama.cpp/Llama-3 both-ends convention on a known-good
    #                       core; head3 is the falsification arm (early was the
    #                       worst K=11 arm, edge second-worst).
    "mxfp8skip14-mid": list(range(25, 39)),
    "mxfp8skip14-middeep": list(range(32, 46)),
    "mxfp8skip14-midsplit2": list(range(22, 29)) + list(range(32, 39)),
    "mxfp8skip14-midtail3": list(range(26, 37)) + [57, 58, 59],
    "mxfp8skip14-midhead3": [3, 4, 5] + list(range(26, 37)),
    # SPREAD: phase came out null at K=11 (two independent tests), so these vary
    # STEP — hence how much of the shallow stack the lattice gives up. s4 -> s3 ->
    # s2 concentrate monotonically deeper; spreadmid3 is the hybrid, a uniform
    # lattice centred on 31.5 like the mid family.
    "mxfp8skip14-spread": [3, 7, 12, 16, 20, 25, 29, 33, 37, 42, 46, 50, 55, 59],
    "mxfp8skip14-spread4": list(range(7, 60, 4)),
    "mxfp8skip14-spread3": list(range(20, 60, 3)),
    "mxfp8skip14-spread2": list(range(33, 60, 2)),
    "mxfp8skip14-spreadmid3": list(range(12, 52, 3)),
}

# Small files a served checkpoint needs beside the weights. Copied, not linked, so
# a variant dir is self-describing (~17 MB).
AUX_SKIP_SUFFIX = (".safetensors", ".orig", ".log")
AUX_SKIP_NAMES = {"model.safetensors.index.json", "config.json", "hf_quant_config.json"}

CHUNK = 32 << 20

# ------------------------------------------------------------ safetensors I/O


def read_header(path: Path) -> tuple[dict, int]:
    """Return (header dict, absolute byte offset where tensor data starts)."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    return header, 8 + n


_hdr_cache: dict[Path, tuple[dict, int]] = {}


def header_cached(path: Path) -> tuple[dict, int]:
    if path not in _hdr_cache:
        _hdr_cache[path] = read_header(path)
    return _hdr_cache[path]


def write_safetensors(out: Path, plan: list[tuple[str, dict, Path, int]]) -> int:
    """Write a safetensors file by streaming byte ranges out of source files.

    plan: (tensor_name, source_header_entry, source_path, source_data_start).
    Payloads are copied verbatim — no dtype interpretation, no torch. Offsets are
    emitted contiguously from 0 because safetensors rejects gaps in the data
    section, and the header is space-padded to a multiple of 8 to keep the data
    section 8-byte aligned (matching the reference serializer).
    """
    header: dict = {"__metadata__": {"format": "pt"}}
    cursor = 0
    for name, entry, _src, _start in plan:
        s, e = entry["data_offsets"]
        nbytes = e - s
        header[name] = {
            "dtype": entry["dtype"],
            "shape": entry["shape"],
            "data_offsets": [cursor, cursor + nbytes],
        }
        cursor += nbytes
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)

    tmp = out.with_suffix(out.suffix + ".partial")
    with open(tmp, "wb") as w:
        w.write(struct.pack("<Q", len(blob)))
        w.write(blob)
        for _name, entry, src, start in plan:
            s, e = entry["data_offsets"]
            remaining = e - s
            with open(src, "rb") as r:
                r.seek(start + s)
                while remaining:
                    buf = r.read(min(CHUNK, remaining))
                    if not buf:
                        raise IOError(f"short read in {src} for {_name}")
                    w.write(buf)
                    remaining -= len(buf)
    os.replace(tmp, out)
    return cursor


# --------------------------------------------------------------- tensor names


def expert_tensor_names(layer: int, mxfp8: bool) -> list[str]:
    """The routed-expert tensor names for one MoE layer, in the given format.

    MXFP8: weight (F8_E4M3) + weight_scale_inv (U8/E8M0, block 32) -> 768 tensors.
    NVFP4: weight (U8, 2x fp4 packed) + weight_scale (F8_E4M3, group 16)
           + weight_scale_2 (F32) + input_scale (F32)               -> 1536 tensors.
    """
    leaves = ("weight", "weight_scale_inv") if mxfp8 else (
        "weight",
        "weight_scale",
        "weight_scale_2",
        "input_scale",
    )
    return [
        f"{LP}.{layer}.block_sparse_moe.experts.{e}.{w}.{leaf}"
        for e in range(N_EXPERTS)
        for w in PROJ
        for leaf in leaves
    ]


def exclude_modules(mxfp8_layers: list[int]) -> list[str]:
    """The `exclude_modules` list for a variant: the donor's non-expert exclusions
    plus, per MXFP8 layer, the two whole-layer globs and 128x3 per-expert module
    names. Sorted, which reproduces alt3x6's 8702-entry list byte-for-byte at
    K=21 (see selftest). Length is exactly 596 + 386*K.
    """
    base = json.loads((DONOR / "config.json").read_text())["quantization_config"][
        "exclude_modules"
    ]
    out = list(base)
    for layer in mxfp8_layers:
        out += [f"{LP}.{layer}", f"{LP}.{layer}.*"]
        out += [
            f"{LP}.{layer}.block_sparse_moe.experts.{e}.{w}"
            for e in range(N_EXPERTS)
            for w in PROJ
        ]
    return sorted(out)


def shard_name(n: int) -> str:
    return f"model-{n:05d}-of-{NUM_SHARDS:05d}.safetensors"


def bank_name(layer: int) -> str:
    return f"layer-{layer:03d}.mxfp8.safetensors"


def check_layers(layers: list[int]) -> list[int]:
    bad = [x for x in layers if x not in MOE_LAYERS]
    if bad:
        sys.exit(f"not MoE layers (must be 3..59): {bad}")
    if len(set(layers)) != len(layers):
        sys.exit(f"duplicate layers: {layers}")
    return sorted(layers)


# ------------------------------------------------------------- bank extraction


def cmd_extract_bank(args) -> None:
    layers = check_layers(args.layers or list(MOE_LAYERS))
    BANK.mkdir(parents=True, exist_ok=True)
    src_index = json.loads((MXFP8_SRC / "model.safetensors.index.json").read_text())
    wm = src_index["weight_map"]

    total_written = 0
    t_all = time.time()
    for layer in layers:
        out = BANK / bank_name(layer)
        if out.exists() and not args.force:
            print(f"  layer {layer:2d}: exists, skipping ({out.stat().st_size / 2**30:.2f} GiB)")
            continue
        want = expert_tensor_names(layer, mxfp8=True)
        missing = [t for t in want if t not in wm]
        if missing:
            sys.exit(
                f"layer {layer}: {len(missing)} expert tensors absent from "
                f"{MXFP8_SRC} (e.g. {missing[0]}) — is this really uniform MXFP8?"
            )
        plan = []
        for name in sorted(want):
            src = MXFP8_SRC / wm[name]
            hdr, start = header_cached(src)
            plan.append((name, hdr[name], src, start))
        srcs = sorted({p[2].name for p in plan})
        t0 = time.time()
        nbytes = write_safetensors(out, plan)
        os.chmod(out, 0o444)  # variants share this inode; never edit in place
        dt = time.time() - t0
        total_written += nbytes
        print(
            f"  layer {layer:2d}: {len(plan)} tensors, {nbytes / 2**30:.2f} GiB "
            f"in {dt:5.1f}s from {len(srcs)} shard(s) -> {out.name}"
        )
    print(
        f"bank: wrote {total_written / 2**30:.1f} GiB in {time.time() - t_all:.0f}s "
        f"({BANK})"
    )


def cmd_verify_bank(args) -> None:
    """Prove the extractor against a known-good artifact: for any layer in
    alt3x6's skip set, the bank's tensors must be byte-identical to the
    corresponding shard in alt3x6, which was built by the vendor tooling.
    """
    layers = check_layers(args.layers or [3, 4, 5])
    for layer in layers:
        if layer not in ALT3X6_LAYERS:
            print(f"  layer {layer}: not in alt3x6's skip set, cannot cross-check")
            continue
        bank_file = BANK / bank_name(layer)
        if not bank_file.exists():
            print(f"  layer {layer}: {bank_file.name} not extracted yet — skipping")
            continue
        ref = ALT3X6 / shard_name(layer + SHARD_OFFSET)
        bh, bstart = read_header(bank_file)
        rh, rstart = read_header(ref)
        bn = {k for k in bh if k != "__metadata__"}
        rn = {k for k in rh if k != "__metadata__"}
        if bn != rn:
            sys.exit(f"layer {layer}: tensor-name mismatch vs {ref.name} "
                     f"(+{len(bn - rn)} / -{len(rn - bn)})")
        bad = 0
        with open(bank_file, "rb") as bf, open(ref, "rb") as rf:
            for name in sorted(bn):
                be, re_ = bh[name], rh[name]
                if be["dtype"] != re_["dtype"] or be["shape"] != re_["shape"]:
                    sys.exit(f"layer {layer}: {name} dtype/shape differs")
                bs, bend = be["data_offsets"]
                rs, rend = re_["data_offsets"]
                if bend - bs != rend - rs:
                    sys.exit(f"layer {layer}: {name} byte length differs")
                bf.seek(bstart + bs)
                rf.seek(rstart + rs)
                remaining = bend - bs
                while remaining:
                    n = min(CHUNK, remaining)
                    if bf.read(n) != rf.read(n):
                        bad += 1
                        break
                    remaining -= n
        if bad:
            sys.exit(f"layer {layer}: {bad} tensors differ in payload from {ref.name}")
        print(f"  layer {layer:2d}: {len(bn)} tensors byte-identical to alt3x6 {ref.name}  OK")


# ---------------------------------------------------------------- build a variant


def cmd_build(args) -> None:
    if args.arm:
        if args.arm not in ARMS:
            sys.exit(f"unknown arm {args.arm!r}; known: {', '.join(ARMS)}")
        layers = ARMS[args.arm]
        out = Path(args.out) if args.out else OUT_ROOT / f"MiniMax-M3-NVFP4-{args.arm}"
    else:
        if not args.layers or not args.out:
            sys.exit("need --arm, or both --layers and --out")
        layers = args.layers
        out = Path(args.out)
    build_variant(check_layers(layers), out, dry_run=args.dry_run, force=args.force)


def build_variant(mxfp8_layers: list[int], out: Path, dry_run=False, force=False) -> None:
    k = len(mxfp8_layers)
    print(f"\n{out.name}: K={k} MXFP8 layers {mxfp8_layers}")

    missing = [l for l in mxfp8_layers if not (BANK / bank_name(l)).exists()]
    if missing:
        msg = (f"expert bank missing layers {missing} — run `extract-bank "
               f"--layers {','.join(map(str, missing))}` first")
        if not dry_run:
            sys.exit(msg)
        print(f"  WARNING: {msg}")
    if out.exists() and not force and not dry_run:
        sys.exit(f"{out} exists; pass --force to rebuild")

    donor_index = json.loads((DONOR / "model.safetensors.index.json").read_text())
    donor_wm = donor_index["weight_map"]
    mxfp8_set = set(mxfp8_layers)
    swapped_shards = {shard_name(l + SHARD_OFFSET) for l in mxfp8_layers}

    # --- links: one entry per shard, exactly one source each
    links: list[tuple[str, Path]] = []
    for n in range(1, NUM_SHARDS + 1):
        name = shard_name(n)
        layer = n - SHARD_OFFSET
        if layer in mxfp8_set:
            links.append((name, BANK / bank_name(layer)))
        else:
            links.append((name, DONOR / name))
    assert len(links) == NUM_SHARDS

    # --- index: donor entries minus the swapped layers' NVFP4 expert tensors,
    #     plus the bank files' actual MXFP8 tensor names.
    weight_map: dict[str, str] = {}
    for tensor, fname in donor_wm.items():
        if fname in swapped_shards:
            continue  # that shard is replaced wholesale; its tensors are re-listed below
        weight_map[tensor] = fname
    for layer in mxfp8_layers:
        fname = shard_name(layer + SHARD_OFFSET)
        bank_file = BANK / bank_name(layer)
        if bank_file.exists():
            bh, _ = read_header(bank_file)
            tensors = [t for t in bh if t != "__metadata__"]
        else:  # dry run against a not-yet-extracted bank
            tensors = expert_tensor_names(layer, mxfp8=True)
        for tensor in tensors:
            if tensor in weight_map:
                sys.exit(f"duplicate tensor {tensor} — refusing to build")
            weight_map[tensor] = fname

    # sanity: expected per-layer tensor counts
    for layer in MOE_LAYERS:
        want = expert_tensor_names(layer, mxfp8=layer in mxfp8_set)
        absent = [t for t in want if t not in weight_map]
        if absent:
            sys.exit(
                f"layer {layer}: {len(absent)}/{len(want)} expected "
                f"{'MXFP8' if layer in mxfp8_set else 'NVFP4'} expert tensors missing "
                f"(e.g. {absent[0]})"
            )

    excl = exclude_modules(mxfp8_layers)
    expect = 596 + 386 * k
    if len(excl) != expect:
        sys.exit(f"exclude_modules has {len(excl)} entries, expected {expect}")

    print(f"  links: {len(links)} shards ({len(swapped_shards)} from bank, "
          f"{len(links) - len(swapped_shards)} from donor)")
    print(f"  index: {len(weight_map)} tensors | exclude_modules: {len(excl)} entries")
    if dry_run:
        print("  (dry run — nothing written)")
        return

    out.mkdir(parents=True, exist_ok=True)
    for name, target in links:
        link = out / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target.resolve())  # absolute: /data and /scratch are
        # bind-mounted at identical paths in the serving container

    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": donor_index.get("metadata", {"format": "pt"}),
                    "weight_map": weight_map}, indent=1)
    )

    cfg = json.loads((DONOR / "config.json").read_text())
    cfg["quantization_config"]["exclude_modules"] = excl
    cfg["quantization_config"]["ignore"] = excl  # alt3x6 mirrors it; match the
    # known-serving skip-family checkpoint
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    hqc = json.loads((DONOR / "hf_quant_config.json").read_text())
    hqc["quantization"]["exclude_modules"] = excl
    (out / "hf_quant_config.json").write_text(json.dumps(hqc, indent=2))

    n_aux = 0
    for src in sorted(DONOR.iterdir()):
        if not src.is_file() or src.name in AUX_SKIP_NAMES:
            continue
        if src.name.endswith(AUX_SKIP_SUFFIX):
            continue
        shutil.copy2(src, out / src.name)
        n_aux += 1
    print(f"  wrote 3 JSON files + {n_aux} aux files -> {out}")


def cmd_build_all(args) -> None:
    for arm, layers in ARMS.items():
        build_variant(check_layers(layers), OUT_ROOT / f"MiniMax-M3-NVFP4-{arm}",
                      dry_run=args.dry_run, force=args.force)


# ------------------------------------------------------------------- validation


def cmd_validate(args) -> None:
    d = Path(args.dir)
    fails: list[str] = []

    def check(ok: bool, label: str, detail: str = "") -> None:
        print(f"  [{'ok ' if ok else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
        if not ok:
            fails.append(label)

    print(f"\nvalidating {d}")
    index = json.loads((d / "model.safetensors.index.json").read_text())
    wm = index["weight_map"]

    # 1. index <-> files, both directions, following symlinks
    listed = set(wm.values())
    on_disk = {p.name for p in d.glob("*.safetensors")}
    dangling = sorted(n for n in listed if not (d / n).exists())
    check(not dangling, "every index file resolves",
          f"{len(dangling)} dangling: {dangling[:3]}" if dangling else f"{len(listed)} files")
    check(listed == on_disk, "index file set == *.safetensors on disk",
          f"only-on-disk={sorted(on_disk - listed)[:3]} only-in-index={sorted(listed - on_disk)[:3]}")

    # 2. no tensor in two files + every header tensor is indexed
    hdr_tensors: dict[str, str] = {}
    dupes: list[str] = []
    for name in sorted(on_disk):
        if not (d / name).exists():
            continue
        h, _ = read_header(d / name)
        for t in h:
            if t == "__metadata__":
                continue
            if t in hdr_tensors:
                dupes.append(t)
            hdr_tensors[t] = name
    check(not dupes, "no tensor name appears in two files",
          f"{len(dupes)} duplicated: {dupes[:3]}" if dupes else f"{len(hdr_tensors)} tensors")
    mismatch = [t for t, f in wm.items() if hdr_tensors.get(t) != f]
    check(not mismatch, "index tensor->file agrees with the shard headers",
          f"{len(mismatch)} wrong: {mismatch[:2]}" if mismatch else "")
    unlisted = sorted(set(hdr_tensors) - set(wm))
    check(not unlisted, "no tensor present on disk but absent from the index",
          f"{len(unlisted)}: {unlisted[:3]}" if unlisted else "")

    # 3. dtype census: every MoE layer is wholly NVFP4 or wholly MXFP8
    fmt_of: dict[int, set[str]] = {l: set() for l in MOE_LAYERS}
    for t in hdr_tensors:
        if ".block_sparse_moe.experts." not in t or f"{LP}." not in t:
            continue
        try:
            layer = int(t.split(f"{LP}.")[1].split(".")[0])
        except (IndexError, ValueError):
            continue
        if layer not in fmt_of:
            continue
        if t.endswith(".weight_scale_inv"):
            fmt_of[layer].add("MXFP8")
        elif t.endswith(".weight_scale_2"):
            fmt_of[layer].add("NVFP4")
    mixed = sorted(l for l, s in fmt_of.items() if len(s) > 1)
    check(not mixed, "no MoE layer mixes both formats", f"mixed: {mixed}" if mixed else "")
    mxfp8 = sorted(l for l, s in fmt_of.items() if s == {"MXFP8"})
    nvfp4 = sorted(l for l, s in fmt_of.items() if s == {"NVFP4"})
    check(len(mxfp8) + len(nvfp4) == len(list(MOE_LAYERS)),
          "all 57 MoE layers classified", f"MXFP8={len(mxfp8)} NVFP4={len(nvfp4)}")
    print(f"         MXFP8 layers (K={len(mxfp8)}): {mxfp8}")

    # every layer has its full complement of tensors
    short = []
    for layer in MOE_LAYERS:
        want = expert_tensor_names(layer, mxfp8=layer in set(mxfp8))
        n_absent = sum(1 for t in want if t not in hdr_tensors)
        if n_absent:
            short.append((layer, n_absent, len(want)))
    check(not short, "every MoE layer has its complete expert tensor set",
          f"short: {short[:3]}" if short else "")

    # 4. exclude_modules agrees with the tensors, in both files, with the exact length
    for fname, path_key in (("config.json", ("quantization_config",)),
                            ("hf_quant_config.json", ("quantization",))):
        blob = json.loads((d / fname).read_text())
        for key in path_key:
            blob = blob[key]
        excl = blob["exclude_modules"]
        want = exclude_modules(mxfp8)
        check(excl == want, f"{fname}: exclude_modules matches the built layer set",
              f"{len(excl)} entries vs {len(want)} expected")
    cfg_q = json.loads((d / "config.json").read_text())["quantization_config"]
    check(cfg_q.get("ignore") == cfg_q["exclude_modules"],
          "config.json: ignore mirrors exclude_modules")
    check(cfg_q.get("quant_algo") == "NVFP4" and cfg_q.get("group_size") == 16,
          "config.json: quant_algo NVFP4, group_size 16")

    # 5. family invariant: every surviving NVFP4 expert input_scale is exactly 1.0
    scales: set[float] = set()
    n = 0
    checked_layers = nvfp4 if not args.quick else nvfp4[:3]
    for layer in checked_layers:
        names = [t for t in hdr_tensors if t.startswith(f"{LP}.{layer}.block_sparse_moe.experts.")
                 and t.endswith(".input_scale")]
        if args.quick:
            names = names[:24]
        by_file: dict[str, list[str]] = {}
        for t in names:
            by_file.setdefault(hdr_tensors[t], []).append(t)
        for fname, ts in by_file.items():
            h, start = header_cached(d / fname)
            with open(d / fname, "rb") as fh:
                for t in ts:
                    s, _e = h[t]["data_offsets"]
                    fh.seek(start + s)
                    scales.add(struct.unpack("<f", fh.read(4))[0])
                    n += 1
    check(scales == {1.0}, "NVFP4 expert input_scale == 1.0 (family invariant)",
          f"{n} checked, distinct={sorted(scales)[:4]}")

    # aux files present
    need = ["tokenizer.json", "tokenizer_config.json", "generation_config.json"]
    absent = [f for f in need if not (d / f).exists()]
    check(not absent, "tokenizer / generation aux files present", f"absent: {absent}" if absent else "")

    # cross-check against the arm table when the name matches
    arm = d.name.replace("MiniMax-M3-NVFP4-", "")
    if arm in ARMS:
        check(mxfp8 == sorted(ARMS[arm]), f"layer set matches ARMS[{arm!r}]",
              f"built {mxfp8} vs table {sorted(ARMS[arm])}")

    if fails:
        sys.exit(f"\n{len(fails)} check(s) FAILED: {fails}")
    print("  all checks passed")


# --------------------------------------------------------------------- selftest


def cmd_selftest(args) -> None:
    print("selftest (headers + JSON only, no bulk I/O)")
    fails = []

    def check(ok, label, detail=""):
        print(f"  [{'ok ' if ok else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
        if not ok:
            fails.append(label)

    # 1. exclude_modules generation reproduces alt3x6 byte-for-byte at K=21
    got = exclude_modules(ALT3X6_LAYERS)
    want = json.loads((ALT3X6 / "config.json").read_text())["quantization_config"][
        "exclude_modules"
    ]
    check(got == want, "exclude_modules(alt3x6 layers) == alt3x6's own list",
          f"{len(got)} vs {len(want)} entries")
    check(len(got) == 596 + 386 * 21, "length formula 596 + 386*K", f"{len(got)}")

    # 2. layer-atomic layout + shard formula hold in both donors
    for D, lab in ((DONOR, "donor"), (ALT3X6, "alt3x6")):
        wm = json.loads((D / "model.safetensors.index.json").read_text())["weight_map"]
        per: dict[int, set[str]] = {}
        for t, f in wm.items():
            if ".block_sparse_moe.experts." in t and f"{LP}." in t:
                layer = int(t.split(f"{LP}.")[1].split(".")[0])
                per.setdefault(layer, set()).add(f)
        one = {l: next(iter(fs)) for l, fs in per.items() if len(fs) == 1}
        check(len(one) == len(per) == 57, f"{lab}: each MoE layer's experts in ONE shard",
              f"{len(one)}/{len(per)} layers")
        ok = all(f == shard_name(l + SHARD_OFFSET) for l, f in one.items())
        check(ok, f"{lab}: shard number == layer + {SHARD_OFFSET}")
        # no non-expert tensor shares an expert shard
        expert_files = set(one.values())
        intruders = [t for t, f in wm.items()
                     if f in expert_files and ".block_sparse_moe.experts." not in t]
        check(not intruders, f"{lab}: expert shards hold nothing else",
              f"{len(intruders)} intruders" if intruders else "")

    # 3. the arm table is internally consistent. K comes from the arm's own name
    # (mxfp8skip<K>-<shape>), so a layer set that disagrees with the name it is
    # filed under — the likeliest way to typo this table — is caught here.
    bad = {}
    for arm, ls in ARMS.items():
        m = re.match(r"mxfp8skip(\d+)-", arm)
        want = int(m.group(1)) if m else len(ls)
        if len(ls) != want or len(set(ls)) != len(ls) or any(x not in MOE_LAYERS for x in ls):
            bad[arm] = f"{len(ls)} layers, expected {want}"
    check(not bad, "every ARMS entry is K distinct MoE layers, K per its name",
          f"{bad}" if bad else f"{len(ARMS)} arms: " +
          ", ".join(f"K={k}×{sum(1 for a in ARMS if f'skip{k}-' in a)}"
                    for k in sorted({len(v) for v in ARMS.values()})))

    if fails:
        sys.exit(f"\n{len(fails)} selftest check(s) FAILED: {fails}")
    print("  selftest passed")


# ------------------------------------------------------------------------- cli


def _layers(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("extract-bank", help="write layer-atomic MXFP8 expert shards from 0602")
    p.add_argument("--layers", type=_layers, help="default: all 57 MoE layers")
    p.add_argument("--force", action="store_true", help="re-extract layers already present")
    p.set_defaults(func=cmd_extract_bank)

    p = sub.add_parser("verify-bank", help="byte-compare bank layers against alt3x6")
    p.add_argument("--layers", type=_layers, help="default: 3,4,5")
    p.set_defaults(func=cmd_verify_bank)

    p = sub.add_parser("build", help="assemble one variant dir")
    p.add_argument("--arm", help=f"one of: {', '.join(ARMS)}")
    p.add_argument("--layers", type=_layers)
    p.add_argument("--out")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("build-all", help="assemble every arm in the ARMS table")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_build_all)

    p = sub.add_parser("validate", help="check a built variant dir (do this before serving)")
    p.add_argument("dir")
    p.add_argument("--quick", action="store_true", help="sample input_scale instead of all")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("selftest", help="verify the generator against known-good artifacts")
    p.set_defaults(func=cmd_selftest)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
