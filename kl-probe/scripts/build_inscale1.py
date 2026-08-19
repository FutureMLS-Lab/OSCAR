"""Copy an NVFP4 checkpoint with every `*.input_scale` tensor forced to 1.0.

Usage:
    uv run python scripts/build_inscale1.py SRC_DIR DST_DIR

Byte-patches the copies in place instead of loading tensors (no torch needed):
for each safetensors shard, copy it, parse the 8-byte-length + JSON header, and
overwrite the scalar payload of every tensor named `*.input_scale` with 1.0 in
its stored dtype. All other files are copied verbatim (the index stays valid —
tensor layout is unchanged).

Earlier variants (0717-inscale1, mxfp8skip21-inscale1, ...) were built by
one-off scratchpad scripts that were lost to /scratch cleanups; this is the
durable replacement.
"""

import json
import shutil
import struct
import sys
from pathlib import Path

ONE = {
    "F32": struct.pack("<f", 1.0),
    "F64": struct.pack("<d", 1.0),
    "F16": (0x3C00).to_bytes(2, "little"),  # fp16 1.0
    "BF16": (0x3F80).to_bytes(2, "little"),  # bf16 1.0
}


def patch_shard(path: Path) -> int:
    with open(path, "r+b") as f:
        (hdr_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hdr_len))
        data_start = 8 + hdr_len
        n = 0
        for name, meta in header.items():
            if name == "__metadata__" or not name.endswith(".input_scale"):
                continue
            dtype, (begin, end) = meta["dtype"], meta["data_offsets"]
            one = ONE[dtype]  # KeyError = unexpected dtype, fail loudly
            count, rem = divmod(end - begin, len(one))
            assert rem == 0, f"{name}: size {end - begin} not a {dtype} multiple"
            f.seek(data_start + begin)
            f.write(one * count)  # input_scale is scalar, but handle any shape
            n += 1
    return n


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    assert src.is_dir(), src
    dst.mkdir(parents=True, exist_ok=True)
    total = 0
    for p in sorted(src.iterdir()):
        out = dst / p.name
        if p.is_dir():
            if not out.exists():
                shutil.copytree(p, out)
            continue
        # Copy to a temp name, patch, then rename — a finished file is always
        # complete AND patched, so an interrupted build can just be re-run.
        if out.exists():
            print(f"{p.name}: exists, skipping", flush=True)
            continue
        tmp = out.with_name(out.name + ".tmp")
        shutil.copyfile(p, tmp)
        if p.suffix == ".safetensors":
            n = patch_shard(tmp)
            total += n
            print(f"{p.name}: {n} input_scale -> 1.0", flush=True)
        tmp.rename(out)
    print(f"done: {total} input_scale tensors patched into {dst}")


if __name__ == "__main__":
    main()
