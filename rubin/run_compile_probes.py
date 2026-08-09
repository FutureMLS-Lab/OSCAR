#!/usr/bin/env python3
"""Compile-only CUDA 13.4 probes for Rubin LUT-B ISA compatibility."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


HEADER = """.version 9.4
.target sm_107a
.address_size 64

"""

REGISTER_SETUP = """    .reg .pred %p;
    .reg .b16 %h<2>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<4>;
    mov.u32 %r0, 0;
    mov.u32 %r1, 0;
    mov.u32 %r2, 0;
    mov.u32 %r3, 0;
    mov.u32 %r4, 0;
    mov.u32 %r5, 0;
    mov.u32 %r6, 0;
    mov.u32 %r7, 0;
    mov.u32 %r8, 0;
    mov.u64 %rd0, 0;
    mov.u64 %rd1, 0;
    mov.u64 %rd2, 0;
    mov.f32 %f0, 1.0;
    mov.f32 %f1, 0.5;
    setp.eq.u32 %p, %r0, %r0;
"""


def entry(name: str, body: str) -> str:
    return (
        HEADER
        + f".visible .entry {name}() {{\n"
        + REGISTER_SETUP
        + body
        + "    ret;\n}\n"
    )


def output_entry(name: str, body: str) -> str:
    return (
        HEADER
        + f".visible .entry {name}(.param .u64 output) {{\n"
        + REGISTER_SETUP
        + "    ld.param.u64 %rd7, [output];\n"
        + body
        + "    st.global.u16 [%rd7], %h0;\n"
        + "    ret;\n}\n"
    )


PROBES = {
    "sm107a_target": {
        "expect": "accept",
        "ptx": entry("sm107a_target", ""),
        "claim": "PTX 9.4 and sm_107a target are accepted",
    },
    "ue5m3_cvt": {
        "expect": "accept",
        "ptx": output_entry(
            "ue5m3_cvt",
            "    cvt.rn.satfinite.ue5m3x2.f32 %h0, %f0, %f1;\n",
        ),
        "claim": "Rubin UE5M3 conversion is accepted",
    },
    "lutb_e4m3_dense": {
        "expect": "accept",
        "ptx": entry(
            "lutb_e4m3_dense",
            (
                "    tcgen05.mma.cta_group::1.kind::f8f6f4"
                ".decompress::lut::b\n"
                "        [%r0], %rd0, %rd1, [%r8], %r5, "
                "{%r1, %r2, %r3, %r4}, %p;\n"
            ),
        ),
        "claim": "Dense E4M3 A x LUT-B E4M3 B is accepted",
    },
    "lutb_mxf8_ue8m0_block32": {
        "expect": "accept",
        "ptx": entry(
            "lutb_mxf8_ue8m0_block32",
            (
                "    tcgen05.mma.cta_group::1.kind::mxf8f6f4"
                ".block_scale.decompress::lut::b.block32\n"
                "        [%r0], %rd0, %rd1, [%r8], %r5, "
                "[%r6], [%r7], %p;\n"
            ),
        ),
        "claim": (
            "Block-32 MXFP8 A x LUT-B B is accepted; PTX Table 69 "
            "defines its scale type as UE8M0"
        ),
    },
    "lutb_collector_b_fill": {
        "expect": "accept",
        "ptx": entry(
            "lutb_collector_b_fill",
            (
                "    tcgen05.mma.cta_group::1.kind::f8f6f4"
                ".decompress::lut::b.collector::b::fill\n"
                "        [%r0], %rd0, %rd1, [%r8], %r5, "
                "{%r1, %r2, %r3, %r4}, %p;\n"
            ),
        ),
        "claim": "LUT-B collector::b::fill is accepted",
    },
    "lutb_collector_b_reuse": {
        "expect": "accept",
        "ptx": entry(
            "lutb_collector_b_reuse",
            (
                "    tcgen05.mma.cta_group::1.kind::f8f6f4"
                ".decompress::lut::b.collector::b::fill\n"
                "        [%r0], %rd0, %rd1, [%r8], %r5, "
                "{%r1, %r2, %r3, %r4}, %p;\n"
                "    tcgen05.mma.cta_group::1.kind::f8f6f4"
                ".decompress::lut::b.collector::b::lastuse\n"
                "        [%r0], %rd0, %rd1, [%r8], %r5, "
                "{%r1, %r2, %r3, %r4}, %p;\n"
            ),
        ),
        "claim": "LUT-B collector B fill-to-lastuse reuse sequence is accepted",
    },
    "sparse_a_dense_b": {
        "expect": "accept",
        "ptx": entry(
            "sparse_a_dense_b",
            (
                "    tcgen05.mma.sp.cta_group::1.kind::f8f6f4\n"
                "        [%r0], %rd0, %rd1, [%r6], %r5, "
                "{%r1, %r2, %r3, %r4}, %p;\n"
            ),
        ),
        "claim": "Sparse-A tcgen05.mma.sp with ordinary dense B is accepted",
    },
    "sparse_a_lutb_b": {
        "expect": "reject",
        "ptx": entry(
            "sparse_a_lutb_b",
            (
                "    tcgen05.mma.sp.cta_group::1.kind::f8f6f4"
                ".decompress::lut::b\n"
                "        [%r0], %rd0, %rd1, [%r6], [%r8], %r5, "
                "{%r1, %r2, %r3, %r4}, %p;\n"
            ),
        ),
        "claim": (
            "PTX 9.4 does not define sparse-A tcgen05.mma.sp with LUT-B"
        ),
    },
}


SPEC_ONLY = {
    "lutb_mxf8_ue5m3_block32": {
        "status": "REJECT",
        "evidence": (
            "PTX 9.4 Table 69 allows only UE8M0 scales for "
            "kind::mxf8f6f4; UE5M3 is only listed for "
            "kind::mxf4nvf4"
        ),
    },
    "lutb_transpose_b": {
        "status": "REJECT",
        "evidence": (
            "PTX 9.4 states that matrix B does not support transpose "
            "when decompress::lut::b is specified"
        ),
    },
}


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", type=Path, default=Path("/usr/local/cuda-13.4"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ptxas = args.cuda / "bin" / "ptxas"
    nvdisasm = args.cuda / "bin" / "nvdisasm"
    cuobjdump = args.cuda / "bin" / "cuobjdump"
    args.output.mkdir(parents=True, exist_ok=True)

    version = run([str(ptxas), "--version"]).stdout.strip()
    results: dict[str, dict] = {}
    for name, probe in PROBES.items():
        ptx_path = args.output / f"{name}.ptx"
        cubin_path = args.output / f"{name}.cubin"
        ptx_path.write_text(probe["ptx"])
        compiled = run(
            [
                str(ptxas),
                "-arch=sm_107a",
                "-v",
                str(ptx_path),
                "-o",
                str(cubin_path),
            ]
        )
        accepted = compiled.returncode == 0
        expected = probe["expect"]
        passed = accepted if expected == "accept" else not accepted
        compatibility = (
            "SUPPORTED"
            if expected == "accept" and accepted
            else "UNSUPPORTED"
            if expected == "reject" and not accepted
            else "PROBE_FAILURE"
        )
        result = {
            "status": "PASS" if passed else "FAIL",
            "compatibility": compatibility,
            "expected": expected,
            "compiler_accepted": accepted,
            "claim": probe["claim"],
            "ptxas_output": compiled.stdout,
        }
        if accepted:
            sass = run([str(nvdisasm), str(cubin_path)])
            sass_path = args.output / f"{name}.sass"
            sass_path.write_text(sass.stdout)
            elf = run([str(cuobjdump), "--dump-elf", str(cubin_path)])
            (args.output / f"{name}.elf.txt").write_text(elf.stdout)
            result["cubin_sha256"] = hashlib.sha256(
                cubin_path.read_bytes()
            ).hexdigest()
            result["nvdisasm_returncode"] = sass.returncode
        results[name] = result

    for name, evidence in SPEC_ONLY.items():
        results[name] = {
            "status": evidence["status"],
            "compatibility": "UNSUPPORTED",
            "expected": "spec_only",
            "compiler_accepted": None,
            "claim": evidence["evidence"],
        }

    summary = {
        "toolchain": version,
        "target": "sm_107a",
        "ptx_isa": "9.4",
        "runtime_executed": False,
        "results": results,
    }
    (args.output / "results.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    lines = [
        "| Probe | Compatibility | Probe | Evidence |",
        "|---|---:|---:|---|",
    ]
    for name, result in results.items():
        lines.append(
            f"| `{name}` | {result['compatibility']} "
            f"| {result['status']} | {result['claim']} |"
        )
    (args.output / "results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))
    if any(
        result["status"] == "FAIL"
        for result in results.values()
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
