#!/usr/bin/env python3
"""Compare Q/P-as-A with the transposed KV-as-A decode orientation."""

from __future__ import annotations

import json
import statistics

import torch

try:
    from sgl_kernel import fp8_blockwise_scaled_mm
    from sglang.srt.layers.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
        sglang_per_token_group_quant_fp8_row_padded,
    )
except Exception:
    fp8_blockwise_scaled_mm = None


def measure(fn, *, warmup: int = 20, repeats: int = 100) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)) * 1000.0)
    return {
        "median_us": statistics.median(values),
        "p20_us": sorted(values)[round(0.2 * (len(values) - 1))],
        "p80_us": sorted(values)[round(0.8 * (len(values) - 1))],
    }


def scaled_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    scale_a = torch.ones(1, device="cuda", dtype=torch.float32)
    scale_b = torch.ones(1, device="cuda", dtype=torch.float32)
    return torch._scaled_mm(
        a,
        b,
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=torch.bfloat16,
        use_fast_accum=True,
    )


def make_blockscaled_mm(
    a: torch.Tensor, b: torch.Tensor
):
    if fp8_blockwise_scaled_mm is None:
        raise RuntimeError("sgl_kernel fp8_blockwise_scaled_mm is unavailable")
    qa, sa = sglang_per_token_group_quant_fp8_row_padded(a.contiguous(), 32)
    qb_rows, sb_rows = sglang_per_token_group_quant_fp8(
        b.t().contiguous(), 32
    )
    qb = qb_rows.t()
    sb = sb_rows.t()
    return lambda: fp8_blockwise_scaled_mm(
        qa, qb, sa, sb, out_dtype=torch.bfloat16
    )


def run_case(
    name: str,
    a_shape: tuple[int, int],
    b_shape: tuple[int, int],
    logical_flops: int,
) -> dict:
    a_bf16 = torch.randn(a_shape, device="cuda", dtype=torch.bfloat16)
    # cuBLASLt FP8 requires B to be column-major.
    b_bf16 = torch.randn(
        (b_shape[1], b_shape[0]), device="cuda", dtype=torch.bfloat16
    ).t()
    physical_flops = 2 * a_shape[0] * a_shape[1] * b_shape[1]
    result = {
        "name": name,
        "a_shape": a_shape,
        "b_shape": b_shape,
        "logical_flops": logical_flops,
        "physical_flops": physical_flops,
        "logical_utilization": logical_flops / physical_flops,
    }
    bf16 = measure(lambda: torch.mm(a_bf16, b_bf16))
    bf16["logical_tflops"] = logical_flops / (bf16["median_us"] * 1e6)
    bf16["physical_tflops"] = physical_flops / (bf16["median_us"] * 1e6)
    result["bf16"] = bf16

    try:
        a_fp8 = a_bf16.to(torch.float8_e4m3fn)
        b_fp8 = b_bf16.to(torch.float8_e4m3fn)
        fp8 = measure(lambda: scaled_mm(a_fp8, b_fp8))
        fp8["logical_tflops"] = logical_flops / (fp8["median_us"] * 1e6)
        fp8["physical_tflops"] = physical_flops / (fp8["median_us"] * 1e6)
        result["fp8_e4m3"] = fp8
    except Exception as error:
        result["fp8_error"] = repr(error)
    try:
        blockscaled_fn = make_blockscaled_mm(a_bf16, b_bf16)
        blockscaled = measure(blockscaled_fn)
        blockscaled["logical_tflops"] = logical_flops / (
            blockscaled["median_us"] * 1e6
        )
        blockscaled["physical_tflops"] = physical_flops / (
            blockscaled["median_us"] * 1e6
        )
        result["blockscaled_fp8"] = blockscaled
    except Exception as error:
        result["blockscaled_fp8_error"] = repr(error)
    return result


def main() -> None:
    torch.manual_seed(20260807)
    props = torch.cuda.get_device_properties(0)
    output = {
        "gpu": props.name,
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "cases": [],
    }
    rows = 4  # Qwen3-4B GQA ratio.
    head_dim = 128
    for context in (8192, 32768):
        qk_flops = 2 * rows * context * head_dim
        output["cases"].extend(
            [
                run_case(
                    f"qk_q_as_a_{context}",
                    (rows, head_dim),
                    (head_dim, context),
                    qk_flops,
                ),
                run_case(
                    f"qk_kv_as_a_{context}",
                    (context, head_dim),
                    (head_dim, rows),
                    qk_flops,
                ),
                run_case(
                    f"pv_p_as_a_{context}",
                    (rows, context),
                    (context, head_dim),
                    qk_flops,
                ),
                run_case(
                    f"pv_v_as_a_{context}",
                    (head_dim, context),
                    (context, rows),
                    qk_flops,
                ),
                # Explicitly materialize the tcgen05-style padding. MXFP8 uses
                # M=128; B300 cuBLASLt requires N to be a multiple of 16.
                run_case(
                    f"qk_q_as_a_padded_m128_{context}",
                    (128, head_dim),
                    (head_dim, context),
                    qk_flops,
                ),
                run_case(
                    f"qk_kv_as_a_padded_n16_{context}",
                    (context, head_dim),
                    (head_dim, 16),
                    qk_flops,
                ),
                run_case(
                    f"pv_p_as_a_padded_m128_{context}",
                    (128, context),
                    (context, head_dim),
                    qk_flops,
                ),
                run_case(
                    f"pv_v_as_a_padded_n16_{context}",
                    (head_dim, context),
                    (context, 16),
                    qk_flops,
                ),
            ]
        )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
