#!/usr/bin/env python3
"""Does the packed-INT2 MLA latent kernel survive CUDA-graph capture/replay?

Why this needs its own probe rather than "read the code and reason".
``quantize_dequantize_reuse`` returns *views into a module-level scratch dict*
(``mla_latent_int2._SCRATCH``) that the next call overwrites, and
``mla_int2_kv_pool._apply_fake_int2_c_kv`` ``.clone()``s the view immediately.
Three separate hazards live in that shape, and only the first is settled by
reading:

1. serial aliasing -- 78 layers share one ``out`` buffer per forward. The clone
   is what makes that safe. (Control arm ``noclone`` below deliberately removes
   it, so a green result cannot be a test that measures nothing.)
2. capture-time allocation -- the scratch is allocated lazily on first call. If
   that first call happens *inside* a capture region the buffer comes from the
   graph's private memory pool.
3. capacity growth after capture -- a later eager call with more groups than the
   captured buffer holds makes ``_scratch`` allocate a *new* buffer and drop the
   old one, so the captured graph replays against a pointer whose Python owner
   is gone. Whether that is still correct is an allocator question, not a
   source-reading question.

The probe mirrors the real write path: 78 layers, real GLM-5.2 rotations,
rotate -> quantize -> unrotate -> store into a pool row, with the whole 78-layer
loop captured in one graph the way sglang captures a decode forward.

Reference is the torch fake-quant (``_fake_quant_int2_groupwise``), i.e. the
path every published GLM number came from.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch


def load_rotations(d: str, n: int, device, dtype=torch.float32):
    fs = sorted(glob.glob(os.path.join(d, "layer_*.pt")),
                key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
    assert len(fs) >= n, (len(fs), n)
    out = []
    for p in fs[:n]:
        r = torch.load(p, map_location="cpu")
        if not torch.is_tensor(r):
            r = list(r.values())[0]
        out.append(r.to(device=device, dtype=dtype).contiguous())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rot-dir", default="/shared/zz-premerge/glm52-rotations")
    ap.add_argument("--layers", type=int, default=78)
    ap.add_argument("--bs", type=int, default=32, help="decode batch (tokens/layer)")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--lloyd-max", type=int, default=1)
    ap.add_argument("--big-tokens", type=int, default=8192,
                    help="post-capture eager prefill size that forces scratch growth")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    from sglang.QuantKernel.mla_latent_int2 import (
        _SCRATCH,
        quantize_dequantize_reuse,
    )
    from sglang.srt.mem_cache.mla_int2_kv_pool import _fake_quant_int2_groupwise

    dev = torch.device("cuda")
    L, B, D = a.layers, a.bs, 512
    lm = bool(a.lloyd_max)
    R = load_rotations(a.rot_dir, L, dev)
    res: dict = {"layers": L, "bs": B, "group": a.group, "lloyd_max": lm,
                 "torch": torch.__version__,
                 "gpu": torch.cuda.get_device_name(0),
                 "capability": list(torch.cuda.get_device_capability(0))}

    torch.manual_seed(0)
    # Distinct per-layer data: an aliasing bug then shows up as "every layer
    # equals the last layer", which identical data would hide completely.
    x = [(torch.randn(B, D, device=dev, dtype=torch.bfloat16) * (1 + 0.1 * i))
         for i in range(L)]

    def ref_pool():
        out = torch.zeros(L, B, D, device=dev, dtype=torch.bfloat16)
        for i in range(L):
            c = x[i].to(torch.float32) @ R[i]
            q = _fake_quant_int2_groupwise(c, a.group, lm).to(torch.bfloat16)
            out[i] = (q.to(torch.float32) @ R[i].T).to(torch.bfloat16)
        return out

    def kernel_pool(src, pool, clone=True):
        """Same algebra through the packed kernel. Writes into `pool`."""
        for i in range(L):
            c = src[i].to(torch.float32) @ R[i]
            _codes, _params, q = quantize_dequantize_reuse(c, a.group, lm)
            if clone:
                q = q.clone()
            q = q.to(torch.bfloat16)
            pool[i] = (q.to(torch.float32) @ R[i].T).to(torch.bfloat16)

    def cmp(got, want, tag):  # noqa: D401
        d = (got.to(torch.float32) - want.to(torch.float32))
        rel = (d.norm() / want.to(torch.float32).norm()).item()
        per = [(d[i].abs().max().item()) for i in range(L)]
        # aliasing signature: every layer equal to the last layer written
        alias = sum(
            1 for i in range(L - 1)
            if torch.equal(got[i], got[L - 1])
        )
        res[tag] = {"relL2": rel, "max_abs": max(per),
                    "layers_equal_to_last": alias,
                    "layers_mismatched": sum(1 for i in range(L)
                                             if per[i] > 1e-3)}
        print(f"[probe] {tag}: {json.dumps(res[tag])}", flush=True)
        return rel

    ref = ref_pool()

    # ---- arm 1: eager kernel, no graph -------------------------------------
    p1 = torch.zeros_like(ref)
    kernel_pool(x, p1)
    torch.cuda.synchronize()
    cmp(p1, ref, "eager_kernel")

    # ---- arm 2: negative control, eager kernel WITHOUT the clone ----------
    p2 = torch.zeros_like(ref)
    kernel_pool(x, p2, clone=False)
    torch.cuda.synchronize()
    cmp(p2, ref, "eager_kernel_noclone")

    # ---- arm 3: CUDA graph capture + replay -------------------------------
    # Static inputs, exactly like sglang's captured decode buffers.
    xs = [t.clone() for t in x]
    p3 = torch.zeros_like(ref)
    # Warm up (JIT the Triton kernel and allocate scratch OUTSIDE capture, which
    # is what really happens: prefill runs eager before any graph is replayed).
    kernel_pool(xs, p3)
    torch.cuda.synchronize()
    res["scratch_keys_before_capture"] = [
        [k[0], str(k[1]), k[2], v[0]] for k, v in _SCRATCH.items()
    ]

    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        kernel_pool(xs, p3)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    try:
        with torch.cuda.graph(g):
            kernel_pool(xs, p3)
        res["capture"] = "ok"
    except Exception as e:                       # noqa: BLE001
        res["capture"] = f"EXCEPTION {type(e).__name__}: {e}"
        print(json.dumps(res, indent=2))
        return 1

    # Fresh data into the static inputs, then replay: the pool must match the
    # torch reference for the NEW data, not the captured data.
    torch.manual_seed(1)
    x2 = [(torch.randn(B, D, device=dev, dtype=torch.bfloat16) * (1 + 0.1 * i))
          for i in range(L)]
    for i in range(L):
        xs[i].copy_(x2[i])
    x_save = x
    x = x2
    ref2 = ref_pool()
    x = x_save
    p3.zero_()
    g.replay()
    torch.cuda.synchronize()
    cmp(p3, ref2, "graph_replay")

    # ---- arm 4: replay again AFTER an eager call that grows the scratch ----
    # This is hazard 3: the captured graph's scratch buffer loses its Python
    # owner while the graph still points at it.
    big = torch.randn(a.big_tokens, D, device=dev, dtype=torch.float32) @ R[0]
    _c, _p, _q = quantize_dequantize_reuse(big, a.group, lm)
    del _c, _p, _q, big
    torch.cuda.synchronize()
    res["scratch_keys_after_growth"] = [
        [k[0], str(k[1]), k[2], v[0]] for k, v in _SCRATCH.items()
    ]
    p3.zero_()
    g.replay()
    torch.cuda.synchronize()
    cmp(p3, ref2, "graph_replay_after_scratch_growth")

    # ---- arm 5: many replays interleaved with eager traffic ---------------
    ok = True
    for it in range(8):
        junk = torch.randn(4096, D, device=dev, dtype=torch.float32) @ R[3]
        quantize_dequantize_reuse(junk, a.group, lm)
        p3.zero_()
        g.replay()
        torch.cuda.synchronize()
        r = (p3.to(torch.float32) - ref2.to(torch.float32)).norm().item()
        if r > 1e-3 * ref2.to(torch.float32).norm().item():
            ok = False
            res[f"replay_iter_{it}_relL2"] = r
    res["repeated_replay_stable"] = ok

    verdict = (
        res["capture"] == "ok"
        and res["eager_kernel"]["relL2"] < 1e-3
        and res["graph_replay"]["relL2"] < 1e-3
        and res["graph_replay_after_scratch_growth"]["relL2"] < 1e-3
        and res["repeated_replay_stable"]
    )
    res["VERDICT"] = "PASS" if verdict else "FAIL"
    res["negative_control_fires"] = res["eager_kernel_noclone"]["relL2"] > 1e-3
    print(json.dumps(res, indent=2))
    return 0 if verdict else 2


if __name__ == "__main__":
    sys.exit(main())
