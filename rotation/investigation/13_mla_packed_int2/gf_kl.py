#!/usr/bin/env python3
"""Validate the group-factored packed-MLA decode kernel at the distribution level.

serve_probe already checks that the group-factored arm produces readable text,
comparing an ascii ratio between arms. That is a garbling detector, not a
correctness test: a kernel can be subtly wrong -- a mis-scaled tail, a dropped
partial, an off-by-one in the split merge -- and still emit fluent prose. The
kernel-level gate covers exactness on synthetic operands; what is missing is
evidence on the real serving path, and that is the thing standing between the
group-factored kernel and being on by default.

Teacher forcing supplies it. The same texts are scored under both kernels with
max_new_tokens=0, so every token position is a paired sample and a few dozen
prompts give thousands of them. Two kernels computing the same attention must
agree to within reduction-order noise; anything larger is a real difference in
what the model computes, visible long before it would show up as a score drop.

Both arms are launched through serve_probe.launch, so the server arguments are
constructed by the same code and cannot drift between them -- the only
difference is SGLANG_OSCAR_MLA_PACKED_GF.

Reported per arm: the dump path. At the end: mean/p99/max |dlogprob| and
symmetric KL, via kl_compare, plus a verdict. The prefix cache is disabled for
both arms because input logprobs from position 0 force a full recompute and a
cached prefix silently returns fewer positions.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import serve_probe as sp  # noqa: E402
import kl_compare as klc  # noqa: E402

OUT = os.environ.get("OUT_DIR", "/shared/mlapacked_gf_kl")
TOP_K = int(os.environ.get("TOP_K", "8"))


def _wait_healthy(p, log_path: str, timeout: float = 1800.0) -> str | None:
    """Same readiness contract as serve_probe.drive, returning an error string."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if p.poll() is not None:
            tail = "".join(open(log_path, errors="ignore").readlines()[-25:])
            print(f"  server exited early rc={p.returncode}; tail:\n{tail}",
                  flush=True)
            return f"server exited rc={p.returncode}"
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{sp.PORT}/health_generate", timeout=5)
            return None
        except Exception:  # noqa: BLE001
            time.sleep(5)
    return "server never became healthy"


def _kl_drive(p, log_path: str) -> dict:
    """Replaces serve_probe.drive: score the shared texts instead of sampling."""
    err = _wait_healthy(p, log_path)
    if err:
        return {"error": err}
    tag = _kl_drive.tag
    out_path = os.path.join(OUT, f"lp.{tag}.json")
    t0 = time.time()
    klc.dump(f"http://127.0.0.1:{sp.PORT}", out_path, TOP_K)
    return {"wall_s": round(time.time() - t0, 1), "dump": out_path}


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    sp.OUT = OUT
    sp.drive = _kl_drive
    # Set before launch, because serve_probe reads it while building the args.
    os.environ["DISABLE_RADIX"] = "1"

    # Teacher forcing needs the prefix cache off in BOTH arms, or the two dumps
    # cover different position counts and kl_compare refuses to compare them.
    common = {"SGLANG_OSCAR_MLA_PACKED_GF_CHECK": "0"}
    arms = [
        ("prod", {"SGLANG_OSCAR_MLA_PACKED_GF": "0"}),
        ("gf", {"SGLANG_OSCAR_MLA_PACKED_GF": "1"}),
    ]
    infos = {}
    for tag, extra in arms:
        _kl_drive.tag = tag
        env = dict(common)
        env.update(extra)
        infos[tag] = sp.launch(tag, packed=True, extra_env=env)
        print(f"[gf_kl] {tag}: {json.dumps({k: v for k, v in infos[tag].items() if k != 'samples'})[:300]}",
              flush=True)

    a, b = infos["prod"].get("dump"), infos["gf"].get("dump")
    if not (a and b):
        print("[gf_kl] one arm produced no dump; nothing to compare")
        return 2
    print("\n[gf_kl] ===== production kernel vs group-factored =====", flush=True)
    rc = klc.cmp(a, b, "prod", "gf")
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "samples"}
                   for k, v in infos.items()}, f, indent=2)
    return rc


if __name__ == "__main__":
    sys.exit(main())
