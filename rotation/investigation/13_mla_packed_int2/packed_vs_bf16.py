#!/usr/bin/env python3
"""How much does packed 2-bit + OSCAR rotation change the model, on a model that can take it?

Kimi-K3 answered the other half of the question: its latent is not a 2-bit
tensor, and lat2 costs 13.4 GPQA points (23.3 vs 36.7 at four bits, against 37.5
unpacked). That is a property of K3's c_kv, measured three independent ways.

This asks whether the PATH is sound on a model whose latent does fit in two
bits. DeepSeek-V2-Lite is non-hybrid MLA, so it also exercises the layer-id
mapping that a hybrid model cannot: a hybrid inner pool has start_layer 0, so
the offset in `start_layer + _local_layer_index(id)` could be wrong and K3 would
still pass.

Not gf_kl.py, which compares two KERNELS with packing on in both arms and would
answer a question nobody asked here. This toggles `packed` instead: identical
model, identical prompts, identical server arguments, 2-bit packed storage with
the OSCAR rotation against ordinary BF16.

Teacher-forced logprobs rather than a benchmark score, because a 16B model's
GPQA is near chance and would hide a real regression. Every token position is a
paired sample, so a few dozen prompts give thousands.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import kl_compare as klc  # noqa: E402
import serve_probe as sp  # noqa: E402

OUT = os.environ.get("OUT_DIR", "/shared/mlapacked_pvb")
TOP_K = int(os.environ.get("TOP_K", "8"))


def _wait_healthy(p, log_path: str, timeout: float = 1800.0) -> str | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if p.poll() is not None:
            tail = "".join(open(log_path, errors="ignore").readlines()[-25:])
            print(f"  server exited rc={p.returncode}; tail:\n{tail}", flush=True)
            return f"server exited rc={p.returncode}"
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{sp.PORT}/health_generate", timeout=5)
            return None
        except Exception:  # noqa: BLE001
            time.sleep(5)
    return "server never became healthy"


def _drive(p, log_path: str) -> dict:
    err = _wait_healthy(p, log_path)
    if err:
        return {"error": err}
    out = os.path.join(OUT, f"lp.{_drive.tag}.json")
    t0 = time.time()
    klc.dump(f"http://127.0.0.1:{sp.PORT}", out, TOP_K)
    return {"wall_s": round(time.time() - t0, 1), "dump": out}


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    sp.OUT = OUT
    sp.drive = _drive
    # Teacher forcing needs the prefix cache off in both arms, or the dumps
    # cover different position counts and are not comparable.
    os.environ["DISABLE_RADIX"] = "1"

    infos = {}
    for tag, packed in (("bf16", False), ("packed2", True)):
        _drive.tag = tag
        infos[tag] = sp.launch(tag, packed=packed, extra_env={
            "SGLANG_OSCAR_MLA_PACKED_GF": "0",
            "SGLANG_OSCAR_MLA_KV_BITS": "2",
        })
        print(f"[pvb] {tag}: {json.dumps({k: v for k, v in infos[tag].items() if k != 'samples'})[:220]}",
              flush=True)

    a, b = infos["bf16"].get("dump"), infos["packed2"].get("dump")
    if not (a and b):
        print("[pvb] an arm produced no dump")
        return 2
    print("\n[pvb] ===== BF16 unpacked vs packed 2-bit + OSCAR rotation =====",
          flush=True)
    rc = klc.cmp(a, b, "bf16", "packed2")
    for tag in ("bf16", "packed2"):
        log = infos[tag].get("log")
        if log and os.path.exists(log):
            n = sum(1 for ln in open(log, errors="ignore")
                    if "packed latent storage" in ln)
            print(f"[pvb] {tag}: packed-pool lines = {n}"
                  + ("  <-- packing did NOT engage" if tag == "packed2" and not n
                     else ""))
    return rc


if __name__ == "__main__":
    sys.exit(main())
