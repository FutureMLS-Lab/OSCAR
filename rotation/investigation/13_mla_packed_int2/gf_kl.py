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


GEN_TOKENS = int(os.environ.get("GEN_TOKENS", "128"))
PAD_TOKENS = int(os.environ.get("PAD_TOKENS", "4000"))


def _gen_dump(url: str, out_path: str) -> None:
    """Score GENERATED tokens, which is the only way to reach a decode kernel.

    max_new_tokens=0 scores the input, and the input is prefill -- so the first
    version of this driver compared two decode kernels without ever running
    one, and reported max |dlogprob| = 0.0000 across 566 positions. The exact
    zero was the tell: two different kernels cannot agree to the last bit by
    accident, and they had not been asked to.

    temperature=0 is deliberate here even though greedy decoding is a trap for
    judging INT2 output *quality*: both arms carry identical quantization and
    identical rotations, so the only free variable is the kernel, and greedy
    removes sampling noise so a token-sequence divergence is attributable. Low
    quality output is fine; agreement is what is under test.

    The prompt is padded so decode reads a substantial cache -- an unpadded
    20-token prompt exercises the kernel on almost no keys, which is where it
    would be least likely to differ.
    """
    pad = ("The following is background material that should be ignored. " * 80)
    rows = []
    for i, text in enumerate(klc.TEXTS):
        prompt = (pad * max(1, PAD_TOKENS // 800))[:PAD_TOKENS * 4] + "\n\n" + text
        body = json.dumps({
            "text": prompt,
            "sampling_params": {"max_new_tokens": GEN_TOKENS, "temperature": 0.0},
            "return_logprob": True,
        }).encode()
        try:
            r = json.loads(urllib.request.urlopen(urllib.request.Request(
                f"{url}/generate", data=body,
                headers={"Content-Type": "application/json"}), timeout=900).read())
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}] request failed: {type(e).__name__}: {e}", flush=True)
            rows.append({"i": i, "error": str(e)[:200]})
            continue
        one = r[0] if isinstance(r, list) else r
        meta = one.get("meta_info", {})
        rows.append({
            "i": i,
            "text": text,
            # kl_compare reads this key; feeding it the OUTPUT logprobs lets the
            # same comparison code report on the decode path.
            "input_token_logprobs": meta.get("output_token_logprobs"),
            "input_top_logprobs": meta.get("output_top_logprobs"),
            "gen": one.get("text", "")[:200],
        })
        n = len(meta.get("output_token_logprobs") or [])
        print(f"  [{i}] {n} generated positions", flush=True)
    with open(out_path, "w") as f:
        json.dump(rows, f)
    print(f"wrote {out_path}")


def _kl_drive(p, log_path: str) -> dict:
    """Replaces serve_probe.drive: score the shared texts instead of sampling."""
    err = _wait_healthy(p, log_path)
    if err:
        return {"error": err}
    tag = _kl_drive.tag
    t0 = time.time()
    pre_path = os.path.join(OUT, f"lp.pre.{tag}.json")
    gen_path = os.path.join(OUT, f"lp.gen.{tag}.json")
    # Prefill agreement is kept as a cheap sanity check, clearly labelled as
    # NOT a test of the decode kernel.
    klc.dump(f"http://127.0.0.1:{sp.PORT}", pre_path, TOP_K)
    _gen_dump(f"http://127.0.0.1:{sp.PORT}", gen_path)
    return {"wall_s": round(time.time() - t0, 1),
            "dump": pre_path, "gen_dump": gen_path}


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

    # Report the decode comparison FIRST and label the prefill one for what it
    # is, so a clean prefill number cannot be mistaken for kernel validation.
    ga, gb = infos["prod"].get("gen_dump"), infos["gf"].get("gen_dump")
    if not (ga and gb):
        print("[gf_kl] one arm produced no generated-token dump; the decode "
              "kernel was NOT tested")
        return 2
    print("\n[gf_kl] ===== DECODE path: production kernel vs group-factored ====="
          "\n[gf_kl] this is the comparison that exercises the kernel under test",
          flush=True)
    rc = klc.cmp(ga, gb, "prod-decode", "gf-decode")

    a, b = infos["prod"].get("dump"), infos["gf"].get("dump")
    if a and b:
        print("\n[gf_kl] ===== PREFILL path (sanity only) ====="
              "\n[gf_kl] both arms share the prefill kernel, so agreement here "
              "says nothing about the decode kernel", flush=True)
        klc.cmp(a, b, "prod-prefill", "gf-prefill")

    for tag in ("prod", "gf"):
        log = infos[tag].get("log")
        if log and os.path.exists(log):
            n = sum(1 for ln in open(log, errors="ignore") if "GF-ENTRY" in ln)
            print(f"[gf_kl] {tag}: GF-ENTRY lines = {n}")
            if tag == "gf" and n <= 1:
                print("[gf_kl] !! the group-factored launcher was entered at "
                      "most once, which is the health-check warmup -- the "
                      "comparison above did NOT run the kernel under test")
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "samples"}
                   for k, v in infos.items()}, f, indent=2)
    return rc


if __name__ == "__main__":
    sys.exit(main())
