#!/usr/bin/env python3
"""Does the group-factored path turn itself on, and only where it should?

The auto rule is the one code path every previous job hid. gf_speed sets
SGLANG_OSCAR_MLA_PACKED_GF explicitly in BOTH arms -- that is what makes it a
clean single-variable comparison -- so `is_set()` was True every time and the
default branch has literally never executed on a GPU. A rule that is only ever
overridden is not a tested rule.

So launch with the variable UNSET and vary only --context-length across the
8192 threshold, then read GF-ENTRY out of the server log:

    ctx-len 4096   -> expect GF-ENTRY == 0   (production kernel)
    ctx-len 32640  -> expect GF-ENTRY >  0   (group-factored)

GF-ENTRY is the kernel's own entry log, not an intention: the same counter that
caught what would otherwise have been a vacuous A/B in gf_speed (prod 0, gf 8).

A third arm pins the variable to 0 at the long context, because "auto turned it
on" and "it is on regardless" produce identical logs at one context. Without
that arm a rule that ignores the env entirely would pass.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import serve_probe as sp  # noqa: E402

OUT = os.environ.get("OUT_DIR", "/shared/mlapacked_gfauto")


def _drive(p, log_path: str) -> dict:
    deadline = time.time() + float(os.environ.get("HEALTH_TIMEOUT", "3000"))
    while time.time() < deadline:
        if p.poll() is not None:
            tail = "".join(open(log_path, errors="ignore").readlines()[-25:])
            print(f"  server exited rc={p.returncode}; tail:\n{tail}", flush=True)
            return {"error": f"server exited rc={p.returncode}"}
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{sp.PORT}/health_generate", timeout=5)
            break
        except Exception:  # noqa: BLE001
            time.sleep(5)
    else:
        return {"error": "server never became healthy"}

    # One decode long enough to run the kernel a few times. The question is
    # WHICH kernel ran, not how fast, so this is deliberately tiny.
    body = json.dumps({
        "text": "Explain in two sentences why a rotation helps 2-bit quantization.",
        "sampling_params": {"max_new_tokens": 48, "temperature": 0.0},
    }).encode()
    try:
        r = json.loads(urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{sp.PORT}/generate", data=body,
            headers={"Content-Type": "application/json"}), timeout=900).read())
        one = r[0] if isinstance(r, list) else r
        txt = (one.get("text") or "")[:120]
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    return {"sample": txt}


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    sp.OUT = OUT
    sp.drive = _drive

    # (tag, context length, env, what the auto rule should decide)
    arms = [
        ("short", "4096", {}, False),
        ("long", "32640", {}, True),
        ("long_pinned_off", "32640", {"SGLANG_OSCAR_MLA_PACKED_GF": "0"}, False),
    ]
    results = []
    for tag, ctx, extra, expect_gf in arms:
        os.environ["CTX_LEN"] = ctx
        sp.CTX = ctx
        # Belt and braces: a leaked value from a previous arm would silently
        # turn the auto arms into pinned ones, which is the exact failure this
        # script exists to rule out.
        if "SGLANG_OSCAR_MLA_PACKED_GF" not in extra:
            os.environ.pop("SGLANG_OSCAR_MLA_PACKED_GF", None)
        info = sp.launch(tag, packed=True, extra_env=extra)
        log = info.get("log")
        n = (sum(1 for ln in open(log, errors="ignore") if "GF-ENTRY" in ln)
             if log and os.path.exists(log) else 0)
        info.update({"ctx_len": ctx, "gf_entry": n, "expect_gf": expect_gf,
                     "pinned": "SGLANG_OSCAR_MLA_PACKED_GF" in extra})
        results.append(info)
        print(f"[gf_auto] {tag}: ctx-len={ctx} GF-ENTRY={n} "
              f"expect_gf={expect_gf}", flush=True)

    print("\n[gf_auto] ===== auto rule =====")
    print(f"{'arm':>16} {'ctx-len':>8} {'pinned':>7} {'GF-ENTRY':>9} "
          f"{'expect':>7}  verdict")
    ok = True
    for r in results:
        got = r["gf_entry"] > 0
        good = got == r["expect_gf"] and not r.get("error")
        ok &= good
        print(f"{r['tag']:>16} {r['ctx_len']:>8} {str(r['pinned']):>7} "
              f"{r['gf_entry']:>9} {str(r['expect_gf']):>7}  "
              f"{'ok' if good else 'WRONG'}"
              + (f"  ({r['error']})" if r.get("error") else ""))
    print(f"\n[gf_auto] VERDICT: {'PASS' if ok else 'FAIL'}")
    # Distinguishing the two failure shapes is the whole point of arm 3.
    s, l = results[0], results[1]
    if s["gf_entry"] > 0 and l["gf_entry"] > 0:
        print("[gf_auto] gf ran at BOTH contexts -- the threshold is not being "
              "consulted, so the rule is 'always on', not 'auto'")
    if s["gf_entry"] == 0 and l["gf_entry"] == 0:
        print("[gf_auto] gf ran at NEITHER context -- the auto branch is not "
              "reached at all, so the rule is 'always off', not 'auto'")
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
