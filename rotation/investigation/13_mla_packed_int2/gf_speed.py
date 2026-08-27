#!/usr/bin/env python3
"""Measure group-factored vs production packed-MLA decode end to end, by context.

The group-factored kernel is 3.76x the production kernel in stage-1 and 1.12x
end to end at long context. Neither number licenses turning it on by default,
because it runs a SECOND pass over the BF16 window arena. That pass is a fixed
cost per decode step, so its share grows as the packed body shrinks -- exactly
the short-context regime the long-context measurement never touched. A default
that helps at 32k and hurts at 1k is not a default, it is a footgun.

So sweep the context length and report the ratio at each. Decode throughput is
the metric that matters: generation is where the kernel runs, and prefill is
shared between the arms.

Both arms launch through serve_probe.launch, so the server arguments come from
one code path; the only difference is SGLANG_OSCAR_MLA_PACKED_GF. Each context
is measured with the same prompt padded to length, the same generation budget,
and a warmup request that is discarded -- the first request after startup pays
for autotuning and JIT and would otherwise be attributed to the kernel.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import serve_probe as sp  # noqa: E402

OUT = os.environ.get("OUT_DIR", "/shared/mlapacked_gf_speed")
CTXS = [int(x) for x in os.environ.get("CTXS", "1000,4000,16000,32000").split(",")]
GEN = int(os.environ.get("GEN_TOKENS", "128"))
REPS = int(os.environ.get("REPS", "3"))


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


# A LIST, not a scalar. The point of measuring concurrency is that gf's window
# pass is a fixed per-STEP cost while the packed body scales with the batch, so
# the ratio should improve with concurrency -- and one number cannot show a
# trend. Sweeping inside one server launch also matters practically: a 355B FP8
# model takes ~20 minutes to come up, so a scalar CONC would mean relaunching
# per point and turning a 2-hour job into a 6-hour one.
CONCS = [int(x) for x in os.environ.get("CONCS", os.environ.get("CONC", "1")).split(",")]


def _many(prompt: str, n: int) -> tuple[float, int]:
    """Fire n identical requests together and return wall time and total tokens.

    Every speed number for this path so far is batch-1, which measures the
    kernel and nothing else. The packed pool's reason to exist is CAPACITY --
    288 B/token/layer against BF16's 1152 -- and capacity only shows up when
    enough requests are in flight to feel the KV limit. A kernel that loses at
    batch 1 can still win the deployment.
    """
    import threading
    res = [None] * n
    def one(i):
        try:
            res[i] = _one(prompt)
        except Exception:  # noqa: BLE001
            res[i] = None
    t0 = time.time()
    ts = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    dt = time.time() - t0
    got = [r for r in res if r]
    return dt, sum(r[1] for r in got)


def _one(prompt: str) -> tuple[float, int]:
    body = json.dumps({
        "text": prompt,
        # Greedy so every repetition decodes the same tokens and the timing is
        # comparable; sampling would vary the length and the work.
        "sampling_params": {"max_new_tokens": GEN, "temperature": 0.0},
    }).encode()
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{sp.PORT}/generate", data=body,
        headers={"Content-Type": "application/json"}), timeout=900).read())
    dt = time.time() - t0
    one = r[0] if isinstance(r, list) else r
    n = (one.get("meta_info", {}) or {}).get("completion_tokens") or GEN
    return dt, n


def _speed_drive(p, log_path: str) -> dict:
    err = _wait_healthy(p, log_path)
    if err:
        return {"error": err}
    filler = "The quick brown fox jumps over the lazy dog. " * 400
    res = {}
    for ctx in CTXS:
        # ~4 chars/token is close enough; the exact prompt length is identical
        # across arms, which is what the comparison needs.
        prompt = (filler * (ctx // 400 + 1))[: ctx * 4] + "\nSummarize:"
        for conc in CONCS:
            # One context that the server refuses -- a prompt longer than
            # --context-length returns HTTP 400 -- must not discard the points
            # that already measured cleanly. The first version let the exception
            # propagate and threw away three good points to report nothing.
            try:
                _many(prompt, conc)  # warmup, discarded: autotuning and JIT
                ts = []
                for _ in range(REPS):
                    dt, n = _many(prompt, conc)
                    ts.append(n / dt if dt > 0 else 0.0)
            except Exception as e:  # noqa: BLE001
                print(f"  ctx={ctx:>6} conc={conc:>3} SKIPPED: "
                      f"{type(e).__name__}: {str(e)[:80]}", flush=True)
                continue
            # String key: json.dump turns tuple keys into a TypeError and int
            # keys into strings, so the summary would not survive the round
            # trip through summary.json in a comparable form either way.
            res[f"{ctx}:{conc}"] = round(statistics.median(ts), 2)
            print(f"  ctx={ctx:>6} conc={conc:>3} decode "
                  f"{res[f'{ctx}:{conc}']:>9.2f} tok/s (median of {REPS})",
                  flush=True)
    return {"tps": res}


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    sp.OUT = OUT
    sp.drive = _speed_drive
    # The server must be able to hold the longest context swept, plus the
    # generation. Leaving this to serve_probe's default silently capped the
    # sweep: a 32000-token prompt against --context-length 16384 comes back as
    # HTTP 400, which reads like a driver bug rather than a configuration one.
    need = max(CTXS) + GEN + 512
    have = int(os.environ.get("CTX_LEN", sp.CTX))
    if have < need:
        print(f"[gf_speed] raising context length {have} -> {need} to cover "
              f"the longest context in the sweep")
        os.environ["CTX_LEN"] = str(need)
        sp.CTX = str(need)

    infos = {}
    for tag, gf in (("prod", "0"), ("gf", "1")):
        infos[tag] = sp.launch(tag, packed=True, extra_env={
            "SGLANG_OSCAR_MLA_PACKED_GF": gf,
            "SGLANG_OSCAR_MLA_PACKED_GF_CHECK": "0",
        })

    a, b = infos["prod"].get("tps"), infos["gf"].get("tps")
    if not (a and b):
        print("[gf_speed] an arm produced no timings")
        return 2
    print("\n[gf_speed] decode tok/s, production vs group-factored")
    print(f"{'ctx':>8} {'conc':>5} {'prod':>10} {'gf':>10} {'ratio':>8}  verdict")
    worst = None
    for ctx in CTXS:
        for conc in CONCS:
            k = f"{ctx}:{conc}"
            pa, pb = a.get(k), b.get(k)
            if not (pa and pb):
                continue
            r = pb / pa
            v = "gf faster" if r > 1.02 else ("gf SLOWER" if r < 0.98 else "tie")
            if worst is None or r < worst[1]:
                worst = (k, r)
            print(f"{ctx:>8} {conc:>5} {pa:>10.2f} {pb:>10.2f} {r:>8.3f}  {v}")
    # The trend across concurrency is the actual question, so state it rather
    # than leaving it to be eyeballed out of the table: gf's window pass is a
    # fixed per-step cost, so if the mechanism is what I think it is, the ratio
    # rises with concurrency at fixed context.
    if len(CONCS) > 1:
        print(f"\n[gf_speed] ratio vs concurrency at fixed context")
        for ctx in CTXS:
            row = []
            for conc in CONCS:
                k = f"{ctx}:{conc}"
                if a.get(k) and b.get(k):
                    row.append(f"c{conc}={b[k] / a[k]:.3f}")
            if row:
                print(f"  ctx={ctx:>6}  " + "  ".join(row))
    for tag in ("prod", "gf"):
        log = infos[tag].get("log")
        n = (sum(1 for ln in open(log, errors="ignore") if "GF-ENTRY" in ln)
             if log and os.path.exists(log) else 0)
        print(f"[gf_speed] {tag}: GF-ENTRY lines = {n} "
              f"({'live' if n else 'NEVER ENTERED'})")
        if tag == "gf" and n == 0:
            print("[gf_speed] !! the kernel under test never ran")
            return 2
    if worst:
        print(f"\n[gf_speed] worst context for gf: {worst[0]} at {worst[1]:.3f}x")
        print("[gf_speed] default-on is defensible only if this is >= ~1.0; a "
              "kernel that wins at long context and loses at short is a knob, "
              "not a default.")
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump({"prod": a, "gf": b}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
