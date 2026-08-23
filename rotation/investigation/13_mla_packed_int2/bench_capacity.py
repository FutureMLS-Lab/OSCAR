#!/usr/bin/env python3
"""Where the 2.92x pool actually pays: concurrency and context, not GPQA.

GPQA structurally cannot show this feature's benefit, and that is a property of
the benchmark rather than of the feature. Measured on all three GLM-5.2 arms:
peak KV occupancy 341,184 / 344,960 / 343,616 tokens and **zero retracts** in
every arm. The workload is 16 concurrent long generations; a pool of 645,056
tokens is never the binding constraint, so a pool of 1,882,304 buys nothing
there. It buys something when concurrency x context exceeds the smaller pool.

So this sweeps that axis directly, on the same model both arms already serve
(DeepSeek-V2-Lite, 1 GPU, the same latent geometry and the same pool/kernel
code as GLM-5.2), and reports the three things that decide deployment:

1. the concurrency at which the **BF16 arm starts retracting or queueing** while
   the packed arm does not;
2. **aggregate** output tokens/s at that concurrency -- which is where a 2.92x
   pool converts into throughput even with a slower per-token kernel;
3. the largest context each arm serves at fixed concurrency before it retracts.

The memory budget is held identical between arms (same ``--mem-fraction-static``)
because the pool-size difference *is* the feature; pinning ``--max-total-tokens``
would erase exactly what is being measured. It is set low enough that the BF16
pool is genuinely reachable -- at the default the pool is millions of tokens and
neither arm is ever pressured, which is the same mistake as measuring on GPQA.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request

MODEL = os.environ.get("MODEL", "deepseek-ai/DeepSeek-V2-Lite")
PORT = int(os.environ.get("PORT", "31431"))
OUT = os.environ.get("OUT_DIR", "/tmp/cap")
MEM_FRAC = os.environ.get("MEM_FRAC", "0.22")
CTX = int(os.environ.get("CTX_LEN", "32768"))
PROMPT_TOK = int(os.environ.get("PROMPT_TOK", "6000"))
GEN_TOK = int(os.environ.get("GEN_TOK", "128"))
CONCURRENCY = [int(x) for x in os.environ.get("CONC", "8,16,32,64,96").split(",")]
CTX_SWEEP = [int(x) for x in os.environ.get("CTX_SWEEP", "4000,8000,16000").split(",")]


# Common short words, ~1 token each. The first version of this used synthetic
# tokens like "w48211", which tokenize to 4-5 tokens apiece, so a request asked
# for at "16000 tokens" actually carried 50-60K and was rejected outright by the
# 32,768 context limit -- both arms returned 0/32 in 2.9 s and the ctx sweep
# measured request rejection rather than pool capacity.
_WORDS = ("time year people way day man thing woman life child world school "
          "state family student group country problem hand part place case "
          "week company system program question work government number night "
          "point home water room mother area money story fact month lot right "
          "study book eye job word business issue side kind head house service "
          "friend father power hour game line end member law car city name").split()


def make_prompt(n_tok: int, salt: int) -> str:
    """Distinct-per-request filler so the radix cache cannot collapse the sweep.

    A shared prefix would let both arms serve N requests out of one copy of the
    KV, which is the opposite of the pressure being measured.
    """
    n = len(_WORDS)
    body = " ".join(_WORDS[(salt * 7919 + i) % n] for i in range(n_tok))
    return f"Document {salt}. {body}\nSummarize the document above."


def wait_healthy(p, log_path: str, timeout: int = 1800) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if p.poll() is not None:
            print(f"    server exited rc={p.returncode}")
            print("".join(open(log_path).readlines()[-20:]))
            return False
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{PORT}/health_generate", timeout=5)
            return True
        except Exception:
            time.sleep(5)
    return False


_ERRORS: list = []


def one_request(prompt: str, gen: int) -> tuple[bool, int]:
    body = json.dumps({
        "text": prompt,
        "sampling_params": {"temperature": 0.7, "top_p": 0.95,
                            "max_new_tokens": gen, "ignore_eos": True},
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/generate", data=body,
        headers={"Content-Type": "application/json"})
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
        d = r if isinstance(r, dict) else r[0]
        n = (d.get("meta_info") or {}).get("completion_tokens") or gen
        return True, int(n)
    except Exception as e:  # noqa: BLE001
        _ERRORS.append(f"{type(e).__name__}: {str(e)[:160]}")
        return False, 0


def log_counters(path: str, since: int) -> dict:
    """Retract / queue / occupancy, read from the scheduler's own counters."""
    lines = open(path, errors="ignore").readlines()[since:]
    t = "".join(lines)
    q = [int(x) for x in re.findall(r"#queue-req: (\d+)", t)]
    tok = [int(x) for x in re.findall(r"#token: (\d+)", t)]
    usage = [float(x) for x in re.findall(r"token usage: ([0-9.]+)", t)]
    return {
        "retract": len(re.findall(r"[Rr]etract", t)),
        "max_queue": max(q) if q else 0,
        "peak_token": max(tok) if tok else 0,
        "peak_usage": max(usage) if usage else 0.0,
        "lines": len(lines),
    }


def sweep(tag: str, packed: bool) -> dict:
    os.makedirs(OUT, exist_ok=True)
    log_path = os.path.join(OUT, f"server.{tag}.log")
    env = dict(os.environ)
    env.update({
        "SGLANG_ENABLE_MIXED_KV_WINDOWS": "1",
        "SGLANG_OSCAR_MLA_KV_ROTATION_PATH": "hadamard",
        "SGLANG_OSCAR_MLA_KV_GROUP_SIZE": "128",
        "SGLANG_LLOYD_MAX": "1",
        "SGLANG_MIXED_KV_PREFIX_TOKENS": "64",
        "SGLANG_MIXED_KV_RECENT_TOKENS": "512",
        "SGLANG_OSCAR_MLA_KV_PACKED": "1" if packed else "0",
        "SGLANG_OSCAR_MLA_PACKED_SELFCHECK": "0",
        "HF_HUB_OFFLINE": "1",
    })
    # The BF16 arm is "no quantization at all", which is what the published
    # 82.32 reference did by pointing the rotation at a path that does not
    # exist. Without that, "BF16" would still be the fake-quant pool.
    if tag == "bf16":
        env["SGLANG_OSCAR_MLA_KV_ROTATION_PATH"] = ""
    args = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--trust-remote-code",
        "--tp", "1", "--port", str(PORT), "--host", "127.0.0.1",
        "--attention-backend", "triton",
        "--prefill-attention-backend", "triton",
        "--decode-attention-backend", "triton",
        "--kv-cache-dtype", "bfloat16",
        "--mem-fraction-static", MEM_FRAC,
        "--max-running-requests", "128",
        "--cuda-graph-max-bs", "128",
        "--context-length", str(CTX),
        "--disable-piecewise-cuda-graph",
    ]
    print(f"\n########## {tag} (packed={packed}) ##########", flush=True)
    res: dict = {"tag": tag, "packed": packed, "conc": [], "ctx": []}
    with open(log_path, "w") as lf:
        p = subprocess.Popen(args, stdout=lf, stderr=subprocess.STDOUT, env=env,
                             preexec_fn=os.setsid)
        try:
            if not wait_healthy(p, log_path):
                res["error"] = "server never healthy"
                return res
            t = open(log_path, errors="ignore").read()
            m = re.search(r"KV Cache is allocated\. #tokens: (\d+)", t)
            res["pool_tokens"] = int(m.group(1)) if m else None
            print(f"  pool = {res['pool_tokens']:,} tokens", flush=True)

            for conc in CONCURRENCY:
                mark = len(open(log_path, errors="ignore").readlines())
                prompts = [make_prompt(PROMPT_TOK, i + conc * 1000)
                           for i in range(conc)]
                t0 = time.time()
                with cf.ThreadPoolExecutor(max_workers=conc) as ex:
                    outs = list(ex.map(lambda s: one_request(s, GEN_TOK), prompts))
                el = time.time() - t0
                ok = sum(1 for o, _ in outs if o)
                gen = sum(n for _, n in outs)
                c = log_counters(log_path, mark)
                if _ERRORS:
                    # A failure is either a retract (capacity) or a rejection
                    # (harness). Printing the text is what tells them apart --
                    # the previous run's 0/32 in 2.9 s was rejection and got
                    # reported as if it were a capacity limit.
                    print(f"    request errors x{len(_ERRORS)}: {_ERRORS[0]}",
                          flush=True)
                    _ERRORS.clear()
                row = {"concurrency": conc, "ok": ok, "of": conc,
                       "wall_s": round(el, 1),
                       "agg_out_tok_s": round(gen / max(el, 1e-9), 1),
                       "kv_needed": conc * (PROMPT_TOK + GEN_TOK), **c}
                res["conc"].append(row)
                print(f"  conc {conc:>4}: ok {ok}/{conc}  wall {el:6.1f}s  "
                      f"agg_out {row['agg_out_tok_s']:>7.1f} tok/s  "
                      f"retract {c['retract']:>4}  max_queue {c['max_queue']:>3}  "
                      f"peak_kv {c['peak_token']:>9,}  usage {c['peak_usage']:.2f}  "
                      f"(needs ~{row['kv_needed']:,})", flush=True)

            for ctx in CTX_SWEEP:
                mark = len(open(log_path, errors="ignore").readlines())
                conc = 32
                prompts = [make_prompt(ctx, 90000 + i + ctx) for i in range(conc)]
                t0 = time.time()
                with cf.ThreadPoolExecutor(max_workers=conc) as ex:
                    outs = list(ex.map(lambda s: one_request(s, GEN_TOK), prompts))
                el = time.time() - t0
                ok = sum(1 for o, _ in outs if o)
                gen = sum(n for _, n in outs)
                c = log_counters(log_path, mark)
                if _ERRORS:
                    print(f"    request errors x{len(_ERRORS)}: {_ERRORS[0]}",
                          flush=True)
                    _ERRORS.clear()
                row = {"ctx": ctx, "concurrency": conc, "ok": ok,
                       "wall_s": round(el, 1),
                       "agg_out_tok_s": round(gen / max(el, 1e-9), 1),
                       "kv_needed": conc * (ctx + GEN_TOK), **c}
                res["ctx"].append(row)
                print(f"  ctx {ctx:>6} x conc {conc}: ok {ok}/{conc}  "
                      f"wall {el:6.1f}s  agg_out {row['agg_out_tok_s']:>7.1f} tok/s  "
                      f"retract {c['retract']:>4}  peak_kv {c['peak_token']:>9,}  "
                      f"(needs ~{row['kv_needed']:,})", flush=True)
        finally:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                p.wait(timeout=60)
            except Exception:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception:
                    pass
    return res


def main() -> None:
    print(f"mem-fraction {MEM_FRAC} (identical between arms -- the pool-size "
          f"difference IS the feature), ctx {CTX}, prompt ~{PROMPT_TOK} tok, "
          f"gen {GEN_TOK} tok")
    out = [sweep("bf16", False), sweep("packed", True)]
    with open(os.path.join(OUT, "capacity.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("\n================ CAPACITY ================")
    a, b = out[0], out[1]
    print(f"pool: BF16 {a.get('pool_tokens'):,} -> packed {b.get('pool_tokens'):,}"
          if a.get("pool_tokens") and b.get("pool_tokens") else "pool: n/a")
    print(f"\n{'conc':>6} | {'BF16 retract':>12} {'BF16 agg tok/s':>14} "
          f"{'BF16 peak kv':>12} | {'packed retract':>14} {'pk agg tok/s':>12} "
          f"{'pk peak kv':>10}")
    for ra, rb in zip(a.get("conc", []), b.get("conc", [])):
        print(f"{ra['concurrency']:>6} | {ra['retract']:>12} "
              f"{ra['agg_out_tok_s']:>14.1f} {ra['peak_token']:>12,} | "
              f"{rb['retract']:>14} {rb['agg_out_tok_s']:>12.1f} "
              f"{rb['peak_token']:>10,}")
    first_a = next((r['concurrency'] for r in a.get("conc", [])
                    if r['retract'] or r['max_queue']), None)
    first_b = next((r['concurrency'] for r in b.get("conc", [])
                    if r['retract'] or r['max_queue']), None)
    print(f"\nfirst concurrency with retract/queue: BF16 {first_a}  packed {first_b}")
    print(f"\n{'ctx':>7} | {'BF16 ok/retract/agg':>26} | {'packed ok/retract/agg':>26}")
    for ra, rb in zip(a.get("ctx", []), b.get("ctx", [])):
        print(f"{ra['ctx']:>7} | {ra['ok']:>3}/{ra['retract']:>4}/"
              f"{ra['agg_out_tok_s']:>10.1f} tok/s | {rb['ok']:>3}/{rb['retract']:>4}/"
              f"{rb['agg_out_tok_s']:>10.1f} tok/s")


if __name__ == "__main__":
    main()
