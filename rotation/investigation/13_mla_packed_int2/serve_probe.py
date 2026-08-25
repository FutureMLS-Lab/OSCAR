#!/usr/bin/env python3
"""Serve one MLA model twice -- fake-quant and packed -- and read the pool size back.

The question this answers is the one that decides whether the storage change
landed, and it is answered by a line of server log, not by a score:
``max_total_num_tokens``. On GLM-5.2 it was **645,056 in the BF16 arm and in
both INT2 arms alike**, because the fake-quant pool dequantizes straight back
into a BF16 cache. If the packed arm reports the same number, nothing changed.

Run on the *same* model, in the *same* process invocation, back to back, so the
two numbers differ in exactly one environment variable. DeepSeek-V2-Lite has
GLM-5.2's latent geometry (kv_lora_rank 512, qk_rope_head_dim 64) and the same
pool, kernel and cell-size code paths, on one GPU instead of eight -- which is
what makes this answerable while a 355B model waits for a node.

What it does NOT answer: accuracy. The rotation here is the uncalibrated
``hadamard`` fallback, so the INT2 arms are expected to be worse than BF16.
Scoring is GLM-5.2's job with its calibrated rotation.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

MODEL = os.environ.get("MODEL", "deepseek-ai/DeepSeek-V2-Lite")
PORT = int(os.environ.get("PORT", "31331"))
CTX = os.environ.get("CTX_LEN", "8192")
OUT = os.environ.get("OUT_DIR", "/tmp/probe")
# The window arena audit needs the ring to WRAP, which needs more generated
# tokens than the recent window is wide (512). 700 wraps it once; a GPQA answer
# wraps it ~20 times, so MAX_NEW is a knob rather than a constant.
MAX_NEW = int(os.environ.get("MAX_NEW", "700"))
# A long shared prefix on purpose: the failure being hunted is a prefix-cache
# hit whose BF16 window rows belong to a different request, so the requests have
# to share a prefix long enough to cover the 64-token sink and then diverge.
_SHARED = (
    "You are a careful assistant. Answer precisely, show your reasoning, and "
    "state any assumption you make rather than leaving it implicit. Prefer a "
    "concrete example over an abstract restatement, and if a question has more "
    "than one reasonable reading, say which one you took. "
)
PROMPTS = [_SHARED + q for q in (
    "Explain why a 2-bit KV cache saves memory but can cost accuracy.",
    "Explain why an attention sink matters more than a middle token.",
    "List the first eight prime numbers, then add them up and show the total.",
    "Describe what a rotation does to outliers before quantization.",
    "Explain why the newest tokens in a sequence are the most sensitive.",
    "Give three reasons long-context serving is memory bound.",
)]


def launch(tag: str, packed: bool, extra_env: dict) -> dict:
    os.makedirs(OUT, exist_ok=True)
    log_path = os.path.join(OUT, f"server.{tag}.log")
    env = dict(os.environ)
    env.update(
        {
            # Knob, not a constant: with windows OFF the pool has no arena, so
            # the group-factored path has nothing to exclude and runs no window
            # pass. That splits the remaining question in half -- if gf still
            # garbles without an arena the base kernel is wrong on real data,
            # and if it comes out clean the fault is the exclusion or the window
            # pass, which is where the bs=1 evidence already points.
            "SGLANG_ENABLE_MIXED_KV_WINDOWS": os.environ.get("WINDOWS", "1"),
            "SGLANG_OSCAR_MLA_KV_ROTATION_PATH": "hadamard",
            "SGLANG_OSCAR_MLA_KV_GROUP_SIZE": "128",
            "SGLANG_LLOYD_MAX": "1",
            # These are what actually gate the arena: the pool sets
            # _latent_windows from (prefix > 0 or recent > 0), NOT from
            # SGLANG_ENABLE_MIXED_KV_WINDOWS. Setting that flag to 0 left
            # has_hp=True and the no-arena experiment silently did not run.
            "SGLANG_MIXED_KV_PREFIX_TOKENS": os.environ.get("HP_PREFIX", "64"),
            "SGLANG_MIXED_KV_RECENT_TOKENS": os.environ.get("HP_RECENT", "512"),
            "SGLANG_OSCAR_MLA_KV_PACKED": "1" if packed else "0",
        }
    )
    env.update(extra_env)
    args = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--trust-remote-code",
        "--tp", "1", "--port", str(PORT), "--host", "127.0.0.1",
        "--attention-backend", "triton",
        "--prefill-attention-backend", "triton",
        "--decode-attention-backend", "triton",
        "--kv-cache-dtype", "bfloat16",
        "--mem-fraction-static", os.environ.get("MEM_FRAC", "0.80"),
        # Pinned, not defaulted: the window arena is sized per request slot and
        # the default is derived from the pool size, which this change moves.
        "--max-running-requests", "64",
    ]
    # Under CUDA graph *replay* the pool's Python does not execute at all -- the
    # captured kernels do -- so any host-side audit or self-check inside the
    # write path can only ever see eager forwards. The ring is still maintained
    # (it is all captured torch ops), but auditing it requires eager decode.
    if os.environ.get("DISABLE_CG", "0") == "1":
        args += ["--disable-cuda-graph"]
    else:
        args += ["--cuda-graph-max-bs", os.environ.get("CG_MAX_BS", "8")]
    args += [
        "--context-length", CTX,
        # DeepSeek-V2-Lite routes its attention layers through the piecewise
        # CUDA graph, i.e. through dynamo. GLM-5.2 is on the
        # piecewise-disabled arch list, so the probe matches it rather than
        # measuring a compile path the target model never takes.
        "--disable-piecewise-cuda-graph",
    ]
    print(f"\n=== {tag} (packed={packed}) ===", flush=True)
    with open(log_path, "w") as lf:
        p = subprocess.Popen(args, stdout=lf, stderr=subprocess.STDOUT, env=env,
                             preexec_fn=os.setsid)
        info = {"tag": tag, "packed": packed, "log": log_path}
        try:
            info.update(drive(p, log_path))
        finally:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                p.wait(timeout=60)
            except Exception:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception:
                    pass
    info.update(scrape(log_path))
    return info


def ask_one(prompt: str, max_new: int | None = None) -> str:
    body = json.dumps({
        "text": prompt,
        "sampling_params": {"temperature": 0.7, "top_p": 0.95,
                            "max_new_tokens": MAX_NEW if max_new is None
                            else max_new},
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/generate", data=body,
        headers={"Content-Type": "application/json"})
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
        return r["text"] if isinstance(r, dict) else r[0]["text"]
    except Exception as e:  # noqa: BLE001
        return f"<<request failed: {e}>>"


def drive(p, log_path: str) -> dict:
    deadline = time.time() + 1800
    while time.time() < deadline:
        if p.poll() is not None:
            print(f"  server exited early rc={p.returncode}; tail:", flush=True)
            print("".join(open(log_path).readlines()[-25:]), flush=True)
            return {"error": f"server exited rc={p.returncode}"}
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health_generate", timeout=5)
            break
        except Exception:
            time.sleep(5)
    else:
        return {"error": "server never became healthy"}

    outs = []
    t0 = time.time()
    if os.environ.get("CONCURRENT", "0") == "1":
        # Sequential requests cannot exercise prefix sharing across *different*
        # req indices: a lone request gets its own index back and lands on the
        # rows it wrote itself. Firing them together is what makes one request
        # read a prefix whose window rows belong to another.
        import threading
        outs = [None] * len(PROMPTS)

        # Firing them all at once does NOT produce prefix sharing: they arrive
        # together, so none can hit a cache the others have not written yet.
        # A first run measured #cached-token max 5 over six requests sharing a
        # ~55-token prefix and read 100% clean, which said nothing at all.
        # Populate the cache with one request first, and let it FINISH so its
        # req index is freed and handed to one of the concurrent batch -- the
        # reclamation is the thing under test, not merely the hit.
        warm = ask_one(PROMPTS[0], max_new=int(os.environ.get("WARM_NEW", "64")))
        print(f"  [warmup] {str(warm)[:120]!r}", flush=True)

        def one(i, prompt):
            outs[i] = ask_one(prompt)

        ts = [threading.Thread(target=one, args=(i, p))
              for i, p in enumerate(PROMPTS)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        for i, txt in enumerate(outs):
            print(f"  [{i}] {str(txt)[:220]!r}", flush=True)
        return {"wall_s": round(time.time() - t0, 1), "samples": outs}
    for i, prompt in enumerate(PROMPTS):
        body = json.dumps({
            "text": prompt,
            # Greedy makes every INT2 config produce the same whitespace soup,
            # so probe with real sampling params.
            "sampling_params": {"temperature": 0.7, "top_p": 0.95, "max_new_tokens": MAX_NEW},
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/generate", data=body,
            headers={"Content-Type": "application/json"})
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=900).read())
            txt = r["text"] if isinstance(r, dict) else r[0]["text"]
        except Exception as e:  # noqa: BLE001
            txt = f"<<request failed: {e}>>"
        outs.append(txt)
        print(f"  [{i}] {txt[:220]!r}", flush=True)
    return {"wall_s": round(time.time() - t0, 1), "samples": outs}


def scrape(log_path: str) -> dict:
    t = open(log_path, errors="ignore").read()

    def one(pat, cast=str):
        m = re.search(pat, t)
        return cast(m.group(1)) if m else None

    g_true = len(re.findall(r"cuda graph: True", t))
    g_false = len(re.findall(r"cuda graph: False", t))
    cached = [int(x) for x in re.findall(r"#cached-token: (\d+)", t)]
    new = [int(x) for x in re.findall(r"#new-token: (\d+)", t)]
    tp = [float(x) for x in re.findall(r"gen throughput \(token/s\): ([0-9.]+)", t)]
    tp_s = sorted(tp)
    return {
        # Two sources: the server_args echo, and the pool's own allocation
        # line. The second is authoritative -- it is what the pool actually
        # built -- and it is the one that survives a server that dies before
        # the args echo.
        "max_total_num_tokens": one(r"max_total_num_tokens=(\d+)", int),
        "pool_tokens": one(r"KV Cache is allocated\. #tokens: (\d+)", int),
        "kv_cache_dtype": one(r"kv_cache_dtype='([a-z0-9_]+)'"),
        "disable_radix_cache": one(r"disable_radix_cache=(\w+)"),
        "captured_bs": one(r"Capture cuda graph bs (\[[^\]]*\])"),
        "graph_true": g_true, "graph_false": g_false,
        "kv_alloc_gb": one(r"KV Cache is allocated\. #tokens: \d+, KV size: ([0-9.]+) GB",
                           float),
        "packed_line": one(r"(\[MLAPacked\] packed latent storage: [^\n]*)"),
        "write_path": one(r"(\[Int2HPKVPool\] write path=[^\n]*)"),
        "selfcheck": one(r"(\[MLAPacked\] selfcheck [^\n]*worst_rel=[^\n]*)"),
        "cache_hit": (
            f"{100.0 * sum(cached) / (sum(cached) + sum(new)):.2f}%"
            if cached and new and sum(cached) + sum(new) else None
        ),
        "decode_tok_s_median": round(tp_s[len(tp_s) // 2], 1) if tp else None,
        "error_lines": [l for l in t.splitlines()
                        if "Traceback" in l or "Error:" in l][:3],
    }


def main() -> None:
    results = []
    # The gf arm differs from `packed` in exactly one variable, so any
    # difference between them is the group-factored two-pass read path and
    # nothing else. It is here because that path is validated only as a KERNEL:
    # the equivalence gate compares stage-1+stage-2 in isolation, and says
    # nothing about whether the backend wiring -- the borrowed split slot, the
    # num_kv_splits-1 bookkeeping, CUDA-graph capture over a second kernel --
    # is right in a live server. A microbenchmark cannot fail that way, so a
    # server has to.
    arms = [
        ("fake", False, {}),
        ("packed", True, {"SGLANG_OSCAR_MLA_PACKED_SELFCHECK": "1"}),
    ]
    if os.environ.get("PROBE_GF", "1") == "1":
        arms.append(("gf", True, {"SGLANG_OSCAR_MLA_PACKED_GF": "1"}))
    for tag, packed, extra in arms:
        results.append(launch(tag, packed, extra))

    print("\n================ RESULT ================")
    for r in results:
        print(f"\n--- {r['tag']} (packed={r['packed']}) ---")
        for k in ("error", "pool_tokens", "max_total_num_tokens", "kv_alloc_gb", "kv_cache_dtype",
                  "disable_radix_cache", "captured_bs", "graph_true", "graph_false",
                  "cache_hit", "decode_tok_s_median", "wall_s", "packed_line",
                  "write_path", "selfcheck", "error_lines"):
            if r.get(k) not in (None, [], ""):
                print(f"  {k}: {r[k]}")
    a = next((r for r in results if not r["packed"]), None)
    b = next((r for r in results if r["packed"]), None)
    ka = (a or {}).get("pool_tokens") or (a or {}).get("max_total_num_tokens")
    kb = (b or {}).get("pool_tokens") or (b or {}).get("max_total_num_tokens")
    if ka and kb:
        ratio = kb / ka
        print(f"\nPOOL: fake-quant {ka:,} -> packed {kb:,}  = {ratio:.2f}x")
        print("VERDICT:", "storage landed" if ratio > 1.5 else
              "STORAGE DID NOT CHANGE -- the pool is still BF16-sized")
    # gf must match packed on POOL SIZE (it changes only the read path) and
    # should beat it on decode throughput. Text is compared too: the failure
    # this arm exists to catch is a wiring bug that still produces numbers.
    g = next((r for r in results if r["tag"] == "gf"), None)
    if g and b:
        pg = g.get("pool_tokens") or g.get("max_total_num_tokens")
        print(f"\nGF: pool {pg:,}" if pg else "\nGF: pool unknown", end="")
        print(f" (packed {kb:,})" if kb else "")
        if pg and kb and pg != kb:
            print("  !! gf changed the POOL SIZE; it must only change the read "
                  "path, so something other than the kernel moved")
        tb, tg = b.get("decode_tok_s_median"), g.get("decode_tok_s_median")
        if tb and tg:
            print(f"  decode tok/s: packed {tb} -> gf {tg} = {tg / tb:.2f}x")
        if g.get("error_lines"):
            print(f"  !! gf errors: {g['error_lines']}")
        # The field is `samples`, and sampling is temperature 0.7, so equality
        # across arms was never the right test -- these will differ even when
        # both are perfect. What matters is whether gf's text is STRUCTURALLY
        # like the packed arm's, because the wiring failure to fear is one that
        # still returns fluent-shaped noise, which is exactly how the K3 latent
        # path failed. Report the shape statistics rather than a verdict, and
        # flag only on a clear break.
        def shape(samples):
            ss = [x for x in (samples or []) if isinstance(x, str)]
            if not ss:
                return None
            n = len(ss)
            L = sum(len(x) for x in ss) / n
            asc = sum(sum(c.isascii() for c in x) for x in ss)
            tot = max(1, sum(len(x) for x in ss))
            words = [w for x in ss for w in x.split()]
            wl = sum(len(w) for w in words) / max(1, len(words))
            empty = sum(1 for x in ss if not x.strip())
            return {"n": n, "mean_chars": round(L), "ascii": round(asc / tot, 3),
                    "mean_word": round(wl, 2), "empty": empty}
        sb, sg = shape(b.get("samples")), shape(g.get("samples"))
        print(f"  text shape  packed {sb}")
        print(f"  text shape  gf     {sg}")
        if sg and sg["empty"] == sg["n"]:
            print("  !! gf produced EMPTY completions -- wiring is broken "
                  "regardless of throughput")
        elif sb and sg and abs(sb["ascii"] - sg["ascii"]) > 0.25:
            print("  !! gf text is structurally unlike the packed arm's "
                  "(ascii ratio moved a lot) -- suspect garbling, not sampling")

    with open(os.path.join(OUT, "probe.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
