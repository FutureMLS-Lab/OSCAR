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
PROMPTS = [
    "Explain in three sentences why a 2-bit KV cache saves memory but can cost accuracy.",
    "Explain in three sentences why a 2-bit KV cache saves memory but can cost accuracy.",
    "List the first eight prime numbers, then add them up and show the total.",
]


def launch(tag: str, packed: bool, extra_env: dict) -> dict:
    os.makedirs(OUT, exist_ok=True)
    log_path = os.path.join(OUT, f"server.{tag}.log")
    env = dict(os.environ)
    env.update(
        {
            "SGLANG_ENABLE_MIXED_KV_WINDOWS": "1",
            "SGLANG_OSCAR_MLA_KV_ROTATION_PATH": "hadamard",
            "SGLANG_OSCAR_MLA_KV_GROUP_SIZE": "128",
            "SGLANG_LLOYD_MAX": "1",
            "SGLANG_MIXED_KV_PREFIX_TOKENS": "64",
            "SGLANG_MIXED_KV_RECENT_TOKENS": "512",
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
        "--cuda-graph-max-bs", "8",
        "--context-length", CTX,
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
    for i, prompt in enumerate(PROMPTS):
        body = json.dumps({
            "text": prompt,
            # Greedy makes every INT2 config produce the same whitespace soup,
            # so probe with real sampling params.
            "sampling_params": {"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 700},
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
        "max_total_num_tokens": one(r"max_total_num_tokens=(\d+)", int),
        "kv_cache_dtype": one(r"kv_cache_dtype='([a-z0-9_]+)'"),
        "disable_radix_cache": one(r"disable_radix_cache=(\w+)"),
        "captured_bs": one(r"Capture cuda graph bs (\[[^\]]*\])"),
        "graph_true": g_true, "graph_false": g_false,
        "kv_alloc": one(r"KV Cache is allocated\. #tokens: \d+, KV size: ([0-9.]+) GB"),
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
    for tag, packed, extra in (
        ("fake", False, {}),
        ("packed", True, {"SGLANG_OSCAR_MLA_PACKED_SELFCHECK": "1"}),
    ):
        results.append(launch(tag, packed, extra))

    print("\n================ RESULT ================")
    for r in results:
        print(f"\n--- {r['tag']} (packed={r['packed']}) ---")
        for k in ("error", "max_total_num_tokens", "kv_alloc", "kv_cache_dtype",
                  "disable_radix_cache", "captured_bs", "graph_true", "graph_false",
                  "cache_hit", "decode_tok_s_median", "wall_s", "packed_line",
                  "write_path", "selfcheck", "error_lines"):
            if r.get(k) not in (None, [], ""):
                print(f"  {k}: {r[k]}")
    a = next((r for r in results if not r["packed"]), None)
    b = next((r for r in results if r["packed"]), None)
    if a and b and a.get("max_total_num_tokens") and b.get("max_total_num_tokens"):
        ratio = b["max_total_num_tokens"] / a["max_total_num_tokens"]
        print(f"\nPOOL: fake-quant {a['max_total_num_tokens']:,} -> "
              f"packed {b['max_total_num_tokens']:,}  = {ratio:.2f}x")
        print("VERDICT:", "storage landed" if ratio > 1.5 else
              "STORAGE DID NOT CHANGE -- the pool is still BF16-sized")
    with open(os.path.join(OUT, "probe.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
