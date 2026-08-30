#!/usr/bin/env python3
"""Is the packed INT2 MLA kernel correct under CUDA graph REPLAY?

Every correctness check so far runs host-side Python in the pool's write path,
and a graph replay executes no Python -- so the arena audit, the self-check and
the window-fidelity measurement all only ever observed eager forwards. "The real
kernel is correct under replay" has been inferred from coherent output and a
plausible score, never tested.

This tests it directly and with one variable. Same packed pool, same rotation,
same windows, greedy decoding; the only difference is --disable-cuda-graph. A
graph replay that reads or writes the packed cache wrongly cannot produce the
same greedy token sequence as the eager path.

Greedy is right here for the same reason it was right for the radix transparency
test: this is one config against itself, so identical output is the passing
result, not a collapsed comparison between two configs.
"""
from __future__ import annotations
import json, os, re, signal, subprocess, sys, time, urllib.request

MODEL = os.environ.get("MODEL", "deepseek-ai/DeepSeek-V2-Lite")
PORT = int(os.environ.get("PORT", "31341"))
OUT = os.environ.get("OUT_DIR", "/tmp/gc")
MAX_NEW = int(os.environ.get("MAX_NEW", "256"))
QS = [
    "Explain why a 2-bit KV cache saves memory but can cost accuracy.",
    "Explain why an attention sink matters more than a middle token.",
    "List the first eight prime numbers, then add them up.",
    "Describe what a rotation does to outliers before quantization.",
    "Give three reasons long-context serving is memory bound.",
    "Explain the difference between a prefill and a decode step.",
]

def post(p):
    body = json.dumps({"text": p,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": MAX_NEW},
        "return_logprob": True}).encode()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{PORT}/generate", data=body,
        headers={"Content-Type": "application/json"}), timeout=1200).read())
    return r[0] if isinstance(r, list) else r

def toks(r):
    lp = (r.get("meta_info") or {}).get("output_token_logprobs") or []
    return [e[1] for e in lp if e]

def launch(tag, graph):
    os.makedirs(OUT, exist_ok=True)
    log = os.path.join(OUT, f"server.{tag}.log")
    env = dict(os.environ); env.update({
        "SGLANG_ENABLE_MIXED_KV_WINDOWS": "1",
        "SGLANG_OSCAR_MLA_KV_ROTATION_PATH": "hadamard",
        "SGLANG_OSCAR_MLA_KV_GROUP_SIZE": "128", "SGLANG_LLOYD_MAX": "1",
        "SGLANG_MIXED_KV_PREFIX_TOKENS": "64",
        "SGLANG_MIXED_KV_RECENT_TOKENS": "512",
        "SGLANG_OSCAR_MLA_KV_PACKED": "1"})
    args = [sys.executable, "-m", "sglang.launch_server", "--model-path", MODEL,
        "--trust-remote-code", "--tp", "1", "--port", str(PORT),
        "--host", "127.0.0.1", "--attention-backend", "triton",
        "--prefill-attention-backend", "triton",
        "--decode-attention-backend", "triton",
        "--kv-cache-dtype", "bfloat16", "--mem-fraction-static", "0.80",
        "--max-running-requests", "64", "--context-length", "8192",
        "--disable-piecewise-cuda-graph"]
    args += ["--cuda-graph-max-bs", "8"] if graph else ["--disable-cuda-graph"]
    print(f"\n=== {tag} (cuda graph={graph}) ===", flush=True)
    with open(log, "w") as lf:
        p = subprocess.Popen(args, stdout=lf, stderr=subprocess.STDOUT, env=env,
                             preexec_fn=os.setsid)
        try:
            dl = time.time() + 2400
            while time.time() < dl:
                if p.poll() is not None:
                    print("".join(open(log).readlines()[-25:]), flush=True)
                    return {"error": f"rc={p.returncode}"}
                try:
                    urllib.request.urlopen(
                        f"http://127.0.0.1:{PORT}/health_generate", timeout=5)
                    break
                except Exception:
                    time.sleep(5)
            else:
                return {"error": "never healthy"}
            out = {}
            for i, q in enumerate(QS):
                out[i] = toks(post(q))
                print(f"  q{i}: {len(out[i])} tokens", flush=True)
            return {"toks": out}
        finally:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM); p.wait(timeout=90)
            except Exception:
                try: os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception: pass

def main():
    t = open(os.path.join(OUT, "x"), "w") if False else None
    g = launch("graph", True)
    e = launch("eager", False)
    print("\n" + "=" * 68)
    if "error" in g or "error" in e:
        print("graph:", g.get("error", "ok"), " eager:", e.get("error", "ok"))
        return
    ident = 0
    for i in range(len(QS)):
        a, b = g["toks"][i], e["toks"][i]
        n = min(len(a), len(b))
        first = next((j for j in range(n) if a[j] != b[j]), None)
        same = first is None and len(a) == len(b)
        ident += same
        print(f"q{i}: identical={same} first_div={first} "
              f"len graph={len(a)} eager={len(b)}")
    print(f"\ngraph-vs-eager identical {ident}/{len(QS)}")
    print("A graph replay that read or wrote the packed cache wrongly could not"
          " reproduce the eager greedy sequence, so identical == the packed"
          " kernel is correct under replay." if ident == len(QS) else
          "DIVERGENCE: the packed kernel behaves differently under graph replay"
          " than eager. That is a correctness bug in the shipped path.")

if __name__ == "__main__":
    main()
