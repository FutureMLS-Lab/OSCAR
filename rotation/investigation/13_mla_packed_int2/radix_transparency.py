#!/usr/bin/env python3
"""Is the prefix cache transparent for the packed MLA pool?

The packed arm loses **12.20 pp** on GPQA with the radix cache on and **1.96 pp**
with it off (n=51, exact McNemar p = 1.0 against both fake-quant and BF16). So
the cache is the variable. Two hypotheses about *how* have now been measured and
both refuted:

* the window arena's owner tag -- 125 samples with real 57- and 292-token prefix
  hits across four req indices, 100% accepted, sink 64/64 every time;
* the read path's numerics -- teacher-forced NLL puts packed at R = 0.05 of the
  cost of quantizing at all, on a corpus calibrated to reproduce the known
  +2.7% PPL.

And the NLL harness **cannot** be extended to cover this, which is worth stating
rather than discovering twice: asking for input logprobs from position 0 makes
sglang recompute every position, so ``#cached-token`` is 0 in every arm even
with ``disable_radix_cache=False``. Teacher forcing and prefix caching are
mutually exclusive by construction.

What does work is a transparency test, and it needs no second server and no
cross-run pairing. A prefix cache is supposed to be *invisible*: with greedy
decoding, sending the same prompt twice must produce the same tokens, because
the second request differs only in reading KV the first one wrote. Any
divergence is the cache failing to be transparent for this pool.

Greedy is normally a trap here -- ``temperature=0`` makes every INT2 config emit
the same degenerate text, so it cannot separate two configs. This is not that
comparison. It is one config against itself with one variable, and identical
output is the *passing* result, which is exactly what greedy is for.

Run both pools. Fake-quant is the control: it keeps a windowed token in the row
it already owns, so sharing a slot shares its value and the cache must be
transparent. If fake-quant is transparent and packed is not, that is the gap.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request

MODEL = os.environ.get("MODEL", "deepseek-ai/DeepSeek-V2-Lite")
PORT = int(os.environ.get("PORT", "31339"))
OUT = os.environ.get("OUT_DIR", "/tmp/radix")
CTX = os.environ.get("CTX_LEN", "16384")
MAX_NEW = int(os.environ.get("MAX_NEW", "256"))
# The shared prefix has to be longer than the sink (64) for the sink itself to
# come from the cache, which is the tier the packed pool addresses differently.
PREFIX_REPS = int(os.environ.get("PREFIX_REPS", "6"))

_P = (
    "You are a careful assistant. Answer precisely, show your reasoning, and "
    "state any assumption you make rather than leaving it implicit. Prefer a "
    "concrete example over an abstract restatement. "
)
PREFIX = _P * PREFIX_REPS
QUESTIONS = [
    "Explain why a 2-bit KV cache saves memory but can cost accuracy.",
    "Explain why an attention sink matters more than a middle token.",
    "Describe what a rotation does to outliers before quantization.",
]


def post(prompt: str) -> dict:
    body = json.dumps({
        "text": prompt,
        # Greedy: the whole point is that the two passes must agree exactly.
        "sampling_params": {"temperature": 0.0, "max_new_tokens": MAX_NEW},
        "return_logprob": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/generate", data=body,
        headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=1200).read())
    return r[0] if isinstance(r, list) else r


def toks(r: dict):
    lp = (r.get("meta_info") or {}).get("output_token_logprobs") or []
    return [(e[1], e[0]) for e in lp if e]


def launch(tag: str, packed: bool) -> dict:
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
    })
    args = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--trust-remote-code",
        "--tp", "1", "--port", str(PORT), "--host", "127.0.0.1",
        "--attention-backend", "triton",
        "--prefill-attention-backend", "triton",
        "--decode-attention-backend", "triton",
        "--kv-cache-dtype", "bfloat16",
        "--mem-fraction-static", os.environ.get("MEM_FRAC", "0.80"),
        "--max-running-requests", "64",
        "--context-length", CTX,
        "--disable-piecewise-cuda-graph",
    ]
    if os.environ.get("DISABLE_CG", "0") == "1":
        args += ["--disable-cuda-graph"]
    else:
        args += ["--cuda-graph-max-bs", os.environ.get("CG_MAX_BS", "8")]

    print(f"\n=== {tag} (packed={packed}) ===", flush=True)
    with open(log_path, "w") as lf:
        p = subprocess.Popen(args, stdout=lf, stderr=subprocess.STDOUT,
                             env=env, preexec_fn=os.setsid)
        try:
            out = drive(p, log_path)
        finally:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                p.wait(timeout=90)
            except Exception:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception:
                    pass
    out["tag"] = tag
    t = open(log_path, errors="ignore").read()
    m = re.search(r"max_total_num_tokens=(\d+)", t)
    out["pool"] = int(m.group(1)) if m else None
    out["max_cached"] = max(
        [int(x) for x in re.findall(r"#cached-token: (\d+)", t)] or [0]
    )
    return out


def drive(p, log_path: str) -> dict:
    deadline = time.time() + 2400
    while time.time() < deadline:
        if p.poll() is not None:
            print(f"  server exited rc={p.returncode}; tail:", flush=True)
            print("".join(open(log_path).readlines()[-30:]), flush=True)
            return {"error": f"server exited rc={p.returncode}"}
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{PORT}/health_generate", timeout=5)
            break
        except Exception:
            time.sleep(5)
    else:
        return {"error": "server never became healthy"}

    res = []
    for qi, q in enumerate(QUESTIONS):
        prompt = PREFIX + q
        try:
            a = post(prompt)          # cold: writes every position itself
            b = post(prompt)          # warm: prefix served from the cache
        except Exception as e:  # noqa: BLE001
            res.append({"q": qi, "error": str(e)})
            continue
        ta, tb = toks(a), toks(b)
        n = min(len(ta), len(tb))
        first = next((i for i in range(n) if ta[i][0] != tb[i][0]), None)
        maxdlp = max(
            (abs(ta[i][1] - tb[i][1]) for i in range(n)), default=0.0
        )
        res.append({
            "q": qi,
            "n_a": len(ta), "n_b": len(tb),
            "first_divergence": first,
            "max_abs_dlogprob": maxdlp,
            "identical": first is None and len(ta) == len(tb),
            "text_a": (a.get("text") or "")[:160],
            "text_b": (b.get("text") or "")[:160],
        })
        print(f"  q{qi}: identical={res[-1]['identical']} "
              f"first_div={first} max|dlp|={maxdlp:.3e}", flush=True)
    return {"results": res}


def main() -> None:
    out = []
    for tag, packed in (("fake", False), ("packed", True)):
        out.append(launch(tag, packed))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "radix_transparency.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 72)
    for r in out:
        if "error" in r:
            print(f"{r['tag']:>8}: {r['error']}")
            continue
        rs = r.get("results", [])
        ident = sum(1 for x in rs if x.get("identical"))
        print(f"{r['tag']:>8}: pool={r['pool']} max_cached_token={r['max_cached']} "
              f"identical {ident}/{len(rs)}")
        for x in rs:
            if not x.get("identical"):
                print(f"           q{x.get('q')} first_div="
                      f"{x.get('first_divergence')} "
                      f"max|dlp|={x.get('max_abs_dlogprob', 0):.3e}")

    # The precondition, checked and printed rather than assumed. Three separate
    # runs in this investigation have produced a clean result that turned out to
    # mean "the test never happened": two concurrent probes with no cache hits,
    # and an NLL run where input logprobs silently disable prefix reuse.
    bad = [r for r in out if "error" not in r and r.get("max_cached", 0) == 0]
    if bad:
        print("\nPRECONDITION FAILED: #cached-token is 0 in "
              f"{[r['tag'] for r in bad]} -- the prefix cache never served "
              "anything, so an 'identical' result here is not evidence of "
              "transparency. Do not read a conclusion off this run.")


if __name__ == "__main__":
    main()
