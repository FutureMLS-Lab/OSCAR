#!/usr/bin/env python3
"""Is the packed read path numerically worse than the fake-quant one it replaces?

The packed arm scores **-12.20 pp against fake-quant** on GPQA, the gap sits in
accuracy-given-answered rather than in truncation, and the discordant pairs are
0/5 and 0/6 -- packed is never uniquely right. That is a systematic quality
loss, but a 198-question score on a 355B model costs eight GPUs for eight hours
per arm, which is far too slow a loop to debug with, and it cannot separate
"the read path returns worse numbers" from "the read path is fine and the
sampled trajectory diverged".

Teacher-forced NLL separates exactly those two, and DeepSeek-V2-Lite runs it on
one GPU: same ``kv_lora_rank`` 512 / ``qk_rope_head_dim`` 64 latent geometry,
same pool, same kernel, same window arena. The rotation is the uncalibrated
``hadamard`` fallback, so all three arms are worse than a calibrated GLM-5.2
would be -- which does not matter, because the quantity of interest is
**packed minus fake-quant** and both arms carry the identical rotation. Any
delta between them is the read path.

Two things this has to get right or it measures nothing:

* **Chunked prefill is mandatory.** With a single-chunk prefill the attention
  reads the freshly-written BF16 values in registers, never the pool, so a
  prompt shorter than the chunk measures the BF16 path in all three arms and
  reports a delta of zero. ``--chunked-prefill-size`` is pinned well below the
  prompt length and only positions past the first chunk are scored.
* **Only positions whose cache read was quantized count.** Position ``i``'s
  logprob depends on the cache for ``[0, i)``; inside the first chunk that
  cache is the current forward's own values. Scoring starts at ``SKIP``.

``max_new_tokens: 1`` with ``logprob_start_len: 0`` returns one logprob per
input position, so the whole measurement is a single request per arm and there
is no sampling variance to average away.
"""
from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request

MODEL = os.environ.get("MODEL", "deepseek-ai/DeepSeek-V2-Lite")
PORT = int(os.environ.get("PORT", "31337"))
OUT = os.environ.get("OUT_DIR", "/tmp/nll")
CHUNK = int(os.environ.get("CHUNK", "512"))
SKIP = int(os.environ.get("SKIP", "1024"))
CTX = os.environ.get("CTX_LEN", "8192")
# Long enough that most scored positions sit past the recent window (512) and
# so are served from the quantized tier rather than the BF16 arena.
TARGET_TOKENS = int(os.environ.get("TARGET_TOKENS", "4096"))
# "repeat" = the fixed passage below (copy task, sharp on retrieval);
# "docs" = non-repeating prose from the tree (restores dynamic range).
CORPUS = os.environ.get("CORPUS", "repeat")


def corpus() -> str:
    """A fixed, self-contained passage repeated to length.

    Deliberately not a downloaded dataset: the delta between two arms on the
    same text is the measurement, and a network fetch is a way for the two arms
    to silently score different text.
    """
    para = (
        "The key-value cache stores, for every attention layer and every token "
        "already processed, the key and value projections that the next token "
        "must attend to. Its size grows linearly with sequence length and with "
        "batch size, so at long context it dominates the memory footprint of "
        "serving and eventually decides how many requests fit on a device at "
        "once. Quantizing it to two bits per element therefore buys capacity "
        "directly. The difficulty is that attention is a weighted average, so "
        "an error in a single key changes the weights of every other token, and "
        "the newest tokens and the attention sink are far more sensitive than "
        "the bulk of the sequence. A rotation applied before quantization "
        "spreads outliers across the channel axis, which is what makes two bits "
        "survivable at all; keeping a small window in full precision covers the "
        "positions the rotation cannot rescue. "
    )
    # ~161 tokens per repetition, measured: the first run assumed 80 and built
    # an 8,535-token prompt against an 8,192-token context, so all three arms
    # died on the same HTTP 400 and the job measured nothing. Budget from the
    # real rate and leave headroom.
    rep = para * max(2, TARGET_TOKENS // 161)
    if CORPUS != "docs":
        return rep

    # The repeated passage measures BF16 at NLL 0.0024 -- the model is copying,
    # which probes cache *retrieval* very sharply but leaves almost no dynamic
    # range if the local n-gram statistics alone suffice. Non-repeating prose
    # restores the range. Taken from markdown already in the tree rather than
    # fetched: a network read is a way for two arms to score different text.
    import glob
    chunks = []
    total = 0
    root = os.environ.get(
        "DOCS_ROOT",
        os.path.join(os.path.dirname(__file__), "..", "..", "..",
                     "sglang-research", "docs"),
    )
    for path in sorted(glob.glob(os.path.join(root, "**", "*.md"),
                                 recursive=True)):
        try:
            t = open(path, errors="ignore").read()
        except Exception:  # noqa: BLE001
            continue
        # Prose only: code fences and tables are low-entropy in a way that
        # mimics the repeated-passage problem this is meant to fix.
        t = re.sub(r"```.*?```", " ", t, flags=re.S)
        t = re.sub(r"\s+", " ", t).strip()
        if len(t) < 400:
            continue
        chunks.append(t)
        total += len(t)
        if total > TARGET_TOKENS * 5:
            break
    if not chunks:
        print("  WARNING: no docs found, falling back to the repeated passage",
              flush=True)
        return rep
    return " ".join(chunks)


def launch(tag: str, env_over: dict) -> dict:
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
    })
    env.update(env_over)
    args = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--trust-remote-code",
        "--tp", "1", "--port", str(PORT), "--host", "127.0.0.1",
        "--attention-backend", "triton",
        "--prefill-attention-backend", "triton",
        "--decode-attention-backend", "triton",
        "--kv-cache-dtype", "bfloat16",
        # Input logprobs materialize a [prompt_len, vocab] logits tensor, which
        # is several GB on its own; the pool has to leave room for it.
        "--mem-fraction-static", os.environ.get("MEM_FRAC", "0.70"),
        "--max-running-requests", "64",
        "--cuda-graph-max-bs", "8",
        "--context-length", CTX,
        "--chunked-prefill-size", str(CHUNK),
        "--disable-piecewise-cuda-graph",
        # A prefix hit would serve scored positions from a previous arm's
        # cached KV, which is the one thing that would make the arms share a
        # read path instead of differing by one.
        "--disable-radix-cache",
    ]
    print(f"\n=== {tag} ===", flush=True)
    with open(log_path, "w") as lf:
        p = subprocess.Popen(args, stdout=lf, stderr=subprocess.STDOUT, env=env,
                             preexec_fn=os.setsid)
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
    out["pool"] = pool_size(log_path)
    return out


def pool_size(log_path: str):
    t = open(log_path, errors="ignore").read()
    m = re.search(r"max_total_num_tokens=(\d+)", t)
    return int(m.group(1)) if m else None


def drive(p, log_path: str) -> dict:
    deadline = time.time() + 2400
    while time.time() < deadline:
        if p.poll() is not None:
            print(f"  server exited early rc={p.returncode}; tail:", flush=True)
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

    body = json.dumps({
        "text": corpus(),
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
        "return_logprob": True,
        "logprob_start_len": 0,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/generate", data=body,
        headers={"Content-Type": "application/json"})
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    except Exception as e:  # noqa: BLE001
        return {"error": f"request failed: {e}"}
    if isinstance(r, list):
        r = r[0]
    lps = r.get("meta_info", {}).get("input_token_logprobs")
    if not lps:
        return {"error": f"no input_token_logprobs; meta={list(r.get('meta_info', {}))}"}
    # Entries are [logprob, token_id, text]; the first position has no
    # prediction and comes back as null.
    vals = [e[0] for e in lps if e and e[0] is not None]
    if len(vals) <= SKIP:
        return {"error": f"only {len(vals)} scored positions, need > SKIP={SKIP}"}
    kept = vals[SKIP:]
    nll = -sum(kept) / len(kept)
    return {
        "n_positions": len(vals),
        "n_scored": len(kept),
        "nll": nll,
        "ppl": math.exp(nll),
    }


def main() -> None:
    arms = [
        ("bf16", {"SGLANG_ENABLE_MIXED_KV_WINDOWS": "0",
                  "SGLANG_OSCAR_MLA_KV_PACKED": "0",
                  "SGLANG_OSCAR_MLA_KV_ROTATION_PATH": ""}),
        ("fakequant", {"SGLANG_OSCAR_MLA_KV_PACKED": "0"}),
        ("packed", {"SGLANG_OSCAR_MLA_KV_PACKED": "1"}),
    ]
    res = []
    for tag, over in arms:
        r = launch(tag, over)
        res.append(r)
        print(f"  {tag}: {json.dumps(r)}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "nll.json"), "w") as f:
        json.dump(res, f, indent=2)

    print("\n" + "=" * 72)
    print(f"{'arm':>10} {'pool tokens':>13} {'scored':>8} {'NLL':>9} {'PPL':>10}")
    by = {}
    for r in res:
        if "error" in r:
            print(f"{r['tag']:>10} {'-':>13} {'-':>8} {'-':>9} {'-':>10}  "
                  f"{r['error']}")
            continue
        by[r["tag"]] = r
        print(f"{r['tag']:>10} {r['pool'] or '-':>13} {r['n_scored']:>8} "
              f"{r['nll']:>9.4f} {r['ppl']:>10.3f}")

    # An absolute NLL threshold is the wrong test and the first run proved it:
    # on the repeated passage BF16 scores 0.00242 and fake-quant 0.00243, so the
    # corpus cannot resolve the quantization cost that GPQA measures at -2.02 pp.
    # A threshold calibrated in absolute NLL would have called any packed result
    # "agrees" -- including a genuinely broken one.
    #
    # The scale-free statistic is the ratio: fake-quant minus BF16 *is* the cost
    # of quantizing, measured on this corpus with this rotation, so it is the
    # natural unit for whatever packed adds on top. R below 1 means packed adds
    # less than quantization itself; R of 5 means it adds five times more.
    have = {"packed", "fakequant", "bf16"} <= set(by)
    if "packed" in by and "fakequant" in by:
        d = by["packed"]["nll"] - by["fakequant"]["nll"]
        print(f"\npacked - fakequant: dNLL = {d:+.3e}")
    if "fakequant" in by and "bf16" in by:
        q = by["fakequant"]["nll"] - by["bf16"]["nll"]
        print(f"fakequant - bf16:   dNLL = {q:+.3e}  "
              f"(the cost of quantizing at all -- the unit below)")
    if have:
        q = by["fakequant"]["nll"] - by["bf16"]["nll"]
        d = by["packed"]["nll"] - by["fakequant"]["nll"]
        if q <= 0:
            print("\n  quantization measured as free or better on this corpus, "
                  "so it gives no unit and no ratio can be formed. The corpus "
                  "is too easy -- rerun with CORPUS=docs before concluding "
                  "anything about packed.")
        else:
            r = d / q
            print(f"\n  R = (packed - fakequant) / (fakequant - bf16) = {r:.2f}")
            if q < 1e-4:
                print(f"    CAUTION: the unit itself is {q:.1e}, far below the "
                      "-2.02 pp this rotation costs on GPQA, so this corpus "
                      "barely resolves quantization at all. Treat R as a "
                      "direction, not a magnitude, and confirm with "
                      "CORPUS=docs.")
            if r > 0.5:
                print("    -> packed adds real damage on top of quantization. "
                      "The read path is implicated, not exonerated.")
            else:
                print("    -> packed adds little on top of quantization; look "
                      "at trajectory divergence, the window arena's hit rate "
                      "under sampling, or prefill.")


if __name__ == "__main__":
    main()
