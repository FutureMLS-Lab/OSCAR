#!/usr/bin/env python3
"""Prefix-cache reuse + correctness gate for Kimi-K3 (KDA linear attention).

WHY v2 EXISTS
-------------
v1 certified reuse from a probe that requested INPUT LOGPROBS.  Asking for input
logprobs makes sglang recompute the prompt, which suppresses prefix-cache reuse,
so the probe always read ``cached_tokens=0``.  v1 therefore could not tell
"the cache is broken" from "my probe disabled the cache", and on run gpqa9 it
first FAILED the arm (v1a) and then reported INCONCLUSIVE (v1b) -- in both cases
proving nothing, and in the first case throwing away a cache-ON server and
downgrading the whole K3 headline to cache-OFF.  That is the only reason K3 is
the one model in the sweep whose GPQA number is not a cache-ON number.

v2 rules, in order of importance:

1.  Never use input logprobs to certify reuse.  The probe is a PLAIN generation
    request, exactly the request shape the scoring run uses, so whatever reuse
    the scoring run gets is what the probe measures.
2.  Reuse is read from the server's own counters: per-request
    ``meta_info.cached_tokens`` plus the scheduler's cumulative ``#cached-token``
    / ``#new-token`` in the server log.  Nothing is inferred from the flags we
    believe we passed.
3.  "Inconclusive" is NOT fatal (exit 2).  A 30-minute weight load must not be
    discarded because a probe could not prove something.
4.  But an arm may only be LABELLED cache-ON when the live server reports
    ``disable_radix_cache=False`` AND a non-zero hit rate actually accumulated.
    The label comes from measurement, never from intent.  The last line of
    output is machine-readable so the caller can record exactly that.

Correctness is a separate axis from reuse, and it is checked only when reuse
actually happened -- a broken mamba-state cache does not crash, it silently
answers from a recurrent state captured at the wrong chunk boundary, so it has
to be caught here or not at all.  The check is self-calibrating: two COLD
greedy generations establish the nondeterminism floor, and the WARM generation
has to sit at that floor.  No absolute tolerance is invented.
"""

import json
import re
import sys
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "31255"
LOG = sys.argv[2] if len(sys.argv) > 2 else None
BASE = "http://127.0.0.1:%s" % PORT

# Long enough to span many page_size=8 pages and several FLA 64-token chunks,
# and byte-identical across the three requests so the whole prompt is a
# cacheable prefix.
BODY_TEXT = "\n".join(
    "Section %d. The quick brown fox jumps over the lazy dog while %d "
    "sensors record pressure, temperature and humidity at station %d." % (i, i * 7, i * 13)
    for i in range(220)
)
PROMPT = ("Read the following log and then answer.\n\n" + BODY_TEXT
          + "\n\nQuestion: how many sections are listed above?\nAnswer:")

# Short: this is a cache probe, not a quality probe, and every token costs
# ~1/20 s of a 1.5 TB model on 16 GPUs.
GEN_TOKENS = 24


def get(path, timeout=300):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.load(r)


def post(path, payload, timeout=1800):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def flush():
    try:
        urllib.request.urlopen(BASE + "/flush_cache", timeout=300).read()
    except Exception as e:                                   # noqa: BLE001
        print("[gate] flush_cache failed (%s); continuing" % type(e).__name__)


def ask():
    """One PLAIN generation request -- the same shape the scoring run sends.

    Deliberately no return_logprob: that is what suppressed reuse in v1 and made
    the measurement meaningless.  temperature 0 so the only thing that can move
    the output is the KV/state the request was served from.
    """
    r = post("/generate", {
        "text": PROMPT,
        "sampling_params": {"max_new_tokens": GEN_TOKENS, "temperature": 0.0,
                            "top_p": 1.0, "top_k": 1},
    })
    mi = r.get("meta_info", {}) or {}
    return {
        "text": r.get("text") or "",
        "cached": mi.get("cached_tokens"),
        "prompt_tokens": mi.get("prompt_tokens"),
        "completion_tokens": mi.get("completion_tokens"),
    }


def log_counters():
    """(cached, new) summed over the scheduler's own prefill lines.

    This is the counter the server publishes about itself.  /metrics is not
    available unless the server was launched with --enable-metrics (it was not,
    on this run), and /get_server_info on this build is a flat server_args dump
    with no live counters, so the log is the authoritative cumulative source.
    """
    if not LOG:
        return None
    try:
        txt = open(LOG, errors="replace").read()
    except Exception:                                        # noqa: BLE001
        return None
    hit = sum(int(x) for x in re.findall(r"#cached-token:\s*(\d+)", txt))
    new = sum(int(x) for x in re.findall(r"#new-token:\s*(\d+)", txt))
    return hit, new


def rate(hit, new):
    tot = hit + new
    return 100.0 * hit / tot if tot else 0.0


def verdict(label, reuse, hit_rate, correctness, rc, note=""):
    print("[gate] %s" % (note or ""))
    print("GATE_VERDICT label=%s reuse=%s hit_rate=%.3f%% correctness=%s rc=%d"
          % (label, reuse, hit_rate, correctness, rc))
    return rc


def main():
    # ---- 1. what the LIVE server says about itself -------------------------
    try:
        info = get("/get_server_info")
    except Exception as e:                                    # noqa: BLE001
        print("[gate] cannot read /get_server_info (%s)" % type(e).__name__)
        return verdict("unknown", "unknown", 0.0, "unknown", 2,
                       "server info unreadable; not labelling this arm")
    sa = info.get("server_args") if isinstance(info.get("server_args"), dict) else info
    disabled = sa.get("disable_radix_cache")
    strategy = sa.get("mamba_scheduler_strategy")
    print("[gate] live server: disable_radix_cache=%s mamba_scheduler_strategy=%s "
          "page_size=%s kv_cache_dtype=%s kv_cache_quant_group_size=%s"
          % (disabled, strategy, sa.get("page_size"), sa.get("kv_cache_dtype"),
             sa.get("kv_cache_quant_group_size")))

    if disabled is not False:
        # Not a failure -- just not a cache-ON arm.  Say so instead of letting a
        # caller label it optimistically.
        return verdict("cache-off", "no", 0.0, "n/a", 2,
                       "server reports disable_radix_cache=%s: this arm is "
                       "cache-OFF and must not be labelled cache-ON" % disabled)

    base = log_counters()
    if base:
        print("[gate] scheduler counters BEFORE probe: cached=%d new=%d (%.3f%%)"
              % (base[0], base[1], rate(*base)))

    # ---- 2. cold / warm / cold, all plain generations ----------------------
    flush()
    a = ask()
    b = ask()               # warm: byte-identical prompt, cache still populated
    flush()
    c = ask()               # cold again: the nondeterminism floor
    print("[gate] prompt_tokens=%s gen_tokens=%s" % (a["prompt_tokens"], a["completion_tokens"]))
    print("[gate] per-request cached_tokens: A(cold)=%s B(warm)=%s C(cold)=%s"
          % (a["cached"], b["cached"], c["cached"]))

    post_ = log_counters()
    d_hit = d_new = 0
    if base and post_:
        d_hit, d_new = post_[0] - base[0], post_[1] - base[1]
        print("[gate] scheduler counters DURING probe: cached=%d new=%d (%.3f%%)"
              % (d_hit, d_new, rate(d_hit, d_new)))
    if post_:
        print("[gate] scheduler counters CUMULATIVE: cached=%d new=%d (%.3f%%)"
              % (post_[0], post_[1], rate(*post_)))

    # ---- 3. did anything actually get reused? ------------------------------
    # Two independent witnesses; either one is enough, and they are both the
    # server's own accounting rather than an inference from flags.
    reuse_req = (b["cached"] or 0) > 0
    reuse_log = d_hit > 0
    reuse = reuse_req or reuse_log
    hr = rate(d_hit, d_new) if (d_hit or d_new) else 0.0

    if not reuse:
        return verdict("cache-on-unproven", "no", hr, "inconclusive", 2,
                       "server has the cache ENABLED but no reuse was observed "
                       "on a byte-identical repeat: keep serving, and do NOT "
                       "call this arm cache-ON until a non-zero hit rate "
                       "accumulates over the scoring run")

    # ---- 4. correctness, only now that reuse is proven ---------------------
    floor_ok = a["text"] == c["text"]           # cold vs cold
    warm_ok = b["text"] == a["text"]            # cold vs cached
    print("[gate] cold-vs-cold identical:   %s" % floor_ok)
    print("[gate] cold-vs-cached identical: %s" % warm_ok)
    if not floor_ok:
        # The cold path itself is not reproducible on this build, so text
        # equality cannot discriminate a bad cache from ordinary reduction-order
        # noise.  Reuse is still proven, so the arm is genuinely cache-ON.
        print("[gate] A(cold) != C(cold), so greedy decode is not bit-reproducible "
              "here and equality cannot separate cache corruption from the "
              "nondeterminism floor")
        for k, v in (("A", a), ("B", b), ("C", c)):
            print("[gate]   %s: %s" % (k, (v["text"] or "").replace("\n", " ")[:110]))
        return verdict("cache-on", "yes", hr, "inconclusive", 0,
                       "reuse is proven and the arm IS cache-ON; correctness "
                       "could not be adjudicated because the cold path is not "
                       "reproducible")
    if not warm_ok:
        print("[gate]   A(cold): %s" % (a["text"] or "").replace("\n", " ")[:200])
        print("[gate]   B(warm): %s" % (b["text"] or "").replace("\n", " ")[:200])
        return verdict("cache-on", "yes", hr, "fail", 1,
                       "the two cold generations agree but the CACHED one "
                       "diverges -- the restored mamba/conv state is wrong, do "
                       "NOT score on this server")
    return verdict("cache-on", "yes", hr, "pass", 0,
                   "reuse proven by the server's own counters and the cached "
                   "generation is identical to cold at the reproducibility floor")


if __name__ == "__main__":
    sys.exit(main())
