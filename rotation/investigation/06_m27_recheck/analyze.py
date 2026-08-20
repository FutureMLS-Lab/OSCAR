#!/usr/bin/env python3
"""Per-arm metrics for the MiniMax-M2.7 re-check.

Same eight-item report the other models are held to, with two differences that
M2.7 forced:

* The captured CUDA-graph batch sizes are read back from ``cuda_graph_bs=[...]``
  in the server log instead of being hardcoded. M2.7's published config used
  ``--cuda-graph-max-bs 1`` (capture list ``[1]``), so a hardcoded
  ``{1,2,4,8,12}`` would have mislabelled the exposure completely.
* M2.7 does not emit ``<think>`` tags in its API response at all, so an
  "unclosed think" count of 0 is structural, not evidence. The column reports
  how many responses contained any think tag so the metric can be read
  honestly, and adds a repetition/runaway proxy that does not depend on tags.
"""
import argparse
import json
import re
import zlib
from collections import Counter
from pathlib import Path

ANSWER = re.compile(r"(?i)Answer\s*:\s*\**\s*([A-D])\b")
THINK_OPEN = re.compile(r"<(mm:)?think>")
THINK_CLOSE = re.compile(r"</(mm:)?think>")

PREFILL = re.compile(r"#new-token:\s*(\d+),\s*#cached-token:\s*(\d+)")
DECODE = re.compile(r"Decode batch.*?#running-req:\s*(\d+).*?cuda graph:\s*(True|False)")
GRAPH_BS = re.compile(r"cuda_graph_bs=\[([0-9,\s]*)\]")
RADIX = re.compile(r"disable_radix_cache=(True|False)")
MAXBS = re.compile(r"cuda_graph_max_bs=(\d+|None)")
WINDOWS = re.compile(r"Enable unified mixed KV \(int2\): prefix=(\d+) recent=(\d+)")
ROT = re.compile(r"Loaded Oscar rotation from (\S+)")
CLIP = re.compile(r"Oscar rotation enabled \(k_clip=([\d.]+) v_clip=([\d.]+) lloyd_max=(\w+)\)")
HP_ALLOC = re.compile(r"HP_PREFIX_ALLOC.*?pages=\[([^\]]*)\]")
AUDIT = re.compile(r"\[mixed-kv-audit\]\s*([A-Z_]+)")


def compressibility(text: str) -> float:
    """Ratio of zlib-compressed to raw size on the response tail.

    Verbatim repetition -- the observed bug-1 failure mode -- compresses far
    better than prose. Low value = repetitive.
    """
    tail = text[-8000:].encode("utf-8", "replace")
    if len(tail) < 512:
        return 1.0
    return len(zlib.compress(tail, 6)) / len(tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--arm", default="")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    d = Path(a.run_dir)
    out = {"arm": a.arm or d.name, "run_dir": str(d)}

    m = d / "metrics.json"
    if m.is_file():
        met = json.loads(m.read_text())
        out["score"] = met.get("score")
        out["chars"] = met.get("chars")

    ev = d / "eval.log"
    if ev.is_file():
        mm = re.search(r"\(elapsed:\s*([\d.]+)s\)", ev.read_text())
        if mm:
            out["eval_elapsed_s"] = float(mm.group(1))

    io = d / "io_log.jsonl"
    if io.is_file():
        n = answered = unclosed = any_think = empty = repetitive = 0
        cap_hits = 0
        max_tokens = None
        lens = []
        for line in io.open():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            r = rec.get("response") or ""
            max_tokens = rec.get("max_tokens", max_tokens)
            n += 1
            lens.append(len(r))
            if ANSWER.search(r):
                answered += 1
            if THINK_OPEN.search(r) or THINK_CLOSE.search(r):
                any_think += 1
            if len(THINK_OPEN.findall(r)) > len(THINK_CLOSE.findall(r)):
                unclosed += 1
            if not r.strip():
                empty += 1
            if compressibility(r) < 0.12:
                repetitive += 1
            # Runaway proxy: no extractable answer and a very long response.
            # 2.5 chars/token is conservative for this tokenizer, so a response
            # past 2.5*max_tokens chars cannot have stopped early.
            if max_tokens and not ANSWER.search(r) and len(r) > 2.0 * max_tokens:
                cap_hits += 1
        out["responses"] = n
        out["answered"] = answered
        out["max_tokens"] = max_tokens
        out["responses_with_think_tag"] = any_think
        out["unclosed_think"] = unclosed
        out["empty_responses"] = empty
        out["repetitive_tail"] = repetitive
        out["runaway_unanswered_long"] = cap_hits
        if lens:
            lens.sort()
            out["resp_chars_median"] = lens[len(lens) // 2]
            out["resp_chars_max"] = lens[-1]

    srv = d / "server.log"
    if srv.is_file():
        new = cached = 0
        dec = Counter()
        graph = Counter()
        audit = Counter()
        alloc_pages = set()
        captured = None
        head = []
        with srv.open(errors="replace") as f:
            for i, line in enumerate(f):
                if i < 4000:
                    head.append(line)
                pm = PREFILL.search(line)
                if pm:
                    new += int(pm.group(1))
                    cached += int(pm.group(2))
                dm = DECODE.search(line)
                if dm:
                    dec[int(dm.group(1))] += 1
                    graph[dm.group(2)] += 1
                am = AUDIT.search(line)
                if am:
                    audit[am.group(1)] += 1
                hm = HP_ALLOC.search(line)
                if hm:
                    for tok in hm.group(1).split(","):
                        tok = tok.strip()
                        if tok.isdigit():
                            alloc_pages.add(int(tok))
                if captured is None:
                    gm = GRAPH_BS.search(line)
                    if gm:
                        captured = sorted(
                            int(x) for x in gm.group(1).split(",") if x.strip()
                        )
        blob = "".join(head)
        for rx, key in (
            (RADIX, "disable_radix_cache"),
            (MAXBS, "cuda_graph_max_bs"),
        ):
            mm = rx.search(blob)
            out[key] = mm.group(1) if mm else None
        mm = WINDOWS.search(blob)
        if mm:
            out["mixed_kv_prefix_tokens"] = int(mm.group(1))
            out["mixed_kv_recent_tokens"] = int(mm.group(2))
        mm = CLIP.search(blob)
        if mm:
            out["k_clip"], out["v_clip"], out["lloyd_max"] = mm.groups()
        mm = ROT.search(blob)
        if mm:
            out["rotation"] = mm.group(1)

        tot = new + cached
        out["prefill_new_tokens"] = new
        out["prefill_cached_tokens"] = cached
        out["cache_hit_rate"] = round(cached / tot, 4) if tot else None
        out["captured_bs"] = captured
        out["decode_bs_hist"] = dict(sorted(dec.items()))
        out["decode_steps"] = sum(dec.values())
        out["decode_cuda_graph"] = dict(graph)
        if captured:
            cs = set(captured)
            nc = sum(c for bs, c in dec.items() if bs not in cs)
            out["decode_steps_noncaptured_bs"] = nc
            out["padded_replay_fraction"] = (
                round(nc / sum(dec.values()), 4) if dec else None
            )
        out["audit_events"] = dict(audit)
        out["decode_write_into_hp_prefix"] = audit.get("DECODE_WRITE_INTO_HP_PREFIX", 0)
        out["kv_content_changed"] = audit.get("KV_CONTENT_CHANGED", 0)
        if alloc_pages:
            out["hp_prefix_alloc_min_page"] = min(alloc_pages)
            out["hp_prefix_page0_ever_allocated"] = 0 in alloc_pages
            out["hp_prefix_alloc_distinct_pages"] = len(alloc_pages)

    if a.json:
        print(json.dumps(out, indent=2))
    else:
        for k, v in out.items():
            print(f"{k:34s} {v}")


if __name__ == "__main__":
    main()
