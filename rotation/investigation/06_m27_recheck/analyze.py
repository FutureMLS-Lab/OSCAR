#!/usr/bin/env python3
"""Per-arm metrics for the MiniMax-M2.7 re-check.

Same eight-item report the other models are held to, with two differences that
M2.7 forced:

* The captured CUDA-graph batch sizes are read back from ``cuda_graph_bs=[...]``
  in the server log instead of being hardcoded. M2.7's published config used
  ``--cuda-graph-max-bs 1`` (capture list ``[1]``), so a hardcoded
  ``{1,2,4,8,12}`` would have mislabelled the exposure completely.
* The project-standard "unclosed ``<think>``" count is ``open > close``, and on
  M2.7 that is **structurally always 0**: the opening tag lives in the chat
  template, so the response contains only ``</think>``. Measured on the three
  historical runs: ``has_open`` = 0/198 in all of them, ``has_close`` = 174 /
  160 / 116. The metric that carries the same information here is
  ``no_think_close`` -- responses that never closed their reasoning -- and it
  tracks the answered-count almost exactly (24 vs 24 unanswered on the clean
  arm, 82 vs 77 on the corrupted one). Both are reported; read
  ``no_think_close``, and read it as *excess over the paired clean arm*, since
  its floor is set by the token budget (24/198 at 32K with zero corruption).
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
        no_close = 0
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
            if not THINK_CLOSE.search(r):
                no_close += 1
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
        out["unclosed_think"] = unclosed          # open>close: always 0 on M2.7
        out["no_think_close"] = no_close          # the metric that carries info
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
        dec_graphed = Counter()
        graph = Counter()
        audit = Counter()
        audit_serving = Counter()
        alloc_pages = set()
        alloc_pages_serving = set()
        # CUDA-graph *capture* runs warmup forwards with out_cache_loc filled
        # with hp_global_offset (cuda_graph_runner.py:976), outside the capture
        # region, so the auditor legitimately sees padded writes before the
        # server serves anything -- 32 of them (8 per TP rank, the _report rate
        # limit) on every arm, all at lines before the first Prefill. Gate on
        # the first "Prefill batch" rather than "The server is fired up": the
        # fired-up line is emitted by the launcher process, so on a multi-rank
        # log it can land after the scheduler ranks have already started
        # working, and gating on it silently dropped real serving-phase events.
        serving = False
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
                    # Padding exists only on a graph *replay*. A non-captured
                    # bs with `cuda graph: False` ran eager and padded nothing,
                    # so the two must be counted jointly -- counting
                    # non-captured sizes alone reported the --cuda-graph-max-bs
                    # 1 arm as 99.5% padded when its true exposure is ZERO.
                    dec_graphed[(int(dm.group(1)), dm.group(2))] += 1
                if not serving and "Prefill batch" in line:
                    serving = True
                am = AUDIT.search(line)
                if am:
                    audit[am.group(1)] += 1
                    if serving:
                        audit_serving[am.group(1)] += 1
                hm = HP_ALLOC.search(line)
                if hm:
                    for tok in hm.group(1).split(","):
                        tok = tok.strip()
                        if tok.isdigit():
                            alloc_pages.add(int(tok))
                            if serving:
                                alloc_pages_serving.add(int(tok))
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
        out["decode_steps_eager"] = graph.get("False", 0)
        if captured:
            cs = set(captured)
            nc = sum(c for bs, c in dec.items() if bs not in cs)
            out["decode_steps_noncaptured_bs"] = nc
            # The real exposure: replayed a graph at a size it was not captured
            # at, so the runner padded and the padded locs went to HP-prefix 0.
            padded = sum(
                c for (bs, g), c in dec_graphed.items()
                if g == "True" and bs not in cs
            )
            out["decode_steps_padded_replay"] = padded
            tot = sum(dec.values())
            out["padded_replay_fraction"] = round(padded / tot, 4) if tot else None
            out["noncaptured_but_eager"] = nc - padded
        out["audit_events_all"] = dict(audit)
        out["audit_events_serving"] = dict(audit_serving)
        # Rate limited to _MAX_REPORTS=8 per kind in mixed_kv_audit._report, so
        # these are "did it happen", not a census. Quantitative exposure comes
        # from padded_replay_fraction above.
        out["decode_write_into_hp_prefix"] = audit_serving.get(
            "DECODE_WRITE_INTO_HP_PREFIX", 0
        )
        out["kv_content_changed"] = audit_serving.get("KV_CONTENT_CHANGED", 0)
        if alloc_pages:
            out["hp_prefix_alloc_min_page"] = min(alloc_pages)
            out["hp_prefix_page0_ever_allocated"] = 0 in alloc_pages
            out["hp_prefix_alloc_distinct_pages"] = len(alloc_pages)
            out["hp_prefix_alloc_pages_sample"] = sorted(alloc_pages)[:12]

    if a.json:
        print(json.dumps(out, indent=2))
    else:
        for k, v in out.items():
            print(f"{k:34s} {v}")


if __name__ == "__main__":
    main()
