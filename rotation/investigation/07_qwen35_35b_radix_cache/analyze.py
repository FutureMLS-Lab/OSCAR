#!/usr/bin/env python3
"""Per-seed metrics for the Qwen3.5-35B-A3B prefix-cache A/B.

One seed directory (io_log.jsonl, metrics.json, eval.log, server.log,
wall_seconds) in, one JSON row out.

Derived from investigation/06_qwen35_radix_cache/analyze.py (the 4B A/B) with
three corrections that the 4B run and the M2.7 re-check between them forced:

1. **CUDA-graph exposure is a joint condition.** The 4B analyzer counted
   ``decode_bs_hist`` and ``decode_cuda_graph`` in two independent counters and
   called "bs not in captured" the padded-replay count. That overcounts: a
   non-captured batch size running *eager* pads nothing. Exposure is
   ``cuda graph: True`` AND ``bs not in captured`` on the *same* line, so the
   histogram here is keyed on the pair. ``noncaptured_but_eager`` is reported
   separately so the two populations stay visible.

2. **The captured list is read from ``Capture cuda graph bs [...]``**, the list
   the runner actually captured, not from the ``cuda_graph_bs=[...]`` server-arg
   echo. They differ: ``get_batch_sizes_to_capture`` appends
   ``req_to_token_pool.size`` (which tracks ``--max-running-requests``) when it
   exceeds the generated list's max, so a probe concurrency chosen against the
   arg echo can silently be a captured size.

3. **"Runaway" is defined here, not read from a field.** ``io_log.jsonl`` has no
   ``finish_reason`` (keys are max_tokens/messages/model/response/temperature/
   top_k/top_p), so the usual ``finish_reason == "length"`` count is
   structurally 0 and means nothing. A runaway is a response that never emitted
   ``Answer:`` *and* ran to >= ``RUNAWAY_CHARS`` characters -- i.e. it spent the
   whole budget without converging. Both halves are required: a short
   unanswered response is a refusal or a parse miss, not a runaway.

Also note ``unclosed_think``. The project-standard count is ``open > close``,
and on Qwen3.5-35B-A3B that is structurally always 0 because the opening tag
comes from the chat template and never appears in the response: measured
0/594 ``<think>`` against 594/594 ``</think>`` on the cache-off arm. The metric
carrying the same information is ``no_think_close``. Both are reported; read
``no_think_close``. (The 4B sibling differs -- it emitted *neither* tag in 1782
responses -- so do not carry a conclusion about this column across the two.)
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

#: A response with no ``Answer:`` that reached this many characters is counted a
#: runaway. Matches the 4B A/B so the two models' columns are comparable.
RUNAWAY_CHARS = 80000

ANSWER = re.compile(r"(?i)Answer\s*:\s*\**\s*([A-D])\b")
THINK_OPEN = re.compile(r"<think>")
THINK_CLOSE = re.compile(r"</think>")

PREFILL = re.compile(r"#new-token:\s*(\d+),\s*#cached-token:\s*(\d+)")
DECODE = re.compile(r"Decode batch.*?#running-req:\s*(\d+).*?cuda graph:\s*(True|False)")
CAPTURE_BS = re.compile(r"Capture cuda graph bs \[([0-9,\s]*)\]")
ARG_GRAPH_BS = re.compile(r"cuda_graph_bs=\[([0-9,\s]*)\]")
RADIX = re.compile(r"disable_radix_cache=(True|False)")
GROUP = re.compile(r"kv_cache_quant_group_size=(\d+)")
STRATEGY = re.compile(r"mamba_scheduler_strategy='([^']*)'")
OVERLAP = re.compile(r"disable_overlap_schedule=(True|False)")
PAGE = re.compile(r"page_size=(\d+)")
# Two wordings in the tree: the plain unified pool logs "Enable unified mixed
# KV (int2): prefix=P recent=R"; the hybrid/mamba path logs "Enable hybrid mixed
# KV (int2) for mambaish model: full_attn_layers=N prefix=P recent=R N_Q=8".
# Qwen3.5-35B-A3B takes the second, so a pattern anchored on the first silently
# reports no windows at all.
WINDOWS = re.compile(r"mixed KV \(int2\).*?prefix=(\d+) recent=(\d+)")
FULL_ATTN = re.compile(r"full_attn_layers=(\d+)")
ROT = re.compile(r"Loaded Oscar rotation from (\S+) for layers (\[[^\]]*\])")
CLIP = re.compile(r"Oscar rotation enabled \(k_clip=([\d.]+) v_clip=([\d.]+) lloyd_max=(\w+)\)")
HP_ALLOC = re.compile(r"HP_PREFIX_ALLOC pages=\[([^\]]*)\]")
AUDIT = re.compile(r"\[mixed-kv-audit\]\s*([A-Z_]+)")
MIXIN = re.compile(r"\[mixed-kv-prefix\]\s*(\w+): enabled=(\w+) hp_prefix_tokens=(\d+)")
TIERCAP = re.compile(r"tier cap #\d+ truncated match (\d+) -> (\d+)")


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

    ev = d / "eval.log"
    if ev.is_file():
        mm = re.search(r"\(elapsed:\s*([\d.]+)s\)", ev.read_text())
        if mm:
            out["eval_elapsed_s"] = float(mm.group(1))

    w = d / "wall_seconds"
    if w.is_file():
        out["seed_wall_s"] = int(w.read_text().strip() or 0)

    obs = d / "observed_disable_radix_cache"
    if obs.is_file():
        out["observed_disable_radix_cache"] = obs.read_text().strip()

    io = d / "io_log.jsonl"
    if io.is_file():
        n = answered = unclosed = no_close = empty = runaway = 0
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
            n += 1
            lens.append(len(r))
            has_ans = bool(ANSWER.search(r))
            if has_ans:
                answered += 1
            if len(THINK_OPEN.findall(r)) > len(THINK_CLOSE.findall(r)):
                unclosed += 1
            if not THINK_CLOSE.search(r):
                no_close += 1
            if not r.strip():
                empty += 1
            if (not has_ans) and len(r) >= RUNAWAY_CHARS:
                runaway += 1
        out["responses"] = n
        out["answered"] = answered
        out["unanswered"] = n - answered
        out["unclosed_think"] = unclosed
        out["no_think_close"] = no_close
        out["empty_responses"] = empty
        out["runaway"] = runaway
        out["runaway_definition"] = f"no Answer: and >= {RUNAWAY_CHARS} chars"
        if lens:
            lens.sort()
            out["resp_chars_median"] = lens[len(lens) // 2]
            out["resp_chars_max"] = lens[-1]

    srv = d / "server.log"
    if srv.is_file():
        new = cached = 0
        dec = Counter()          # bs -> steps
        graph = Counter()        # "True"/"False" -> steps
        dec_graphed = Counter()  # (bs, graph) -> steps   <- the joint one
        audit = Counter()
        alloc_pages = set()
        rot_lines, mixin_lines, tiercaps = [], [], []
        capture_bs = arg_graph_bs = None
        radix_flag = group_size = strategy = overlap = None
        pages = set()
        with srv.open(errors="replace") as f:
            for line in f:
                pm = PREFILL.search(line)
                if pm:
                    new += int(pm.group(1))
                    cached += int(pm.group(2))
                dm = DECODE.search(line)
                if dm:
                    bs, g = int(dm.group(1)), dm.group(2)
                    dec[bs] += 1
                    graph[g] += 1
                    dec_graphed[(bs, g)] += 1
                if capture_bs is None:
                    cm = CAPTURE_BS.search(line)
                    if cm:
                        capture_bs = sorted(
                            int(x) for x in cm.group(1).split(",") if x.strip()
                        )
                if arg_graph_bs is None:
                    am = ARG_GRAPH_BS.search(line)
                    if am:
                        arg_graph_bs = sorted(
                            int(x) for x in am.group(1).split(",") if x.strip()
                        )
                if radix_flag is None:
                    rm = RADIX.search(line)
                    if rm:
                        radix_flag = rm.group(1)
                if group_size is None:
                    gm = GROUP.search(line)
                    if gm:
                        group_size = int(gm.group(1))
                if strategy is None:
                    sm = STRATEGY.search(line)
                    if sm:
                        strategy = sm.group(1)
                if overlap is None:
                    om = OVERLAP.search(line)
                    if om:
                        overlap = om.group(1)
                for pg in PAGE.findall(line):
                    pages.add(int(pg))
                wm = WINDOWS.search(line)
                if wm:
                    out["hp_prefix_tokens"] = int(wm.group(1))
                    out["hp_recent_tokens"] = int(wm.group(2))
                    fa = FULL_ATTN.search(line)
                    if fa:
                        out["full_attn_layers"] = int(fa.group(1))
                cm2 = CLIP.search(line)
                if cm2:
                    out["k_clip"], out["v_clip"], out["lloyd_max"] = cm2.groups()
                rl = ROT.search(line)
                if rl and len(rot_lines) < 4:
                    rot_lines.append([rl.group(1).rsplit("/", 1)[-1], rl.group(2)])
                ml = MIXIN.search(line)
                if ml and len(mixin_lines) < 4:
                    mixin_lines.append(list(ml.groups()))
                tc = TIERCAP.search(line)
                if tc and len(tiercaps) < 8:
                    tiercaps.append([int(tc.group(1)), int(tc.group(2))])
                al = AUDIT.search(line)
                if al:
                    audit[al.group(1)] += 1
                ha = HP_ALLOC.search(line)
                if ha:
                    for x in ha.group(1).split(","):
                        x = x.strip()
                        if x:
                            alloc_pages.add(int(x))

        tot = new + cached
        out["server_disable_radix_cache"] = radix_flag
        out["server_quant_group_size"] = group_size
        out["mamba_scheduler_strategy"] = strategy
        out["disable_overlap_schedule"] = overlap
        out["page_sizes_seen"] = sorted(pages)
        out["rotation_files"] = rot_lines
        out["mixin_report"] = mixin_lines
        out["tier_cap_examples"] = tiercaps
        out["prefill_new_tokens"] = new
        out["prefill_cached_tokens"] = cached
        out["cache_hit_rate"] = round(cached / tot, 6) if tot else None
        out["captured_bs"] = capture_bs
        out["arg_cuda_graph_bs"] = arg_graph_bs
        out["decode_bs_hist"] = dict(sorted(dec.items()))
        out["decode_steps"] = sum(dec.values())
        out["decode_cuda_graph"] = dict(graph)
        if capture_bs:
            cs = set(capture_bs)
            nc = sum(c for bs, c in dec.items() if bs not in cs)
            out["decode_steps_noncaptured_bs"] = nc
            # The real exposure: a graph *replayed* at a size it was not
            # captured at, so the runner padded, and the padded out_cache_loc
            # entries point at HP-prefix slot 0.
            padded = sum(
                c for (bs, g), c in dec_graphed.items()
                if g == "True" and bs not in cs
            )
            out["decode_steps_padded_replay"] = padded
            st = sum(dec.values())
            out["padded_replay_fraction"] = round(padded / st, 4) if st else None
            out["noncaptured_but_eager"] = nc - padded
            out["probe_bs_was_captured"] = None  # filled by the caller if known
        out["audit_events"] = dict(audit)
        # Rate limited to _MAX_REPORTS=8 per kind in mixed_kv_audit._report, so
        # these are "did it happen", not a census.
        out["audit_violations"] = {
            k: v for k, v in audit.items()
            if not k.startswith("HP_PREFIX_")
        }
        out["audit_violation_total"] = sum(out["audit_violations"].values())
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
