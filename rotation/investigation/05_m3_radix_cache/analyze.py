#!/usr/bin/env python3
"""Per-arm metrics for the M3 radix-cache A/B.

Reads one arm's head RUN_DIR (io_log.jsonl, metrics.json, eval.log, server.log)
and prints the row: score, answered, unclosed-think, runaway, cache hit rate,
decode batch-size mix (evidence that CUDA-graph padded replays happened) and
the mixed-KV auditor verdict.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

ANSWER = re.compile(r"(?i)Answer\s*:\s*([A-D])")
# M3 emits <think>; the VL build also uses <mm:think>. Count both families.
THINK_OPEN = re.compile(r"<(mm:)?think>")
THINK_CLOSE = re.compile(r"</(mm:)?think>")

PREFILL = re.compile(r"#new-token:\s*(\d+),\s*#cached-token:\s*(\d+)")
DECODE = re.compile(r"Decode batch.*?#running-req:\s*(\d+).*?cuda graph:\s*(True|False)")
FINISH_LEN = re.compile(r"'finish_reason':\s*\{'type':\s*'length'")
FINISH_ANY = re.compile(r"'finish_reason':\s*\{'type':\s*'(\w+)'")
CAPTURED = re.compile(r"Capture cuda graph.*")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--arm", default="")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    d = Path(a.run_dir)
    out = {"arm": a.arm or d.parent.name, "run_dir": str(d)}

    m = d / "metrics.json"
    if m.is_file():
        met = json.loads(m.read_text())
        out["score"] = met.get("score")
        out["chars"] = met.get("chars")

    ev = d / "eval.log"
    if ev.is_file():
        t = ev.read_text()
        mm = re.search(r"\(elapsed:\s*([\d.]+)s\)", t)
        if mm:
            out["eval_elapsed_s"] = float(mm.group(1))

    w = d / "wall_seconds"
    if w.is_file():
        out["pod_wall_s"] = int(w.read_text().strip() or 0)

    io = d / "io_log.jsonl"
    if io.is_file():
        n = answered = unclosed = empty = 0
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
            if ANSWER.search(r):
                answered += 1
            if len(THINK_OPEN.findall(r)) > len(THINK_CLOSE.findall(r)):
                unclosed += 1
            if len(r.strip()) == 0:
                empty += 1
        out["responses"] = n
        out["answered"] = answered
        out["unclosed_think"] = unclosed
        out["empty_responses"] = empty
        if lens:
            lens.sort()
            out["resp_chars_median"] = lens[len(lens) // 2]
            out["resp_chars_max"] = lens[-1]

    srv = d / "server.log"
    if srv.is_file():
        new = cached = 0
        dec = Counter()
        graph = Counter()
        finish = Counter()
        audit_events = Counter()
        audit_lines = []
        capt = None
        with srv.open(errors="replace") as f:
            for line in f:
                pm = PREFILL.search(line)
                if pm:
                    new += int(pm.group(1))
                    cached += int(pm.group(2))
                dm = DECODE.search(line)
                if dm:
                    dec[int(dm.group(1))] += 1
                    graph[dm.group(2)] += 1
                fm = FINISH_ANY.search(line)
                if fm:
                    finish[fm.group(1)] += 1
                if "[mixed-kv-audit]" in line:
                    kind = line.split("[mixed-kv-audit]", 1)[1].strip().split(":")[0]
                    audit_events[kind] += 1
                    if len(audit_lines) < 8:
                        audit_lines.append(line.strip()[-300:])
                if "Capture cuda graph bs" in line and capt is None:
                    capt = line.strip()[-160:]
        tot = new + cached
        out["prefill_new_tokens"] = new
        out["prefill_cached_tokens"] = cached
        out["cache_hit_rate"] = round(cached / tot, 4) if tot else None
        out["decode_bs_hist"] = dict(sorted(dec.items()))
        out["decode_cuda_graph"] = dict(graph)
        # capture list for cuda_graph_max_bs=8 is [1,2,4,8]; anything else is a
        # padded replay (or eager if graph:False).
        cap = {1, 2, 4, 8, 12}
        out["decode_samples_noncaptured_bs"] = sum(
            c for bs, c in dec.items() if bs not in cap
        )
        out["finish_reasons"] = dict(finish)
        out["runaway_length_finish"] = finish.get("length", 0)
        out["audit_events"] = dict(audit_events)
        out["decode_write_into_hp_prefix"] = audit_events.get(
            "DECODE_WRITE_INTO_HP_PREFIX", 0
        )
        out["audit_samples"] = audit_lines
        out["radix_disabled_flag"] = "disable_radix_cache=True" in srv.read_text()[:200000]

    if a.json:
        print(json.dumps(out, indent=2))
    else:
        for k, v in out.items():
            print(f"{k:34s} {v}")


if __name__ == "__main__":
    main()
