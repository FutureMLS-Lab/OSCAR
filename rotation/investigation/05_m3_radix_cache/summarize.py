#!/usr/bin/env python3
"""Collect the M3 radix-cache A/B into one table.

Layout produced by run_arm.sh + the patched harness:

    out/<arm>/head/server.log          one server lifetime per arm
    out/<arm>/head/seed<N>/{eval.log,metrics.json,io_log.jsonl,runner.log}

Per seed: score, answered (`Answer: [ABCD]`), unclosed <think>/<mm:think>,
runaway (finish_reason == length), empty responses, elapsed.
Per arm:  server config read back from the log (disable_radix_cache, kv dtype,
mixed-KV sink/recent, rotation files, Lloyd-Max, dtype), prefix-cache token hit
rate, decode batch-size mix, auditor events.
"""
import argparse
import json
import re
import statistics
from pathlib import Path

ANSWER = re.compile(r"(?i)Answer\s*:\s*([A-D])")
THINK_OPEN = re.compile(r"<(mm:)?think>")
THINK_CLOSE = re.compile(r"</(mm:)?think>")
PREFILL = re.compile(r"#new-token:\s*(\d+),\s*#cached-token:\s*(\d+)")
DECODE = re.compile(r"Decode batch.*?#running-req:\s*(\d+).*?cuda graph:\s*(True|False)")
FINISH = re.compile(r"'finish_reason':\s*\{'type':\s*'(\w+)'")
CAPTURED_BS = {1, 2, 4, 8, 12}


def scan_server_log(p: Path):
    out = {}
    new = cached = 0
    dec, graph, finish = {}, {}, {}
    finish_seq = []
    audit = {}
    audit_lines = []
    if not p.is_file():
        return out
    head = ""
    with p.open(errors="replace") as f:
        for i, raw in enumerate(f):
            line = raw.replace("\r", "\n")
            if i < 40:
                head += line
            m = PREFILL.search(line)
            if m:
                new += int(m.group(1))
                cached += int(m.group(2))
            m = DECODE.search(line)
            if m:
                bs = int(m.group(1))
                dec[bs] = dec.get(bs, 0) + 1
                graph[m.group(2)] = graph.get(m.group(2), 0) + 1
            m = FINISH.search(line)
            if m:
                finish[m.group(1)] = finish.get(m.group(1), 0) + 1
                finish_seq.append(m.group(1))
            if "[mixed-kv-audit]" in line:
                kind = line.split("[mixed-kv-audit]", 1)[1].strip().split(":")[0]
                audit[kind] = audit.get(kind, 0) + 1
                if len(audit_lines) < 6:
                    audit_lines.append(line.strip()[-260:])
            if "Enable unified mixed KV" in line and "mixed_kv_line" not in out:
                out["mixed_kv_line"] = line.strip()[-200:]
            if "Loaded Oscar rotation" in line:
                out.setdefault("rotation_lines", [])
                s = line.strip()
                if len(out["rotation_lines"]) < 2 and s[-40:] not in "".join(
                    out["rotation_lines"]
                ):
                    out["rotation_lines"].append(s.split("Loaded Oscar rotation")[-1][:160])
            if "KV Cache is allocated" in line and "kv_cache_line" not in out:
                out["kv_cache_line"] = line.strip()[-180:]
    for key in (
        "disable_radix_cache",
        "kv_cache_dtype",
        "dtype",
        "cuda_graph_max_bs",
        "page_size",
        "enable_cache_report",
        "max_running_requests",
        "mem_fraction_static",
        "tp_size",
        "nnodes",
        "context_length",
    ):
        m = re.search(rf"\b{key}=([^,)]+)", head)
        if m:
            out[key] = m.group(1).strip("'")
    tot = new + cached
    out["prefill_new_tokens"] = new
    out["prefill_cached_tokens"] = cached
    out["cache_hit_rate"] = round(cached / tot, 4) if tot else None
    out["decode_bs_hist"] = dict(sorted(dec.items()))
    out["decode_cuda_graph"] = graph
    out["decode_samples_padded_bs"] = sum(
        c for bs, c in dec.items() if bs not in CAPTURED_BS
    )
    out["finish_reasons"] = finish
    out["_finish_seq"] = finish_seq
    out["audit_events"] = audit
    out["decode_write_into_hp_prefix"] = audit.get("DECODE_WRITE_INTO_HP_PREFIX", 0)
    out["audit_samples"] = audit_lines
    return out


def scan_seed(d: Path):
    r = {"seed": d.name}
    m = d / "metrics.json"
    if m.is_file():
        r["score"] = json.loads(m.read_text()).get("score")
    ev = d / "eval.log"
    if ev.is_file():
        mm = re.search(r"\(elapsed:\s*([\d.]+)s\)", ev.read_text())
        if mm:
            r["elapsed_s"] = round(float(mm.group(1)))
    io = d / "io_log.jsonl"
    n = answered = unclosed = empty = 0
    lens = []
    if io.is_file():
        for line in io.open():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = rec.get("response") or ""
            n += 1
            lens.append(len(t))
            if ANSWER.search(t):
                answered += 1
            if len(THINK_OPEN.findall(t)) > len(THINK_CLOSE.findall(t)):
                unclosed += 1
            if not t.strip():
                empty += 1
    r.update(responses=n, answered=answered, unclosed_think=unclosed, empty=empty)
    if lens:
        lens.sort()
        r["chars_median"] = lens[len(lens) // 2]
        r["chars_max"] = lens[-1]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="out/ directory")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    result = {}
    for arm_dir in sorted(root.iterdir()):
        if not arm_dir.is_dir():
            continue
        head = arm_dir / "head"
        if not head.is_dir():
            continue
        arm = scan_server_log(head / "server.log")
        seeds = [scan_seed(d) for d in sorted(head.glob("seed*")) if d.is_dir()]
        # Requests run strictly sequentially per seed, so the finish-reason
        # stream splits cleanly by each seed's response count.
        seq = arm.pop("_finish_seq", [])
        off = 0
        for s in seeds:
            k = s.get("responses", 0)
            chunk = seq[off : off + k]
            off += k
            s["runaway_length"] = sum(1 for x in chunk if x == "length")
            s["finish_reasons"] = {x: chunk.count(x) for x in set(chunk)}
        scores = [s["score"] for s in seeds if s.get("score") is not None]
        if scores:
            arm["score_mean"] = round(statistics.mean(scores) * 100, 2)
            if len(scores) > 1:
                arm["score_sd_pp"] = round(statistics.stdev(scores) * 100, 2)
                arm["score_sem_pp"] = round(
                    statistics.stdev(scores) * 100 / len(scores) ** 0.5, 2
                )
            arm["scores_pp"] = [round(x * 100, 2) for x in scores]
        w = head / "wall_seconds"
        if w.is_file():
            arm["pod_wall_s"] = int(w.read_text().strip() or 0)
        arm["seeds"] = seeds
        result[arm_dir.name] = arm

    if a.json:
        print(json.dumps(result, indent=2))
        return
    for name, arm in result.items():
        print("=" * 78)
        print(f"ARM {name}")
        for k in (
            "disable_radix_cache",
            "kv_cache_dtype",
            "dtype",
            "page_size",
            "cuda_graph_max_bs",
            "max_running_requests",
            "context_length",
            "tp_size",
            "nnodes",
            "mixed_kv_line",
            "rotation_lines",
            "kv_cache_line",
            "cache_hit_rate",
            "prefill_cached_tokens",
            "prefill_new_tokens",
            "decode_cuda_graph",
            "decode_samples_padded_bs",
            "decode_bs_hist",
            "audit_events",
            "decode_write_into_hp_prefix",
            "audit_samples",
            "scores_pp",
            "score_mean",
            "score_sd_pp",
            "score_sem_pp",
            "pod_wall_s",
        ):
            if k in arm:
                print(f"  {k:28s} {arm[k]}")
        for s in arm.get("seeds", []):
            print(
                f"   {s['seed']:>7s} score={s.get('score')} n={s.get('responses')} "
                f"answered={s.get('answered')} unclosed={s.get('unclosed_think')} "
                f"runaway={s.get('runaway_length')} empty={s.get('empty')} "
                f"elapsed={s.get('elapsed_s')}s finish={s.get('finish_reasons')}"
            )


if __name__ == "__main__":
    main()
