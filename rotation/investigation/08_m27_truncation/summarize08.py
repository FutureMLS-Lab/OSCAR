#!/usr/bin/env python3
"""One table per arm: score WITH answered, capped, loops and length together.

A score alone cannot distinguish quality loss from truncation, which is the
entire question on this model, so nothing here prints a score without the
truncation and length columns next to it.

``score`` comes from the arm's own ``metrics.json`` whenever it exists -- that
is the authoritative number -- and falls back to the local re-scorer only for
arms still in flight, marked with a ``~``. The re-scorer agrees with
metrics.json to within 2/198 on the nine complete arms it was checked against.

``capped`` is ``finish_reason == "length"``, straight from the server, and is
split into loops and coherent-but-long by 5-gram multiplicity, so "ran out of
budget while making sense" and "degenerated" are never summed into one number.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import length_repetition as lr
import rescore_gpqa as rs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--variant", default="diamond")
    ap.add_argument("--total", type=int, default=198)
    a = ap.parse_args()

    key = rs.build_key_to_answer(a.variant)
    hdr = ("arm", "score", "n/198", "answered", "capped", "cap_loop",
           "cap_coh", "loops", "words_p50", "words_p90")
    print(("{:<17}" + "{:>9}" * (len(hdr) - 1)).format(*hdr))
    for rd in a.run_dirs:
        d = Path(rd)
        if not (d / "io_log.jsonl").is_file():
            continue
        n = correct = answered = capped = loops = cap_loop = 0
        words: list[int] = []
        for line in (d / "io_log.jsonl").open(errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = ""
            for m in rec.get("messages") or []:
                if isinstance(m, dict) and m.get("role") == "user":
                    prompt = str(m.get("content", ""))
                    break
            gold = key.get(prompt.strip())
            resp = rec.get("response") or ""
            n += 1
            words.append(len(lr.tokens(resp)))
            ms = rs.STRICT.search(resp)
            if ms:
                answered += 1
            if gold is not None and ms and ms.group(1) == gold:
                correct += 1
            rep = lr.repetition(resp, prompt)
            if rep["loop"]:
                loops += 1
            if rec.get("finish_reason") == "length":
                capped += 1
                if rep["loop"]:
                    cap_loop += 1
        mj = d / "metrics.json"
        if mj.is_file():
            score = f"{100 * json.loads(mj.read_text())['score']:.2f}"
        else:
            score = f"~{100 * correct / a.total:.2f}"
        words.sort()
        print(("{:<17}" + "{:>9}" * (len(hdr) - 1)).format(
            d.name[:17], score, f"{n}/{a.total}", answered, capped, cap_loop,
            capped - cap_loop, loops,
            words[len(words) // 2] if words else 0,
            words[9 * len(words) // 10] if words else 0))


if __name__ == "__main__":
    main()
