"""Re-score finished HumanEval runs from their saved io_log.jsonl.

Generation is the expensive part and it is already on disk, so a broken grader
does not require new GPU time. Applies humaneval_eval.find_code to every saved
response, re-executes the tests, and writes metrics_rescored.json next to the
original metrics.json (left untouched so the discrepancy stays visible).

Usage: python rescore_humaneval.py <out_root>   # e.g. .../07_perhead_bench/out
"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "third_party"))
sys.path.insert(0, HERE)

from humaneval_eval import find_code  # noqa: E402
from human_eval.data import read_problems  # noqa: E402
from human_eval.execution import check_correctness  # noqa: E402

PROBLEMS = list(read_problems().values())


def _match(messages):
    txt = messages[0]["content"] if isinstance(messages, list) else str(messages)
    for p in PROBLEMS:
        key = p["prompt"].strip()[:120]
        if key and key in txt:
            return p
    return None


def rescore(io_log):
    rows = [json.loads(l) for l in open(io_log)]
    passed = matched = 0
    for r in rows:
        p = _match(r["messages"])
        if p is None:
            continue
        matched += 1
        passed += int(check_correctness(p, find_code(r["response"]), 6.0)["passed"])
    return matched, passed


def main(root):
    found = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        if "io_log.jsonl" not in filenames or os.sep + "humaneval" not in dirpath:
            continue
        found += 1
        io_log = os.path.join(dirpath, "io_log.jsonl")
        matched, passed = rescore(io_log)
        if not matched:
            print(f"{dirpath}: no problems matched, skipped")
            continue
        score = passed / matched
        old = None
        mpath = os.path.join(dirpath, "metrics.json")
        if os.path.exists(mpath):
            try:
                old = json.load(open(mpath)).get("score")
            except Exception:
                pass
        json.dump({"score": score, "pass@1": score, "n": matched,
                   "score_as_originally_graded": old,
                   "note": "re-scored with humaneval_eval.find_code"},
                  open(os.path.join(dirpath, "metrics_rescored.json"), "w"), indent=1)
        rel = os.path.relpath(dirpath, root)
        print(f"{rel}: n={matched} rescored={score:.4f}"
              + (f" (was {old:.4f})" if isinstance(old, float) else ""))
    if not found:
        print(f"no humaneval io_log.jsonl found under {root}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
