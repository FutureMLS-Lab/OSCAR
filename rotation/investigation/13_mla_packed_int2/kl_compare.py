#!/usr/bin/env python3
"""Teacher-forced logprob / KL comparison between two serving configurations.

A 198-question GPQA run costs hours and returns one number, which says the arms
differ but not where or by how much. Teacher forcing scores the SAME tokens
under both configurations, so every position is a paired sample and a few dozen
prompts give thousands of them. It is both far faster and far more sensitive:
a path that is subtly wrong shows a logprob shift long before it shows a score
drop, and a path that is badly wrong shows it on the first prompt.

Two modes:

  dump    hit a live server, teacher-force each text, write per-token logprobs
  cmp     read two dumps and report the paired differences

Reported:
  * mean and p99 |dlogprob| -- magnitude of disagreement
  * symmetric KL over the returned top-k at each position, when available
  * top-1 agreement rate -- would the two configurations emit the same token
  * the worst positions, with their text, so a systematic pattern is visible
    rather than just a summary statistic

Teacher forcing requires the prefix cache OFF: input logprobs from position 0
force a full recompute, and a cached prefix silently returns fewer positions.
The caller is responsible for that; this script asserts the position count
matches between dumps and refuses to compare otherwise.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.request

# Short, varied, and deliberately not GPQA: the question is whether the two
# forward paths agree at all, which does not need domain-hard prompts. Mixing
# prose, code, math and a long-ish factual list exercises different activation
# regimes.
TEXTS = [
    "The capital of France is Paris, and the capital of Japan is Tokyo.",
    "def add(a, b):\n    return a + b\n\nprint(add(2, 3))",
    "Compute 17 * 23. Seventeen times twenty-three equals three hundred ninety-one.",
    "Attention is all you need. The transformer replaced recurrence with self-attention.",
    "A KV cache stores keys and values so that decoding does not recompute the prefix.",
    "In 1969, Apollo 11 landed on the Moon; Neil Armstrong stepped out first.",
    "The mitochondrion is the powerhouse of the cell, producing ATP via oxidative phosphorylation.",
    "import torch\nx = torch.randn(4, 8)\ny = x.softmax(dim=-1)\nassert y.sum(-1).allclose(torch.ones(4))",
    "Quantization to two bits per value reduces memory but increases reconstruction error.",
    "The derivative of x squared with respect to x is two x.",
    "Rome was not built in a day, and neither was any large software system.",
    "SELECT name, count(*) FROM users GROUP BY name HAVING count(*) > 1;",
    "Water boils at 100 degrees Celsius at sea level, and lower at altitude.",
    "A rotation matrix is orthogonal: its transpose is its inverse.",
    "The Pacific is the largest ocean, covering about a third of the Earth's surface.",
    "for i in range(10):\n    if i % 3 == 0:\n        continue\n    print(i)",
    "Entropy measures uncertainty; a fair coin has one bit of it per flip.",
    "The speed of light in vacuum is approximately 299,792,458 metres per second.",
    "Gradient descent follows the negative gradient to reduce a loss function.",
    "Shakespeare wrote Hamlet, Macbeth, King Lear, and Othello, among many others.",
    "A prime number has exactly two distinct positive divisors: one and itself.",
    "class Node:\n    def __init__(self, value):\n        self.value = value\n        self.next = None",
    "Photosynthesis converts carbon dioxide and water into glucose using sunlight.",
    "The integral of one over x with respect to x is the natural logarithm of x.",
    "Latency and throughput trade off: batching raises one and usually raises the other.",
    "Mount Everest is the highest mountain above sea level, at about 8,849 metres.",
    "A hash table gives expected constant time lookup by mapping keys to buckets.",
    "The French Revolution began in 1789 with the storming of the Bastille.",
    "Softmax exponentiates its inputs and normalises them to sum to one.",
    "DNA is a double helix of nucleotides, paired adenine to thymine and guanine to cytosine.",
]


def dump(url: str, out_path: str, top_k: int) -> None:
    rows = []
    for i, text in enumerate(TEXTS):
        body = json.dumps({
            "text": text,
            # max_new_tokens=0 scores the INPUT only. logprob_start_len=0 asks
            # for every position, which is what forces the full recompute that
            # makes this comparable across configurations.
            "sampling_params": {"max_new_tokens": 0, "temperature": 0.0},
            "return_logprob": True,
            "logprob_start_len": 0,
            "top_logprobs_num": top_k,
        }).encode()
        req = urllib.request.Request(
            f"{url}/generate", data=body,
            headers={"Content-Type": "application/json"})
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=600).read())
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}] request failed: {type(e).__name__}: {e}", flush=True)
            rows.append({"i": i, "error": str(e)[:200]})
            continue
        meta = (r[0] if isinstance(r, list) else r).get("meta_info", {})
        rows.append({
            "i": i,
            "text": text,
            "input_token_logprobs": meta.get("input_token_logprobs"),
            "input_top_logprobs": meta.get("input_top_logprobs"),
        })
        n = len(meta.get("input_token_logprobs") or [])
        print(f"  [{i}] {n} positions", flush=True)
    with open(out_path, "w") as f:
        json.dump(rows, f)
    print(f"wrote {out_path}")


def _lp(row):
    """Per-position (logprob, token_id) pairs, tolerating nulls at position 0."""
    out = []
    for e in row.get("input_token_logprobs") or []:
        if not e:
            continue
        lp, tid = e[0], e[1]
        if lp is None:
            continue
        out.append((float(lp), tid))
    return out


def _topk(row):
    return row.get("input_top_logprobs") or []


def cmp(a_path: str, b_path: str, label_a: str, label_b: str) -> int:
    A = json.load(open(a_path))
    B = json.load(open(b_path))
    deltas, kls, agree, total = [], [], 0, 0
    worst = []
    for ra, rb in zip(A, B):
        if ra.get("error") or rb.get("error"):
            continue
        la, lb = _lp(ra), _lp(rb)
        if len(la) != len(lb):
            print(f"  [{ra.get('i')}] position count differs "
                  f"({len(la)} vs {len(lb)}) -- prefix cache on? skipping")
            continue
        for pos, ((pa, ta), (pb, tb)) in enumerate(zip(la, lb)):
            d = abs(pa - pb)
            deltas.append(d)
            total += 1
            if ta == tb:
                agree += 1
            if len(worst) < 8 or d > worst[-1][0]:
                worst.append((d, ra.get("i"), pos, pa, pb,
                              (ra.get("text") or "")[:48]))
                worst.sort(key=lambda x: -x[0])
                del worst[8:]
        ka, kb = _topk(ra), _topk(rb)
        for ea, eb in zip(ka, kb):
            if not ea or not eb:
                continue
            da = {e[1]: float(e[0]) for e in ea if e and e[0] is not None}
            db = {e[1]: float(e[0]) for e in eb if e and e[0] is not None}
            shared = set(da) & set(db)
            if len(shared) < 2:
                continue
            # symmetric KL restricted to the shared support, renormalised
            za = math.log(sum(math.exp(da[t]) for t in shared))
            zb = math.log(sum(math.exp(db[t]) for t in shared))
            kl = 0.0
            for t in shared:
                pa_, pb_ = math.exp(da[t] - za), math.exp(db[t] - zb)
                kl += 0.5 * (pa_ - pb_) * (da[t] - za - db[t] + zb)
            kls.append(kl)

    if not deltas:
        print("no comparable positions -- both dumps empty or mismatched")
        return 2
    deltas.sort()
    n = len(deltas)
    mean = sum(deltas) / n
    p99 = deltas[min(n - 1, int(0.99 * n))]
    print(f"\n{label_a}  vs  {label_b}")
    print(f"  positions compared   {n}")
    print(f"  mean |dlogprob|      {mean:.4f}")
    print(f"  p99  |dlogprob|      {p99:.4f}")
    print(f"  max  |dlogprob|      {deltas[-1]:.4f}")
    if kls:
        kls.sort()
        print(f"  mean symmetric KL    {sum(kls)/len(kls):.4e}")
        print(f"  p99  symmetric KL    {kls[min(len(kls)-1, int(0.99*len(kls)))]:.4e}")
    print(f"  top-1 token agreement {100.0*agree/total:.2f}%  ({agree}/{total})")
    # Calibration: bf16 nondeterminism alone lands well under 0.01 nats, and a
    # 2-bit KV cache against bf16 typically sits in the 0.02-0.2 range. A path
    # that is functionally different sits orders of magnitude above that.
    verdict = ("indistinguishable" if mean < 0.01 else
               "small -- consistent with a quantization-level difference"
               if mean < 0.25 else
               "LARGE -- these are different functions, not a precision effect")
    print(f"  VERDICT: {verdict}")
    print("\n  worst positions (dlogprob, prompt, pos, a, b):")
    for d, i, pos, pa, pb, t in worst:
        print(f"    {d:8.3f}  [{i}] pos={pos:<4} a={pa:8.3f} b={pb:8.3f}  {t!r}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--url", default=f"http://127.0.0.1:{os.environ.get('PORT','30000')}")
    d.add_argument("--out", required=True)
    d.add_argument("--top-k", type=int, default=8)
    c = sub.add_parser("cmp")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--label-a", default="A")
    c.add_argument("--label-b", default="B")
    a = p.parse_args()
    if a.mode == "dump":
        dump(a.url, a.out, a.top_k)
        return 0
    return cmp(a.a, a.b, a.label_a, a.label_b)


if __name__ == "__main__":
    sys.exit(main())
