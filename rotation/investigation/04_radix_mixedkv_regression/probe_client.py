"""Probe: sequential prefix-hit + concurrent repeats against one server arm.

Degeneration score per completion = max fraction of the tail occupied by a
repeating short n-gram (simple loop detector) + emptiness/</think> checks.
"""
import argparse, json, concurrent.futures as cf
import urllib.request

AP = argparse.ArgumentParser()
AP.add_argument("--port", type=int, required=True)
AP.add_argument("--arm", required=True)
AP.add_argument("--out", required=True)
A = AP.parse_args()

URL = f"http://127.0.0.1:{A.port}/v1/chat/completions"

# ~2.6k-token prompt: filler well beyond sink64+recent256 so the shared bulk is INT2.
FILLER = " ".join(
    f"Fact {i}: the {i}-th sample in the calibration suite measured a latency of {i*7%97} ms and a residual of 0.{i%9}{i%7}."
    for i in range(400)
)
PROBLEM = (
    "Compute the sum of the first 40 positive integers, then subtract 20. "
    "Please reason step by step, and put your final answer within \\boxed{}."
)
PROMPT = f"Context notes:\n{FILLER}\n\nQuestion: {PROBLEM}"


def call(seed: int, max_tokens: int = 2048) -> str:
    body = json.dumps({
        "model": "q3",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.6, "top_p": 0.95,
        "seed": seed,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["message"]["content"] or ""


def degen_score(text: str) -> float:
    """Fraction of the last 1500 chars consumed by a repeating 3-40 char unit."""
    tail = text[-1500:]
    if len(tail) < 60:
        return 0.0
    best = 0.0
    for n in (3, 5, 8, 12, 20, 40):
        counts = {}
        for i in range(0, len(tail) - n, n):
            g = tail[i:i + n]
            counts[g] = counts.get(g, 0) + 1
        if counts:
            top = max(counts.values())
            best = max(best, top * n / len(tail))
    return round(best, 3)


results = {}

# 0) divergent-prefix: share ~2.6k-token header, diverge mid-page with different questions.
#    Sequential first (commit shared prefix), then 4 concurrent divergent extensions.
def divergent_prompt(i: int) -> str:
    return (f"Context notes:\n{FILLER}\n\nQuestion {i}: Compute the sum of the first "
            f"{30+i} positive integers, then subtract {i}. "
            "Please reason step by step, and put your final answer within \\boxed{}.")

def call_prompt(prompt: str, seed: int, max_tokens: int = 2048) -> str:
    body = json.dumps({"model": "q3", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.6, "top_p": 0.95, "seed": seed}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["message"]["content"] or ""

seed_txt = call_prompt(divergent_prompt(0), seed=2)          # commit shared header
with cf.ThreadPoolExecutor(max_workers=4) as ex:
    div_outs = list(ex.map(lambda i: call_prompt(divergent_prompt(i), seed=20+i), range(1, 5)))
recheck = call_prompt(divergent_prompt(0), seed=2)            # re-read the shared page after others wrote
results["divergent_seed"] = {"degen": degen_score(seed_txt), "len": len(seed_txt), "boxed": "\\boxed" in seed_txt}
results["divergent"] = [{"degen": degen_score(t), "len": len(t), "boxed": "\\boxed" in t} for t in div_outs]
results["divergent_recheck"] = {"degen": degen_score(recheck), "len": len(recheck), "boxed": "\\boxed" in recheck}

# 1) sequential: first request (cold) then same prompt again (radix hit if enabled)
first = call(seed=1)
second = call(seed=1)
results["seq_cold"] = {"degen": degen_score(first), "len": len(first), "boxed": "\\boxed" in first}
results["seq_hit"] = {"degen": degen_score(second), "len": len(second), "boxed": "\\boxed" in second}

# 2) concurrent repeats (N_REPEATS style): 5 at once, same prompt, different seeds
with cf.ThreadPoolExecutor(max_workers=5) as ex:
    outs = list(ex.map(lambda s: call(seed=s), range(10, 15)))
results["concurrent"] = [
    {"degen": degen_score(t), "len": len(t), "boxed": "\\boxed" in t} for t in outs
]

with open(f"{A.out}/probe_{A.arm}.json", "w") as f:
    json.dump(results, f, indent=1)
for i, t in enumerate([first, second] + outs):
    with open(f"{A.out}/text_{A.arm}_{i}.txt", "w") as f:
        f.write(t)

bad = sum(1 for r in results["concurrent"] if r["degen"] > 0.5 or not r["boxed"])
print(f"[{A.arm}] seq cold/hit degen={results['seq_cold']['degen']}/{results['seq_hit']['degen']} "
      f"boxed={results['seq_cold']['boxed']}/{results['seq_hit']['boxed']} | "
      f"concurrent bad={bad}/5 degens={[r['degen'] for r in results['concurrent']]}")
