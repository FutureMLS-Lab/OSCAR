import argparse, json, urllib.request, time
AP = argparse.ArgumentParser()
AP.add_argument("--port", type=int, required=True)
AP.add_argument("--out", required=True)
AP.add_argument("--tag", default="sweep")
A = AP.parse_args()
URL = "http://127.0.0.1:%d/v1/chat/completions" % A.port

def call(content, mt=80):
    body = json.dumps({"model": "q3", "messages": [{"role": "user", "content": content}],
                       "max_tokens": mt, "temperature": 0}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=300))
            return r["choices"][0]["message"]["content"] or ""
        except Exception as e:
            time.sleep(4)
            if attempt == 2:
                return "<FAILED %s>" % e

def alpha_frac(t):
    body = t.replace("<think>", "")[:400].strip()
    if not body:
        return 0.0
    return round(sum(c.isalpha() for c in body) / len(body), 3)

res = {}
for nwords in (4, 60, 140, 220, 300, 600, 2000):
    filler = " ".join("note %d says %d." % (i, i % 7) for i in range(nwords))
    p = (filler + " " if nwords > 4 else "") + "What is 3+4? Reply briefly."
    approx = int(len(p.split()) * 1.35)
    out = call(p)
    res[nwords] = {"approx_prompt_tok": approx, "alpha_frac": alpha_frac(out),
                   "head": out[:70].replace("\n", " ")}
    print("prompt~%5d tok: alpha=%.3f | %s" % (approx, res[nwords]["alpha_frac"], res[nwords]["head"]))

json.dump(res, open("%s/sweep_%s.json" % (A.out, A.tag), "w"), indent=1)
