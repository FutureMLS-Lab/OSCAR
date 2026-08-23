import collections, json, re, sys, urllib.request
port, out, arm = sys.argv[1], sys.argv[2], sys.argv[3]
filler = " ".join("note %d says %d." % (i, i % 7) for i in range(600))   # ~2.5k tok -> real quant tier
prompts = [("short", "What is 3+4? Reply briefly.", "7"),
           ("long",  filler + " Now: compute the sum of the first 40 positive integers, then subtract 20. Put the final answer in \\boxed{}.", "800")]


def judge(t, prompt="", expect=None):
    """Detect the failure modes INT2 KV actually produces: digit/symbol soup and
    repetition loops. A letter-ratio threshold cannot do this -- markdown-heavy
    but coherent replies ('**Analyze the Request:**') score as low as garbage.

    Two traps this has already fallen into, hence the care:
      * scoring the request-failure string itself as fluent prose;
      * measuring repetition over letters only, so a model faithfully
        enumerating a repetitive prompt ('note 7 says 0, note 8 says 1') reads
        as 'note says note says ...' -- a fake loop. Numbers are kept as
        tokens, and n-grams the prompt already contains are not counted.
    """
    if t.startswith("<FAILED") or not t.strip():
        return False, {"reason": "no usable response"}
    # A terse correct answer is a pass, and no prose metric applies to it:
    # Gemma-4-12B-it is not a thinking model and answers "What is 3+4? Reply
    # briefly." with "7", which has zero words to measure.
    if expect and expect in t and len(t.strip()) <= 40:
        return True, {"terse_correct": True, "text_len": len(t)}
    if len(t) < 60:
        return False, {"reason": "response too short to judge"}
    body = t.replace("<think>", " ").replace("</think>", " ")
    words = re.findall(r"[A-Za-z][A-Za-z']*", body)
    nw = len(words)
    mean_len = (sum(map(len, words)) / nw) if nw else 0.0
    alnum = [c for c in body if c.isalnum()]
    digit_frac = (sum(c.isdigit() for c in alnum) / len(alnum)) if alnum else 1.0

    tok = lambda s: [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z']*|\d+", s)]
    seq = tok(body)
    grams = lambda xs: [tuple(xs[i:i + 5]) for i in range(max(0, len(xs) - 4))]
    from_prompt = set(grams(tok(prompt)))
    mine = [g for g in grams(seq) if g not in from_prompt]
    c = collections.Counter(mine)
    top_count = c.most_common(1)[0][1] if mine else 0
    top_gram = (top_count / max(1, len(mine))) if mine else 0.0

    m = {"words": nw, "mean_word_len": round(mean_len, 2),
         "digit_frac": round(digit_frac, 3), "top_5gram_frac": round(top_gram, 3)}
    shape = {"wordlike": 2.2 <= mean_len <= 9.0,  # not char soup, not glued tokens
             "not_digit_soup": digit_frac < 0.55,
             # A loop means some n-gram recurs. With every gram distinct the
             # ratio is 1/N, which alone exceeds 0.06 below ~21 words -- that is
             # the metric bottoming out, not a repetition loop.
             "no_repeat_loop": top_count < 2 or top_gram < 0.06}
    # enough_words exists to reject a stub. A response that states the expected
    # answer AND passes every shape check is not a stub, however briefly it puts
    # it -- MiniMax-M2.7 answers 3+4 in 24 words, one under the threshold. The
    # waiver cannot rescue garbled output: garbage does not carry the right
    # answer and still pass wordlike / digit / repetition.
    answered = bool(expect) and expect in body and all(shape.values())
    checks = dict(shape, enough_words=(nw >= 25 or answered))
    m["failed_checks"] = [k for k, v in checks.items() if not v]
    return all(checks.values()), m


res = {}
for tag, content, expect in prompts:
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": content}],
                       "max_tokens": 400, "temperature": 0.6, "top_p": 0.95, "top_k": 20}).encode()
    req = urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions" % port,
                                 data=body, headers={"Content-Type": "application/json"})
    try:
        t = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"] or ""
    except Exception as e:
        t = "<FAILED %s>" % e
    ok, m = judge(t, content, expect)
    m["has_expected_answer"] = expect in t
    res[tag] = dict(coherent=ok, len=len(t), text=t, **m)
    print("[smoke] %-10s %-5s coherent=%s len=%d words=%s wl=%s dig=%s rep=%s %s | %s" % (
        arm, tag, ok, len(t), m.get("words"), m.get("mean_word_len"), m.get("digit_frac"),
        m.get("top_5gram_frac"), ("BAD:" + ",".join(m.get("failed_checks", []))) if not ok else "",
        t[:100].replace("\n", " ")), flush=True)
json.dump(res, open("%s/smoke.json" % out, "w"), indent=1)
ok = all(r["coherent"] for r in res.values())
print("[smoke] %s VERDICT=%s (coherent prose, no digit soup, no repetition loop, both prompts)" % (
    arm, "PASS" if ok else "FAIL"))
