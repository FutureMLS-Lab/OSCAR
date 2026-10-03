#!/usr/bin/env python3
"""Judge the verbatim-recall probe: did the answer's tail quote the correct
option, allowing a small model's paraphrase but not a different option?

usage: recall_judge.py <response.json> <correct option> <other option>...
prints OK | MISS | CAP | ERR

An exact (whitespace- and case-insensitive) copy in the last 1500 characters
is OK. Otherwise the tail is compared to every option word by word: number
words are folded to digits and punctuation dropped, and the share of the
option's words found in order in the tail (difflib matching blocks) is the
score. OK needs the correct option to score at least 0.85 and to beat every
other option by 0.25 -- Qwen3-4B-Thinking copies "eleven months" as
"11 months" and "through baleen plates" as "with baleen plates" (score 0.9),
while a wrong option scores about 0.3 against the right one.
"""
import difflib, json, re, sys

NUMBERS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
           "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
           "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
           "nineteen": "19", "twenty": "20"}


def norm(t):
    return re.sub(r"\s+", " ", t).strip().lower()


def words(t):
    return [NUMBERS.get(w, w) for w in re.findall(r"[a-z0-9]+", t.lower())]


def score(option, tail):
    ow, tw = words(option), words(tail)
    if not ow:
        return 0.0
    sm = difflib.SequenceMatcher(None, ow, tw, autojunk=False)
    return sum(b.size for b in sm.get_matching_blocks()) / len(ow)


def judge(d, correct, others):
    final = d.get("content") or d.get("text") or ""
    if d.get("error"):
        return "ERR"
    tail = final[-1500:]
    if norm(correct) in norm(tail):
        return "OK"
    s_ok = score(correct, tail)
    s_other = max((score(o, tail) for o in others), default=0.0)
    if s_ok >= 0.85 and s_ok - s_other >= 0.25:
        return "OK"
    if d.get("finish_reason") == "length":
        return "CAP"
    return "MISS"


if __name__ == "__main__":
    try:
        d = json.loads(open(sys.argv[1]).read()) if sys.argv[1].endswith(".json") and not sys.argv[1].startswith("{") else json.loads(sys.argv[1])
    except Exception:
        print("MISS"); raise SystemExit
    print(judge(d, sys.argv[2], sys.argv[3:]))
