"""Generation coherence: shared vs per-head rotation under INT2 fake-quant."""
import argparse, glob, json, sys, torch
sys.path.insert(0, "/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/oscar/rotation/investigation/05_qwen3moe_absorb_fix")
import fq_common as FQ
from transformers import AutoModelForCausalLM, AutoTokenizer

AP = argparse.ArgumentParser()
AP.add_argument("--model-glob", required=True)
AP.add_argument("--pairs", required=True, help="name=krot,vrot;name2=...")
AP.add_argument("--out", required=True)
AP.add_argument("--tag", required=True)
A = AP.parse_args()

M = glob.glob(A.model_glob)[0]
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()
FQ.install(model)
nl = model.config.num_hidden_layers
dev = next(model.parameters()).device

FILLER = " ".join("Reference item %d records value %d." % (i, i % 23) for i in range(600))
PROMPT = ("Background notes (ignore unless relevant):\n" + FILLER +
          "\n\nPlease reason step by step, and put your final answer within \\boxed{}.\n"
          "Compute the sum of the first 40 positive integers, then subtract 20.")
msgs = [{"role": "user", "content": PROMPT}]
enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
ids = enc["input_ids"] if hasattr(enc, "keys") else enc      # transformers v5 returns BatchEncoding
ids = ids.to(dev)

def alpha_frac(t):
    b = t.replace("<think>", "")[:600].strip()
    return round(sum(c.isalpha() for c in b) / max(1, len(b)), 3)

res = {}
arms = [("bf16", None, None)] + [
    (p.split("=")[0], p.split("=")[1].split(",")[0], p.split("=")[1].split(",")[1])
    for p in A.pairs.split(";") if p]
for name, krot, vrot in arms:
    if krot is None:
        FQ.CFG["enabled"] = False
    else:
        FQ.CFG["R_k"] = FQ.load_rotations(krot, nl)
        FQ.CFG["R_v"] = FQ.load_rotations(vrot, nl)
        FQ.CFG["enabled"] = True
        FQ.CFG["sink"] = 64
        FQ.CFG["recent"] = 256          # production tiering, tracked per step
    torch.manual_seed(0)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=300, do_sample=True,
                             temperature=0.6, top_p=0.95, top_k=20)
    txt = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    res[name] = {"alpha": alpha_frac(txt), "boxed": "\\boxed" in txt,
                 "quantized_rows": FQ.STATS["quantized_rows"], "head": txt[:130].replace("\n", " ")}
    print("[gen] %-9s alpha=%.3f boxed=%s qrows=%d | %s" % (
        name, res[name]["alpha"], res[name]["boxed"], FQ.STATS["quantized_rows"], res[name]["head"]), flush=True)
    FQ.STATS["quantized_rows"] = 0

json.dump(res, open("%s/gen_%s.json" % (A.out, A.tag), "w"), indent=1)
