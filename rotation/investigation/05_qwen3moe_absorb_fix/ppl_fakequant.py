"""Chunked teacher-forced PPL with OSCAR fake-quant KV.

Emulates production tiering per query chunk: sink[0:64] bf16, the previous
chunk (recent window) bf16, everything between quantized. Chunk size == recent
window, so the bf16 recent span is exact.
"""
import argparse, json, math, sys, time
import torch
sys.path.insert(0, "/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/oscar/rotation/investigation/05_qwen3moe_absorb_fix")
import fq_common as FQ

AP = argparse.ArgumentParser()
AP.add_argument("--model", required=True)
AP.add_argument("--k-rot", default=None)
AP.add_argument("--v-rot", default=None)
AP.add_argument("--mode", default="bf16")          # bf16 | quant
AP.add_argument("--lloyd-max", action="store_true")
AP.add_argument("--window", type=int, default=2048)
AP.add_argument("--chunk", type=int, default=256)   # == recent window
AP.add_argument("--sink", type=int, default=64)
AP.add_argument("--num-windows", type=int, default=8)
AP.add_argument("--out", required=True)
AP.add_argument("--tag", required=True)
A = AP.parse_args()

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

tok = AutoTokenizer.from_pretrained(A.model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    A.model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model.eval()
n_layers = model.config.num_hidden_layers
FQ.install(model)

if A.mode == "quant":
    FQ.CFG["R_k"] = FQ.load_rotations(A.k_rot, n_layers)
    FQ.CFG["R_v"] = FQ.load_rotations(A.v_rot, n_layers)
    FQ.CFG["sink"] = A.sink
    FQ.CFG["lloyd_max"] = A.lloyd_max
    sh = FQ.CFG["R_k"][0].shape
    print("[ppl] rotation shape per layer:", tuple(sh), "(per-head)" if len(sh) == 3 else "(shared)")

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(ds["text"])
ids = tok(text, return_tensors="pt").input_ids[0]
print("[ppl] tokenized:", ids.numel())

dev = next(model.parameters()).device
nll_sum, ntok = 0.0, 0
t0 = time.time()
for w in range(A.num_windows):
    beg = w * A.window
    win = ids[beg: beg + A.window]
    if win.numel() < A.window:
        break
    win = win.to(dev).unsqueeze(0)
    for cs in range(A.chunk, A.window, A.chunk):     # predict tokens [cs, cs+chunk)
        ce = min(cs + A.chunk, A.window)
        FQ.CFG["enabled"] = (A.mode == "quant")
        FQ.CFG["quant_end"] = max(A.sink, cs - A.chunk)   # recent chunk stays bf16
        with torch.no_grad():
            out = model(win[:, :ce])
        logits = out.logits[0, cs - 1: ce - 1].float()
        tgt = win[0, cs: ce]
        nll = torch.nn.functional.cross_entropy(logits, tgt, reduction="sum")
        nll_sum += float(nll); ntok += tgt.numel()
    print("[ppl] window %d/%d  running ppl=%.4f  (%.0fs)" % (
        w + 1, A.num_windows, math.exp(nll_sum / max(1, ntok)), time.time() - t0), flush=True)

ppl = math.exp(nll_sum / max(1, ntok))
print("[ppl] hook calls=%d quantized_rows=%d" % (FQ.STATS["calls"], FQ.STATS["quantized_rows"]))
if A.mode == "quant" and FQ.STATS["quantized_rows"] == 0:
    raise SystemExit("[ppl] ABORT: quant mode but nothing was quantized -- hook is dead")
print("[ppl] TAG=%s MODE=%s PPL=%.4f tokens=%d" % (A.tag, A.mode, ppl, ntok))
json.dump({"tag": A.tag, "mode": A.mode, "ppl": ppl, "tokens": ntok,
           "quantized_rows": FQ.STATS["quantized_rows"],
           "lloyd_max": A.lloyd_max, "k_rot": A.k_rot},
          open("%s/ppl_%s.json" % (A.out, A.tag), "w"), indent=1)
