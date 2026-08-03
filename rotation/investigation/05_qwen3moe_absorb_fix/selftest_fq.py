"""Prove the fake-quant hook actually fires and changes logits."""
import sys, torch
sys.path.insert(0, "/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/oscar/rotation/investigation/05_qwen3moe_absorb_fix")
import fq_common as FQ
from transformers import AutoModelForCausalLM, AutoTokenizer
import glob
M = glob.glob("/shared/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/")[0]
ZOO = "/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/oscar/rotation/OSCAR-RotationZoo/Qwen3-8B/seq20000_prompt83_group128"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()
FQ.install(model)
print("[selftest] attn impl now:", model.config._attn_implementation)
ids = tok("The capital of France is Paris. " * 120, return_tensors="pt").input_ids[:, :1024].to(next(model.parameters()).device)
FQ.CFG["enabled"] = False
with torch.no_grad(): a = model(ids).logits[0, -1].float()
calls_after_bf16 = FQ.STATS["calls"]
FQ.CFG["R_k"] = FQ.load_rotations(f"{ZOO}/k_rotation_qqt_r_h_pbr.pt", model.config.num_hidden_layers)
FQ.CFG["R_v"] = FQ.load_rotations(f"{ZOO}/v_rotation_sst_r_h_pbr.pt", model.config.num_hidden_layers)
FQ.CFG["enabled"] = True; FQ.CFG["sink"] = 64; FQ.CFG["quant_end"] = 768
with torch.no_grad(): b = model(ids).logits[0, -1].float()
print("[selftest] hook calls:", FQ.STATS["calls"], "(bf16 pass:", calls_after_bf16, ")")
print("[selftest] quantized rows:", FQ.STATS["quantized_rows"])
print("[selftest] logit max|delta| =", float((a-b).abs().max()), " cos =", float(torch.nn.functional.cosine_similarity(a,b,dim=0)))
print("[selftest] VERDICT:", "HOOK ACTIVE" if FQ.STATS["quantized_rows"] > 0 and float((a-b).abs().max()) > 1e-3 else "HOOK DEAD")
