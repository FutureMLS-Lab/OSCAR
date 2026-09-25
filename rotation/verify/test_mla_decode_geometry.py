#!/usr/bin/env python3
"""Does the stock triton MLA decode compute the right thing at K3's geometry?

K3's absorbed MLA decode is a grouped (MQA-like) attention: 12 local query
heads against 1 KV head, query width 576 (kv_lora 512 + rope 64) and value
width 512 -- the value is the first 512 columns of the same latent row.

The BF16 arm takes this kernel; the 2-bit arm takes our packed decode instead,
and only the BF16 arm degenerates. So the question is whether this kernel is
correct at these dims. Compared against a plain torch softmax attention.
"""
import sys, torch
sys.path.insert(0, "/oscar/src/sglang-research/python")
from sglang.srt.layers.attention.triton_ops.decode_attention import decode_attention_fwd

dev = "cuda"
H, QK, V = 12, 576, 512          # local q heads, q width, value width
SCALE = 1.0 / (192 ** 0.5)       # layer.scaling as logged: 0.072169
MAX_SPLITS = 8

def run(bs, ctx, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    pool = bs * ctx + 16
    latent = torch.randn(pool, 1, QK, device=dev, dtype=torch.bfloat16, generator=g) * 0.3
    k_buf = latent
    v_buf = latent[..., :V]                      # value is the c_kv half
    q = torch.randn(bs, H, QK, device=dev, dtype=torch.bfloat16, generator=g) * 0.3
    o = torch.empty(bs, H, V, device=dev, dtype=torch.bfloat16)

    kv_indptr = torch.arange(0, (bs + 1) * ctx, ctx, device=dev, dtype=torch.int32)
    kv_indices = torch.arange(0, bs * ctx, device=dev, dtype=torch.int32)
    attn_logits = torch.empty(bs, H, MAX_SPLITS, V, device=dev, dtype=torch.float32)
    attn_lse = torch.empty(bs, H, MAX_SPLITS, device=dev, dtype=torch.float32)
    num_kv_splits = torch.full((bs,), MAX_SPLITS, device=dev, dtype=torch.int32)

    decode_attention_fwd(
        q, k_buf, v_buf, o, kv_indptr, kv_indices,
        attn_logits, attn_lse, num_kv_splits, MAX_SPLITS,
        SCALE, 1.0, 1.0, logit_cap=0.0, sinks=None, xai_temperature_len=-1,
    )
    torch.cuda.synchronize()

    # reference
    ref = torch.empty_like(o, dtype=torch.float32)
    for b in range(bs):
        idx = kv_indices[b * ctx : (b + 1) * ctx].long()
        k = k_buf[idx, 0, :].float()             # [ctx, 576]
        v = v_buf[idx, 0, :].float()             # [ctx, 512]
        s = (q[b].float() @ k.T) * SCALE         # [H, ctx]
        p = torch.softmax(s, dim=-1)
        ref[b] = p @ v
    d = (o.float() - ref).abs()
    rel = (d.max() / ref.abs().max().clamp(min=1e-6)).item()
    cos = torch.nn.functional.cosine_similarity(
        o.float().flatten().unsqueeze(0), ref.flatten().unsqueeze(0)).item()
    tag = "ok  " if rel < 3e-2 else "BAD "
    print(f"  {tag} bs={bs:<3d} ctx={ctx:<6d} rel_max={rel:8.4f}  cos={cos:.6f}")
    return rel < 3e-2

print("stock triton MLA decode vs torch, at K3's absorbed geometry "
      f"(H={H} q={QK} v={V} scale={SCALE:.6f})")
ok = True
for bs, ctx in ((1, 128), (1, 4096), (6, 1024), (12, 2048), (6, 16384)):
    ok &= run(bs, ctx)
print("\nVERDICT:", "kernel is correct here" if ok else "STOCK MLA DECODE IS WRONG AT K3 GEOMETRY")


# Result on B200, 2026-09-21: correct at every size tried
#   bs=1  ctx=128    rel 0.0033   cos 0.999997
#   bs=1  ctx=4096   rel 0.0029   cos 0.999997
#   bs=6  ctx=1024   rel 0.0034   cos 0.999997
#   bs=12 ctx=2048   rel 0.0027   cos 0.999997
#   bs=6  ctx=16384  rel 0.0035   cos 0.999998
#
# This was run while chasing K3's BF16 arm, which degenerates into "!!!!"
# repetition in 34/48 GPQA questions (score 18.75, below the 25.0 floor of a
# four-way choice) while the 2-bit arm on the same questions scores 83.33 with
# 0/48 degenerate. The two arms differ in exactly two server args -- the
# rendezvous address and the random seed -- so the kernel was a prime suspect.
# It is not the kernel.
#
# Also eliminated by measurement, not argument, in the same pass:
#   g_proj output gate  -- [MLA-GATE] g_proj=True, 88 lines, in BOTH arms
#   scaling             -- 0.072169 = 1/sqrt(192) in both
#   skip_rope           -- True in ours and upstream; K3's MLA is NoPE
#   layer-id mapping    -- both go through _transfer_full_attention_id
#   kv_cache_dtype      -- bfloat16 in both
#   kv_a_layernorm      -- applied on every branch of the absorbed path
#   mixed-KV radix mixin-- inert when no OSCAR pool is installed
#   head_dim = 256      -- the codebase-wide MLA convention, not a K3 quirk
#   CUDA-graph vs eager -- both arms replay graphs for >99% of decode steps
#   long-context drift  -- degeneration starts at a median of 4,040 chars and
#                          as early as char 5, so it is not a sink/length effect
