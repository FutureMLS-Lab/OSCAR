import torch, sys
sys.path.insert(0, "/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/oscar/sglang-research/python")
from sglang.srt.mem_cache.kv_quant_kernels import _groupwise_dequantize_int2_torch
O = "/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/oscar/rotation/investigation/05_qwen3moe_absorb_fix/out_v7"

for arm in ("had", "zoo"):
    d = torch.load(f"{O}/dump_{arm}/dbg_step0_pre.pt", map_location="cpu", weights_only=False)
    post = torch.load(f"{O}/dump_{arm}/dbg_step0_post.pt", map_location="cpu", weights_only=False)
    print(f"\n=== ARM {arm}")
    for k in ("q_rot","hp_k","raw_k","k_sz"):
        print(f"  {k}: {tuple(d[k].shape)} {d[k].dtype}")
    Rk = d["R_k"].float()
    hp_ptr, hp_idx = d["hp_indptr"].long(), d["hp_indices"].long()
    qt_ptr, qt_idx = d["q_indptr"].long(), d["q_indices"].long()
    hs = hp_idx[hp_ptr[0]:hp_ptr[1]]
    qs = qt_idx[qt_ptr[0]:qt_ptr[1]]
    print(f"  req0: hp={len(hs)} quant={len(qs)} slots; hp range [{hs.min()},{hs.max()}], quant range [{qs.min()},{qs.max()}]")
    hp_k = d["hp_k"].float()
    HP_OFF = hp_k.shape[0]  # to inspect index spaces
    # hp slot ids may be offset (HP region); clamp into hp buffer space:
    hs_local = hs - (hs.min() // max(1,hp_k.shape[0]) * hp_k.shape[0]) if hs.max() >= hp_k.shape[0] else hs
    raw_k, k_sz = d["raw_k"], d["k_sz"]
    print(f"  raw_k buf: {tuple(raw_k.shape)}, k_sz: {tuple(k_sz.shape)}")
    kq_rot = _groupwise_dequantize_int2_torch(raw_k[qs % raw_k.shape[0]], k_sz[qs % k_sz.shape[0]], 128, torch.float32)
    hp_rot = hp_k[hs_local % hp_k.shape[0]]
    # unrotate both, compare per-channel magnitude profile (raw K has huge fixed outlier channels)
    kq_raw = kq_rot @ Rk.T
    hp_raw = hp_rot @ Rk.T
    prof_q = kq_raw.abs().amax(dim=0).flatten(0)   # [heads*128] channel max profile
    prof_h = hp_raw.abs().amax(dim=0).flatten(0)
    prof_qn = prof_q.view(-1,128); prof_hn = prof_h.view(-1,128)
    for h in range(min(2, prof_qn.shape[0])):
        c = torch.nn.functional.cosine_similarity(prof_qn[h], prof_hn[h], dim=0)
        print(f"  head{h}: channel-profile cos(quant_unrot, hp_unrot) = {c:.4f} | top3 quant ch {prof_qn[h].topk(3).indices.tolist()} vs hp ch {prof_hn[h].topk(3).indices.tolist()}")
    # ALSO rotated-space profile comparison (are they even in the same space?)
    pr_q = kq_rot.abs().amax(dim=0).view(-1,128); pr_h = hp_rot.abs().amax(dim=0).view(-1,128)
    c2 = torch.nn.functional.cosine_similarity(pr_q[0], pr_h[0], dim=0)
    print(f"  head0: ROTATED-space profile cos = {c2:.4f}")
