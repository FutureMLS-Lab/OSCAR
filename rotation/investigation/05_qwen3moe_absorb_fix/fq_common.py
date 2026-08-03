"""Fake-quant KV emulation of the OSCAR INT2 mixed-KV read path (transformers v5).

Registers an attention function that rotates -> INT2 quantizes -> dequantizes ->
unrotates K/V for cache positions in [sink, quant_end), leaving the sink prefix
and the recent window in bf16, exactly like the serving pool's tiering.
"""
import torch

STATS = {"calls": 0, "quantized_rows": 0}

CFG = {
    "enabled": False,
    "R_k": None,        # dict layer_idx -> [heads,128,128] or [128,128]
    "R_v": None,
    "sink": 64,
    "quant_end": None,  # exclusive cache position; positions >= this stay bf16 (recent window)
    "recent": None,     # if set, quant_end is derived per call as seq_len - recent (generation mode)
    "k_clip": 0.96,
    "v_clip": 0.92,
    "lloyd_max": False,
}
_LM4 = torch.tensor([-1.5104, -0.4528, 0.4528, 1.5104])


def _quant_rows(x, clip, lloyd_max):
    """x: [..., 128] one group per row -> 2-bit round trip."""
    if lloyd_max:
        m = x.mean(-1, keepdim=True)
        sd = x.std(-1, keepdim=True).clamp_min(1e-8)
        z = (x - m) / sd
        lv = _LM4.to(x.device, x.dtype)
        return lv[(z.unsqueeze(-1) - lv).abs().argmin(-1)] * sd + m
    n = x.shape[-1]
    s = x.sort(-1).values
    lo = s[..., max(0, int((1 - clip) * n) - 1)].unsqueeze(-1)
    hi = s[..., min(n - 1, int(clip * n))].unsqueeze(-1)
    x = x.clamp(lo, hi)
    sc = (hi - lo).clamp_min(1e-8) / 3.0
    return ((x - lo) / sc).round().clamp(0, 3) * sc + lo


def _rot(x, R):
    # x: [b, heads, seq, 128]; R: [128,128] shared or [heads,128,128] per-head
    if R.dim() == 2:
        return x @ R
    return torch.einsum("bhsd,hde->bhse", x, R)


def _unrot(x, R):
    if R.dim() == 2:
        return x @ R.T
    return torch.einsum("bhsd,hed->bhse", x, R)


def fake_quant_kv(key, value, layer_idx):
    """key/value: [b, kv_heads, seq, head_dim] in cache order."""
    STATS["calls"] += 1
    if not CFG["enabled"]:
        return key, value
    seq = key.shape[2]
    start = CFG["sink"]
    if CFG["recent"] is not None:
        end = seq - CFG["recent"]
    else:
        end = CFG["quant_end"] if CFG["quant_end"] is not None else seq
    end = min(end, seq)
    if end <= start:
        return key, value
    Rk = CFG["R_k"][layer_idx].to(key.device, torch.float32)
    Rv = CFG["R_v"][layer_idx].to(key.device, torch.float32)
    k_mid = key[:, :, start:end, :].to(torch.float32)
    v_mid = value[:, :, start:end, :].to(torch.float32)
    k_q = _unrot(_quant_rows(_rot(k_mid, Rk), CFG["k_clip"], CFG["lloyd_max"]), Rk)
    v_q = _unrot(_quant_rows(_rot(v_mid, Rv), CFG["v_clip"], CFG["lloyd_max"]), Rv)
    key = key.clone()
    value = value.clone()
    STATS["quantized_rows"] += int(k_mid.shape[0] * k_mid.shape[1] * k_mid.shape[2])
    key[:, :, start:end, :] = k_q.to(key.dtype)
    value[:, :, start:end, :] = v_q.to(value.dtype)
    return key, value


def install(model):
    """Wrap the model's attention implementation with the fake-quant hook."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    base_name = "sdpa" if "sdpa" in ALL_ATTENTION_FUNCTIONS else "eager"
    base_fn = ALL_ATTENTION_FUNCTIONS[base_name]

    def oscar_fq(module, query, key, value, attention_mask, **kwargs):
        key, value = fake_quant_kv(key, value, module.layer_idx)
        return base_fn(module, query, key, value, attention_mask, **kwargs)

    ALL_ATTENTION_FUNCTIONS["oscar_fq"] = oscar_fq
    try:
        model.set_attn_implementation("oscar_fq")
    except Exception as e:
        print("[fq] set_attn_implementation failed:", e)
    model.config._attn_implementation = "oscar_fq"
    for m in model.modules():
        cfg = getattr(m, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = "oscar_fq"
        if hasattr(m, "_attn_implementation"):
            m._attn_implementation = "oscar_fq"
    return model


def load_rotations(path, n_layers):
    """Accept both shared [128,128] and per-head [H,128,128] checkpoints."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    lay = ck["layers"]
    out = {}
    for lid in range(n_layers):
        r = lay[lid]["rotation"] if lid in lay else lay[str(lid)]["rotation"]
        out[lid] = r.float()
    return out
