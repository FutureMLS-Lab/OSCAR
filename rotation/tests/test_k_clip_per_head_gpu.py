"""GPU equivalence tests for the per-head K clip index (CLIP_PER_HEAD) in the INT2 pack kernels and the decode flush.
Per-head tables must reproduce the scalar-ratio kernels bit for bit: a table of one repeated index equals the scalar
launch, and a mixed table equals, head by head, the scalar launch at that head's ratio. Run: python3 rotation/tests/test_k_clip_per_head_gpu.py"""
import os, sys, tempfile
sys.path.insert(0, "python")
import torch
from sglang.srt.environ import envs
from sglang.srt.mem_cache.unified_kv_pool import UnifiedInt2HPKVPool, clip_ratio_to_index
from sglang.srt.runtime_context import get_parallel
from sglang.QuantKernel.oscar_rotation_clip_int2_kv import (
    quantized_set_kv_int2_pretransformed_clip_triton,
    quantized_set_kv_int2_oscar_rotate_k_clip_triton,
    clip_rows_per_head,
    _clip_index,
)
from sglang.QuantKernel.gpu_flush_int2 import gpu_flush_int2_apply, FlushPlan

assert torch.cuda.is_available(), "GPU test"
torch.manual_seed(0)
H, D, L = 4, 128, 2
_DIR = tempfile.mkdtemp()
paths = []
for tag in ("k", "v"):
    p = os.path.join(_DIR, f"{tag}.pt"); torch.save({"layers": {i: {"rotation": torch.eye(D)} for i in range(L)}}, p); paths.append(p)
with (envs.SGLANG_OSCAR_K_ROTATION_PATH.override(paths[0]), envs.SGLANG_OSCAR_V_ROTATION_PATH.override(paths[1]),
      envs.SGLANG_OSCAR_K_CLIP_RATIO.override(0.90), envs.SGLANG_OSCAR_V_CLIP_RATIO.override(0.92),
      envs.SGLANG_OSCAR_CALIBRATION_ACTIVE.override(False), get_parallel().override(attn_tp_rank=0)):
    pool = UnifiedInt2HPKVPool(num_quant_pages=64, hp_dtype=torch.bfloat16, hp_prefix_tokens=32, hp_recent_tokens=128, dtype="int2",
                               head_num=H, head_dim=D, layer_num=L, device="cuda", enable_memory_saver=False, max_req_slots=8,
                               v_head_dim=D, start_layer=0, end_layer=L - 1, model_dtype=torch.bfloat16, kv_cache_quant_group_size=None,
                               scale_dtype=torch.float32, num_hp_prefix_slots=64)
T = 40
k = (torch.randn(T, H, D, device="cuda") * torch.tensor([1.0, 3.0, 0.5, 2.0], device="cuda")[None, :, None]).to(torch.bfloat16)
v = torch.randn(T, H, D, device="cuda").to(torch.bfloat16)
loc = torch.arange(T, device="cuda", dtype=torch.int64)
ratios = [0.80, 0.90, 0.93, 0.85]
idx_mixed = torch.tensor([_clip_index(r, D) for r in ratios], dtype=torch.int32, device="cuda")
assert torch.equal(idx_mixed.cpu(), clip_ratio_to_index(torch.tensor(ratios, dtype=torch.float64), D)), "pool index rule == kernel index rule"

def pack(k_clip_ratio, k_clip_idx, fused):
    kb, vb, ks, vs = (pool.k_buffer[0].clone().zero_(), pool.v_buffer[0].clone().zero_(), pool.k_scales_zeros[0].clone().zero_(), pool.v_scales_zeros[0].clone().zero_())
    if fused:
        R = torch.eye(D, device="cuda", dtype=k.dtype).unsqueeze(0).repeat(H, 1, 1)
        quantized_set_kv_int2_oscar_rotate_k_clip_triton(k, v, R, loc, kb, vb, ks, vs, k_clip_ratio, 0.92, k_clip_idx=k_clip_idx)
    else:
        quantized_set_kv_int2_pretransformed_clip_triton(k, v, loc, kb, vb, ks, vs, k_clip_ratio, 0.92, k_clip_idx=k_clip_idx)
    return kb[:T].clone(), ks[:T].clone(), vb[:T].clone(), vs[:T].clone()

for fused in (False, True):
    name = "fused-rotate" if fused else "pretransformed"
    ref = pack(0.90, None, fused)
    same = pack(0.90, torch.full((H,), _clip_index(0.90, D), dtype=torch.int32, device="cuda"), fused)
    for a, b, what in zip(ref, same, ("k", "k_sz", "v", "v_sz")):
        assert torch.equal(a, b), f"{name}: uniform per-head table != scalar ({what})"
    mixed = pack(0.90, idx_mixed, fused)
    for h, r in enumerate(ratios):
        single = pack(r, None, fused)
        assert torch.equal(mixed[0][:, h], single[0][:, h]) and torch.equal(mixed[1][:, h], single[1][:, h]), f"{name}: head {h} at {r} differs from the scalar launch"
    assert torch.equal(mixed[2], ref[2]) and torch.equal(mixed[3], ref[3]), f"{name}: V must be untouched by the K table"
    # a head whose index differs must actually differ from the 0.90 run (the table is read)
    assert not torch.equal(mixed[0][:, 0], ref[0][:, 0]), f"{name}: head 0 at 0.80 should differ from 0.90"
    print(f"{name}: per-head clip == scalar launches head by head")

# torch reference of the clip itself
kr = clip_rows_per_head(k.float(), idx_mixed)
for h, r in enumerate(ratios):
    i = _clip_index(r, D); thr = k.float()[:, h].abs().sort(-1).values[:, i : i + 1]
    assert torch.equal(kr[:, h], torch.maximum(torch.minimum(k.float()[:, h], thr), -thr)), h
print("clip_rows_per_head: matches the scalar clip per head")

# decode flush: hp rows -> int2 slots through the fused flush kernel, per-head table vs scalar
g = pool._flush_groups[0]
n = 16
src = torch.arange(n, device="cuda", dtype=torch.int64) + 8
dst = torch.arange(n, device="cuda", dtype=torch.int64) + 48
for l in range(L):
    pool.hp_k_buffer[l][src] = (torch.randn(n, H, D, device="cuda") * torch.tensor([1.0, 3.0, 0.5, 2.0], device="cuda")[None, :, None]).to(pool.hp_k_buffer[l].dtype)
    pool.hp_v_buffer[l][src] = torch.randn(n, H, D, device="cuda").to(pool.hp_v_buffer[l].dtype)
def plan():
    return FlushPlan(returned_slot_ids=src.clone(), src_hp_slot=src.clone(), flush_pos=torch.zeros(n, dtype=torch.int32, device="cuda"),
                     valid_mask=torch.ones(n, dtype=torch.int8, device="cuda"), dst_quant_slots=dst.clone(), bs=1, flush_interval=n)
def flush(k_clip_ratio, table):
    for l in range(L):
        pool.k_buffer[l][dst] = 0; pool.k_scales_zeros[l][dst] = 0
    gpu_flush_int2_apply(plan(), req_pool_indices=torch.zeros(1, dtype=torch.int64, device="cuda"), req_to_token=torch.zeros((1, 256), dtype=torch.int32, device="cuda"),
                         hp_k_ptrs=g["hp_k_ptrs"], hp_v_ptrs=g["hp_v_ptrs"], quant_k_ptrs=g["quant_k_ptrs"], quant_v_ptrs=g["quant_v_ptrs"], k_sz_ptrs=g["k_sz_ptrs"], v_sz_ptrs=g["v_sz_ptrs"],
                         hp_k_sample=g["hp_k_sample"], hp_v_sample=g["hp_v_sample"], quant_k_sample=g["k_sample"], quant_v_sample=g["v_sample"], k_sz_sample=g["k_sz_sample"], v_sz_sample=g["v_sz_sample"],
                         hp_k_strides=g["hp_k_stride"], hp_v_strides=g["hp_v_stride"], quant_k_strides=g["quant_k_stride"], quant_v_strides=g["quant_v_stride"], k_sz_strides=g["k_sz_stride"], v_sz_strides=g["v_sz_stride"],
                         num_heads=g["head_num"], head_dim=g["head_dim"], v_head_dim=g["v_head_dim"], k_num_scale_groups=g["k_num_scale_groups"], v_num_scale_groups=g["v_num_scale_groups"],
                         num_layers=g["num_layers"], k_clip_ratio=k_clip_ratio, v_clip_ratio=0.92, lloyd_max=False, apply_remap=False,
                         hp_k_layers=g["hp_k_layers"], hp_v_layers=g["hp_v_layers"], quant_k_layers=g["quant_k_layers"], quant_v_layers=g["quant_v_layers"], k_sz_layers=g["k_sz_layers"], v_sz_layers=g["v_sz_layers"],
                         k_clip_idx=table)
    return [(pool.k_buffer[l][dst].clone(), pool.k_scales_zeros[l][dst].clone()) for l in range(L)]
ref = flush(0.90, None)
same = flush(0.90, torch.full((L, H), _clip_index(0.90, D), dtype=torch.int32, device="cuda"))
assert all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(ref, same)), "flush: uniform table != scalar"
table = torch.stack([idx_mixed, idx_mixed.flip(0)], 0)   # layer 1 uses the reversed assignment
mixed = flush(0.90, table)
for l in range(L):
    for h in range(H):
        r = ratios[h] if l == 0 else ratios[H - 1 - h]
        single = flush(r, None)
        assert torch.equal(mixed[l][0][:, h], single[l][0][:, h]) and torch.equal(mixed[l][1][:, h], single[l][1][:, h]), f"flush: layer {l} head {h} at {r} differs"
print("flush: per-(layer, head) clip == scalar launches head by head")
print("KTEST OK")
