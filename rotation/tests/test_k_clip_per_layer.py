"""CPU tests for the per-layer K clip plumbing (no GPU): env spec parsing and flush-group splitting."""
import json, os, sys, tempfile
sys.path.insert(0, "python")
from sglang.srt.mem_cache.unified_kv_pool import parse_k_clip_per_layer, split_layers_by_clip

ids = list(range(4, 10))  # global layer ids of a pool whose start_layer is 4
assert parse_k_clip_per_layer("", ids, 0.96) == [0.96] * 6
with tempfile.TemporaryDirectory() as d:
    f = os.path.join(d, "clip.json"); json.dump({"4": 0.80, "7": 0.93}, open(f, "w"))
    assert parse_k_clip_per_layer(f, ids, 0.90) == [0.80, 0.90, 0.90, 0.93, 0.90, 0.90], "json object keyed by global id"
    g = os.path.join(d, "list.json"); json.dump([0.9] * 4 + [0.85] * 6, open(g, "w"))
    assert parse_k_clip_per_layer(g, ids, 0.96) == [0.85] * 6, "json list is in global layer order"
    bad = os.path.join(d, "bad.json"); json.dump({"5": 1.5}, open(bad, "w"))
    try:
        parse_k_clip_per_layer(bad, ids, 0.9); raise AssertionError("expected a range error")
    except ValueError:
        pass
assert parse_k_clip_per_layer("0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05", ids, 0.96) == [0.5, 0.4, 0.3, 0.2, 0.1, 0.05], "comma list in global order"
# flush groups: consecutive runs of equal clip, order preserved, one group per run
clips = [0.9, 0.9, 0.85, 0.85, 0.9, 0.93]
assert split_layers_by_clip([0, 1, 2, 3, 4, 5], clips) == [(0.9, [0, 1]), (0.85, [2, 3]), (0.9, [4]), (0.93, [5])]
assert split_layers_by_clip([1, 3, 5], clips) == [(0.9, [1]), (0.85, [3]), (0.93, [5])]
assert split_layers_by_clip([0, 1], [0.96, 0.96]) == [(0.96, [0, 1])], "uniform clip keeps one group (unchanged behaviour)"
print("k_clip_per_layer OK")

# per-head table: JSON {"<layer>,<head>": ratio} over GLOBAL heads, missing heads keep the layer clip; ratio -> kernel index
from sglang.srt.mem_cache.unified_kv_pool import parse_k_clip_per_head, clip_ratio_to_index
import torch
with tempfile.TemporaryDirectory() as d:
    f = os.path.join(d, "ph.json"); json.dump({"4,0": 0.80, "4,3": 0.93, "9,1": 0.85, "99,0": 0.5}, open(f, "w"))
    t = parse_k_clip_per_head(f, ids, 4, [0.9] * 6)
    assert t.shape == (6, 4) and t[0].tolist() == [0.80, 0.9, 0.9, 0.93] and t[5].tolist() == [0.9, 0.85, 0.9, 0.9], t
    assert t[1:5].eq(0.9).all(), "unnamed layers keep the layer clip"
    assert parse_k_clip_per_head("", ids, 4, [0.9] * 6) is None
    bad = os.path.join(d, "bad.json"); json.dump({"4,7": 0.9}, open(bad, "w"))
    try:
        parse_k_clip_per_head(bad, ids, 4, [0.9] * 6); raise AssertionError("expected a head range error")
    except ValueError:
        pass
idx = clip_ratio_to_index(torch.tensor([0.96, 0.90, 0.85, 0.80, 1.0, 0.0], dtype=torch.float64), 128)
assert idx.dtype == torch.int32 and idx.tolist() == [122, 115, 108, 102, 127, 0], idx.tolist()   # int(ratio*128), clamped
print("k_clip_per_head OK")

# TP sharding of the per-head table: [L, 4 global heads] -> rank r of TP2 keeps heads [2r, 2r+2) (the v114 arms on TP2
# models died with "per-head clip index must be int32 [2], got (4,)": the 1-D rows were not sliced)
from sglang.srt.mem_cache.memory_pool import _head_shard_start
tbl = torch.tensor([[0.80, 0.85, 0.90, 0.93]] * 3, dtype=torch.float64)
for rank, want in ((0, [0.80, 0.85]), (1, [0.90, 0.93])):
    beg = _head_shard_start(4, 2, rank, 2); assert tbl[:, beg:beg + 2][0].tolist() == want, (rank, beg)
print("per-head table sharding OK")
