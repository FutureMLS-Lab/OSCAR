#!/usr/bin/env python3
"""Does the GLM-5.3-Flash config parse, and does it label every layer correctly?

"AutoConfig did not raise" is far too weak a check here. The trap this file
exists to catch is an indexing convention, and a wrong one does not raise --
it silently routes layers to the wrong attention type.

KimiLinearConfig.is_kda_layer tests ``(layer_idx + 1) in kda_layers``: Kimi's
list is 1-indexed. GLM-5.3-Flash's is 0-indexed and agrees with ``layer_types``
position for position. Copying Kimi's predicate mislabels 23 of 45 layers. So
this asserts the mapping against ``layer_types`` from the checkpoint itself,
which is positional and therefore carries no convention to disagree about.

Also reports the packed-pool cell arithmetic at qk_rope_head_dim=0, because
that zero is the reason this model is worth the port: k_pe is never quantized,
so at rope 64 it is a fixed 128 B floor of a 288 B row, and at rope 0 it
disappears.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, "/work/oscar/sglang-research/python")

MODEL_DIR = os.environ.get("GLM53_DIR", "")


def main() -> int:
    if not MODEL_DIR:
        import glob
        cands = sorted(glob.glob(
            "/hf/hub/models--zai-org--GLM-5.3-Flash/snapshots/*/"))
        if not cands:
            print("no GLM-5.3-Flash snapshot found under /hf")
            return 2
        d = cands[0]
    else:
        d = MODEL_DIR
    print(f"snapshot: {d}")

    raw = json.load(open(os.path.join(d, "config.json")))
    truth = raw["text_config"]["layer_types"]
    print(f"layer_types from checkpoint: {len(truth)} layers, "
          f"{truth.count('linear_attention')} linear / "
          f"{len(truth) - truth.count('linear_attention')} full")

    # 1. Does sglang's registry now parse it at all?
    from sglang.srt.utils.hf_transformers_utils import get_config
    try:
        cfg = get_config(d, trust_remote_code=False)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: get_config raised {type(e).__name__}: {str(e)[:300]}")
        return 2
    print(f"get_config OK -> {type(cfg).__name__}")

    ok = True

    # 2. Every layer labelled the same as the checkpoint says.
    wrong = [
        i for i in range(len(truth))
        if cfg.is_kda_layer(i) != (truth[i] == "linear_attention")
    ]
    print(f"layer-type mapping: {len(wrong)} mismatches"
          + (f"  e.g. {wrong[:8]}  <-- WRONG" if wrong else "   <- CORRECT"))
    ok &= not wrong

    # 3. And show what Kimi's predicate WOULD have done, so the trap stays
    #    visible to whoever reads this next instead of living in a commit.
    kda = set((raw["text_config"].get("linear_attn_config") or {})
              .get("kda_layers") or [])
    kimi_wrong = [
        i for i in range(len(truth))
        if ((i + 1) in kda) != (truth[i] == "linear_attention")
    ]
    print(f"  (Kimi's `(i+1) in kda_layers` would mislabel "
          f"{len(kimi_wrong)}/{len(truth)} layers)")

    # 4. The geometry the packed pool cares about.
    tc = cfg.text_config
    print(f"\nlatent geometry: kv_lora_rank={tc.kv_lora_rank} "
          f"qk_rope_head_dim={tc.qk_rope_head_dim} "
          f"mla_use_nope={tc.mla_use_nope} "
          f"index_head_dim={tc.index_head_dim} index_topk={tc.index_topk}")
    for name in ("kv_lora_rank", "qk_rope_head_dim", "index_head_dim"):
        top = getattr(cfg, name, None)
        if top != getattr(tc, name, None):
            print(f"  FAIL: top-level {name}={top} disagrees with text_config")
            ok = False
    ok &= tc.kv_lora_rank == 512 and tc.qk_rope_head_dim == 0

    from sglang.srt.mem_cache.mla_packed_kv_pool import (
        packed_latent_bytes_per_token,
    )
    print(f"\n{'model':<24} {'bits':>4} {'cell':>6} {'BF16':>6} {'ratio':>7}")
    for label, R, rope in (("GLM-5.2 / K3", 512, 64),
                           ("GLM-5.3-Flash", tc.kv_lora_rank,
                            tc.qk_rope_head_dim)):
        for bits in (2, 4):
            c = packed_latent_bytes_per_token(R, rope, 128, bits=bits)
            bf = (R + rope) * 2
            print(f"{label:<24} {bits:>4} {c:>6} {bf:>6} {bf / c:>6.2f}x")
    n_full = len(truth) - truth.count("linear_attention")
    print(f"\nBut only {n_full}/{len(truth)} layers have a KV cache at all "
          f"({100.0 * n_full / len(truth):.0f}%) -- the other "
          f"{truth.count('linear_attention')} are KDA linear attention with no "
          f"KV to compress. Do not quote the cell ratio as a model-level saving.")

    # 5. Indexer geometry must agree with the STORED tensor shapes, not just be
    #    present. index_kpool defaults to 16 in the reference and is 4 here, so
    #    a config that merely parses can still size index_kpool_compress_ape
    #    wrong; checking the field against itself would not catch that.
    import struct, collections
    wm = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
    want = {}
    for k in wm:
        for t in ("index_kpool_compress_ape", "index_kpool_compress_gate",
                  "indexer.wq_b.weight", "indexer.wk.weight",
                  "indexer.weights_proj.weight"):
            if k.endswith(t) and t not in want:
                want[t] = k
    by = collections.defaultdict(list)
    for t, k in want.items():
        by[wm[k]].append((t, k))
    shapes = {}
    for shard, items in by.items():
        with open(os.path.join(d, shard), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            meta = json.loads(f.read(n))
        for t, k in items:
            shapes[t] = meta[k]["shape"]
    exp = {
        "index_kpool_compress_ape": [tc.index_kpool, tc.index_head_dim],
        "index_kpool_compress_gate": [tc.index_head_dim, tc.hidden_size],
        "indexer.wq_b.weight": [tc.index_n_heads * tc.index_head_dim, tc.q_lora_rank],
        "indexer.wk.weight": [tc.index_head_dim, tc.hidden_size],
        "indexer.weights_proj.weight": [tc.index_n_heads, tc.hidden_size],
    }
    print(f"\nindexer: kpool={tc.index_kpool} n_heads={tc.index_n_heads} "
          f"tail={tc.index_kpool_always_select_tail} "
          f"types={collections.Counter(tc.indexer_types or [])}")
    for t, e in exp.items():
        got = shapes.get(t)
        good = got == e
        ok &= good
        print(f"  {t:<28} stored {got}  from config {e}  "
              f"{'ok' if good else 'MISMATCH'}")

    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
