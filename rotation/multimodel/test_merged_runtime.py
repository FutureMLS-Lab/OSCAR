#!/usr/bin/env python3
"""GPU smoke test for hybrid LUT-B pool routing."""

from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.lutb_kv_pool import LUTBBlockFakeQuantKVPool
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, KVWriteLoc
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool


def batch_for(loc: torch.Tensor):
    mapping = torch.zeros((1, 512), dtype=torch.int32, device="cuda")
    mapping[0, : len(loc)] = loc
    return SimpleNamespace(
        batch_size=1,
        req_pool_indices=torch.tensor([0], device="cuda"),
        seq_lens_cpu=torch.tensor([len(loc)]),
        seq_lens=torch.tensor([len(loc)], device="cuda"),
        extend_prefix_lens_cpu=[0],
        req_to_token_pool=SimpleNamespace(req_to_token=mapping),
    )


def main() -> None:
    codebook = torch.tensor([-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    path = Path("/tmp/lutb-hybrid-codebook.pt")
    torch.save(
        {
            "format": "lutb_int3_e4m3",
            "format_version": 2,
            "method": "lloyd_max",
            "lloyd_max_rule": "float_centroids_round_once_to_distinct_e4m3",
            "oscar": False,
            "source_grouping": "layer",
            "layers": {
                layer: {"k": codebook, "v": codebook}
                for layer in (0, 3, 5)
            },
        },
        path,
    )
    common = {
        "enable_alt_stream": False,
        "model_dtype": torch.bfloat16,
        "codebook_path": str(path),
        "use_oscar": False,
        "sink_tokens": 64,
        "recent_tokens": 256,
    }
    hybrid = HybridLinearKVPool(
        size=512,
        page_size=1,
        dtype=torch.bfloat16,
        head_num=1,
        head_dim=64,
        full_attention_layer_ids=[3],
        device="cuda",
        mamba_pool=SimpleNamespace(),
        full_kv_pool_class=LUTBBlockFakeQuantKVPool,
        full_kv_pool_kwargs=common,
    )
    loc = torch.arange(1, 385, device="cuda")
    q = torch.randn(384, 1, 64, device="cuda", dtype=torch.bfloat16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    q, k, v = hybrid.prepare_qkv(3, q, k, v)
    hybrid.set_kv_buffer(
        SimpleNamespace(layer_id=3),
        loc,
        k,
        v,
        already_lutb_quantized=True,
    )
    hybrid.seal_blocks(3, batch_for(loc), is_decode=False)
    assert hybrid.full_kv_pool.sealed_k_blocks == 8

    swa = SWAKVPool(
        size=512,
        size_swa=512,
        page_size=1,
        dtype=torch.bfloat16,
        head_num=1,
        head_dim=64,
        swa_attention_layer_ids=[0],
        full_attention_layer_ids=[5],
        device="cuda",
        token_to_kv_pool_class=LUTBBlockFakeQuantKVPool,
        token_to_kv_pool_kwargs=common,
        swa_window_size=512,
    )
    identity = torch.arange(514, dtype=torch.int64, device="cuda")
    identity[-1] = -1
    swa.register_mapping(identity)
    q, k, v = torch.randn_like(q), torch.randn_like(q), torch.randn_like(q)
    q, k, v = swa.prepare_qkv(0, q, k, v)
    swa.set_kv_buffer(
        SimpleNamespace(layer_id=0),
        KVWriteLoc(loc, loc),
        k,
        v,
        already_lutb_quantized=True,
    )
    swa.seal_blocks(0, batch_for(loc), is_decode=False)
    assert swa.swa_kv_pool.sealed_k_blocks == 8
    print("merged hybrid LUT-B unit PASS")


if __name__ == "__main__":
    main()
