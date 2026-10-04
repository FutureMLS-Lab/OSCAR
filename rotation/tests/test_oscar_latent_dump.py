"""The MLA latent dump keeps raw rows and sampled queries until the budget and
writes one file per layer with the shapes the joint latent fitter reads."""
import os
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
from sglang.srt.mem_cache.oscar_calibration import OscarLatentDump  # noqa: E402


def test_dump_writes_once_per_layer_at_budget():
    d = Path(tempfile.mkdtemp(prefix="latent_dump_"))
    dump = OscarLatentDump(d, token_budget=100, rank=0, query_stride=16)
    gen = torch.Generator().manual_seed(0)
    for start in (0, 64):  # 64 + 64 rows offered, 100 kept
        n = 64
        dump.observe(layer_id=3, c_kv=torch.randn(n, 1, 32, generator=gen), k_pe=torch.randn(n, 1, 8, generator=gen),
                     q_nope=torch.randn(n, 4, 32, generator=gen), q_pe=torch.randn(n, 4, 8, generator=gen), positions=torch.arange(start, start + n))
    path = d / "layer_3_rank0.pt"
    assert path.exists()
    p = torch.load(path)
    assert p["tokens"] == 100 and tuple(p["c_kv"].shape) == (100, 32) and tuple(p["k_pe"].shape) == (100, 8)
    assert p["q_rows"].tolist() == list(range(0, 100, 16)) and tuple(p["q_nope"].shape) == (7, 4, 32) and tuple(p["q_pe"].shape) == (7, 4, 8)
    assert p["positions"].tolist() == list(range(100))
    # further rows for a written layer are ignored, other layers keep collecting
    dump.observe(layer_id=3, c_kv=torch.zeros(5, 32), k_pe=torch.zeros(5, 8), q_nope=torch.zeros(5, 4, 32), q_pe=torch.zeros(5, 4, 8), positions=torch.arange(5))
    assert torch.load(path)["tokens"] == 100
    dump.observe(layer_id=4, c_kv=torch.zeros(5, 32), k_pe=torch.zeros(5, 8), q_nope=torch.zeros(5, 4, 32), q_pe=torch.zeros(5, 4, 8), positions=torch.arange(5))
    assert not (d / "layer_4_rank0.pt").exists() and dump._counts[4] == 5
