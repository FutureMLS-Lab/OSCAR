"""GPU: the INT2 dense extend's FlashInfer ragged prefill (used where FlashAttention 3 is unavailable)
matches exact attention over [prefix + extend] rows: GQA, the causal mask aligned to each request's
last key (prefix then extend), a sliding window, and a request with no prefix."""
import math

import pytest
import torch

flashinfer = pytest.importorskip("flashinfer")
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from sglang.srt.layers.attention.triton_backend import TritonAttnBackend  # noqa: E402

DEV = torch.device("cuda")
HQ, HKV, D = 32, 8, 128


def _reference(q, k, v, q_lens, k_lens, scale, window_left):
    out, qo, ko = [], 0, 0
    for ql, kl in zip(q_lens, k_lens):
        qq = q[qo:qo + ql].float().transpose(0, 1)  # [HQ, ql, D]
        kk = k[ko:ko + kl].float().repeat_interleave(HQ // HKV, 1).transpose(0, 1)
        vv = v[ko:ko + kl].float().repeat_interleave(HQ // HKV, 1).transpose(0, 1)
        s = qq @ kk.transpose(1, 2) * scale  # [HQ, ql, kl]
        qpos = torch.arange(ql, device=DEV)[:, None] + (kl - ql)
        kpos = torch.arange(kl, device=DEV)[None, :]
        mask = kpos <= qpos
        if window_left >= 0:
            mask &= kpos >= qpos - window_left
        s = s.masked_fill(~mask, float("-inf"))
        out.append((torch.softmax(s, -1) @ vv).transpose(0, 1))
        qo, ko = qo + ql, ko + kl
    return torch.cat(out)


@pytest.mark.parametrize("window_left", [-1, 1023])
def test_ragged_prefill_matches_exact_attention(window_left):
    torch.manual_seed(0)
    prefixes, extends = [0, 3000, 20000], [257, 512, 1000]
    q_lens = extends
    k_lens = [p + e for p, e in zip(prefixes, extends)]
    q = torch.randn(sum(q_lens), HQ, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(sum(k_lens), HKV, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    qo_indptr = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32, device=DEV)
    kv_indptr = torch.tensor([0] + list(torch.tensor(k_lens).cumsum(0)), dtype=torch.int32, device=DEV)
    backend = TritonAttnBackend.__new__(TritonAttnBackend)
    backend.device = DEV
    backend._int2_prefill_flashinfer = None
    scale = 1 / math.sqrt(D)
    got = backend._flashinfer_ragged_prefill(
        q, k, v, qo_indptr, kv_indptr, sm_scale=scale, causal=True, window_left=window_left, logits_soft_cap=None
    )
    ref = _reference(q, k, v, q_lens, k_lens, scale, window_left)
    rel = ((got.float() - ref).norm() / ref.norm()).item()
    assert rel < 1e-2, rel
