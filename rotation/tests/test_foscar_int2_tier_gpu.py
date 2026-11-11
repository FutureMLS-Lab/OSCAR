"""GPU (SM90 / SM100 with the foscar package): the FOSCAR INT2 tier and its decode, without a server.

* FoscarInt2Tier.write (per-head K pre-clamp, V clip) reads back through dequantize() as OSCAR's
  extend quantizer writes it (a handful of rounding ties aside);
* the eager decode matches exact attention over those rows, and an empty INT2 tier returns -inf LSE;
* a CUDA-graph bucket replays the eager result, with the LSE of empty and padded rows masked.
SM100 decodes with an fp8 query and fp8 P, so its tolerance against fp32 is a few percent.
"""
import math

import pytest
import torch

foscar = pytest.importorskip("foscar")
if not torch.cuda.is_available() or not foscar.is_supported():
    pytest.skip("needs an SM90 or SM100 GPU", allow_module_level=True)

from sglang.srt.layers.attention.foscar_int2_decode import FoscarInt2Decode  # noqa: E402
from sglang.srt.mem_cache.foscar_int2_tier import FOSCAR_PAGE_TOKENS as P  # noqa: E402
from sglang.srt.mem_cache.foscar_int2_tier import FoscarInt2Tier  # noqa: E402

DEV = torch.device("cuda")
H, D, HQ, L = 8, 128, 32, 2
PAGES = [3, 0, 5]  # INT2 pages of three requests (the second has none)
CLIPS = [0.8, 0.85, 0.9, 0.9, 0.8, 0.85, 0.9, 0.93]
V_CLIP = 0.92


def _quantize_like_oscar(x, idx):
    t = x.abs().sort(-1).values.gather(-1, idx.view(1, -1, 1).expand(x.shape[0], -1, 1))
    x = torch.maximum(torch.minimum(x, t), -t)
    mn, mx = x.min(-1, keepdim=True).values, x.max(-1, keepdim=True).values
    scale = torch.clamp(mx - mn, min=1e-8) / 3
    zero = -mn / scale
    return (torch.clamp(torch.floor(x / scale + zero + 0.5), 0, 3) - zero) * scale


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    k_idx = [torch.tensor([min(int(c * D), D - 1) for c in CLIPS], dtype=torch.int32, device=DEV) for _ in range(L)]
    tier = FoscarInt2Tier(num_quant_pages=32, layer_num=L, head_num=H, head_dim=D, v_head_dim=D, device="cuda",
                          k_clip_layer=[0.9] * L, k_clip_idx_layer=k_idx, v_clip=V_CLIP)
    page_ids, indptr, slots_all, nxt = [], [0], [], 1  # page 0 stays reserved, as in the pool
    for n in PAGES:
        ids = list(range(nxt, nxt + n))
        nxt += n
        page_ids.append(ids)
        indptr.append(indptr[-1] + n * P)
        slots_all += [p * P + i for p in ids for i in range(P)]
    rows = {}
    for li in range(L):
        for r, ids in enumerate(page_ids):
            if not ids:
                continue
            n = len(ids) * P
            k = (torch.randn(n, H, D, device=DEV) * torch.linspace(0.2, 3, D, device=DEV)).to(torch.bfloat16)
            v = torch.randn(n, H, D, device=DEV).to(torch.bfloat16)
            slots = torch.tensor([p * P + i for p in ids for i in range(P)], dtype=torch.int64, device=DEV)
            tier.write(li, k, v, slots)
            rows[(li, r)] = (
                _quantize_like_oscar(k.float(), k_idx[li].long()),
                _quantize_like_oscar(v.float(), torch.full((H,), int(V_CLIP * D), device=DEV)),
                ids,
            )
    token_indptr = torch.tensor(indptr + [indptr[-1]], dtype=torch.int32, device=DEV)  # row 3 = graph padding
    token_indices = torch.tensor(slots_all + [0] * P, dtype=torch.int64, device=DEV)
    return tier, rows, page_ids, indptr, token_indptr, token_indices


def test_read_back_matches_the_oscar_quantizer(setup):
    tier, rows, *_ = setup
    for (li, _r), (kq, vq, ids) in rows.items():
        n = len(ids) * P
        ko = torch.empty(n, H, D, dtype=torch.bfloat16, device=DEV)
        vo = torch.empty_like(ko)
        tier.dequantize(li, ko, vo, torch.tensor([0, len(ids)], dtype=torch.int32, device=DEV),
                        torch.tensor(ids, dtype=torch.int32, device=DEV), torch.tensor([n], dtype=torch.int32, device=DEV),
                        torch.tensor([0], dtype=torch.int32, device=DEV))
        for out, ref in ((ko, kq), (vo, vq)):
            off = ((out.float() - ref).abs() > 1e-2 * ref.abs().amax()).float().mean().item()
            assert off < 5e-3, off


def test_eager_and_graph_decode(setup):
    tier, rows, page_ids, indptr, token_indptr, token_indices = setup
    scale = 1 / math.sqrt(D)
    dec = FoscarInt2Decode(tier, num_qo_heads=HQ, sm_scale=scale, max_context_len=8 * P, device=DEV)
    q = torch.randn(4, HQ, D, device=DEV).to(torch.bfloat16)

    def exact(li, r):
        kq, vq, _ = rows[(li, r)]
        qq = q[r].float()
        if dec.fp8_query:
            qd = qq.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448.0
            qq = (qq / qd).to(torch.float8_e4m3fn).float() * qd
        s = torch.einsum("hd,htd->ht", qq, kq.repeat_interleave(HQ // H, 1).transpose(0, 1)) * scale
        return torch.einsum("ht,htd->hd", torch.softmax(s, -1), vq.repeat_interleave(HQ // H, 1).transpose(0, 1)), torch.logsumexp(s, -1)

    tol = (0.10, 0.06) if dec.fp8_query else (0.02, 0.02)
    out = torch.empty(4, HQ, D, dtype=torch.bfloat16, device=DEV)
    lse = torch.empty(4, HQ, dtype=torch.float32, device=DEV)
    dec.plan_eager(token_indptr[:4], token_indices, num_tokens=indptr[-1], max_tokens=max(PAGES) * P, batch_size=3)
    eager = {}
    for li in range(L):
        dec.run(li, q[:3].contiguous(), out[:3], lse[:3])
        eager[li] = (out[:3].clone(), lse[:3].clone())
        for r in range(3):
            if not page_ids[r]:
                assert torch.isneginf(lse[r]).all()
                continue
            o_x, l_x = exact(li, r)
            assert ((out[r].float() - o_x).norm() / o_x.norm()).item() < tol[0]
            assert (lse[r] - l_x).abs().max().item() < tol[1]

    dec.reserve_graph_workspace(4)
    out_g = torch.zeros(4, HQ, D, dtype=torch.bfloat16, device=DEV)
    lse_g = torch.zeros(4, HQ, dtype=torch.float32, device=DEV)
    graphs = []
    for li in range(L):
        dec.capture_graph_bucket(4, token_indptr, token_indices)
        dec.run(li, q, out_g, lse_g)  # warm-up outside the graph
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            dec.run(li, q, out_g, lse_g)
        graphs.append(g)
    lengths = [n * P for n in PAGES] + [0]
    for li in range(L):
        dec.update_graph_bucket(4, token_indptr, token_indices, lengths)
        graphs[li].replay()
        torch.cuda.synchronize()
        for r in range(4):
            if r < 3 and page_ids[r]:
                o_e, l_e = eager[li]
                assert ((out_g[r].float() - o_e[r].float()).norm() / o_e[r].float().norm()).item() < 1e-2
                assert (lse_g[r] - l_e[r]).abs().max().item() < 1e-2
            else:
                assert torch.isneginf(lse_g[r]).all()
