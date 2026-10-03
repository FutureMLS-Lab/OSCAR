"""``fast_rotate_rows`` must agree with the torch rotation it replaces.

Decode rotates Q by ``R_k`` and the output by ``R_v^T`` on every layer; the
batched kernel does that in one launch. Both paths are bf16 in, fp32
accumulate, bf16 out, so they must agree to bf16 rounding (one ulp of the
largest value) on shared and per-KV-head rotations, with and without the
transpose, in place, and across a token tail that does not fill a block.
"""
import pytest
import torch

from sglang.QuantKernel.oscar_rotate_rows import fast_rotate_rows, rotate_rows_supported
from sglang.srt.layers.attention.quantized_kv_prefill import _apply_oscar_rotation

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _orthogonal(*shape, device):
    gen = torch.Generator(device="cpu").manual_seed(11)
    m = torch.randn(*shape, generator=gen)
    q, _ = torch.linalg.qr(m)
    return q.to(torch.bfloat16).to(device)


def _assert_bf16_close(got, ref):
    assert got.dtype == ref.dtype and got.shape == ref.shape
    tol = 2.0 ** -7 * ref.abs().max().item()  # one bf16 ulp of the largest value
    diff = (got.float() - ref.float()).abs()
    assert diff.max().item() <= tol, f"max |d| {diff.max().item():.4g} > {tol:.4g}"


def test_supported_dims():
    assert rotate_rows_supported(128) and rotate_rows_supported(256) and rotate_rows_supported(64)
    assert not rotate_rows_supported(96) and not rotate_rows_supported(8) and not rotate_rows_supported(512)


@gpu
@pytest.mark.parametrize("tokens", [1, 16, 17, 45])
@pytest.mark.parametrize("head_dim", [128, 256])
def test_shared_rotation_matches_torch(tokens, head_dim):
    heads = 16
    R = _orthogonal(head_dim, head_dim, device="cuda")
    x = torch.randn(tokens, heads, head_dim, dtype=torch.bfloat16, device="cuda")
    got = fast_rotate_rows(x, R)
    _assert_bf16_close(got, _apply_oscar_rotation(x, R))
    got_t = fast_rotate_rows(x, R, trans=True)
    _assert_bf16_close(got_t, (x @ R.T).contiguous())


@gpu
def test_per_head_rotation_maps_query_heads_to_kv_heads():
    tokens, kv_heads, group, head_dim = 9, 4, 8, 128
    R = _orthogonal(kv_heads, head_dim, head_dim, device="cuda")
    x = torch.randn(tokens, kv_heads * group, head_dim, dtype=torch.bfloat16, device="cuda")
    got = fast_rotate_rows(x, R, kv_group_num=group)
    _assert_bf16_close(got, _apply_oscar_rotation(x, R, group))
    # the inverse on the attention output, in place, per KV head
    o = torch.randn(tokens, kv_heads * group, head_dim, dtype=torch.bfloat16, device="cuda")
    Rh = R.repeat_interleave(group, dim=0)
    ref = torch.einsum("thd,hed->the", o, Rh)
    fast_rotate_rows(o, R, trans=True, out=o, kv_group_num=group)
    _assert_bf16_close(o, ref)


@gpu
def test_in_place_and_non_contiguous_input():
    R = _orthogonal(128, 128, device="cuda")
    base = torch.randn(6, 8, 256, dtype=torch.bfloat16, device="cuda")
    x = base[:, :, :128]  # last dim contiguous, rows strided
    ref = _apply_oscar_rotation(x, R)
    got = fast_rotate_rows(x, R)
    _assert_bf16_close(got, ref)
    y = x.contiguous()
    fast_rotate_rows(y, R, out=y)
    _assert_bf16_close(y, ref)
    empty = torch.empty(0, 8, 128, dtype=torch.bfloat16, device="cuda")
    assert fast_rotate_rows(empty, R).shape == (0, 8, 128)
