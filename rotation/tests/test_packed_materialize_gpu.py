"""GPU equivalence of the one-launch packed-latent materialize against the
two-kernel reference (gather_dequant_rows + assemble_rows): identical bits,
with and without the BF16 window arena, with -1 holes. Needs CUDA."""
import pytest
import torch

from sglang.QuantKernel.mla_latent_int2 import (
    assemble_rows,
    gather_dequant_assemble_rows,
    gather_dequant_rows,
    quantize_pack,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("lloyd", [False, True])
@pytest.mark.parametrize("has_hp", [False, True])
def test_fused_materialize_matches_two_kernels(lloyd, has_hp):
    torch.manual_seed(0)
    dev = "cuda"
    n_slots, R, rope, gs, bits = 2048, 512, 64, 128, 2
    x = torch.randn(n_slots, R, device=dev, dtype=torch.bfloat16)
    codes, params = quantize_pack(x, group_size=gs, lloyd_max=lloyd, bits=bits)
    # the pool keeps one row per slot: [slots, D/4] codes and [slots, 2*NG] params
    codes = codes.reshape(n_slots, R // (8 // bits)).contiguous()
    params = params.reshape(n_slots, 2 * (R // gs)).contiguous()
    rope_buf = torch.randn(n_slots, rope, device=dev, dtype=torch.bfloat16)
    if has_hp:
        n_hp = 256
        hp_buf = torch.randn(n_hp, R, device=dev, dtype=torch.bfloat16)
        hp_row = torch.full((n_slots,), -1, dtype=torch.int32, device=dev)
        owners = torch.randperm(n_slots, device=dev)[:n_hp].to(torch.int32)
        hp_row[owners.long()] = torch.arange(n_hp, dtype=torch.int32, device=dev)
        hp_owner = owners.clone()
        hp_owner[::3] = -5  # some arena rows now owned by someone else: must NOT be used
    else:
        hp_buf = hp_row = hp_owner = None
    slots = torch.randint(-1, n_slots, (1500,), dtype=torch.int32, device=dev)
    slots[:7] = -1
    n = slots.numel()
    ref = torch.empty(n, R + rope, device=dev, dtype=torch.bfloat16)
    scratch = torch.empty(n, R, device=dev, dtype=torch.bfloat16)
    gather_dequant_rows(slots, codes, params, scratch, gs, lloyd, bits)
    assemble_rows(scratch, slots, rope_buf, hp_buf, hp_row, hp_owner, ref)
    out = torch.empty_like(ref)
    table = torch.full((n,), -9, dtype=torch.int32, device=dev)
    gather_dequant_assemble_rows(slots, codes, params, rope_buf, hp_buf, hp_row, hp_owner, out, gs, lloyd, bits,
                                 table=table)
    torch.cuda.synchronize()
    assert torch.equal(out, ref), (out.float() - ref.float()).abs().max()
    # the in-launch staging table equals stage_decode's where(valid, arange, slots)
    ar = torch.arange(n, dtype=torch.int32, device=dev)
    assert torch.equal(table, torch.where(slots >= 0, ar, slots))
