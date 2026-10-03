"""CPU: stage_decode_fused returns what stage_decode returns, given a
materialize that writes the table the fused kernel writes."""
import torch

from sglang.srt.layers.attention.nsa.packed_staging import stage_decode, stage_decode_fused


def test_fused_glue_matches_reference():
    torch.manual_seed(0)
    bs, topk, D, mult = 3, 8, 16, 4
    pt = torch.randint(-1, 50, (bs, topk), dtype=torch.int32)
    pt[0, :3] = -1
    pool = torch.randn(50, 1, D)

    def materialize(slots, out):
        out.copy_(pool[slots.long()])

    def materialize_table(slots, out, table):
        valid = slots >= 0
        out.copy_(pool[torch.where(valid, slots, torch.zeros_like(slots)).long()])
        out[~valid] = 0
        table.copy_(torch.where(valid, torch.arange(slots.numel(), dtype=torch.int32), slots))

    n = bs * topk
    buf_ref = torch.zeros(32, 1, D); buf_f = torch.zeros(32, 1, D)
    rows_ref, table_ref = stage_decode(materialize, pt, torch.arange(32, dtype=torch.int32), buf_ref, mult)
    rows_f, table_f = stage_decode_fused(materialize_table, pt, buf_f, torch.full((32,), -9, dtype=torch.int32), mult)
    assert rows_ref.shape == rows_f.shape == (-(-n // mult) * mult, 1, D)
    assert torch.equal(table_ref, table_f)
    live = (pt >= 0).reshape(-1)
    assert torch.equal(rows_ref[:n][live], rows_f[:n][live])
