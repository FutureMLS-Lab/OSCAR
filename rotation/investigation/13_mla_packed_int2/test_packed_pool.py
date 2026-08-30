#!/usr/bin/env python3
"""Drive ``MLAPackedInt2KVPool`` directly: ring addressing, ageing, padding.

The kernel tests in ``test_packed_latent.py`` prove the arithmetic. This proves
the *bookkeeping*, which is where a windowed pool actually goes wrong:

* a token inside the sink or the recent window comes back BF16-exact, a token
  between them comes back dequantized;
* a token that ages out of the recent window is served from the packed tier
  **without any explicit eviction step** -- writing position L-1 reuses the ring
  row position L-1-R held, and the owner tag is what turns that into a demote;
* a padded CUDA-graph row (``out_cache_loc == 0``, stale ``req_pool_indices``)
  does not touch a live request's ring;
* two requests do not share ring rows.

All of that is one GPU and a few seconds, against 25 minutes of weight load to
find the same bug end to end.
"""
from __future__ import annotations

import sys
import tempfile
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.mla_int2_kv_pool import _fake_quant_int2_groupwise
from sglang.srt.mem_cache.mla_packed_kv_pool import MLAPackedInt2KVPool

R, ROPE, GS = 512, 64, 128
P, W = 4, 8                      # tiny windows so ageing happens in a few steps
LAYERS, SIZE, PAGE = 2, 512, 1
MAX_REQS = 2

FAILURES: list = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILURES.append(name)


def make_pool(tmp):
    for i in range(LAYERS):
        torch.save(torch.eye(R), f"{tmp}/layer_{i}.pt")
    import os

    os.environ["SGLANG_MIXED_KV_PREFIX_TOKENS"] = str(P)
    os.environ["SGLANG_MIXED_KV_RECENT_TOKENS"] = str(W)
    return MLAPackedInt2KVPool(
        size=SIZE,
        page_size=PAGE,
        dtype=torch.bfloat16,
        kv_lora_rank=R,
        qk_rope_head_dim=ROPE,
        layer_num=LAYERS,
        device="cuda",
        enable_memory_saver=False,
        start_layer=0,
        end_layer=LAYERS - 1,
        rotation_path=tmp,
        group_size=GS,
        lloyd_max=False,
        max_reqs=MAX_REQS,
    )


class _Layer:
    def __init__(self, lid):
        self.layer_id = lid


def fb(is_decode, seq_lens, req_pool_indices, positions=None, extend_seq_lens=None):
    dev = "cuda"
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode_or_idle=lambda: is_decode),
        positions=positions,
        seq_lens=torch.tensor(seq_lens, device=dev, dtype=torch.int64),
        req_pool_indices=torch.tensor(req_pool_indices, device=dev, dtype=torch.int64),
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.zeros((MAX_REQS, 4096), dtype=torch.int32, device=dev)
        ),
        extend_seq_lens=extend_seq_lens,
        extend_seq_lens_cpu=None,
        extend_prefix_lens_cpu=None,
    )


def main() -> None:
    dev = "cuda"
    tmp = tempfile.mkdtemp()
    pool = make_pool(tmp)
    check("pool built with windows on", pool._latent_windows,
          f"sink={pool._win_p} recent={pool._win_r}")

    L = 20                       # first request: 20 tokens, so positions 0..19
    x = torch.randn(L, 1, R, device=dev, dtype=torch.bfloat16)
    pe = torch.randn(L, 1, ROPE, device=dev, dtype=torch.bfloat16)
    slots = torch.arange(64, 64 + L, device=dev, dtype=torch.int64)

    pool.note_forward_batch(
        fb(
            False,
            [L],
            [0],
            positions=torch.arange(L, device=dev, dtype=torch.int64),
            extend_seq_lens=torch.tensor([L], device=dev, dtype=torch.int64),
        )
    )
    pool.set_kv_buffer(_Layer(0), slots, torch.cat([x, pe], dim=-1), None)

    got = pool.materialize_rows(0, slots)[:, 0, :]
    xf = x.reshape(L, R).float()
    deq = _fake_quant_int2_groupwise(xf, GS, False).to(torch.bfloat16)
    want = deq.clone()
    win = [p for p in range(L) if p < P or p >= L - W]
    want[win] = x.reshape(L, R)[win]
    # Window rows must be bit-exact -- they never went through the quantizer.
    bad_win = (got[win, :R].float() - want[win].float()).abs().max().item()
    check(
        "extend: sink+recent rows are BF16-exact",
        bad_win < 1e-6,
        f"window={win[:3]}..{win[-3:]} max_abs={bad_win:.2e}",
    )
    # Quantized rows are held to the one-step criterion, not to equality: the
    # kernel and torch break .5 ties differently on bf16-coarse input, which is
    # a rounding disagreement rather than a wrong code (see test_packed_latent).
    mid_all = [p for p in range(L) if P <= p < L - W]
    dm = (got[mid_all, :R].float() - want[mid_all].float()).abs()
    ref = want[mid_all].float().reshape(-1, GS)
    step = ((ref.amax(-1) - ref.amin(-1)) / 3.0).clamp(min=1e-6)
    step = step.repeat_interleave(GS).reshape(dm.shape)
    check(
        "extend: middle rows match the torch reference to within one code",
        int((dm > step * 1.01).sum()) == 0,
        f"codes>1 step=0 of {dm.numel()}",
    )
    check("extend: k_pe passthrough",
          torch.equal(got[:, R:], pe.reshape(L, ROPE)), "")

    # A middle token must NOT be exact -- otherwise "windows work" would be
    # indistinguishable from "nothing is quantized".
    mid = [p for p in range(L) if P <= p < L - W]
    moved = (got[mid, :R].float() - x.reshape(L, R)[mid].float()).abs().max().item()
    check("extend: middle tokens really are 2-bit", moved > 1e-3,
          f"max_abs vs original={moved:.3e} over {len(mid)} rows")

    # ---- decode: one new token per step, ring laps, old tokens age out ------
    aged = []
    for step in range(1, W + 2):
        pos = L - 1 + step
        seq = pos + 1
        nx = torch.randn(1, 1, R, device=dev, dtype=torch.bfloat16)
        npe = torch.randn(1, 1, ROPE, device=dev, dtype=torch.bfloat16)
        nslot = torch.tensor([64 + L + step - 1], device=dev, dtype=torch.int64)
        pool.note_forward_batch(fb(True, [seq], [0]))
        pool.set_kv_buffer(_Layer(0), nslot, torch.cat([nx, npe], dim=-1), None)
        aged.append(pos - W)     # the position that just left the window

    got2 = pool.materialize_rows(0, slots)[:, 0, :R]
    # positions in `aged` that are >= P should now come back dequantized
    demoted = [p for p in aged if P <= p < L]
    still = (got2[demoted].float() - x.reshape(L, R)[demoted].float()).abs()
    check(
        "decode: tokens that aged out of the recent window are demoted",
        bool(len(demoted) > 0 and still.max().item() > 1e-3),
        f"positions {demoted} now differ from BF16 by {still.max().item():.3e}",
    )
    sink_ok = (got2[:P].float() - x.reshape(L, R)[:P].float()).abs().max().item()
    check("decode: the sink is never demoted", sink_ok < 1e-6, f"max_abs={sink_ok:.2e}")

    # ---- a padded graph replay must not disturb a live ring ----------------
    before = pool.hp_owner_of_row.clone()
    padded = torch.zeros(1, dtype=torch.int64, device=dev)     # out_cache_loc = 0
    pool.note_forward_batch(fb(True, [1], [0]))
    pool.set_kv_buffer(
        _Layer(0),
        padded,
        torch.cat([torch.randn(1, 1, R, device=dev, dtype=torch.bfloat16),
                   torch.randn(1, 1, ROPE, device=dev, dtype=torch.bfloat16)], dim=-1),
        None,
    )
    check(
        "padded replay (slot 0, stale req index) leaves the ring untouched",
        torch.equal(before, pool.hp_owner_of_row),
        "",
    )

    # ---- two requests must not share ring rows -----------------------------
    L2 = 12
    x2 = torch.randn(L2, 1, R, device=dev, dtype=torch.bfloat16)
    pe2 = torch.randn(L2, 1, ROPE, device=dev, dtype=torch.bfloat16)
    slots2 = torch.arange(200, 200 + L2, device=dev, dtype=torch.int64)
    pool.note_forward_batch(
        fb(False, [L2], [1],
           positions=torch.arange(L2, device=dev, dtype=torch.int64),
           extend_seq_lens=torch.tensor([L2], device=dev, dtype=torch.int64))
    )
    pool.set_kv_buffer(_Layer(0), slots2, torch.cat([x2, pe2], dim=-1), None)
    sink_a = (
        pool.materialize_rows(0, slots)[:P, 0, :R].float()
        - x.reshape(L, R)[:P].float()
    ).abs().max().item()
    check("request 1's window does not evict request 0's",
          sink_a < 1e-6, f"req0 sink max_abs={sink_a:.2e}")

    # ---- layers are independent -------------------------------------------
    l1 = pool.materialize_rows(1, slots)[:, 0, :R]
    check("layer 1 is a separate buffer (untouched -> zero)",
          bool(l1.abs().max().item() == 0.0), "")

    print()
    print(f"pool KV bytes: {pool.get_kv_size_bytes()/2**20:.2f} MiB for "
          f"{SIZE + PAGE} rows x {LAYERS} layers "
          f"(BF16 would be {(SIZE+PAGE)*(R+ROPE)*2*LAYERS/2**20:.2f} MiB)")
    if FAILURES:
        print("FAILED:", ", ".join(FAILURES))
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("needs a GPU")
        sys.exit(1)
    main()
