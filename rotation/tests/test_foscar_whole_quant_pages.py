"""CPU: with whole quant pages (FOSCAR, 64-token pages) every request's INT2 tier is a run of
full pages in logical order across chunked prefill and the decode flushes that follow, HP-recent
never exceeds the ring (hp_recent + N_Q - 1), and the flush countdown seeded at the final chunk
demotes exactly the page that starts at the INT2 tier's end."""
import random

from sglang.srt.mem_cache.allocation import _mixed_extend_layout_counts

N_Q, P, R = 64, 128, 512


def _simulate(prompt_len: int, prefix_len: int, chunk: int, decode_steps: int):
    """Positions the allocator puts in the INT2 tier, chunk by chunk and flush by flush; a radix
    hit of ``prefix_len`` tokens brings [0, P) as HP-prefix and [P, prefix_len) as INT2."""
    quant = list(range(P, prefix_len))
    pre, counter, hp_recent = prefix_len, None, 0
    while pre < prompt_len:
        end = prompt_len if prompt_len - pre <= chunk else (pre + chunk) // N_Q * N_Q
        final = end >= prompt_len
        hp_prefix_count, hp_recent_count, quant_count, quant_alloc, counter = _mixed_extend_layout_counts(
            pre, end, P, R, N_Q, is_final_chunk=final, whole_quant_pages=True
        )
        assert quant_alloc == quant_count, "no slack: the allocation is the quant count"
        # Logical order inside a chunk: [HP-prefix part][INT2][HP-recent].
        q0 = pre + hp_prefix_count
        quant += list(range(q0, q0 + quant_count))
        hp_recent = hp_recent_count if final else 0
        pre = end
    seq = prompt_len
    for _ in range(decode_steps):
        # advance_flush_counters, then the plan's demote range at the current length.
        if counter == 0:
            first = seq - R - (N_Q - 1)
            quant += list(range(first, first + N_Q))
            hp_recent -= N_Q
            counter = N_Q - 1
        else:
            counter -= 1
        seq += 1
        hp_recent += 1
        assert 0 <= hp_recent <= R + N_Q - 1, (seq, hp_recent)
    return quant


def test_final_chunk_moves_the_remainder_to_hp_recent():
    for seq_len in [700, 1000, 4161, 16449, 65601]:
        hp_prefix, hp_recent, quant, alloc, counter = _mixed_extend_layout_counts(
            0, seq_len, P, R, N_Q, is_final_chunk=True, whole_quant_pages=True
        )
        assert quant % N_Q == 0 and alloc == quant
        assert hp_prefix + quant + hp_recent == seq_len
        assert R <= hp_recent <= R + N_Q - 1
        # The flush fires `counter` steps later, when HP-recent holds R + N_Q - 1 tokens.
        assert hp_recent + counter == R + N_Q - 1


def test_without_the_flag_the_layout_is_unchanged():
    for seq_len in [700, 1000, 4161]:
        assert _mixed_extend_layout_counts(0, seq_len, P, R, 8, is_final_chunk=True) == _mixed_extend_layout_counts(
            0, seq_len, P, R, 8, is_final_chunk=True, whole_quant_pages=False
        )


def test_int2_tier_is_contiguous_whole_pages_through_chunks_and_flushes():
    rng = random.Random(0)
    for _ in range(400):
        prompt = rng.randint(1, 40000)
        prefix = rng.choice([0, 0, 64, 128, 192, 1024, 4096])
        prefix = min(prefix, max(0, (prompt - R - N_Q) // N_Q * N_Q))
        chunk = rng.choice([512, 2048, 8192, 1 << 20])
        positions = _simulate(prompt, prefix, chunk, decode_steps=rng.randint(0, 400))
        assert positions == list(range(P, P + len(positions))), "INT2 tier must be [P, P + n) in order"
        assert len(positions) % N_Q == 0, "INT2 tier must be whole pages"
