"""The strided logits dump must address pruned logprob rows by their absolute
position: a chunk of a chunked prefill starts at the request's prefix length,
the pruned block of a sequence starts at its logprob start (one earlier when
only the sampled row is left), and a row's logits predict the next position.
Wrong by one here and two arms would be compared at different tokens."""
from sglang.srt.layers.logprob_processor import logits_dump_rows


def test_docstring_example_positions():
    # the example from LogitsProcessor._get_pruned_states
    token_to_seq_idx = [0, 0, 0, 0, 1, 2, 2, 2]
    rows = [0, 1, 2, 3, 5, 6, 7]
    sel, pos, in_idx = logits_dump_rows(
        rows, token_to_seq_idx, extend_seq_lens_cpu=[4, 5, 6], extend_logprob_start_lens_cpu=[0, 5, 3],
        extend_prefix_lens_cpu=[0, 10, 100], from_pos=0, stride=1,
    )
    assert sel == list(range(7))
    assert pos == [1, 2, 3, 4, 104, 105, 106]
    assert in_idx == [0, 1, 2, 3, 12, 13, 14]


def test_stride_and_from_filter_predicted_positions():
    token_to_seq_idx = [0] * 8
    rows = list(range(8))
    sel, pos, in_idx = logits_dump_rows(
        rows, token_to_seq_idx, extend_seq_lens_cpu=[8], extend_logprob_start_lens_cpu=[0],
        extend_prefix_lens_cpu=[16384], from_pos=0, stride=4,
    )
    assert pos == [16388, 16392]
    assert sel == [3, 7] and in_idx == [3, 7]


def test_chunk_before_logprob_start_contributes_nothing():
    # start == extend_len: only the sampled row survives pruning and it is never a logprob row
    sel, pos, in_idx = logits_dump_rows([], [0], [5], [5], [0], 0, 1)
    assert sel == [] and pos == [] and in_idx == []
