"""MiniMax-M3 block-sparse attention kernels (Triton).

Ported from sglang upstream ``sglang/kernels/ops/attention/minimax_sparse``
(Copyright 2025 XunhaoLai, Apache-2.0). Two stages per layer:

  index   ``flash_*_with_topk_index``: the lightning indexer scores every key
          block against the query (max over the block's tokens) and keeps the
          top-k blocks plus the local block;
  main    ``flash_*_with_gqa_share_sparse``: GQA attention restricted to the
          selected blocks, reading K/V by ``req_to_token[slot_ids[b], pos]``.

Unchanged except for import paths. The packed/INT2 integration lives in
``minimax_sparse_backend.py``, which stages dequantized rows into a BF16
buffer and hands the kernels a remapped ``req_to_token``.
"""
