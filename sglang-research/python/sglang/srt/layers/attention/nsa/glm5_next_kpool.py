"""GLM-5.3-Flash's k-pool indexer selection stage.

GLM-5.2 and GLM-5.3-Flash are both DSA, and the bulk of sglang's 1608-line
``Indexer`` carries over unchanged: ``wq_b`` / ``wk`` / ``k_norm`` /
``weights_proj`` have the same names, shapes and roles (verified against the
checkpoint), and the output is still a ``topk_indices`` tensor of width
``index_topk``, so the sparse-attention consumer sees the same interface.

What is new is the SELECTION stage. GLM-5.2 scores every key; GLM-5.3-Flash
groups keys into pools of ``index_kpool`` (4 here), scores the pools, takes
``index_topk // index_kpool`` of them, and expands each back into its member
tokens -- plus an always-selected tail. ``grep -i pool`` over
``nsa_indexer.py`` finds 26 hits and every one of them is memory-pool naming
(``token_to_kv_pool``, ``req_to_token_pool``); there is no k-pooling in it.

Ported from ``transformers.models.glm5_next`` (5.16.1) rather than derived, for
the same reason the mHC port was: the details are not recoverable from shapes.

Four of them, each of which produces a working-but-wrong selector:

  * Pools start at the **first real token**, not at slot 0. With left padding
    ``[P, P, A, B, C, D]`` and kpool 4, pool 0 must be ``[A, B, C, D]``, not
    ``[P, P, A, B]``.
  * The pool's key is a **learned weighted average**: ``gate_scores`` plus the
    per-slot ``ape``, softmaxed **over the token axis inside the pool**, then
    used to weight the member keys. Not a mean, and not a softmax over pools.
  * ``pool_valid`` is ``.all(-1)`` -- a pool counts only if **every** member is
    valid. ``.any(-1)`` also runs and quietly admits partial pools.
  * A fully-invalid pool softmaxes ``-inf`` over the whole axis and yields NaN,
    which ``nan_to_num`` clears. Without it the NaN reaches the scores and the
    top-k picks garbage.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def build_pools(
    keys: torch.Tensor,
    gate_scores: torch.Tensor,
    valid_keys: torch.Tensor,
    ape: torch.Tensor,
    index_kpool: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group [B, S, D] keys into pools and return (pool_keys, pool_indices, pool_valid).

    ``ape`` is [index_kpool, D]. Shapes follow the reference: pool_keys
    [B, P, D], pool_indices [B, P, index_kpool], pool_valid [B, P].
    """
    batch_size, seq_len = keys.shape[:2]
    device = keys.device
    number_of_pools = (seq_len + index_kpool - 1) // index_kpool

    # Pools are anchored on the first VALID key, so left padding does not shift
    # every pool boundary by the pad width.
    first_key = torch.where(
        valid_keys.any(-1),
        valid_keys.long().argmax(-1),
        torch.full((batch_size,), seq_len, dtype=torch.long, device=device),
    )
    pool_offsets = torch.arange(number_of_pools * index_kpool, device=device)
    pool_offsets = pool_offsets.view(1, number_of_pools, index_kpool)
    pool_indices = first_key[:, None, None] + pool_offsets

    batch_idx = torch.arange(batch_size, device=device)[:, None, None]
    safe_indices = pool_indices.clamp(0, seq_len - 1)

    grouped_keys = keys[batch_idx, safe_indices]
    grouped_gate = gate_scores[batch_idx, safe_indices]
    grouped_valid = valid_keys[batch_idx, safe_indices]

    # clamp() above made out-of-range slots alias the last token; mask them.
    grouped_valid = grouped_valid & (pool_indices < seq_len)
    pool_valid = grouped_valid.all(-1)          # ALL, not ANY
    pool_indices = pool_indices.masked_fill(~grouped_valid, -1)

    logits = grouped_gate.float() + ape.float()[None, None]
    logits = logits.masked_fill(~grouped_valid[..., None], float("-inf"))
    # dim=2 is the token axis INSIDE a pool. A fully invalid pool is all -inf,
    # whose softmax is NaN.
    probs = torch.nan_to_num(logits.softmax(dim=2)).to(grouped_keys.dtype)
    pool_keys = (probs * grouped_keys).sum(dim=2)

    keep = pool_valid.any(0)
    return pool_keys[:, keep], pool_indices[:, keep], pool_valid[:, keep]


def visible_tokens(
    valid_keys: torch.Tensor, q_length: int, current_length: int
) -> torch.Tensor:
    """Causal AND not-padding, as [B, q_length, kv_len]."""
    device = valid_keys.device
    kv_positions = torch.arange(valid_keys.shape[-1], device=device)
    q_positions = current_length - q_length + torch.arange(q_length, device=device)
    causal = kv_positions[None, None, :] <= q_positions[None, :, None]
    return causal & valid_keys[:, None, :]


def select_pools(
    q: torch.Tensor,
    pool_keys: torch.Tensor,
    pool_indices: torch.Tensor,
    pool_valid: torch.Tensor,
    head_weights: torch.Tensor,
    token_visible: torch.Tensor,
    softmax_scale: float,
    index_topk: int,
    index_kpool: int,
    kv_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score pools, take the top ``index_topk // index_kpool``, expand to tokens.

    ``q`` is [B, S, n_heads, D]; ``head_weights`` is [B, S, n_heads] already
    scaled by ``n_heads ** -0.5``.
    """
    batch_size, seq_len = q.shape[:2]

    scores = torch.matmul(q.float(), pool_keys.transpose(-1, -2).float().unsqueeze(1))
    scores = F.relu(scores * softmax_scale)
    index_scores = torch.matmul(head_weights.unsqueeze(-2), scores).squeeze(-2)

    # A pool is selectable only if its LAST token is visible to the query --
    # the pool is a contiguous run, so its final token bounds the whole pool.
    pool_end = pool_indices[..., -1].clamp(0, kv_len - 1)
    pool_visible = token_visible.gather(
        dim=-1, index=pool_end[:, None, :].expand(batch_size, seq_len, -1)
    )
    valid_candidates = pool_visible & pool_valid[:, None]
    index_scores = index_scores.masked_fill(
        ~valid_candidates, torch.finfo(index_scores.dtype).min
    )

    select_k = min(index_topk // index_kpool, index_scores.shape[-1])
    selected = index_scores.topk(select_k, dim=-1).indices
    batch_idx = torch.arange(batch_size, device=q.device)[:, None, None]

    selected_valid = valid_candidates.gather(-1, selected)
    selected_indices = pool_indices[batch_idx, selected]

    topk_indices = selected_indices.flatten(-2)
    topk_indices = topk_indices.masked_fill(
        ~selected_valid[..., None].expand_as(selected_indices).flatten(-2), -1
    )
    return topk_indices, selected_valid


def append_visible_tail(
    topk_indices: torch.Tensor,
    token_visible: torch.Tensor,
    key_valid: torch.Tensor,
    index_kpool: int,
) -> torch.Tensor:
    """Append the current INCOMPLETE pool as raw token indices.

    Pools cover whole groups of ``index_kpool``; whatever is left over at the
    end of the visible range belongs to no complete pool and would never be
    selectable, so it is appended verbatim. With kpool 4 and visible
    ``[A B C D E F]``, the pools give ``[A B C D]`` and the tail gives
    ``[E F]``.

    The width is ``index_kpool - 1``, not ``index_kpool``: a leftover of
    exactly ``index_kpool`` tokens is a COMPLETE pool and is already
    selectable, so the remainder is always in ``[0, index_kpool)``.

    ``tail_start`` counts from ``first_key``, not from 0, for the same reason
    pool boundaries do -- left padding must not shift it.
    """
    max_tail_width = index_kpool - 1
    if max_tail_width == 0:
        return topk_indices

    batch_size, _, kv_length = token_visible.shape
    device = token_visible.device

    first_key = torch.where(
        key_valid.any(-1),
        key_valid.long().argmax(-1),
        torch.full((batch_size,), kv_length, dtype=torch.long, device=device),
    )
    visible_count = token_visible.long().sum(-1)
    tail_count = visible_count.remainder(index_kpool)
    tail_offsets = torch.arange(max_tail_width, device=device)

    tail_start = first_key[:, None] + visible_count - tail_count
    tail_indices = tail_start[..., None] + tail_offsets

    # Three independent reasons a tail slot is dead: it is past this row's
    # remainder, it runs off the end of the cache, or it lands on a padded /
    # not-yet-visible token. All three have to be masked -- dropping any one
    # leaves a plausible index that silently points at the wrong token.
    tail_valid = (tail_offsets[None, None, :] < tail_count[..., None]) & tail_indices.lt(
        kv_length
    )
    kv_idx = tail_indices.clamp(0, kv_length - 1)
    tail_visible = token_visible.gather(dim=-1, index=kv_idx)
    tail_indices = tail_indices.masked_fill(~(tail_valid & tail_visible), -1)

    return torch.cat([topk_indices, tail_indices], dim=-1)
