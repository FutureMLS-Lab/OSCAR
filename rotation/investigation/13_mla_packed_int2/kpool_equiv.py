#!/usr/bin/env python3
"""Is the ported k-pool selection stage the reference's?

Same method as the mHC test, for the same reason: the details are not
recoverable from shapes, and every wrong variant still returns a correctly
shaped index tensor that the sparse-attention kernel will happily consume. A
selector that picks the wrong 2048 tokens does not crash -- it degrades quality
in a way that looks like "2-bit KV is lossy".

Compares against the real ``Glm5NextTextIndexer`` methods, with left padding in
the batch so the first-real-token anchoring is actually exercised (with no
padding, anchoring at slot 0 and at the first valid key agree, and the test
would pass against a wrong implementation).
"""
from __future__ import annotations

import os
import sys

import torch


def main() -> int:
    torch.manual_seed(0)
    ref_path = os.environ.get("REF_TF", "")
    if ref_path and ref_path not in sys.path:
        sys.path.insert(0, ref_path)

    try:
        from transformers.models.glm5_next.modeling_glm5_next import (
            Glm5NextTextIndexer,
        )
        import transformers
    except Exception as e:  # noqa: BLE001
        print(f"cannot import the reference: {type(e).__name__}: {str(e)[:200]}")
        return 2
    print(f"reference transformers {transformers.__version__}")

    from sglang.srt.layers.attention.nsa.glm5_next_kpool import (
        append_visible_tail,
        build_pools,
        select_pools,
        visible_tokens,
    )

    B, S, D, KP, NH, TOPK = 2, 20, 8, 4, 3, 8
    HIDDEN, QLORA = 16, 12

    class Cfg:
        hidden_size, index_n_heads, index_head_dim = HIDDEN, NH, D
        qk_rope_head_dim, index_topk, q_lora_rank = 0, TOPK, QLORA
        index_kpool, index_kpool_always_select_tail = KP, True

    try:
        ref = Glm5NextTextIndexer(Cfg(), layer_idx=0)
    except TypeError:
        ref = Glm5NextTextIndexer(Cfg())
    ref.eval()

    ape = torch.randn(KP, D)
    with torch.no_grad():
        ref.index_kpool_compress_ape.copy_(ape)

    # LEFT PADDING is the point: without it, anchoring at slot 0 and at the
    # first valid key agree and this test would pass a wrong implementation.
    keys = torch.randn(B, S, D)
    gate = torch.randn(B, S, D)
    valid = torch.ones(B, S, dtype=torch.bool)
    valid[0, :5] = False           # 5 pad tokens on row 0
    valid[1, :2] = False           # 2 on row 1
    packed = torch.cat([keys, gate, valid.to(keys.dtype)[..., None]], dim=-1)

    ok = True
    with torch.no_grad():
        r_pk, r_pi, r_pv = ref.get_pooled_states(packed_states=packed)
        m_pk, m_pi, m_pv = build_pools(keys, gate, valid, ape, KP)

    for name, a, b in (("pool_keys", r_pk, m_pk),
                       ("pool_indices", r_pi, m_pi),
                       ("pool_valid", r_pv, m_pv)):
        if a.shape != b.shape:
            print(f"  {name:<13} SHAPE {tuple(a.shape)} vs {tuple(b.shape)}  MISMATCH")
            ok = False
            continue
        if a.dtype == torch.bool or a.dtype == torch.long:
            same = bool((a == b).all())
            print(f"  {name:<13} exact {'MATCH' if same else 'MISMATCH'}")
            ok &= same
        else:
            d = (a - b).abs().max().item()
            good = d < 1e-6
            ok &= good
            print(f"  {name:<13} max|d| {d:.3e}  {'MATCH' if good else 'MISMATCH'}")

    with torch.no_grad():
        r_vis = ref.get_visible_tokens(valid_keys=valid, q_length=S, current_length=S)
        m_vis = visible_tokens(valid, S, S)
    same = bool((r_vis == m_vis).all())
    ok &= same
    print(f"  visible       exact {'MATCH' if same else 'MISMATCH'}")

    # End to end: the whole selection stage against the reference indexer's
    # forward. This is the one that matters -- the pieces can each match and
    # still be wired together wrongly.
    with torch.no_grad():
        hidden = torch.randn(B, S, HIDDEN)
        q_resid = torch.randn(B, S, QLORA)
        r_topk = ref(hidden_states=hidden, q_resid=q_resid,
                     attention_mask=valid, past_key_values=None)

        # Mine, following the reference's own order of operations.
        q = ref.wq_b(q_resid).view(B, S, -1, D)
        k = ref.k_norm(ref.wk(hidden)).view(B, S, -1, D).squeeze(2)
        gate_scores = torch.nn.functional.linear(hidden, ref.index_kpool_compress_gate)
        vis = visible_tokens(valid, S, S)
        pk, pi, pv = build_pools(k, gate_scores, valid, ape, KP)
        hw = ref.weights_proj(hidden).float() * (NH ** -0.5)
        m_topk, _ = select_pools(q, pk, pi, pv, hw, vis,
                                 ref.softmax_scale, TOPK, KP, S)
        m_topk = append_visible_tail(m_topk, vis, valid, KP)
        width = TOPK + (KP - 1 if Cfg.index_kpool_always_select_tail else 0)
        m_topk = torch.nn.functional.pad(
            m_topk, (0, max(0, width - m_topk.shape[-1])), value=-1)[..., :width]
        m_topk = m_topk.masked_fill(~valid[..., None], -1).to(torch.int32)

    same = r_topk.shape == m_topk.shape and bool((r_topk == m_topk).all())
    ok &= same
    print(f"  END-TO-END   {tuple(r_topk.shape)} vs {tuple(m_topk.shape)}  "
          f"{'exact MATCH' if same else 'MISMATCH'}")
    if not same and r_topk.shape == m_topk.shape:
        diff = (r_topk != m_topk)
        print(f"    differing entries: {int(diff.sum())}/{r_topk.numel()}")

    # Negative controls -- a test that also passes the obvious-but-wrong
    # implementations proves nothing about the port.
    with torch.no_grad():
        # (a) anchor at slot 0 instead of the first real token
        naive_idx = torch.arange(
            ((S + KP - 1) // KP) * KP).view(1, -1, KP).expand(B, -1, -1)
        diff = int((naive_idx[:, : m_pi.shape[1]] != m_pi).sum())
        # (b) plain mean instead of the gated weighted average
        print(f"\n  [control] slot-0 anchoring differs in {diff} pool index entries")
        mean_pk = torch.zeros_like(m_pk)
        print(f"  [control] a plain mean would ignore ape entirely; ape norm = "
              f"{ape.norm():.3f}")
        # (c) .any() instead of .all() for pool validity
        any_valid = (m_pi >= 0).any(-1)
        print(f"  [control] .any() vs .all() pool_valid differs on "
              f"{int((any_valid != m_pv).sum())} pools")

    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
