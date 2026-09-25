"""Runtime invariant auditor for the unified mixed HP+int2 KV pool.

Off by default; enabled with ``SGLANG_MIXED_KV_AUDIT=1``. Every
``SGLANG_MIXED_KV_AUDIT_EVERY`` decode steps (default 25) it validates the
invariants that keep mixed-KV tiering and the radix cache consistent, and
logs the first violation of each kind (rate-limited).

The invariants, and why each one matters:

1. **Flush windows are all-or-nothing.** ``gpu_flush_int2_plan`` allocates one
   whole quant page per flushing request and hands back the *unused* slots of
   that page via ``returned_slot_ids``. ``UnifiedInt2HPKVAllocator.free``
   aggregates quant slot ids to whole pages, so a partially-used page is
   returned to ``free_pages`` while some of its slots are live in
   ``req_to_token``. That page is then handed to another request → two writers
   on the same physical slots → silently corrupted KV. A flushing request must
   therefore have ``valid_mask`` all-ones (page fully consumed) or all-zeros
   (page fully returned).

2. **The HP-recent ring must drain.** Live HP-recent slots are a contiguous
   position suffix of length <= ``hp_recent_ring_size``. Ring slots are only
   reclaimed by the flush; if a flush window is skipped (``valid`` all-zero
   while the request was gated to flush) the ring never drains, wraps, and
   overwrites positions that are still live.

3. **Ring ids are per-request and unique across live positions.** The same ring
   slot id appearing at two live positions of one request means the ring lapped
   a position that was never demoted (invariant 2 violated earlier).

4. **HP-prefix ids only appear below ``hp_prefix_tokens``**, and ring ids only
   inside this request's own slab.

5. **No quant slot id is shared by two requests outside their cached prefix.**
   Sharing inside ``[0, cache_protected_len)`` is exactly what the radix cache
   is for; sharing above it is a double-allocation.

Cost: the flush-plan checks sync twice per decode step, and the ``req_to_token``
scan (invariants 2-5) walks every live token in Python, so it is O(tokens in
flight) every ``SGLANG_MIXED_KV_AUDIT_EVERY`` steps. On a GPQA-198 run at 32
concurrency with the default 25 that cost ~1 minute of the ~13; raise the
interval for long-context runs. Never leave it on for benchmarking.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_ENABLED: Optional[bool] = None
_EVERY = 0
_step = 0
_content_step = 0
_content: dict[int, dict[int, tuple[int, float]]] = {}
_reported: dict[str, int] = {}
_MAX_REPORTS = 8


def audit_enabled() -> bool:
    global _ENABLED, _EVERY
    if _ENABLED is None:
        _ENABLED = os.environ.get("SGLANG_MIXED_KV_AUDIT", "0") not in ("0", "", "false")
        _EVERY = int(os.environ.get("SGLANG_MIXED_KV_AUDIT_EVERY", "25"))
        if _ENABLED:
            logger.warning(
                "SGLANG_MIXED_KV_AUDIT is ON (every %d decode steps). "
                "This syncs the decode hot path; do not use for benchmarking.",
                _EVERY,
            )
    return _ENABLED


def _report(kind: str, msg: str) -> None:
    n = _reported.get(kind, 0)
    if n >= _MAX_REPORTS:
        return
    _reported[kind] = n + 1
    logger.error("[mixed-kv-audit] %s: %s", kind, msg)


def audit_flush_plan(plan, flush_mask: torch.Tensor, seq_lens: torch.Tensor,
                     prefix_lens: torch.Tensor, req_pool_indices: torch.Tensor) -> None:
    """Invariants 1 + 2: every gated flush consumes or returns a whole page."""
    if plan is None or not audit_enabled():
        return
    bs = plan.bs
    fi = plan.flush_interval
    valid = plan.valid_mask.view(bs, fi).sum(dim=1)
    gated = flush_mask.to(torch.int32)
    straddle = (gated == 1) & (valid > 0) & (valid < fi)
    skipped = (gated == 1) & (valid == 0)
    if bool(straddle.any().item()):
        idx = torch.nonzero(straddle).flatten().tolist()
        _report(
            "STRADDLED_FLUSH_PAGE",
            f"reqs={idx} valid={valid[straddle].tolist()} "
            f"seq_lens={seq_lens[straddle].tolist()} "
            f"prefix_lens={prefix_lens[straddle].tolist()} "
            f"req_pool_idx={req_pool_indices[straddle].tolist()} "
            "-> partially-used quant page returned to the free list "
            "(double allocation, corrupts KV)",
        )
    if bool(skipped.any().item()):
        idx = torch.nonzero(skipped).flatten().tolist()
        _report(
            "SKIPPED_FLUSH",
            f"reqs={idx} seq_lens={seq_lens[skipped].tolist()} "
            f"prefix_lens={prefix_lens[skipped].tolist()} "
            f"req_pool_idx={req_pool_indices[skipped].tolist()} "
            "-> HP-recent ring did not drain this cycle (will lap live slots)",
        )


def audit_kv_content(batch, kv_pool) -> None:
    """Content stability: a committed position's KV must never change.

    For every live request we fingerprint layer 0's K row for a sample of
    positions and remember it per ``(req_pool_idx, position, slot)``. A
    position's slot legitimately changes exactly once, when the flush demotes
    it from the HP-recent ring into a quant page (fingerprint re-based). What
    must never happen is the *content* of a slot changing while a live position
    still points at it -- that is another writer on our KV, which is what a
    double-allocated quant page or a lapped ring slot looks like from the
    reader's side.
    """
    global _content_step
    if not audit_enabled():
        return
    _content_step += 1
    if _EVERY <= 0 or (_content_step % _EVERY) != 0:
        return
    try:
        hp_k = kv_pool.get_hp_key_buffer(kv_pool.start_layer)
        q_k = kv_pool.get_raw_key_buffer(kv_pool.start_layer)
    except Exception:  # geometry without these accessors
        return
    hp_off = int(kv_pool.hp_global_offset)
    rtt = batch.req_to_token_pool.req_to_token
    seq_lens = batch.seq_lens.tolist()
    req_pool_indices = batch.req_pool_indices.tolist()

    for i, req in enumerate(batch.reqs):
        seq_len = int(seq_lens[i])
        rpi = int(req_pool_indices[i])
        if seq_len <= 0:
            continue
        # sample: the whole HP-prefix window plus a stride over the rest
        positions = list(range(0, min(seq_len, int(kv_pool.hp_prefix_tokens)), 8))
        positions += list(range(int(kv_pool.hp_prefix_tokens), seq_len, 97))
        if not positions:
            continue
        pos_t = torch.tensor(positions, dtype=torch.int64, device=rtt.device)
        slots = rtt[rpi, pos_t].to(torch.int64)
        # HP rows and int2-packed quant rows have different widths, so
        # summarize each tier separately and stitch the per-position values.
        sig = [0.0] * len(positions)
        hp_sel = slots >= hp_off
        if bool(hp_sel.any()):
            hp_rows = hp_k[(slots[hp_sel] - hp_off)].reshape(int(hp_sel.sum()), -1)
            vals = hp_rows.to(torch.float32).sum(dim=-1).to("cpu").tolist()
            for j, v in zip(torch.nonzero(hp_sel).flatten().tolist(), vals):
                sig[j] = v
        if bool((~hp_sel).any()):
            q_rows = q_k[slots[~hp_sel]].reshape(int((~hp_sel).sum()), -1)
            vals = q_rows.to(torch.float32).sum(dim=-1).to("cpu").tolist()
            for j, v in zip(torch.nonzero(~hp_sel).flatten().tolist(), vals):
                sig[j] = v
        slots_cpu = slots.to("cpu").tolist()
        gen = _content.setdefault(rpi, {})
        for pos, slot, s in zip(positions, slots_cpu, sig):
            prev = gen.get(pos)
            if prev is None or prev[0] != slot:
                gen[pos] = (slot, s)          # first sight, or legitimate demote
            elif prev[1] != s:
                _report(
                    "KV_CONTENT_CHANGED",
                    f"rpi={rpi} pos={pos} slot={slot} seq_len={seq_len} "
                    f"protected={getattr(req, 'cache_protected_len', -1)} "
                    f"sig {prev[1]:.6g} -> {s:.6g}: another writer touched a "
                    "live position's KV",
                )
                gen[pos] = (slot, s)


def report_decode_write_into_hp_prefix(bad_locs, total: int) -> None:
    """A decode KV write aimed at the shared HP-prefix pool.

    Decode allocates from ``alloc_hp_recent``, so every real entry is a
    per-request ring slot. An entry pointing into the shared HP-prefix pool can
    only come from CUDA-graph padding (the graph runner fills padded
    ``out_cache_loc`` entries with ``hp_global_offset``, i.e. HP-prefix slot 0,
    which is a *live, allocatable* page -- only quant page 0 is reserved), and
    the write clobbers whatever request or radix-tree node owns it.
    """
    _report(
        "DECODE_WRITE_INTO_HP_PREFIX",
        f"n_bad={bad_locs.numel()}/{total} locs={bad_locs[:8].tolist()}: "
        "decode wrote into the shared HP-prefix pool (CUDA-graph padding?)",
    )


def audit_hp_prefix_pages(action: str, pages, extra: str = "") -> None:
    """Trace HP-prefix page alloc/free so a page that is handed out while the
    radix tree still owns it can be traced to its releaser.

    HP-prefix pages back the shared prefix window: many live requests read the
    same page through the tree. Freeing one while it is still referenced --
    ``free`` aggregates to whole pages, so freeing a single slack slot of a
    page releases the whole thing -- lets the next ``alloc_hp_prefix`` hand it
    to a new request, whose prefill then overwrites the shared prefix's K/V
    under everyone's feet.
    """
    if not audit_enabled():
        return
    if isinstance(pages, torch.Tensor):
        if pages.numel() == 0:
            return
        page_list = pages.to("cpu").tolist()
    else:
        page_list = list(pages)
        if not page_list:
            return
    where = ""
    if action == "free":
        import traceback

        frames = [
            f"{f.filename.rsplit('/', 1)[-1]}:{f.lineno}:{f.name}"
            for f in traceback.extract_stack()[-7:-1]
        ]
        where = " <- " + " <- ".join(reversed(frames))
    logger.warning(
        "[mixed-kv-audit] HP_PREFIX_%s pages=%s%s%s",
        action.upper(),
        page_list[:16],
        (" " + extra) if extra else "",
        where,
    )


def audit_release(req_pool_idx: int) -> None:
    """Drop per-request fingerprints when a req slot is recycled."""
    if not audit_enabled():
        return
    _content.pop(int(req_pool_idx), None)


#: ``latent_grid_error`` below this means "this row is already on the INT2
#: grid". A quantized row only misses the grid by bf16 storage rounding
#: (~1e-2); an unquantized one sits between levels, i.e. near 0.5.
LATENT_GRID_TOL = 0.25


def latent_grid_error(kv_pool, layer_id: int, rows: torch.Tensor) -> torch.Tensor:
    """Per-row evidence that a stored latent has been through INT2.

    The MLA/NSA pool fake-quantizes into a float cache, so a token's tier is
    not readable from its slot id the way it is in the unified pool -- every
    row is BF16-shaped either way. It is readable from the row's *content*,
    but only in the right basis: the round trip ends with a dense 512x512
    un-rotation, so a quantized row looks like arbitrary floats in storage.
    Rotate it back and each ``group_size`` block collapses onto 4 uniformly
    spaced levels, because that is all 2 bits can hold -- for the Lloyd-Max
    path too, whose dequant is also ``(q - zero) * scale``.

    So: rotate, fit each group's min/max, and ask how far the values are from
    integer level indices. ~0 means "already quantized", ~0.5 means "still
    the bf16 value". Returns one number per row (the worst group).
    """
    rank = int(kv_pool.kv_lora_rank)
    group = int(getattr(kv_pool, "_group_size", 128))
    n = rows.shape[0]
    x = rows[..., :rank].reshape(n, rank).to(torch.float32)
    rotations = getattr(kv_pool, "rotations_f32", None)
    R = rotations.get(layer_id) if rotations else None
    if R is not None:
        x = x @ R
    g = x.reshape(-1, group)
    lo = g.amin(dim=-1, keepdim=True)
    hi = g.amax(dim=-1, keepdim=True)
    scale = torch.where((hi - lo).abs() > 1e-8, (hi - lo) / 3.0, torch.ones_like(lo))
    q = (g - lo) / scale
    return (q - q.round()).abs().amax(dim=-1).reshape(n, -1).amax(dim=-1)


def audit_latent_windows(batch, kv_pool) -> None:
    """Tier check for the MLA/NSA latent pool's BF16 sink + recent windows.

    The invariant, for a request of length ``L`` with sink ``P`` / recent
    ``R``: positions ``[0, P)`` and ``[L-R, L)`` are BF16, positions
    ``[P, L-R)`` are INT2. A violation in the middle band means the demote
    never ran -- the "window" is really "everything after the prompt stays
    BF16", which would inflate the score and read as a win. A violation in
    either window means the write path quantized a row it should have
    skipped. Tiers come from :func:`latent_grid_error`, so this needs no
    reference copy of the latents.
    """
    global _step
    if not audit_enabled():
        return
    if not getattr(kv_pool, "latent_windows_enabled", lambda: False)():
        return
    if getattr(kv_pool, "hp_subspaces", None):
        # The HP-subspace variant adds a full-precision component back after
        # the round trip, so a quantized row is no longer on the grid.
        return
    _step += 1
    if _EVERY <= 0 or (_step % _EVERY) != 0:
        return

    P = int(kv_pool.hp_prefix_tokens)
    R = int(kv_pool.hp_recent_tokens)
    layer_id = int(kv_pool.start_layer)
    buf = kv_pool.get_key_buffer(layer_id)
    rtt = batch.req_to_token_pool.req_to_token
    seq_lens = batch.seq_lens.tolist()
    req_pool_indices = batch.req_pool_indices.tolist()

    for i in range(len(req_pool_indices)):
        seq_len = int(seq_lens[i])
        rpi = int(req_pool_indices[i])
        if seq_len <= 0:
            continue
        # One sample per band; a band that does not exist yet is skipped.
        # ``seq_lens`` here is the length *before* this step's token, which is
        # exactly the length the previous forward's demote worked to.
        probes = []
        if P > 0:
            probes.append(("sink", 0, False))
        mid_lo, mid_hi = P, seq_len - R
        if mid_hi > mid_lo:
            probes.append(("quant", (mid_lo + mid_hi) // 2, True))
            probes.append(("quant_edge", mid_hi - 1, True))
        if R > 0 and seq_len - 1 >= max(P, seq_len - R):
            probes.append(("recent", seq_len - 1, False))
        if not probes:
            continue
        pos_t = torch.tensor(
            [p for _, p, _ in probes], dtype=torch.int64, device=rtt.device
        )
        slots = rtt[rpi, pos_t].to(torch.int64)
        errs = latent_grid_error(kv_pool, layer_id, buf[slots]).tolist()
        for (band, pos, want_quant), err in zip(probes, errs):
            if want_quant and err > LATENT_GRID_TOL:
                _report(
                    "LATENT_NOT_QUANTIZED",
                    f"rpi={rpi} pos={pos} seq_len={seq_len} band={band} "
                    f"grid_err={err:.3f}>{LATENT_GRID_TOL} (sink={P} "
                    f"recent={R}) -> a token outside both BF16 windows was "
                    "never pushed through INT2; the recent window is not "
                    "draining",
                )
            elif (not want_quant) and err <= LATENT_GRID_TOL:
                _report(
                    "LATENT_WINDOW_QUANTIZED",
                    f"rpi={rpi} pos={pos} seq_len={seq_len} band={band} "
                    f"grid_err={err:.3f}<={LATENT_GRID_TOL} (sink={P} "
                    f"recent={R}) -> a token inside a BF16 window was "
                    "quantized anyway",
                )


def audit_req_to_token(batch, kv_pool) -> None:
    """Invariants 2-5, read off ``req_to_token`` for every live request."""
    global _step
    if not audit_enabled():
        return
    _step += 1
    if _EVERY <= 0 or (_step % _EVERY) != 0:
        return

    hp_off = int(kv_pool.hp_global_offset)
    recent_base = hp_off + int(kv_pool.num_hp_prefix_slots)
    ring = int(kv_pool.hp_recent_ring_size)
    hp_prefix_tokens = int(kv_pool.hp_prefix_tokens)
    rtt = batch.req_to_token_pool.req_to_token
    seq_lens = batch.seq_lens.tolist()
    req_pool_indices = batch.req_pool_indices.tolist()

    quant_owner: dict[int, tuple[int, int]] = {}
    for i, req in enumerate(batch.reqs):
        seq_len = int(seq_lens[i])
        rpi = int(req_pool_indices[i])
        if seq_len <= 0:
            continue
        row = rtt[rpi, :seq_len].to("cpu", torch.int64)
        protected = int(getattr(req, "cache_protected_len", 0))

        is_recent = row >= recent_base
        is_hp_prefix = (row >= hp_off) & (row < recent_base)
        recent_pos = torch.nonzero(is_recent).flatten()
        if recent_pos.numel():
            first = int(recent_pos[0])
            last = int(recent_pos[-1])
            n_recent = int(recent_pos.numel())
            if last != seq_len - 1 or (last - first + 1) != n_recent:
                _report(
                    "RING_NOT_SUFFIX",
                    f"rpi={rpi} seq_len={seq_len} first={first} last={last} "
                    f"n={n_recent} protected={protected}",
                )
            if n_recent > ring:
                _report(
                    "RING_OVERFULL",
                    f"rpi={rpi} seq_len={seq_len} n_recent={n_recent} > ring={ring}",
                )
            slab_lo = recent_base + rpi * ring
            ids = row[is_recent]
            if int(ids.min()) < slab_lo or int(ids.max()) >= slab_lo + ring:
                _report(
                    "RING_ID_FOREIGN_SLAB",
                    f"rpi={rpi} ids=[{int(ids.min())},{int(ids.max())}] "
                    f"slab=[{slab_lo},{slab_lo + ring})",
                )
            if int(torch.unique(ids).numel()) != n_recent:
                _report(
                    "RING_ID_ALIASED",
                    f"rpi={rpi} seq_len={seq_len} n_recent={n_recent} "
                    f"unique={int(torch.unique(ids).numel())} protected={protected} "
                    "-> a live position's HP-recent slot was reused by a newer token",
                )
        hp_prefix_pos = torch.nonzero(is_hp_prefix).flatten()
        if hp_prefix_pos.numel() and int(hp_prefix_pos[-1]) >= hp_prefix_tokens:
            _report(
                "HP_PREFIX_ABOVE_WINDOW",
                f"rpi={rpi} last_hp_prefix_pos={int(hp_prefix_pos[-1])} "
                f"hp_prefix_tokens={hp_prefix_tokens}",
            )

        quant_pos = torch.nonzero(row < hp_off).flatten()
        qids = row[row < hp_off]
        if int(torch.unique(qids).numel()) != int(qids.numel()):
            _report(
                "QUANT_ID_ALIASED_IN_REQ",
                f"rpi={rpi} seq_len={seq_len} n_quant={int(qids.numel())} "
                f"unique={int(torch.unique(qids).numel())}",
            )
        for p, s in zip(quant_pos.tolist(), qids.tolist()):
            prev = quant_owner.get(s)
            if prev is None:
                quant_owner[s] = (rpi, p)
            elif prev[0] != rpi and (p >= protected or prev[1] >= protected):
                _report(
                    "QUANT_ID_SHARED_ABOVE_PREFIX",
                    f"slot={s} req({prev[0]})@pos{prev[1]} vs req({rpi})@pos{p} "
                    f"protected={protected} -> two writers on one quant slot",
                )
