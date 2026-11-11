"""FOSCAR-backed INT2 tier of the unified mixed-KV pool (``SGLANG_OSCAR_FOSCAR``).

FOSCAR (github.com/FutureMLS-Lab/FOSCAR) owns the INT2 pages in its own region-major layout:
one ``QuantKVCache`` per layer over 64-token pages, written by ``append`` from rows already in
the OSCAR frame (rotated and centered), read back to bf16 by ``dequantize`` for prefill, and
decoded by ``PagedDecodeWrapper`` (``layers/attention/foscar_int2_decode.py``). The pool keeps
the HP tiers and the slot namespace: quant slot ``s`` is FOSCAR page ``s // 64``, position
``s % 64``, so the pool's quant page size must be 64 tokens.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)

# Fixed by FOSCAR: one code block and one scale block per (page, KV head).
FOSCAR_PAGE_TOKENS = 64


def foscar_supported() -> bool:
    try:
        import foscar
    except ImportError:
        return False
    return bool(foscar.is_supported())


def _clip_index(ratio: float, head_dim: int) -> int:
    """OSCAR's extend-quantizer order statistic (``clip_ratio_to_index``); -1 = no clip."""
    if ratio is None or ratio <= 0.0 or ratio >= 1.0:
        return -1
    return min(int(ratio * head_dim), head_dim - 1)


class FoscarInt2Tier:
    """Per-layer FOSCAR caches for the INT2 tier, plus one dummy page per layer.

    The dummy page (index ``num_quant_pages``) holds 64 zero tokens. Decode points a request
    with no INT2 tokens at it with length 1 and masks the result's LSE, because FOSCAR's graph
    mode takes lengths in ``[1, max_kv_len]``.

    K clip: FOSCAR's writer takes one compile-time clip index per cache, while the OSCAR recipe
    clips per (layer, KV head). The per-head clip is therefore applied here as a clamp to the
    head's order statistic before an unclipped ``append``, which writes the same values as an
    in-writer clip (the clip only clamps the row before its min/max are taken). A layer whose
    heads share one index uses FOSCAR's in-writer clip instead.
    """

    def __init__(
        self,
        *,
        num_quant_pages: int,
        layer_num: int,
        head_num: int,
        head_dim: int,
        v_head_dim: int,
        device: str,
        k_clip_layer: List[float],
        k_clip_idx_layer: Optional[List[torch.Tensor]],
        v_clip: float,
    ):
        from foscar import QuantKVLayout

        if head_dim != v_head_dim:
            raise ValueError(
                f"SGLANG_OSCAR_FOSCAR needs equal K and V head dims; got {head_dim} and {v_head_dim}"
            )
        self.num_quant_pages = int(num_quant_pages)
        self.dummy_page = self.num_quant_pages
        self.layer_num = int(layer_num)
        self.head_num = int(head_num)
        self.head_dim = int(head_dim)
        self.device = torch.device(device)
        self.layout = QuantKVLayout.create(
            bits=2, head_dim=self.head_dim, num_kv_heads=self.head_num, device=self.device
        )
        self.caches = [self.layout.allocate(self.num_quant_pages + 1) for _ in range(self.layer_num)]
        self.v_clip = None if v_clip is None or v_clip <= 0.0 or v_clip >= 1.0 else float(v_clip)
        # Per (layer, head) order-statistic index of the K clip, -1 = no clip; the pool's
        # ``k_clip_idx_layer`` (per-head table) wins over its per-layer ratios.
        k_idx_rows = []
        for li in range(self.layer_num):
            if k_clip_idx_layer is not None:
                k_idx_rows.append([int(i) for i in k_clip_idx_layer[li].tolist()])
            else:
                k_idx_rows.append([_clip_index(float(k_clip_layer[li]), self.head_dim)] * self.head_num)
        # A layer whose heads share one index takes FOSCAR's in-writer clip; the ratio below
        # maps back to exactly that index (``min(int(r * D), D - 1)``).
        self._k_clip_scalar: List[Optional[float]] = [
            (None if row[0] < 0 else (row[0] + 0.5) / self.head_dim) if len(set(row)) == 1 else None
            for row in k_idx_rows
        ]
        self._k_preclip = [len(set(row)) > 1 for row in k_idx_rows]
        self._k_clip_idx = torch.tensor(
            [[self.head_dim - 1 if i < 0 else i for i in row] for row in k_idx_rows],
            dtype=torch.int64,
            device=self.device,
        )
        dummy_slots = torch.arange(
            self.dummy_page * FOSCAR_PAGE_TOKENS,
            (self.dummy_page + 1) * FOSCAR_PAGE_TOKENS,
            dtype=torch.int64,
            device=self.device,
        )
        zeros = torch.zeros(
            FOSCAR_PAGE_TOKENS, self.head_num, self.head_dim, dtype=torch.bfloat16, device=self.device
        )
        for li in range(self.layer_num):
            self.caches[li].append(zeros, zeros, dummy_slots)
            # Compile every writer this layer uses now: a first call compiles, which a CUDA
            # graph capture may not contain.
            self.write(li, zeros, zeros, dummy_slots)
        logger.info(
            "FOSCAR INT2 tier: %d layers x %d pages (+1 dummy) x %d B/page, %d per-head-clip layers, v_clip=%s",
            self.layer_num,
            self.num_quant_pages,
            self.layout.bytes_per_page(),
            sum(self._k_preclip),
            self.v_clip,
        )

    @staticmethod
    def bytes_per_page(*, head_num: int, head_dim: int, device: str) -> int:
        from foscar import QuantKVLayout

        layout = QuantKVLayout.create(bits=2, head_dim=head_dim, num_kv_heads=head_num, device=torch.device(device))
        return layout.bytes_per_page()

    def _clamp_rows(self, layer: int, k: torch.Tensor) -> torch.Tensor:
        """Clamp each (token, head) row to its head's order statistic of |k| (ascending)."""
        idx = self._k_clip_idx[layer].view(1, -1, 1).expand(k.shape[0], -1, 1)
        t = k.abs().sort(dim=-1).values.gather(-1, idx)
        return torch.maximum(torch.minimum(k, t), -t)

    def write(self, layer: int, k: torch.Tensor, v: torch.Tensor, slots: torch.Tensor) -> None:
        """Quantize and store rows already in the OSCAR frame. ``k``/``v``: bf16 [T, H, D] with
        contiguous [H, D] rows; ``slots``: quant slot ids (page * 64 + position), int64 [T]."""
        if k.shape[0] == 0:
            return
        if self._k_preclip[layer]:
            k = self._clamp_rows(layer, k).contiguous()
            k_clip = None
        else:
            k_clip = self._k_clip_scalar[layer]
        self.caches[layer].append(k, v, slots.to(torch.int64), k_clip=k_clip, v_clip=self.v_clip)

    def dequantize(
        self,
        layer: int,
        k_out: torch.Tensor,
        v_out: torch.Tensor,
        page_indptr: torch.Tensor,
        page_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        out_indptr: torch.Tensor,
    ) -> None:
        self.caches[layer].dequantize(k_out, v_out, page_indptr, page_indices, seq_lens, out_indptr)


def foscar_flush_apply(
    plan,
    *,
    tier: FoscarInt2Tier,
    hp_k_buffers: List[torch.Tensor],
    hp_v_buffers: List[torch.Tensor],
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
) -> None:
    """Decode flush into FOSCAR pages: demote each plan row's HP-recent slots (already in the
    OSCAR frame) into its new quant page, layer by layer, then point ``req_to_token`` at the
    quant slots. Same plan and remap as ``gpu_flush_int2_apply``; invalid entries get slot -1,
    which the writer skips. Runs on the schedule stream after the previous forward, never
    inside a CUDA graph."""
    from sglang.QuantKernel.gpu_flush_int2 import _launch_flush_remap

    src = plan.src_hp_slot.clamp(min=0)
    dst = torch.where(
        plan.valid_mask.to(torch.bool), plan.dst_quant_slots, torch.full_like(plan.dst_quant_slots, -1)
    )
    for li in range(tier.layer_num):
        tier.write(li, hp_k_buffers[li].index_select(0, src), hp_v_buffers[li].index_select(0, src), dst)
    _launch_flush_remap(plan, req_pool_indices, req_to_token, plan.bs, plan.flush_interval)
