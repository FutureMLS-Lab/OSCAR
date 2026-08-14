"""
Unified HP + int2 KV cache pool.

Quant arena: paged with ``N_Q`` slots per page. HP arena: shared HP-prefix
pool (paged) followed by per-request HP-recent ring slabs. Slot id namespace
is flat (``[0, num_quant_pages*N_Q)`` quant, ``[HP_OFFSET, ...)`` HP), and
kernels dispatch by ``slot >= HP_OFFSET``.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.QuantKernel.fused_hadamard_int2_kv import (
    quantized_set_kv_int2_pretransformed_triton,
)
from sglang.QuantKernel.oscar_rotation_clip_int2_kv import (
    quantized_set_kv_int2_oscar_rotate_k_clip_triton,
    quantized_set_kv_int2_pretransformed_clip_triton,
)
from sglang.srt.mem_cache.kv_quant_kernels import _get_num_scale_groups
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.dp_attention import get_attention_tp_rank
from sglang.srt.mem_cache.memory_pool import (
    KVCache,
    OscarRotationConfig,
    _set_kv_buffer_impl,
    get_tensor_size_bytes,
    load_oscar_rotation_config,
    load_oscar_rotations,
)

logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024


@triton.jit
def _set_mixed_hp_buffer_kernel(
    src_ptr,
    dst_ptr,
    loc_ptr,
    num_tokens,
    row_dim: tl.constexpr,
    src_stride_token: tl.constexpr,
    src_stride_dim: tl.constexpr,
    dst_stride_loc: tl.constexpr,
    dst_stride_dim: tl.constexpr,
    HP_OFFSET: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    offs = block_idx * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    loc = tl.load(loc_ptr + token_idx)
    is_hp = loc >= HP_OFFSET
    hp_loc = loc - HP_OFFSET
    mask = is_hp & (token_idx < num_tokens) & (offs < row_dim)
    vals = tl.load(
        src_ptr + token_idx * src_stride_token + offs * src_stride_dim,
        mask=mask,
        other=0.0,
    )
    tl.store(
        dst_ptr + hp_loc * dst_stride_loc + offs * dst_stride_dim,
        vals,
        mask=mask,
    )


def _resolve_torch_dtype(name: str, *, kind: str) -> torch.dtype:
    """Map a friendly dtype name (``bf16``/``bfloat16``/``fp16``/``half``/``fp32``)
    to the corresponding ``torch.dtype``. ``kind`` is only used in the error
    message ("scale" / "HP" / etc.) so the caller's intent surfaces in the
    failure mode.
    """
    n = name.lower()
    if n in ("bf16", "bfloat16"):
        return torch.bfloat16
    if n in ("fp16", "float16", "half"):
        return torch.float16
    if n in ("fp32", "float32"):
        return torch.float32
    raise ValueError(
        f"Unsupported {kind} dtype: {name}. Expected bf16/fp16/fp32."
    )


def resolve_scale_dtype(name: str) -> torch.dtype:
    return _resolve_torch_dtype(name, kind="scale")


def resolve_hp_dtype(name: str) -> torch.dtype:
    return _resolve_torch_dtype(name, kind="HP")


def compute_page_geometry(hp_dtype: torch.dtype) -> Tuple[int, int]:
    """Return ``(N_H, N_Q)`` for int2 + ``hp_dtype``.

    ``N_Q`` is the int2 page size used by the paged quant allocator (and
    ``--page-size``). ``N_H`` is retained as ``1`` for documentation and for
    legacy callers, but no longer carries the LCM byte-equivalence invariant —
    HP and quant arenas are decoupled allocations under the slab design.
    """
    hp_itemsize = torch.empty(0, dtype=hp_dtype).element_size()
    return 1, 4 * hp_itemsize


def compute_recent_ring_size(hp_recent_tokens: int, n_q: int) -> int:
    # Max HP-recent occupancy between flushes is hp_recent + (N_Q - 1); the
    # ring reuses slots after the oldest N_Q have been demoted to quant.
    return int(hp_recent_tokens) + int(n_q) - 1


def _shard_rotation_heads(R, local_head_num: int, tp_rank: int):
    """Slice a per-head rotation down to the KV heads this rank owns.

    A V2 checkpoint stores every KV head of the model, but under tensor
    parallelism each rank holds only ``local_head_num`` consecutive heads.
    Shared (2D) rotations are TP-invariant and pass through untouched.
    Accepts either a stacked ``[L, H, hd, hd]`` tensor or a per-layer list.

    A bare 3D tensor is deliberately left alone: ``[L, hd, hd]`` (V1 stacked
    per-layer shared rotations, what this pool actually holds for V1) and
    ``[H, hd, hd]`` (one layer's per-head rotations) are indistinguishable by
    shape. Slicing it as heads mistakes the layer axis for a head axis and
    breaks every V1 model, so per-head sharding only happens where the head
    axis is unambiguous: a per-layer list, or a 4D ``[L, H, hd, hd]``.
    """

    def _slice(m):
        if m.dim() != 3:                      # [hd, hd] shared -> unchanged
            return m
        total = m.shape[0]
        if total == local_head_num:
            return m
        if total % local_head_num != 0:
            raise ValueError(
                f"per-head rotation has {total} KV heads, which is not a "
                f"multiple of this rank's {local_head_num}"
            )
        beg = tp_rank * local_head_num
        return m[beg : beg + local_head_num].contiguous()

    if isinstance(R, (list, tuple)):
        return [_slice(m) for m in R]
    if R.dim() == 4:                          # [L, H, hd, hd]
        total = R.shape[1]
        if total != local_head_num:
            if total % local_head_num != 0:
                raise ValueError(
                    f"per-head rotation has {total} KV heads, not a multiple "
                    f"of this rank's {local_head_num}"
                )
            beg = tp_rank * local_head_num
            R = R[:, beg : beg + local_head_num].contiguous()
    return R


def _rotate_heads(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Apply an Oscar rotation to ``x`` of shape ``[tokens, heads, head_dim]``.

    ``R`` is either ``[hd, hd]`` (V1, one rotation shared by every KV head) or
    ``[heads, hd, hd]`` (V2, one per KV head).
    """
    if R.dim() == 2:
        return x @ R
    # ``.contiguous()`` is load-bearing: einsum can hand back a non-contiguous
    # result, and the downstream store path does ``k.view(-1, row_dim)``, which
    # raises "view size is not compatible with input tensor's size and stride".
    return torch.einsum("thd,hde->the", x, R).contiguous()


class UnifiedInt2HPKVPool(KVCache):
    """Unified HP + int2 MHA KV cache.

    The pool exposes:
      * ``k_buffer[l]``, ``v_buffer[l]``           – quant (int2 packed uint8) views
      * ``hp_k_buffer[l]``, ``hp_v_buffer[l]``     – HP (``hp_dtype``) views
      * ``k_scales_zeros[l]``, ``v_scales_zeros[l]`` – per-group scales+zeros in
        ``scale_dtype`` (bf16/fp16/fp32)

    The quant and HP views alias the same byte arena. Callers must treat a
    physical page as homogeneous (either tier) at any given time; this invariant
    is enforced by the ``UnifiedInt2HPKVAllocator`` that hands out slot ids into
    these views.
    """

    def __init__(
        self,
        num_quant_pages: int,
        hp_dtype: torch.dtype,
        hp_prefix_tokens: int,
        hp_recent_tokens: int,
        dtype: str,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        max_req_slots: int,
        v_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        model_dtype: Optional[torch.dtype] = None,
        kv_cache_quant_group_size: Optional[int] = None,
        scale_dtype: torch.dtype = torch.bfloat16,
        num_hp_prefix_slots: int = 0,
        rotation_layer_ids: Optional[List[int]] = None,
        layer_groups: Optional[List[dict]] = None,
    ):
        assert dtype == "int2", (
            "UnifiedInt2HPKVPool supports only int2 quant tier; got %s" % dtype
        )
        # Work around KVCache.__init__ dtype validation: it stores ``dtype`` as
        # a string and sets ``store_dtype=torch.uint8`` for int2.
        super().__init__(
            size=num_quant_pages,  # used by base class for sizing heuristics only
            page_size=1,
            dtype=dtype,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
            model_dtype=model_dtype,
        )

        self.num_quant_pages = int(num_quant_pages)
        self.hp_dtype = hp_dtype
        self.scale_dtype = scale_dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim if v_head_dim is not None else head_dim
        self.hp_prefix_tokens = int(hp_prefix_tokens)
        self.hp_recent_tokens = int(hp_recent_tokens)
        self.kv_cache_quant_group_size = kv_cache_quant_group_size

        # --- Per-layer geometry (two-group / heterogeneous SWA support) ------
        # Default (layer_groups is None): one uniform group over all layers,
        # geometry == the scalar head_num/head_dim/v_head_dim above. This is
        # the path taken by every existing OSCAR model (qwen3, minimax, glm),
        # so their behavior is unchanged. When ``layer_groups`` is provided
        # (gemma4_unified: full 1x512 + sliding 8x256), every layer carries its
        # own (head_num, head_dim, v_head_dim) and the buffers/flush run per
        # group. ``layer_groups`` entries hold *global* layer ids; we map them
        # to local [0, layer_num) indices via ``start_layer``.
        self._layer_groups = layer_groups
        if layer_groups is None:
            self._layer_head_num = [head_num] * self.layer_num
            self._layer_head_dim = [self.head_dim] * self.layer_num
            self._layer_v_head_dim = [self.v_head_dim] * self.layer_num
            # local-index lists per group (single group = all layers)
            self._group_local_layer_ids = [list(range(self.layer_num))]
        else:
            self._layer_head_num = [None] * self.layer_num
            self._layer_head_dim = [None] * self.layer_num
            self._layer_v_head_dim = [None] * self.layer_num
            self._group_local_layer_ids = []
            for g in layer_groups:
                local_ids = []
                for gid in g["layer_ids"]:
                    local = gid - self.start_layer
                    if not (0 <= local < self.layer_num):
                        continue
                    self._layer_head_num[local] = int(g["head_num"])
                    self._layer_head_dim[local] = int(g["head_dim"])
                    self._layer_v_head_dim[local] = int(g["v_head_dim"])
                    local_ids.append(local)
                self._group_local_layer_ids.append(local_ids)
            missing = [i for i, h in enumerate(self._layer_head_num) if h is None]
            if missing:
                raise ValueError(
                    f"UnifiedInt2HPKVPool: layer_groups did not cover local "
                    f"layer indices {missing} (start_layer={self.start_layer}, "
                    f"layer_num={self.layer_num})"
                )

        self.N_H, self.N_Q = compute_page_geometry(hp_dtype)
        self.hp_recent_ring_size = compute_recent_ring_size(
            self.hp_recent_tokens, self.N_Q
        )
        if num_hp_prefix_slots > 0 and num_hp_prefix_slots % self.N_Q != 0:
            num_hp_prefix_slots = (
                (num_hp_prefix_slots + self.N_Q - 1) // self.N_Q * self.N_Q
            )
        self.num_hp_prefix_slots = int(num_hp_prefix_slots)
        self.slab_size = self.hp_recent_ring_size  # back-compat alias
        self._hp_offset = self.num_quant_pages * self.N_Q
        self._hp_recent_base = self.num_hp_prefix_slots

        # Window sizes must be N_Q-aligned so radix tree (page_size=N_Q) and
        # the flush kernel land on page boundaries.
        if self.hp_prefix_tokens % self.N_Q != 0:
            raise ValueError(
                f"SGLANG_MIXED_KV_PREFIX_TOKENS ({self.hp_prefix_tokens}) "
                f"must be a multiple of N_Q ({self.N_Q})."
            )
        if self.hp_recent_tokens % self.N_Q != 0:
            raise ValueError(
                f"SGLANG_MIXED_KV_RECENT_TOKENS ({self.hp_recent_tokens}) "
                f"must be a multiple of N_Q ({self.N_Q})."
            )
        # Flush every N_Q decode steps demotes exactly N_Q HP-recent slots
        # into one quant page. Per-request counter (initialized at admission
        # to (hp_recent+N_Q-1)-H_0) keeps every flush whole-page.
        self.flush_interval = self.N_Q
        self.max_req_slots = int(max_req_slots)
        self._flush_counter = torch.zeros(
            (self.max_req_slots,), dtype=torch.int32, device=self.device
        )
        self._next_slab_offset = torch.zeros(
            (self.max_req_slots,), dtype=torch.int32, device=self.device
        )

        # Forward-done event stashed each iteration; consumed by
        # ``wait_pending_forward`` inside the flush apply phase.
        self._pending_forward_done = None

        # Grouping for quantization. The scalar ``*_num_scale_groups`` reflect
        # the primary (default-group) geometry; per-layer arrays carry the
        # right grouping for every layer in the two-group case.
        self.k_quant_group_size, self.k_num_scale_groups = self._resolve_quant_grouping(
            self.head_dim, "K"
        )
        self.v_quant_group_size, self.v_num_scale_groups = self._resolve_quant_grouping(
            self.v_head_dim, "V"
        )
        self._layer_k_num_scale_groups = []
        self._layer_v_num_scale_groups = []
        for li in range(self.layer_num):
            hd = self._layer_head_dim[li]
            vhd = self._layer_v_head_dim[li]
            _, k_ng = self._resolve_quant_grouping(hd, "K")
            _, v_ng = self._resolve_quant_grouping(vhd, "V")
            self._layer_k_num_scale_groups.append(k_ng)
            self._layer_v_num_scale_groups.append(v_ng)
            assert hd % 4 == 0, (
                f"head_dim={hd} (layer {li}) must be divisible by 4 for int2 packing"
            )
            assert vhd % 4 == 0, (
                f"v_head_dim={vhd} (layer {li}) must be divisible by 4 for int2 packing"
            )

        self._create_arenas()

        # Cached attributes used by the rest of the stack.
        self.device_module = torch.get_device_module(self.device)
        self.alt_stream = None
        self.row_dim = self.head_num * self.head_dim  # for store_cache helpers
        # Per-layer row_dim / same_kv_dim for the HP store helpers.
        self._layer_row_dim = [
            self._layer_head_num[li] * self._layer_head_dim[li]
            for li in range(self.layer_num)
        ]
        self._layer_same_kv_dim = [
            self._layer_head_dim[li] == self._layer_v_head_dim[li]
            for li in range(self.layer_num)
        ]
        self.same_kv_dim = self.head_dim == self.v_head_dim

        # Oscar rotation + clip. Per-layer orthogonal matrices [head_dim,
        # head_dim] / [v_head_dim, v_head_dim] are loaded in ``hp_dtype`` so
        # the ``rows @ R`` pre-pass and ``result @ R.T`` inverse are plain
        # bf16 GEMMs.
        self._oscar_cfg: OscarRotationConfig = load_oscar_rotation_config()
        self._k_clip_ratio: float = self._oscar_cfg.k_clip_ratio
        self._v_clip_ratio: float = self._oscar_cfg.v_clip_ratio
        self._lloyd_max: bool = envs.SGLANG_LLOYD_MAX.get()
        if rotation_layer_ids is not None and len(rotation_layer_ids) != self.layer_num:
            raise ValueError(
                f"UnifiedInt2HPKVPool: rotation_layer_ids has "
                f"{len(rotation_layer_ids)} entries but layer_num={self.layer_num}"
            )
        self._rotation_layer_ids = (
            list(rotation_layer_ids) if rotation_layer_ids is not None else None
        )
        # Scalar head_dim for uniform models (stacked [L,hd,hd], indexable as
        # ``self._R_k[idx]``); per-layer list for the two-geometry-group case
        # (list of per-layer matrices, indexed the same way).
        k_head_dim_arg = (
            self._layer_head_dim if self._layer_groups is not None else self.head_dim
        )
        v_head_dim_arg = (
            self._layer_v_head_dim if self._layer_groups is not None else self.v_head_dim
        )
        self._R_k = load_oscar_rotations(
            self._oscar_cfg.k_rotation_path,
            layer_num=self.layer_num,
            start_layer=self.start_layer,
            head_dim=k_head_dim_arg,
            device=torch.device(self.device),
            dtype=self.hp_dtype,
            layer_ids=self._rotation_layer_ids,
        )
        self._R_v = load_oscar_rotations(
            self._oscar_cfg.v_rotation_path,
            layer_num=self.layer_num,
            start_layer=self.start_layer,
            head_dim=v_head_dim_arg,
            device=torch.device(self.device),
            dtype=self.hp_dtype,
            layer_ids=self._rotation_layer_ids,
        )
        # Per-head rotations ship every KV head; keep only this rank's slice.
        _tp_rank = get_attention_tp_rank()
        self._R_k = _shard_rotation_heads(self._R_k, self.head_num, _tp_rank)
        self._R_v = _shard_rotation_heads(self._R_v, self.head_num, _tp_rank)
        logger.info(
            "UnifiedInt2HPKVPool: Oscar rotation enabled (k_clip=%.4f v_clip=%.4f lloyd_max=%s)",
            self._k_clip_ratio,
            self._v_clip_ratio,
            self._lloyd_max,
        )

        hp_total_slots = (
            self.num_hp_prefix_slots
            + self.max_req_slots * self.hp_recent_ring_size
        )
        self._finalize_allocation_log(hp_total_slots)
        hp_itemsize = torch.empty(0, dtype=self.hp_dtype).element_size()
        per_layer_hp_elems = sum(
            self._layer_head_num[li]
            * (self._layer_head_dim[li] + self._layer_v_head_dim[li])
            for li in range(self.layer_num)
        )
        hp_bytes = hp_total_slots * per_layer_hp_elems * hp_itemsize
        logger.info(
            "UnifiedInt2HPKVPool: HP arena reserves %.2f GB "
            "(hp_prefix_pool_slots=%d, max_req_slots=%d, recent_ring=%d "
            "= R=%d + N_Q-1=%d, P=%d, layers=%d, head_num=%d, "
            "head_dim+v_head_dim=%d, hp_dtype=%s)",
            hp_bytes / GB,
            self.num_hp_prefix_slots,
            self.max_req_slots,
            self.hp_recent_ring_size,
            self.hp_recent_tokens,
            self.N_Q - 1,
            self.hp_prefix_tokens,
            self.layer_num,
            self.head_num,
            self.head_dim + self.v_head_dim,
            str(self.hp_dtype),
        )

    # -- Configuration accessors -------------------------------------------

    def mixed_kv_enabled(self) -> bool:
        return True

    def stash_pending_forward(self, event) -> None:
        """Record the most recent forward-stream completion event.

        Called once per iteration from the scheduler. The event is consumed
        by :meth:`wait_pending_forward` at the apply boundary inside
        ``_alloc_for_decode_mixed``.
        """
        self._pending_forward_done = event

    def wait_pending_forward(self) -> None:
        """Order the current stream after the stashed forward-done event.

        Must be issued *before* the apply phase of the flush
        (``gpu_flush_int2_apply``), since the remap kernel writes
        ``req_to_token`` at positions the previous forward's attention is
        concurrently reading. Pre-apply work (allocator free, plan kernel)
        runs ahead of this wait so its host syncs don't block on the
        previous forward.
        """
        if self._pending_forward_done is None:
            return
        torch.cuda.current_stream().wait_event(self._pending_forward_done)
        self._pending_forward_done = None

    @property
    def hp_global_offset(self) -> int:
        return self._hp_offset

    @property
    def hp_size(self) -> int:
        return (
            self.num_hp_prefix_slots
            + self.max_req_slots * self.hp_recent_ring_size
        )

    @property
    def quant_size(self) -> int:
        return self.num_quant_pages * self.N_Q

    @property
    def hp_prefix_pool_slots(self) -> int:
        return self.num_hp_prefix_slots

    @property
    def hp_recent_base(self) -> int:
        """First HP-buffer index reserved for per-req recent slabs."""
        return self._hp_recent_base

    def release_req_slab(self, req_pool_idx) -> None:
        # Reset the per-req HP-recent cursor and flush counter so the next
        # request taking over ``req_pool_idx`` starts clean.
        if isinstance(req_pool_idx, torch.Tensor):
            idx = req_pool_idx.to(self._next_slab_offset.device).to(torch.int64)
            if idx.numel() == 0:
                return
            self._next_slab_offset[idx] = 0
            self._flush_counter[idx] = 0
        else:
            i = int(req_pool_idx)
            self._next_slab_offset[i] = 0
            self._flush_counter[i] = 0

    def _resolve_quant_grouping(self, head_dim: int, tensor_name: str) -> tuple[int, int]:
        group_size = (
            head_dim
            if self.kv_cache_quant_group_size is None
            else self.kv_cache_quant_group_size
        )
        if group_size <= 0:
            raise ValueError(
                f"{tensor_name} kv_cache_quant_group_size must be positive, got {group_size}"
            )
        if head_dim % group_size != 0:
            raise ValueError(
                f"{tensor_name} head_dim ({head_dim}) must be divisible by "
                f"kv_cache_quant_group_size ({group_size})"
            )
        return group_size, head_dim // group_size

    # -- Arena construction ------------------------------------------------

    def _create_arenas(self):
        # HP arena layout: [shared prefix pool] [per-req recent slab 0]
        # [per-req recent slab 1] ... Quant arena is paged with N_Q slots
        # per page; scales/zeros are quant-only.
        hp_total_slots = (
            self.num_hp_prefix_slots
            + self.max_req_slots * self.hp_recent_ring_size
        )
        nq = self.num_quant_pages * self.N_Q
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                self.k_buffer = [
                    torch.zeros(
                        (nq, self._layer_head_num[li], self._layer_head_dim[li] // 4),
                        dtype=torch.uint8,
                        device=self.device,
                    )
                    for li in range(self.layer_num)
                ]
                self.v_buffer = [
                    torch.zeros(
                        (nq, self._layer_head_num[li], self._layer_v_head_dim[li] // 4),
                        dtype=torch.uint8,
                        device=self.device,
                    )
                    for li in range(self.layer_num)
                ]
                self.k_scales_zeros = [
                    torch.zeros(
                        (nq, self._layer_head_num[li], 2 * self._layer_k_num_scale_groups[li]),
                        dtype=self.scale_dtype,
                        device=self.device,
                    )
                    for li in range(self.layer_num)
                ]
                self.v_scales_zeros = [
                    torch.zeros(
                        (nq, self._layer_head_num[li], 2 * self._layer_v_num_scale_groups[li]),
                        dtype=self.scale_dtype,
                        device=self.device,
                    )
                    for li in range(self.layer_num)
                ]
                self.hp_k_buffer = [
                    torch.zeros(
                        (hp_total_slots, self._layer_head_num[li], self._layer_head_dim[li]),
                        dtype=self.hp_dtype,
                        device=self.device,
                    )
                    for li in range(self.layer_num)
                ]
                self.hp_v_buffer = [
                    torch.zeros(
                        (hp_total_slots, self._layer_head_num[li], self._layer_v_head_dim[li]),
                        dtype=self.hp_dtype,
                        device=self.device,
                    )
                    for li in range(self.layer_num)
                ]

        # Per-group device pointer arrays + strides for the fused decode-flush
        # kernel. The flush kernel loops over layers inside the kernel and
        # requires *identical* strides across the layers it spans, so it must
        # run once per uniform-geometry group. ``self._group_local_layer_ids``
        # is ``[[0..L)]`` (single group) for uniform models — preserving the
        # exact previous single-launch behavior — and one entry per geometry
        # group otherwise.
        def _base_ptrs(local_ids: List[int], tensors: List[torch.Tensor]) -> torch.Tensor:
            return torch.tensor(
                [tensors[i].data_ptr() for i in local_ids],
                dtype=torch.int64,
                device=self.device,
            )

        def _strides(t: torch.Tensor) -> tuple:
            return (int(t.stride(0)), int(t.stride(1)), int(t.stride(2)))

        self._flush_groups = []
        for local_ids in self._group_local_layer_ids:
            l0 = local_ids[0]
            hp_k_stride = _strides(self.hp_k_buffer[l0])
            hp_v_stride = _strides(self.hp_v_buffer[l0])
            q_k_stride = _strides(self.k_buffer[l0])
            q_v_stride = _strides(self.v_buffer[l0])
            k_sz_stride = _strides(self.k_scales_zeros[l0])
            v_sz_stride = _strides(self.v_scales_zeros[l0])
            for l in local_ids:
                assert _strides(self.hp_k_buffer[l]) == hp_k_stride
                assert _strides(self.hp_v_buffer[l]) == hp_v_stride
                assert _strides(self.k_buffer[l]) == q_k_stride
                assert _strides(self.v_buffer[l]) == q_v_stride
                assert _strides(self.k_scales_zeros[l]) == k_sz_stride
                assert _strides(self.v_scales_zeros[l]) == v_sz_stride
            self._flush_groups.append(
                {
                    # The pointer arrays feed the fused kernel; the tensor lists
                    # feed the unfused fallback for non-power-of-two head dims.
                    "hp_k_layers": [self.hp_k_buffer[l] for l in local_ids],
                    "hp_v_layers": [self.hp_v_buffer[l] for l in local_ids],
                    "quant_k_layers": [self.k_buffer[l] for l in local_ids],
                    "quant_v_layers": [self.v_buffer[l] for l in local_ids],
                    "k_sz_layers": [self.k_scales_zeros[l] for l in local_ids],
                    "v_sz_layers": [self.v_scales_zeros[l] for l in local_ids],
                    "hp_k_ptrs": _base_ptrs(local_ids, self.hp_k_buffer),
                    "hp_v_ptrs": _base_ptrs(local_ids, self.hp_v_buffer),
                    "quant_k_ptrs": _base_ptrs(local_ids, self.k_buffer),
                    "quant_v_ptrs": _base_ptrs(local_ids, self.v_buffer),
                    "k_sz_ptrs": _base_ptrs(local_ids, self.k_scales_zeros),
                    "v_sz_ptrs": _base_ptrs(local_ids, self.v_scales_zeros),
                    "hp_k_stride": hp_k_stride,
                    "hp_v_stride": hp_v_stride,
                    "quant_k_stride": q_k_stride,
                    "quant_v_stride": q_v_stride,
                    "k_sz_stride": k_sz_stride,
                    "v_sz_stride": v_sz_stride,
                    "head_num": self._layer_head_num[l0],
                    "head_dim": self._layer_head_dim[l0],
                    "v_head_dim": self._layer_v_head_dim[l0],
                    "k_num_scale_groups": self._layer_k_num_scale_groups[l0],
                    "v_num_scale_groups": self._layer_v_num_scale_groups[l0],
                    "num_layers": len(local_ids),
                    "k_sample": self.k_buffer[l0],
                    "v_sample": self.v_buffer[l0],
                    "hp_k_sample": self.hp_k_buffer[l0],
                    "hp_v_sample": self.hp_v_buffer[l0],
                    "k_sz_sample": self.k_scales_zeros[l0],
                    "v_sz_sample": self.v_scales_zeros[l0],
                }
            )

        # Back-compat single-group flush metadata (used by the existing
        # common.py flush call for uniform models; the two-group path iterates
        # ``self._flush_groups`` instead).
        g0 = self._flush_groups[0]
        self._flush_hp_k_ptrs = g0["hp_k_ptrs"]
        self._flush_hp_v_ptrs = g0["hp_v_ptrs"]
        self._flush_quant_k_ptrs = g0["quant_k_ptrs"]
        self._flush_quant_v_ptrs = g0["quant_v_ptrs"]
        self._flush_k_sz_ptrs = g0["k_sz_ptrs"]
        self._flush_v_sz_ptrs = g0["v_sz_ptrs"]
        self._flush_hp_k_stride = g0["hp_k_stride"]
        self._flush_hp_v_stride = g0["hp_v_stride"]
        self._flush_quant_k_stride = g0["quant_k_stride"]
        self._flush_quant_v_stride = g0["quant_v_stride"]
        self._flush_k_sz_stride = g0["k_sz_stride"]
        self._flush_v_sz_stride = g0["v_sz_stride"]

    # -- KVCache interface -------------------------------------------------

    def get_kv_size_bytes(self):
        k = sum(get_tensor_size_bytes(t) for t in self.k_buffer)
        k += sum(get_tensor_size_bytes(s) for s in self.k_scales_zeros)
        k += sum(get_tensor_size_bytes(t) for t in self.hp_k_buffer)
        v = sum(get_tensor_size_bytes(t) for t in self.v_buffer)
        v += sum(get_tensor_size_bytes(s) for s in self.v_scales_zeros)
        v += sum(get_tensor_size_bytes(t) for t in self.hp_v_buffer)
        return k, v

    def _layer_index(self, layer_id: int) -> int:
        return layer_id - self.start_layer

    # Per-layer geometry accessors. Equal to the scalar head_num/head_dim/
    # v_head_dim for uniform models; per-layer for the two-group case. Callers
    # that previously read ``kv_pool.head_dim`` directly on a per-layer basis
    # (e.g. dequantize_prefix_kv) must use these.
    def get_layer_head_num(self, layer_id: int) -> int:
        return self._layer_head_num[self._layer_index(layer_id)]

    def get_layer_head_dim(self, layer_id: int) -> int:
        return self._layer_head_dim[self._layer_index(layer_id)]

    def get_layer_v_head_dim(self, layer_id: int) -> int:
        return self._layer_v_head_dim[self._layer_index(layer_id)]

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        # Triton backend asks for the quant view in the mixed path; HP view is
        # accessed via ``get_hp_key_buffer``.
        return self.k_buffer[self._layer_index(layer_id)]

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.v_buffer[self._layer_index(layer_id)]

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def get_raw_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.k_buffer[self._layer_index(layer_id)]

    def get_raw_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.v_buffer[self._layer_index(layer_id)]

    def get_key_scales_zeros(self, layer_id: int) -> torch.Tensor:
        return self.k_scales_zeros[self._layer_index(layer_id)]

    def get_value_scales_zeros(self, layer_id: int) -> torch.Tensor:
        return self.v_scales_zeros[self._layer_index(layer_id)]

    def get_hp_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.hp_k_buffer[self._layer_index(layer_id)]

    def get_hp_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.hp_v_buffer[self._layer_index(layer_id)]

    def get_raw_kv_buffer(self, layer_id: int):
        idx = self._layer_index(layer_id)
        return {
            "k_buffer": self.k_buffer[idx],
            "v_buffer": self.v_buffer[idx],
            "k_scales_zeros": self.k_scales_zeros[idx],
            "v_scales_zeros": self.v_scales_zeros[idx],
            "dtype": "int2",
        }

    def _split_global_locs(self, loc: torch.Tensor):
        loc64 = loc.to(torch.int64)
        hp_mask = loc64 >= self._hp_offset
        quant_loc = loc64[~hp_mask]
        hp_loc_global = loc64[hp_mask] - self._hp_offset
        return quant_loc, hp_loc_global, hp_mask

    def _rotate_kv_inplace(
        self,
        layer_id: int,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        v_rotation_absorbed: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the per-layer Oscar rotation ``rows @ R`` to HP K/V tiles.

        Returns tensors in ``self.hp_dtype`` ready to be stored or packed.
        ``R_k`` / ``R_v`` are ``[head_dim, head_dim]`` bf16 on the KV device,
        loaded in ``__init__``.
        """
        idx = self._layer_index(layer_id)
        k_hp = _rotate_heads(cache_k.to(self.hp_dtype), self._R_k[idx])
        if v_rotation_absorbed:
            v_hp = cache_v.to(self.hp_dtype)
        else:
            v_hp = _rotate_heads(cache_v.to(self.hp_dtype), self._R_v[idx])
        return k_hp, v_hp

    def _prepare_hp_kv_tensors(
        self,
        layer_id: int,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        already_rotated: bool,
        v_rotation_absorbed: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the Oscar rotation to HP K/V and cast to ``hp_dtype``.
        ``already_rotated`` skips the rotation pre-pass.
        """
        if already_rotated:
            return cache_k.to(self.hp_dtype), cache_v.to(self.hp_dtype)
        return self._rotate_kv_inplace(
            layer_id, cache_k, cache_v, v_rotation_absorbed
        )

    def _set_hp_kv_buffer(
        self,
        layer_id: int,
        hp_loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        idx = self._layer_index(layer_id)
        _set_kv_buffer_impl(
            cache_k,
            cache_v,
            self.hp_k_buffer[idx],
            self.hp_v_buffer[idx],
            hp_loc,
            row_dim=self._layer_row_dim[idx],
            store_dtype=self.hp_dtype,
            device_module=self.device_module,
            alt_stream=self.alt_stream,
            same_kv_dim=self._layer_same_kv_dim[idx],
        )

    def _set_quant_kv_buffer_extend(
        self,
        layer_id: int,
        quant_loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        already_hadamard_transformed: bool,
        mixed_hp_offset: Optional[int] = None,
        v_rotation_absorbed: bool = False,
    ):
        """Prefill/extend-only: rotate (oscar R) + optional per-row clip +
        int2-pack + write quant slots.

        Decode-time flushes go through the dedicated GPU flush kernel
        (see ``gpu_flush_int2``); this method is *not* used for those.
        """
        idx = self._layer_index(layer_id)
        clip_on = self._k_clip_ratio > 0.0 or self._v_clip_ratio > 0.0

        # Fused rotate(K) + clip(KV) + quantize(KV) + set(KV). Skips the
        # standalone ``K @ R_k`` GEMM and its bf16 staging tensor by doing
        # the rotation inside the int2 pack kernel via ``tl.dot``. V must
        # already be in R_v space (rotation absorbed) — the kernel does
        # not rotate V. Requires single-scale layout (num_groups == 1) for
        # both K and V scales/zeros.
        if envs.SGLANG_OSCAR_FUSED_ROTATE_CLIP_QUANT.get():
            assert v_rotation_absorbed, (
                "V rotation must be absorbed for fused oscar K-rotation + clip + quant + set"
            )

        use_fused_rotate = (
            envs.SGLANG_OSCAR_FUSED_ROTATE_CLIP_QUANT.get()
            and not already_hadamard_transformed
            and v_rotation_absorbed
            and clip_on
            and _get_num_scale_groups(self.k_scales_zeros[idx]) == 1
            and _get_num_scale_groups(self.v_scales_zeros[idx]) == 1
        )
        if use_fused_rotate:
            quantized_set_kv_int2_oscar_rotate_k_clip_triton(
                cache_k.to(self.hp_dtype),
                cache_v.to(self.hp_dtype),
                self._R_k[idx],
                quant_loc,
                self.k_buffer[idx],
                self.v_buffer[idx],
                self.k_scales_zeros[idx],
                self.v_scales_zeros[idx],
                self._k_clip_ratio,
                self._v_clip_ratio,
                hp_global_offset=mixed_hp_offset,
            )
            return

        if not already_hadamard_transformed:
            cache_k, cache_v = self._rotate_kv_inplace(
                layer_id, cache_k, cache_v, v_rotation_absorbed
            )
        else:
            cache_k = cache_k.to(self.hp_dtype)
            cache_v = cache_v.to(self.hp_dtype)

        if not clip_on:
            quantized_set_kv_int2_pretransformed_triton(
                cache_k,
                cache_v,
                quant_loc,
                self.k_buffer[idx],
                self.v_buffer[idx],
                self.k_scales_zeros[idx],
                self.v_scales_zeros[idx],
                hp_global_offset=mixed_hp_offset,
            )
            return

        quantized_set_kv_int2_pretransformed_clip_triton(
            cache_k,
            cache_v,
            quant_loc,
            self.k_buffer[idx],
            self.v_buffer[idx],
            self.k_scales_zeros[idx],
            self.v_scales_zeros[idx],
            self._k_clip_ratio,
            self._v_clip_ratio,
            hp_global_offset=mixed_hp_offset,
            lloyd_max=self._lloyd_max,
        )

    def _set_mixed_hp_kv_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        idx = self._layer_index(layer_id)

        def _launch(src: torch.Tensor, dst: torch.Tensor):
            src2 = src.reshape(src.shape[0], -1)
            dst2 = dst.reshape(dst.shape[0], -1)
            row_dim = src2.shape[1]
            if row_dim == 0 or src2.shape[0] == 0:
                return
            block_row = min(1024, triton.next_power_of_2(row_dim))
            grid = (src2.shape[0], triton.cdiv(row_dim, block_row))
            _set_mixed_hp_buffer_kernel[grid](
                src2,
                dst2,
                loc,
                src2.shape[0],
                row_dim,
                src2.stride(0),
                src2.stride(1),
                dst2.stride(0),
                dst2.stride(1),
                HP_OFFSET=int(self._hp_offset),
                BLOCK_ROW=block_row,
                num_warps=4,
                num_stages=1,
            )

        _launch(cache_k, self.hp_k_buffer[idx])
        _launch(cache_v, self.hp_v_buffer[idx])

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
        already_hadamard_transformed: bool = False,
        is_decode: bool = False,
    ):
        """Write K/V to the unified pool.

        ``is_decode`` selects the path:
            * ``True`` -- single-token decode write. Caller fills ``loc`` with
              valid HP slot ids (from ``allocator.alloc_hp_recent``); we
              write only the HP buffer, with no boolean masking (safe under
              CUDA-graph capture).
            * ``False`` (extend / prefill) -- mixed write: int2 quant slots get
              the rotated+clipped pack, HP slots get the bf16 row.

        Callers must pass ``is_decode`` based on
        ``forward_batch.forward_mode.is_decode_or_idle()``. Capture state is
        *not* a reliable proxy: piecewise CUDA graph captures parts of prefill
        (would mis-route to the HP-only branch) and ``--disable-cuda-graph``
        runs decode eagerly (would mis-route to the quant+HP branch).
        """
        if loc.numel() == 0:
            return

        layer_id = layer_id_override if layer_id_override is not None else layer.layer_id
        v_rotation_absorbed = bool(getattr(layer, "oscar_v_rotation_absorbed", False))

        if is_decode:
            hp_local = loc.to(torch.int64) - self._hp_offset
            cache_k_hp, cache_v_hp = self._prepare_hp_kv_tensors(
                layer_id,
                cache_k,
                cache_v,
                already_hadamard_transformed,
                v_rotation_absorbed,
            )
            self._set_hp_kv_buffer(layer_id, hp_local, cache_k_hp, cache_v_hp)
            return

        self._set_quant_kv_buffer_extend(
            layer_id,
            loc,
            cache_k,
            cache_v,
            already_hadamard_transformed,
            mixed_hp_offset=int(self._hp_offset),
            v_rotation_absorbed=v_rotation_absorbed,
        )
        cache_k_hp, cache_v_hp = self._prepare_hp_kv_tensors(
            layer_id,
            cache_k,
            cache_v,
            already_hadamard_transformed,
            v_rotation_absorbed,
        )
        self._set_mixed_hp_kv_buffer(layer_id, loc, cache_k_hp, cache_v_hp)

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        if tgt_loc.numel() == 0:
            return
        # Moves only make sense within the same tier. The allocator ensures
        # that; callers should split by tier before calling here.
        tgt_q, tgt_hp, tgt_mask = self._split_global_locs(tgt_loc)
        src_q, src_hp, src_mask = self._split_global_locs(src_loc)
        assert torch.equal(tgt_mask, src_mask), (
            "move_kv_cache requires src/tgt tiers to match"
        )
        for l in range(self.layer_num):
            if tgt_q.numel() > 0:
                self.k_buffer[l][tgt_q] = self.k_buffer[l][src_q]
                self.v_buffer[l][tgt_q] = self.v_buffer[l][src_q]
                self.k_scales_zeros[l][tgt_q] = self.k_scales_zeros[l][src_q]
                self.v_scales_zeros[l][tgt_q] = self.v_scales_zeros[l][src_q]
            if tgt_hp.numel() > 0:
                self.hp_k_buffer[l][tgt_hp] = self.hp_k_buffer[l][src_hp]
                self.hp_v_buffer[l][tgt_hp] = self.hp_v_buffer[l][src_hp]

    def get_cpu_copy(self, indices):
        raise NotImplementedError("CPU offload is not supported by UnifiedInt2HPKVPool")

    def load_cpu_copy(self, kv_cache_cpu, indices):
        raise NotImplementedError("CPU offload is not supported by UnifiedInt2HPKVPool")
