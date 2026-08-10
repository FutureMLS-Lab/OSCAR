"""
Unified HP + int2/int1 KV cache pool.

Quant arena: paged with ``N_Q`` slots per page. HP arena: shared HP-prefix
pool (paged) followed by per-request HP-recent ring slabs. Slot id namespace
is flat (``[0, num_quant_pages*N_Q)`` quant, ``[HP_OFFSET, ...)`` HP), and
kernels dispatch by ``slot >= HP_OFFSET``.

INT2 packs 4 quant slots per byte (head_dim // 4 bytes per row); INT1 packs
8 quant slots per byte (head_dim // 8 bytes per row). The pool is parameterized
by ``pack_factor`` so a single class serves both dtypes.
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
    _launch_grouped_clip_int2,
    _launch_single_clip_int2,
    quantized_set_kv_int2_oscar_rotate_k_clip_triton,
    quantized_set_kv_int2_pretransformed_clip_triton,
)
from sglang.QuantKernel.oscar_rotation_clip_int1_kv import (
    quantized_set_kv_int1_oscar_rotate_k_clip_triton,
    quantized_set_kv_int1_pretransformed_clip_triton,
    quantized_set_kv_int1_pretransformed_triton,
)
from sglang.QuantKernel.oscar_rotation_pq_k_kv import (
    pq_decode_k_at_locs,
    pq_encode_k,
)
from sglang.srt.mem_cache.kv_quant_kernels import _get_num_scale_groups
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.layers.radix_attention import RadixAttention
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


def _validate_pq_codebook_file(
    data: dict, expected_head_dim: int, label: str
) -> Tuple[int, int, int]:
    """Return (n_sub, n_centroids, sub_dim) after strict shape validation."""
    if "codebooks_stage1" in data:
        stage1 = data["codebooks_stage1"]
        stage2 = data.get("codebooks_stage2")
        if stage1.ndim != 4 or stage2 is None or stage2.ndim != 4:
            raise ValueError(
                f"{label} RVQ codebooks must be rank-4 stage1/stage2 tensors"
            )
        if (
            stage1.shape[0] != stage2.shape[0]
            or stage1.shape[1] != stage2.shape[1]
            or stage1.shape[3] != stage2.shape[3]
        ):
            raise ValueError(
                f"{label} RVQ stage shapes are incompatible: "
                f"{tuple(stage1.shape)} vs {tuple(stage2.shape)}"
            )
        n_sub, n_centroids, sub_dim = map(int, stage1.shape[1:])
        n_layers = int(stage1.shape[0])
        if int(stage2.shape[2]) > 256:
            raise ValueError(
                f"{label} RVQ stage2 has {stage2.shape[2]} centroids; "
                "uint8 codes support at most 256"
            )
        if int(stage2.shape[2]) & (int(stage2.shape[2]) - 1):
            raise ValueError(
                f"{label} RVQ stage2 centroid count must be a power of two, "
                f"got {stage2.shape[2]}"
            )
    elif "codebooks_per_layer" in data:
        codebooks = data["codebooks_per_layer"]
        if codebooks.ndim != 4:
            raise ValueError(
                f"{label} codebooks_per_layer must be rank 4, got "
                f"shape={tuple(codebooks.shape)}"
            )
        n_sub, n_centroids, sub_dim = map(int, codebooks.shape[1:])
        n_layers = int(codebooks.shape[0])
    elif "codebooks" in data:
        books = data["codebooks"]
        if not books:
            raise ValueError(f"{label} shared codebook list is empty")
        stacked = torch.stack(books)
        if stacked.ndim != 3:
            raise ValueError(
                f"{label} shared codebooks must stack to rank 3, got "
                f"shape={tuple(stacked.shape)}"
            )
        n_sub, n_centroids, sub_dim = map(int, stacked.shape)
        n_layers = None
    else:
        raise ValueError(f"{label} file contains no supported codebook tensor")

    metadata = {
        "n_sub": n_sub,
        "sub_dim": sub_dim,
    }
    if "n_centroids" in data:
        metadata["n_centroids"] = n_centroids
    for key, actual in metadata.items():
        if key in data and int(data[key]) != actual:
            raise ValueError(
                f"{label} metadata {key}={data[key]} does not match tensor "
                f"shape ({actual})"
            )
    if n_centroids > 256:
        raise ValueError(
            f"{label} has {n_centroids} centroids; uint8 codes support at most 256"
        )
    if n_centroids & (n_centroids - 1):
        raise ValueError(
            f"{label} centroid count must be a power of two, got {n_centroids}"
        )
    if n_sub * sub_dim != expected_head_dim:
        raise ValueError(
            f"{label} codebook reconstructs {n_sub}*{sub_dim}="
            f"{n_sub * sub_dim} dims, expected {expected_head_dim}"
        )
    if n_layers is not None:
        raw_layer_ids = data.get("layer_ids", list(range(n_layers)))
        if len(raw_layer_ids) != n_layers:
            raise ValueError(
                f"{label} has {n_layers} layer codebooks but "
                f"{len(raw_layer_ids)} layer_ids"
            )
        layer_ids = []
        for raw_layer_id in raw_layer_ids:
            try:
                layer_id = int(raw_layer_id)
                exact = float(raw_layer_id) == layer_id
            except (TypeError, ValueError):
                exact = False
            if not exact:
                raise ValueError(f"{label} layer_id {raw_layer_id!r} is not an integer")
            layer_ids.append(layer_id)
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError(f"{label} layer_ids must be unique: {layer_ids}")
    return n_sub, n_centroids, sub_dim


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
    raise ValueError(f"Unsupported {kind} dtype: {name}. Expected bf16/fp16/fp32.")


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
    ):
        assert dtype in ("int2", "int1", "pq_k_int2v"), (
            "UnifiedInt2HPKVPool supports int2/int1/pq_k_int2v quant tiers; got %s"
            % dtype
        )
        # Bit-width-specific packing: int2 → 4 vals/byte,
        # scalar int1 → 8 vals/byte for both K and V, and PQ → one byte per
        # sub-vector code (16 bytes for the default 16x8 codebook).
        # For pq_k_int2v: K uses N_SUB=16 codes/token-head = 16 bytes (same as int1 @ 128-dim),
        # V uses int2 packing (4 vals/byte = 32 bytes). Split into k/v pack factors.
        self._k_pack_factor = 8 if dtype in ("int1", "pq_k_int2v") else 4
        self._v_pack_factor = 8 if dtype == "int1" else 4
        self._pack_factor = (
            self._k_pack_factor
        )  # back-compat alias (used in base asserts)
        # RVQ stage-2 state — MUST be defined before _create_arenas() (which allocates
        # k_buffer2 when _is_rvq). _is_rvq is decided in the early peek below; the cb2
        # codebooks themselves are loaded later (only needed at flush/decode, not alloc).
        self._is_rvq: bool = False
        self._rvq_cb2_per_layer: Optional[list] = None
        self._rvq_cb2_norms_per_layer: Optional[list] = None
        self.k_buffer2: Optional[list] = None
        # Early codebook peek for pq_k_int2v: n_sub may differ from the n_sub=16
        # default (e.g. CQ-16c8b uses n_sub=8 → k_pack_factor=16).  We must know
        # the correct k_pack_factor BEFORE _create_arenas() allocates k_buffer, or
        # the buffer gets the wrong last dimension and pq_decode_k infers the wrong
        # N_SUB, causing OOB reads into the codebook.  Only read the scalar field
        # here; full codebook tensors are loaded to GPU later (lines 303+).
        if dtype == "pq_k_int2v":
            _pq_path_early = envs.SGLANG_PQ_K_CODEBOOK.get()
            if _pq_path_early:
                _hdr = torch.load(
                    _pq_path_early, map_location="cpu", weights_only=False
                )
                _n_sub_early, _, _ = _validate_pq_codebook_file(_hdr, head_dim, "PQ K")
                _early_pack = head_dim // _n_sub_early
                if _early_pack != self._k_pack_factor:
                    self._k_pack_factor = _early_pack
                    self._pack_factor = _early_pack
                # RVQ if the codebook carries a stage-2 (residual) book.
                self._is_rvq = "codebooks_stage1" in _hdr
            # Early peek for PQ V: set v_pack_factor so v_buffer is sized to the PQ
            # codes (v_head_dim // n_sub_v = 16B for n_sub=16), not the INT2 32B —
            # this is the true 1.0-bpe-stored V (clean packing). Must precede _create_arenas.
            _pqv_path_early = envs.SGLANG_PQ_V_CODEBOOK.get()
            if _pqv_path_early:
                _vhdr = torch.load(
                    _pqv_path_early, map_location="cpu", weights_only=False
                )
                _vhd_early = v_head_dim if v_head_dim is not None else head_dim
                _v_n_sub_early, _, _ = _validate_pq_codebook_file(
                    _vhdr, _vhd_early, "PQ V"
                )
                self._v_pack_factor = _vhd_early // _v_n_sub_early
        # Work around KVCache.__init__ dtype validation: it stores ``dtype`` as
        # a string and sets ``store_dtype=torch.uint8`` for int2/int1.
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

        # Grouping for quantization.
        self.k_quant_group_size, self.k_num_scale_groups = self._resolve_quant_grouping(
            self.head_dim, "K"
        )
        self.v_quant_group_size, self.v_num_scale_groups = self._resolve_quant_grouping(
            self.v_head_dim, "V"
        )
        assert self.head_dim % self._k_pack_factor == 0, (
            f"head_dim={self.head_dim} must be divisible by {self._k_pack_factor} "
            f"for {dtype} K packing"
        )
        assert self.v_head_dim % self._v_pack_factor == 0, (
            f"v_head_dim={self.v_head_dim} must be divisible by "
            f"{self._v_pack_factor} for {dtype} V packing"
        )

        self._create_arenas()

        # Cached attributes used by the rest of the stack.
        self.device_module = torch.get_device_module(self.device)
        self.alt_stream = None
        self.row_dim = self.head_num * self.head_dim  # for store_cache helpers
        self.same_kv_dim = self.head_dim == self.v_head_dim

        # Oscar rotation + clip. Per-layer orthogonal matrices [head_dim,
        # head_dim] / [v_head_dim, v_head_dim] are loaded in ``hp_dtype`` so
        # the ``rows @ R`` pre-pass and ``result @ R.T`` inverse are plain
        # bf16 GEMMs.
        self._oscar_cfg: OscarRotationConfig = load_oscar_rotation_config()
        self._k_clip_ratio: float = self._oscar_cfg.k_clip_ratio
        self._v_clip_ratio: float = self._oscar_cfg.v_clip_ratio
        self._lloyd_max: bool = envs.SGLANG_LLOYD_MAX.get()
        self._R_k: torch.Tensor = load_oscar_rotations(
            self._oscar_cfg.k_rotation_path,
            layer_num=self.layer_num,
            start_layer=self.start_layer,
            head_dim=self.head_dim,
            device=torch.device(self.device),
            dtype=self.hp_dtype,
        )
        self._R_v: torch.Tensor = load_oscar_rotations(
            self._oscar_cfg.v_rotation_path,
            layer_num=self.layer_num,
            start_layer=self.start_layer,
            head_dim=self.v_head_dim,
            device=torch.device(self.device),
            dtype=self.hp_dtype,
        )
        logger.info(
            "UnifiedInt2HPKVPool: Oscar rotation enabled (k_clip=%.4f v_clip=%.4f lloyd_max=%s)",
            self._k_clip_ratio,
            self._v_clip_ratio,
            self._lloyd_max,
        )

        # PQ K codebook (only for pq_k_int2v dtype).
        # Supports two file formats:
        #   (a) shared codebook: {"codebooks": list[tensor], "n_sub", "sub_dim", "n_centroids", ...}
        #   (b) per-layer codebooks: {"codebooks_per_layer": Tensor[L, N_SUB, N_CENTS, SUB_DIM], ...}
        # Per-layer is strongly preferred — shared codebooks fail when layers have
        # very different K scales (e.g. Qwen3-8B layer 0 has K ~7x larger than other layers).
        self._pq_codebook: Optional[torch.Tensor] = (
            None  # [N_SUB, N_CENTS, SUB_DIM] (shared)
        )
        self._pq_codebooks_per_layer: Optional[list] = (
            None  # list of [N_SUB, N_CENTS, SUB_DIM]
        )
        self._pq_cb_norms: Optional[torch.Tensor] = (
            None  # [N_SUB, N_CENTS] (shared, or None)
        )
        self._pq_cb_norms_per_layer: Optional[list] = None  # list of [N_SUB, N_CENTS]
        # RVQ stage-2 (residual VQ): _is_rvq / k_buffer2 / cb2 lists are initialized
        # earlier (before _create_arenas, which allocates k_buffer2). The stage-2
        # codebooks are populated in the load block below when codebooks_stage1 present.
        # PQ V (optional): when SGLANG_PQ_V_CODEBOOK is set, V uses PQ instead of INT2.
        # The early codebook peek above changes _v_pack_factor before arena
        # construction, so v_buffer is exactly n_sub bytes wide.
        self._pq_v: bool = False
        self._pq_v_codebooks_per_layer: Optional[list] = None
        self._pq_v_cb_norms_per_layer: Optional[list] = None
        if dtype == "pq_k_int2v":
            _pqv_path = envs.SGLANG_PQ_V_CODEBOOK.get()
            if _pqv_path:
                _vd = torch.load(_pqv_path, map_location="cpu", weights_only=False)
                _validate_pq_codebook_file(_vd, self.v_head_dim, "PQ V")
                _vcbs = _vd["codebooks_per_layer"].to(
                    torch.float16
                )  # [L, N_SUB, N_CENTS, SUB_DIM]
                _vids = _vd.get("layer_ids", list(range(len(_vcbs))))
                _v_l2i = {int(l): i for i, l in enumerate(_vids)}
                _vdev = torch.device(self.device)
                self._pq_v_codebooks_per_layer = []
                self._pq_v_cb_norms_per_layer = []
                for lid in range(self.start_layer, self.start_layer + self.layer_num):
                    vc = _vcbs[_v_l2i[lid]].to(_vdev).contiguous()
                    self._pq_v_codebooks_per_layer.append(vc)
                    self._pq_v_cb_norms_per_layer.append(
                        (vc.float() ** 2).sum(-1).to(dtype=torch.float32, device=_vdev)
                    )
                self._pq_v = True
                logger.info(
                    "UnifiedInt2HPKVPool: PQ V codebooks loaded from %s (n_layers=%d n_sub=%d "
                    "n_cents=%d sqnr_avg=%.2f dB) — V uses PQ, stored in v_buffer[:n_sub]",
                    _pqv_path,
                    self.layer_num,
                    _vd["n_sub"],
                    _vd["n_centroids"],
                    _vd.get("sqnr_avg", float("nan")),
                )
            pq_path = envs.SGLANG_PQ_K_CODEBOOK.get()
            if not pq_path:
                raise ValueError(
                    "SGLANG_PQ_K_CODEBOOK must point to a .pt codebook file when dtype=pq_k_int2v"
                )
            _pq_data = torch.load(pq_path, map_location="cpu", weights_only=False)
            _n_sub, _n_cents, _sub_dim = _validate_pq_codebook_file(
                _pq_data, self.head_dim, "PQ K"
            )
            dev = torch.device(self.device)
            if "codebooks_stage1" in _pq_data:
                # RVQ format: stage1 = PQ codebook, stage2 = residual codebook.
                _s1 = _pq_data["codebooks_stage1"].to(
                    torch.float16
                )  # [L, N_SUB, N_CENTS1, SUB_DIM]
                _s2 = _pq_data["codebooks_stage2"].to(
                    torch.float16
                )  # [L, N_SUB, N_CENTS2, SUB_DIM]
                _layer_ids = _pq_data.get("layer_ids", list(range(len(_s1))))
                _lid_to_cb_idx = {int(lid): i for i, lid in enumerate(_layer_ids)}
                self._pq_codebooks_per_layer = []
                self._pq_cb_norms_per_layer = []
                self._rvq_cb2_per_layer = []
                self._rvq_cb2_norms_per_layer = []
                for lid in range(self.start_layer, self.start_layer + self.layer_num):
                    c1 = _s1[_lid_to_cb_idx[lid]].to(dev).contiguous()
                    c2 = _s2[_lid_to_cb_idx[lid]].to(dev).contiguous()
                    self._pq_codebooks_per_layer.append(c1)
                    self._pq_cb_norms_per_layer.append(
                        (c1.float() ** 2).sum(-1).to(dtype=torch.float32, device=dev)
                    )
                    self._rvq_cb2_per_layer.append(c2)
                    self._rvq_cb2_norms_per_layer.append(
                        (c2.float() ** 2).sum(-1).to(dtype=torch.float32, device=dev)
                    )
                self._is_rvq = True
                logger.info(
                    "UnifiedInt2HPKVPool: RVQ K codebooks loaded from %s (n_layers=%d n_sub=%d "
                    "stage1_cents=%d stage2_cents=%d sub_dim=%d sqnr_stage1=%.2f sqnr_rvq=%.2f dB)",
                    pq_path,
                    self.layer_num,
                    _n_sub,
                    _s1.shape[2],
                    _s2.shape[2],
                    _sub_dim,
                    _pq_data.get("sqnr_stage1_avg", float("nan")),
                    _pq_data.get("sqnr_rvq_avg", float("nan")),
                )
            elif "codebooks_per_layer" in _pq_data:
                # Per-layer codebooks: [n_layers, N_SUB, N_CENTS, SUB_DIM] fp32
                _all_cbs = _pq_data["codebooks_per_layer"].to(
                    torch.float16
                )  # [L, N_SUB, N_CENTS, SUB_DIM]
                _layer_ids = _pq_data.get("layer_ids", list(range(len(_all_cbs))))
                assert len(_layer_ids) == _all_cbs.shape[0]
                # Build layer_index → codebook mapping (same ordering as layer_ids)
                _lid_to_cb_idx = {int(lid): i for i, lid in enumerate(_layer_ids)}
                self._pq_codebooks_per_layer = []
                self._pq_cb_norms_per_layer = []
                for lid in range(self.start_layer, self.start_layer + self.layer_num):
                    cb = _all_cbs[_lid_to_cb_idx[lid]].to(dev).contiguous()
                    self._pq_codebooks_per_layer.append(cb)
                    self._pq_cb_norms_per_layer.append(
                        (cb.float() ** 2).sum(-1).to(dtype=torch.float32, device=dev)
                    )
                sqnr_info = _pq_data.get("sqnr_avg", float("nan"))
                logger.info(
                    "UnifiedInt2HPKVPool: PQ K per-layer codebooks loaded from %s "
                    "(n_layers=%d n_sub=%d n_cents=%d sub_dim=%d avg_sqnr=%.2f dB)",
                    pq_path,
                    self.layer_num,
                    _n_sub,
                    _n_cents,
                    _sub_dim,
                    sqnr_info,
                )
            else:
                # Legacy shared codebook format
                _books = _pq_data["codebooks"]  # list of [N_CENTS, SUB_DIM] tensors
                self._pq_codebook = torch.stack(_books).to(
                    dtype=torch.float16, device=dev
                )
                self._pq_cb_norms = (
                    (self._pq_codebook.float() ** 2)
                    .sum(-1)
                    .to(dtype=torch.float32, device=dev)
                )
                logger.info(
                    "UnifiedInt2HPKVPool: PQ K shared codebook loaded from %s "
                    "(n_sub=%d n_cents=%d sub_dim=%d sqnr_train=%.2f dB) "
                    "[WARNING: shared codebook degrades on layers with different K scales]",
                    pq_path,
                    _n_sub,
                    _n_cents,
                    _sub_dim,
                    _pq_data.get("sqnr_train", float("nan")),
                )
            # Log the effective pack factor (already set via the early peek above).
            new_pack = self.head_dim // _n_sub
            if new_pack != 8:  # non-default → worth calling out
                logger.info(
                    "UnifiedInt2HPKVPool: PQ K n_sub=%d → k_pack_factor=%d (%.2f bpe K)",
                    _n_sub,
                    new_pack,
                    8.0 / (self.head_dim / _n_sub),
                )

        hp_total_slots = (
            self.num_hp_prefix_slots + self.max_req_slots * self.hp_recent_ring_size
        )
        self._finalize_allocation_log(hp_total_slots)
        hp_itemsize = torch.empty(0, dtype=self.hp_dtype).element_size()
        hp_bytes = (
            hp_total_slots
            * self.layer_num
            * self.head_num
            * (self.head_dim + self.v_head_dim)
            * hp_itemsize
        )
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
        return self.num_hp_prefix_slots + self.max_req_slots * self.hp_recent_ring_size

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

    def _resolve_quant_grouping(
        self, head_dim: int, tensor_name: str
    ) -> tuple[int, int]:
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
            self.num_hp_prefix_slots + self.max_req_slots * self.hp_recent_ring_size
        )
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                self.k_buffer = [
                    torch.zeros(
                        (
                            self.num_quant_pages * self.N_Q,
                            self.head_num,
                            self.head_dim // self._k_pack_factor,
                        ),
                        dtype=torch.uint8,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                if self._is_rvq:
                    # RVQ stage-2 codes: identical shape to k_buffer (same n_sub).
                    self.k_buffer2 = [
                        torch.zeros_like(self.k_buffer[i])
                        for i in range(self.layer_num)
                    ]
                self.v_buffer = [
                    torch.zeros(
                        (
                            self.num_quant_pages * self.N_Q,
                            self.head_num,
                            self.v_head_dim // self._v_pack_factor,
                        ),
                        dtype=torch.uint8,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.k_scales_zeros = [
                    torch.zeros(
                        (
                            self.num_quant_pages * self.N_Q,
                            self.head_num,
                            2 * self.k_num_scale_groups,
                        ),
                        dtype=self.scale_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_scales_zeros = [
                    torch.zeros(
                        (
                            self.num_quant_pages * self.N_Q,
                            self.head_num,
                            2 * self.v_num_scale_groups,
                        ),
                        dtype=self.scale_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.hp_k_buffer = [
                    torch.zeros(
                        (hp_total_slots, self.head_num, self.head_dim),
                        dtype=self.hp_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.hp_v_buffer = [
                    torch.zeros(
                        (hp_total_slots, self.head_num, self.v_head_dim),
                        dtype=self.hp_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

        # Cached device pointer arrays for the fused decode-flush kernel. The
        # flush kernel loops over layers inside the kernel, so it needs
        # per-layer base pointers as an int64 GPU tensor. Strides are identical
        # across layers (we enforce that below); the flush kernel reads the
        # single set at launch time via tl.constexpr.
        def _base_ptrs(tensors: List[torch.Tensor]) -> torch.Tensor:
            return torch.tensor(
                [t.data_ptr() for t in tensors],
                dtype=torch.int64,
                device=self.device,
            )

        self._flush_hp_k_ptrs = _base_ptrs(self.hp_k_buffer)
        self._flush_hp_v_ptrs = _base_ptrs(self.hp_v_buffer)
        self._flush_quant_k_ptrs = _base_ptrs(self.k_buffer)
        self._flush_quant_v_ptrs = _base_ptrs(self.v_buffer)
        self._flush_k_sz_ptrs = _base_ptrs(self.k_scales_zeros)
        self._flush_v_sz_ptrs = _base_ptrs(self.v_scales_zeros)

        # Strides (elements, not bytes) for each kind of buffer. The arenas are
        # all contiguous, so every layer shares the same strides; assert to be
        # safe.
        def _strides(t: torch.Tensor) -> tuple:
            return (int(t.stride(0)), int(t.stride(1)), int(t.stride(2)))

        hp_k_stride = _strides(self.hp_k_buffer[0])
        hp_v_stride = _strides(self.hp_v_buffer[0])
        q_k_stride = _strides(self.k_buffer[0])
        q_v_stride = _strides(self.v_buffer[0])
        k_sz_stride = _strides(self.k_scales_zeros[0])
        v_sz_stride = _strides(self.v_scales_zeros[0])
        for l in range(self.layer_num):
            assert _strides(self.hp_k_buffer[l]) == hp_k_stride
            assert _strides(self.hp_v_buffer[l]) == hp_v_stride
            assert _strides(self.k_buffer[l]) == q_k_stride
            assert _strides(self.v_buffer[l]) == q_v_stride
            assert _strides(self.k_scales_zeros[l]) == k_sz_stride
            assert _strides(self.v_scales_zeros[l]) == v_sz_stride

        self._flush_hp_k_stride = hp_k_stride
        self._flush_hp_v_stride = hp_v_stride
        self._flush_quant_k_stride = q_k_stride
        self._flush_quant_v_stride = q_v_stride
        self._flush_k_sz_stride = k_sz_stride
        self._flush_v_sz_stride = v_sz_stride

    # -- KVCache interface -------------------------------------------------

    def get_kv_size_bytes(self):
        k = sum(get_tensor_size_bytes(t) for t in self.k_buffer)
        if self.k_buffer2 is not None:
            k += sum(get_tensor_size_bytes(t) for t in self.k_buffer2)
        k += sum(get_tensor_size_bytes(s) for s in self.k_scales_zeros)
        k += sum(get_tensor_size_bytes(t) for t in self.hp_k_buffer)
        v = sum(get_tensor_size_bytes(t) for t in self.v_buffer)
        v += sum(get_tensor_size_bytes(s) for s in self.v_scales_zeros)
        v += sum(get_tensor_size_bytes(t) for t in self.hp_v_buffer)
        return k, v

    def _layer_index(self, layer_id: int) -> int:
        return layer_id - self.start_layer

    def get_pq_codebook(self, layer_id: int) -> torch.Tensor:
        """Return the PQ K codebook for this layer (per-layer or shared)."""
        if self._pq_codebooks_per_layer is not None:
            return self._pq_codebooks_per_layer[self._layer_index(layer_id)]
        return self._pq_codebook

    def get_pq_cb_norms(self, layer_id: int) -> torch.Tensor:
        """Return the PQ K centroid norms for this layer (per-layer or shared)."""
        if self._pq_cb_norms_per_layer is not None:
            return self._pq_cb_norms_per_layer[self._layer_index(layer_id)]
        return self._pq_cb_norms

    def get_pq_v_codebook(self, layer_id: int) -> Optional[torch.Tensor]:
        """PQ V codebook for this layer, or None if V is not PQ (uses INT2)."""
        if self._pq_v_codebooks_per_layer is None:
            return None
        return self._pq_v_codebooks_per_layer[self._layer_index(layer_id)]

    def get_pq_v_cb_norms(self, layer_id: int) -> Optional[torch.Tensor]:
        if self._pq_v_cb_norms_per_layer is None:
            return None
        return self._pq_v_cb_norms_per_layer[self._layer_index(layer_id)]

    def get_rvq_cb2(self, layer_id: int) -> Optional[torch.Tensor]:
        """RVQ stage-2 (residual) codebook for this layer, or None if not RVQ."""
        if self._rvq_cb2_per_layer is None:
            return None
        return self._rvq_cb2_per_layer[self._layer_index(layer_id)]

    def get_rvq_cb2_norms(self, layer_id: int) -> Optional[torch.Tensor]:
        if self._rvq_cb2_norms_per_layer is None:
            return None
        return self._rvq_cb2_norms_per_layer[self._layer_index(layer_id)]

    def get_raw_key_buffer2(self, layer_id: int) -> Optional[torch.Tensor]:
        """RVQ stage-2 codes buffer for this layer, or None if not RVQ."""
        if self.k_buffer2 is None:
            return None
        return self.k_buffer2[self._layer_index(layer_id)]

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
        buffers = {
            "k_buffer": self.k_buffer[idx],
            "v_buffer": self.v_buffer[idx],
            "k_scales_zeros": self.k_scales_zeros[idx],
            "v_scales_zeros": self.v_scales_zeros[idx],
            "dtype": self.dtype,
        }
        if self.k_buffer2 is not None:
            buffers["k_buffer2"] = self.k_buffer2[idx]
        return buffers

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
        k_hp = cache_k.to(self.hp_dtype) @ self._R_k[idx]
        if v_rotation_absorbed:
            v_hp = cache_v.to(self.hp_dtype)
        else:
            v_hp = cache_v.to(self.hp_dtype) @ self._R_v[idx]
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
        return self._rotate_kv_inplace(layer_id, cache_k, cache_v, v_rotation_absorbed)

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
            row_dim=self.row_dim,
            store_dtype=self.hp_dtype,
            device_module=self.device_module,
            alt_stream=self.alt_stream,
            same_kv_dim=self.same_kv_dim,
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
        use_fused_rotate = (
            envs.SGLANG_OSCAR_FUSED_ROTATE_CLIP_QUANT.get()
            and not already_hadamard_transformed
            and v_rotation_absorbed
            and clip_on
            and self.head_dim == self.v_head_dim
            and _get_num_scale_groups(self.k_scales_zeros[idx]) == 1
            and _get_num_scale_groups(self.v_scales_zeros[idx]) == 1
            and self.dtype != "pq_k_int2v"
        )
        if use_fused_rotate:
            assert v_rotation_absorbed, (
                "V rotation must be absorbed for fused oscar K-rotation + clip + quant + set"
            )
            if self.dtype == "int1":
                quantized_set_kv_int1_oscar_rotate_k_clip_triton(
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
                    lloyd_max=self._lloyd_max,
                )
            else:
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

        if not clip_on and self.dtype != "pq_k_int2v":
            if self.dtype == "int1":
                quantized_set_kv_int1_pretransformed_triton(
                    cache_k,
                    cache_v,
                    quant_loc,
                    self.k_buffer[idx],
                    self.v_buffer[idx],
                    self.k_scales_zeros[idx],
                    self.v_scales_zeros[idx],
                    hp_global_offset=mixed_hp_offset,
                )
            else:
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

        if self.dtype == "pq_k_int2v":
            # K: PQ encode (OSCAR rotation already applied via cache_k = rotated).
            k_for_pq = (
                cache_k if cache_k.dtype == torch.float16 else cache_k.to(torch.float16)
            )
            if k_for_pq.shape[0] > 0:
                pq_encode_k(
                    k_for_pq,
                    quant_loc,
                    self.k_buffer[idx],
                    self.get_pq_codebook(layer_id),
                    self.get_pq_cb_norms(layer_id),
                    hp_global_offset=mixed_hp_offset,
                )
                # RVQ stage 2 encodes the stage-1 reconstruction residual.
                if self._is_rvq:
                    cb1 = self.get_pq_codebook(layer_id)
                    recon1 = pq_decode_k_at_locs(
                        self.k_buffer[idx],
                        quant_loc,
                        cb1,
                        self.head_dim,
                        hp_global_offset=mixed_hp_offset,
                    ).to(k_for_pq.dtype)
                    pq_encode_k(
                        (k_for_pq - recon1).contiguous(),
                        quant_loc,
                        self.k_buffer2[idx],
                        self.get_rvq_cb2(layer_id),
                        self.get_rvq_cb2_norms(layer_id),
                        hp_global_offset=mixed_hp_offset,
                    )

            # V: PQ encode into the compact n_sub-byte buffer when configured;
            # otherwise retain the INT2 Lloyd-Max path.
            if self._pq_v:
                if cache_v.shape[0] > 0:
                    pq_encode_k(
                        cache_v.to(torch.float16),
                        quant_loc,
                        self.v_buffer[idx],
                        self.get_pq_v_codebook(layer_id),
                        self.get_pq_v_cb_norms(layer_id),
                        hp_global_offset=mixed_hp_offset,
                    )
            else:
                v_grouped_ok = _get_num_scale_groups(self.v_scales_zeros[idx]) == 1
                if v_grouped_ok:
                    _launch_single_clip_int2(
                        cache_v,
                        quant_loc,
                        self.v_buffer[idx],
                        self.v_scales_zeros[idx],
                        self._v_clip_ratio,
                        hp_global_offset=mixed_hp_offset,
                        lloyd_max=self._lloyd_max,
                    )
                else:
                    _launch_grouped_clip_int2(
                        cache_v,
                        quant_loc,
                        self.v_buffer[idx],
                        self.v_scales_zeros[idx],
                        self._v_clip_ratio,
                        hp_global_offset=mixed_hp_offset,
                    )
        elif self.dtype == "int1":
            quantized_set_kv_int1_pretransformed_clip_triton(
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
        else:
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

        layer_id = (
            layer_id_override if layer_id_override is not None else layer.layer_id
        )
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
                if self.k_buffer2 is not None:
                    self.k_buffer2[l][tgt_q] = self.k_buffer2[l][src_q]
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
