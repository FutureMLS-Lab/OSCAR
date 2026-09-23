"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

"""
Memory pool.

SGLang has two levels of memory pool.
ReqToTokenPool maps a request to its token locations.
TokenToKVPoolAllocator manages the indices to kv cache data.
KVCache actually holds the physical kv cache.
"""

import abc
import dataclasses
import logging
import math
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import numpy as np
import torch
import triton
import triton.language as tl
from sgl_kernel import hadamard_transform

from sglang.jit_kernel.kvcache import can_use_store_cache, store_cache
from sglang.srt.configs.mamba_utils import BaseLinearStateParams
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.layers.attention.q_rotation import (
    maybe_apply_k_rotation,
    maybe_apply_v_rotation,
)
from sglang.srt.layers.attention.nsa import index_buf_accessor
from sglang.srt.layers.attention.nsa.quant_k_cache import (
    quantize_k_cache,
    quantize_k_cache_separate,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.kv_quant_kernels import (
    quantized_set_kv_int4_triton,
    quantized_set_kv_int8_triton,
)
from sglang.srt.mem_cache.utils import (
    get_mla_kv_buffer_triton,
    maybe_init_custom_mem_pool,
    set_mla_kv_buffer_triton,
    set_mla_kv_scale_buffer_triton,
)
from sglang.srt.utils import (
    cpu_has_amx_support,
    is_cpu,
    is_cuda,
    is_hip,
    is_npu,
    next_power_of_2,
)
from sglang.srt.utils.custom_op import register_custom_op
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

store_cache = register_custom_op(store_cache, mutates_args=["k_cache", "v_cache"])

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import LayerDoneCounter
    from sglang.srt.managers.schedule_batch import Req


logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024
_is_cuda = is_cuda()
_is_npu = is_npu()
_is_cpu = is_cpu()
_cpu_has_amx_support = cpu_has_amx_support()
_is_hip = is_hip()
_hadamard_enabled = 1 if os.environ.get("HADAMARD", "0") in ("1", "true", "True") else 0
_rotate_v_enabled = 1 if os.environ.get("ROTATE_V", "0") in ("1", "true", "True") else 0
_hadamard_order = int(os.environ.get("HADAMARD_ORDER", "16"))
_hadamard_v_raw = os.environ.get("HADAMARD_V")
_hadamard_v_enabled = (
    int(_hadamard_v_raw in ("1", "true", "True"))
    if _hadamard_v_raw is not None
    else (_hadamard_enabled and _rotate_v_enabled)
)
_kv_quant_bits = int(os.environ.get("KV_QUANT_BITS", "4"))
_kv_max_quant_val = (1 << _kv_quant_bits) - 1
_kv_clip_ratio = float(os.environ.get("KV_CLIP_RATIO", "0"))
_kitty_boost_ratio = float(os.environ.get("KITTY_BOOST_RATIO", "0"))


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) in ("1", "true", "True")


# ── Per-layer K/V MSE collection (used to produce CoQuant-vs-baselines plots) ──
_KITTY_MSE_DUMP_ENABLED = _env_flag("SGLANG_KITTY_DUMP_MSE")
_KITTY_MSE_DUMP_DIR = os.environ.get("SGLANG_KITTY_DUMP_DIR", "/tmp/kitty_mse")
# Per-(K|V, layer_id) running [num_sum, den_sum, count].
from collections import defaultdict as _dd
_KITTY_MSE_STATE = {"K": _dd(lambda: [0.0, 0.0, 0]), "V": _dd(lambda: [0.0, 0.0, 0])}


_KITTY_MSE_DUMP_COUNTER = [0]


def _kitty_mse_finalize():
    if not _KITTY_MSE_DUMP_ENABLED:
        return
    if not _KITTY_MSE_STATE["K"] and not _KITTY_MSE_STATE["V"]:
        return
    try:
        from sglang.srt.distributed import get_tensor_model_parallel_rank
        rank = get_tensor_model_parallel_rank()
    except Exception:
        rank = os.environ.get("RANK", "0")
    os.makedirs(_KITTY_MSE_DUMP_DIR, exist_ok=True)
    out = {}
    for kv in ("K", "V"):
        out[kv] = {int(lid): {"num": v[0], "den": v[1], "count": v[2]}
                   for lid, v in _KITTY_MSE_STATE[kv].items()}
    import json as _json
    path = os.path.join(_KITTY_MSE_DUMP_DIR, f"kv_mse_rank{rank}.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        _json.dump(out, f)
    os.replace(tmp, path)


def _kitty_mse_periodic_dump():
    """Called from the fake-quant hot path. Writes JSON every 32 calls so the
    file is always near-current even if atexit doesn't fire under SIGTERM."""
    if not _KITTY_MSE_DUMP_ENABLED:
        return
    _KITTY_MSE_DUMP_COUNTER[0] += 1
    if _KITTY_MSE_DUMP_COUNTER[0] % 32 == 0:
        _kitty_mse_finalize()


import atexit as _atexit
_atexit.register(_kitty_mse_finalize)


@dataclass(frozen=True)
class HistoryFakeQuantConfig:
    enabled: bool
    sink_length: int
    recent_window_length: int
    history_quant_bits: int
    clip_ratio: float
    debug: bool
    k_rotation_path: str = ""
    v_rotation_path: str = ""
    k_clip_ratio: float = 0.0
    v_clip_ratio: float = 0.0
    k_head_group_size: int = 0
    v_head_group_size: int = 0
    k_quant_method: str = "asym"
    v_quant_method: str = "asym"

    def __post_init__(self):
        if self.sink_length < 0:
            raise ValueError(f"sink_length must be non-negative, got {self.sink_length}")
        if self.recent_window_length < 0:
            raise ValueError(
                "recent_window_length must be non-negative, "
                f"got {self.recent_window_length}"
            )
        if not (1 <= self.history_quant_bits <= 16):
            raise ValueError(
                "history_quant_bits must be in [1, 16], "
                f"got {self.history_quant_bits}"
            )
        if not (0.0 <= self.clip_ratio <= 1.0):
            raise ValueError(f"clip_ratio must be in [0, 1], got {self.clip_ratio}")

    def get_k_clip_ratio(self) -> float:
        return self.k_clip_ratio if self.k_clip_ratio > 0 else self.clip_ratio

    def get_v_clip_ratio(self) -> float:
        return self.v_clip_ratio if self.v_clip_ratio > 0 else self.clip_ratio


def _load_kitty_rotations(
    k_path: str, v_path: str
) -> dict[int, dict[str, torch.Tensor]]:
    """Load per-layer Kitty rotation matrices (eigendecomp+BR+Hadamard).

    Returns {layer_id: {"k": R_k, "v": R_v}} where R is orthogonal [head_dim, head_dim].
    """
    result: dict[int, dict[str, torch.Tensor]] = {}
    for path, key in [(k_path, "k"), (v_path, "v")]:
        if not path:
            continue
        state = torch.load(path, map_location="cpu")
        for lid, ldata in state["layers"].items():
            result.setdefault(int(lid), {})[key] = ldata["rotation"].float()
        logger.info("Loaded Kitty %s rotation from %s (%d layers)", key, path, len(state["layers"]))
    return result


def _load_history_fake_quant_config() -> HistoryFakeQuantConfig:
    recent_window_length = os.environ.get("SGLANG_KITTY_RECENT_WINDOW")
    if recent_window_length is None:
        recent_window_length = os.environ.get("SGLANG_KITTY_RECENT_WINDOW_LENGTH", "0")

    history_quant_bits = os.environ.get("SGLANG_KITTY_HISTORY_QUANT_BITS")
    if history_quant_bits is None:
        history_quant_bits = os.environ.get(
            "SGLANG_KITTY_HISTORY_BITS", str(_kv_quant_bits)
        )

    clip_ratio = os.environ.get("SGLANG_KITTY_HISTORY_CLIP_RATIO")
    if clip_ratio is None:
        clip_ratio = os.environ.get("SGLANG_KITTY_CLIP_RATIO", str(_kv_clip_ratio))

    return HistoryFakeQuantConfig(
        enabled=_env_flag("SGLANG_KITTY_HISTORY_FAKE_QUANT"),
        sink_length=int(os.environ.get("SGLANG_KITTY_SINK_LENGTH", "0")),
        recent_window_length=int(recent_window_length),
        history_quant_bits=int(history_quant_bits),
        clip_ratio=float(clip_ratio),
        debug=_env_flag("SGLANG_KITTY_HISTORY_DEBUG"),
        k_rotation_path=os.environ.get("SGLANG_KITTY_K_ROTATION_PATH", ""),
        v_rotation_path=os.environ.get("SGLANG_KITTY_V_ROTATION_PATH", ""),
        k_clip_ratio=float(os.environ.get("SGLANG_KITTY_K_CLIP_RATIO", "0")),
        v_clip_ratio=float(os.environ.get("SGLANG_KITTY_V_CLIP_RATIO", "0")),
        k_head_group_size=int(os.environ.get("SGLANG_KITTY_K_HEAD_GROUP_SIZE", "0")),
        v_head_group_size=int(os.environ.get("SGLANG_KITTY_V_HEAD_GROUP_SIZE", "0")),
        k_quant_method=os.environ.get("SGLANG_KITTY_K_QUANT_METHOD", "asym"),
        v_quant_method=os.environ.get("SGLANG_KITTY_V_QUANT_METHOD", "asym"),
    )

def _clip_quantile(x: torch.Tensor, ratio: float) -> torch.Tensor:
    """Clip per-row to the ratio-th percentile of |x| along last dim."""
    orig_shape = x.shape
    flat = x.reshape(-1, orig_shape[-1])
    thr = torch.quantile(flat.abs().float(), ratio, dim=-1, keepdim=True).to(x.dtype)
    flat = flat.clamp(-thr, thr)
    return flat.view(orig_shape)

def _kitty_fake_quant(x: torch.Tensor, boost_ratio: float, clip_ratio: float) -> torch.Tensor:
    """Kitty-style mixed-precision fake quantization for K cache.
    Top boost_ratio channels (by magnitude) -> INT4, rest -> INT2.
    Clip + per-row asymmetric quantization, returns FP16 with quantization noise baked in.
    """
    orig_shape = x.shape
    flat = x.reshape(-1, orig_shape[-1]).float()
    if clip_ratio > 0:
        thr = torch.quantile(flat.abs(), clip_ratio, dim=-1, keepdim=True)
        flat = flat.clamp(-thr, thr)
    mn = flat.min(-1, keepdim=True).values
    mx = flat.max(-1, keepdim=True).values
    s2 = ((mx - mn) / 3).clamp(min=1e-10)
    result = ((flat - mn) / s2).round().clamp(0, 3) * s2 + mn
    if boost_ratio > 0:
        n_boost = max(1, int(flat.shape[-1] * boost_ratio))
        channel_mag = flat.abs().mean(dim=0)
        _, boost_idx = channel_mag.topk(n_boost)
        s4 = ((mx - mn) / 15).clamp(min=1e-10)
        q4 = ((flat - mn) / s4).round().clamp(0, 15) * s4 + mn
        result[:, boost_idx] = q4[:, boost_idx]
    return result.to(x.dtype).view(orig_shape)

_quant_sim_layers_str = os.environ.get("SGLANG_QUANT_SIM_LAYERS", "")
_quant_sim_layers: set[int] = (
    {int(x.strip()) for x in _quant_sim_layers_str.split(",") if x.strip()}
    if _quant_sim_layers_str
    else set()
)

_quant_sim_rotation_path = os.environ.get("SGLANG_QUANT_SIM_ROTATION_PATH", "")
_quant_sim_rotations: dict[int, torch.Tensor] = {}
_quant_sim_grouping: str = "layer"

if _quant_sim_layers and _quant_sim_rotation_path:
    _sim_state = torch.load(_quant_sim_rotation_path, map_location="cpu")
    _quant_sim_grouping = _sim_state.get("source_grouping", _sim_state.get("grouping", "layer"))
    for lid, ldata in _sim_state["layers"].items():
        _quant_sim_rotations[int(lid)] = ldata["rotation"].to(dtype=torch.float32).contiguous()
    del _sim_state
    logger.info(
        "Quant simulation enabled for layers %s with rotation from %s (grouping=%s)",
        sorted(_quant_sim_layers), _quant_sim_rotation_path, _quant_sim_grouping,
    )
elif _quant_sim_layers:
    logger.info("Quant simulation enabled for layers %s (Hadamard only, no QR)", sorted(_quant_sim_layers))


def _simulate_int4_quantize_dequantize(x: torch.Tensor) -> torch.Tensor:
    """Simulate per-head asymmetric int4 quantization noise (matches Triton int4 kernel)."""
    return _simulate_asymmetric_quantize_dequantize(x, max_quant_val=_kv_max_quant_val)


def _simulate_asymmetric_quantize_dequantize(
    x: torch.Tensor, *, max_quant_val: int
) -> torch.Tensor:
    """Simulate per-row asymmetric quantize-dequantize along the last dim."""
    x_fp32 = x.float()
    val_min = x_fp32.amin(dim=-1, keepdim=True)
    val_max = x_fp32.amax(dim=-1, keepdim=True)
    val_range = (val_max - val_min).clamp(min=1e-8)
    scale = val_range / max_quant_val
    zero = -val_min / scale
    q = (x_fp32 / scale + zero + 0.5).to(torch.int32).clamp(0, max_quant_val)
    return ((q.float() - zero) * scale).to(x.dtype)


def _simulate_bits_quantize_dequantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    if bits >= 16:
        return x
    return _simulate_asymmetric_quantize_dequantize(x, max_quant_val=(1 << bits) - 1)


# Lloyd-Max optimal boundaries/centroids for 4-level N(0,1) quantizer
_NF2_BOUNDARIES = torch.tensor([-0.9816, 0.0, 0.9816])
_NF2_CENTROIDS = torch.tensor([-1.510, -0.4528, 0.4528, 1.510])


def _simulate_nf2_quantize_dequantize(x: torch.Tensor) -> torch.Tensor:
    """NormalFloat-2: Lloyd-Max optimal 4-level quantizer for N(0,1).
    Per-row: normalize to zero-mean unit-variance, quantize, denormalize."""
    x_fp32 = x.float()
    mu = x_fp32.mean(dim=-1, keepdim=True)
    sd = x_fp32.std(dim=-1, keepdim=True).clamp(min=1e-10)
    xn = (x_fp32 - mu) / sd
    bnd = _NF2_BOUNDARIES.to(device=x.device)
    ctr = _NF2_CENTROIDS.to(device=x.device)
    idx = (xn.unsqueeze(-1) > bnd).sum(-1)
    return (ctr[idx] * sd + mu).to(x.dtype)


def _simulate_group_quantize_dequantize(
    x: torch.Tensor, group_size: int, bits: int
) -> torch.Tensor:
    """Per-group asymmetric quantization along the last dim.
    Splits last dim into groups of group_size, independent scale/zero per group."""
    if bits >= 16:
        return x
    orig_shape = x.shape
    D = orig_shape[-1]
    assert D % group_size == 0, f"dim {D} not divisible by group_size {group_size}"
    max_quant_val = (1 << bits) - 1
    x_fp32 = x.float().reshape(*orig_shape[:-1], D // group_size, group_size)
    val_min = x_fp32.amin(dim=-1, keepdim=True)
    val_max = x_fp32.amax(dim=-1, keepdim=True)
    val_range = (val_max - val_min).clamp(min=1e-8)
    scale = val_range / max_quant_val
    zero = -val_min / scale
    q = (x_fp32 / scale + zero + 0.5).to(torch.int32).clamp(0, max_quant_val)
    return ((q.float() - zero) * scale).reshape(orig_shape).to(x.dtype)


_quant_sim_device_cache: dict[tuple[int, str, str], torch.Tensor] = {}


def _sim_get_rotation(layer_id: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (layer_id, str(device), str(dtype))
    cached = _quant_sim_device_cache.get(key)
    if cached is not None:
        return cached
    R = _quant_sim_rotations[layer_id].to(device=device, dtype=dtype)
    _quant_sim_device_cache[key] = R
    return R


def _sim_apply_rotation(k: torch.Tensor, layer_id: int) -> torch.Tensor:
    """Apply QR rotation for simulation (independent of _Q_ROTATION_MANAGER)."""
    R = _sim_get_rotation(layer_id, k.device, k.dtype)
    if _quant_sim_grouping == "layer":
        return torch.matmul(k, R)
    if _quant_sim_grouping == "kv_group":
        return torch.einsum("tgd,gdf->tgf", k, R)
    if _quant_sim_grouping == "head":
        return torch.einsum("thd,hdf->thf", k, R)
    raise ValueError(f"Unsupported sim grouping: {_quant_sim_grouping}")


def _sim_apply_inverse_rotation(k: torch.Tensor, layer_id: int) -> torch.Tensor:
    """Apply inverse QR rotation for simulation."""
    R = _sim_get_rotation(layer_id, k.device, k.dtype)
    if _quant_sim_grouping == "layer":
        return torch.matmul(k, R.t())
    if _quant_sim_grouping == "kv_group":
        return torch.einsum("tgd,gfd->tgf", k, R)
    if _quant_sim_grouping == "head":
        return torch.einsum("thd,hfd->thf", k, R)
    raise ValueError(f"Unsupported sim grouping: {_quant_sim_grouping}")


def get_tensor_size_bytes(t: Union[torch.Tensor, List[torch.Tensor]]):
    if isinstance(t, list):
        return sum(get_tensor_size_bytes(x) for x in t)
    return np.prod(t.shape) * t.dtype.itemsize


def _set_kv_buffer_impl(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    row_dim: int,  # head_num * head_dim
    store_dtype: torch.dtype,
    device_module: Any,
    alt_stream: Optional[torch.cuda.Stream] = None,
    same_kv_dim: bool = True,
) -> None:
    row_bytes = row_dim * store_dtype.itemsize
    if _is_cuda and same_kv_dim and can_use_store_cache(row_bytes):
        return store_cache(
            k.view(-1, row_dim),
            v.view(-1, row_dim),
            k_cache.view(-1, row_dim),
            v_cache.view(-1, row_dim),
            indices,
            row_bytes=row_bytes,
        )

    from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

    if get_is_capture_mode() and alt_stream is not None:
        current_stream = device_module.current_stream()
        alt_stream.wait_stream(current_stream)
        k_cache[indices] = k
        with device_module.stream(alt_stream):
            v_cache[indices] = v
        current_stream.wait_stream(alt_stream)
    else:  # fallback to naive implementation
        k_cache[indices] = k
        v_cache[indices] = v


class ReqToTokenPool:
    """A memory pool that maps a request to its token locations."""

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
    ):
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.size = size
        self.max_context_len = max_context_len
        self.device = device
        with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.req_to_token = torch.zeros(
                (size, max_context_len), dtype=torch.int32, device=device
            )
        self.free_slots = list(range(size))
        self.req_epoch = np.zeros(size, dtype=np.int64)

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, reqs: list[Req]) -> Optional[List[int]]:
        chunked = [i for i, r in enumerate(reqs) if r.req_pool_idx is not None]
        if not any(r.is_dllm() for r in reqs):
            assert (
                len(chunked) <= 1
            ), "only one chunked request may reuse req_pool_idx in a batch"
        assert all(
            reqs[i].is_chunked > 0 or reqs[i].kv_committed_len > 0 for i in chunked
        ), "request has req_pool_idx but is not chunked"

        need_size = len(reqs) - len(chunked)
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        offset = 0
        for r in reqs:
            if r.req_pool_idx is None:
                r.req_pool_idx = select_index[offset]
                self.req_epoch[r.req_pool_idx] += 1
                offset += 1
        return [r.req_pool_idx for r in reqs]

    def free(self, req: Req):
        assert req.req_pool_idx is not None, "request must have req_pool_idx"
        self.free_slots.append(req.req_pool_idx)
        req.req_pool_idx = None

    def clear(self):
        self.free_slots = list(range(self.size))
        self.req_epoch.fill(0)


class MambaPool:
    @dataclass(frozen=True, kw_only=True)
    class State:
        conv: List[torch.Tensor]
        temporal: torch.Tensor

        def at_layer_idx(self, layer: int):
            kwargs = {}
            for k, v in vars(self).items():
                if k == "conv" or k == "intermediate_conv_window":
                    kwargs[k] = [conv[layer] for conv in v]
                else:
                    kwargs[k] = v[layer]
            return type(self)(**kwargs)

        def mem_usage_bytes(self):
            return sum(
                get_tensor_size_bytes(getattr(self, f.name))
                for f in dataclasses.fields(self)
            )

    @dataclass(frozen=True, kw_only=True)
    class SpeculativeState(State):
        intermediate_ssm: torch.Tensor
        intermediate_conv_window: List[torch.Tensor]

    def __init__(
        self,
        *,
        size: int,
        spec_state_size: int,
        cache_params: BaseLinearStateParams,
        device: str,
        enable_memory_saver: bool = False,
        speculative_num_draft_tokens: Optional[int] = None,
    ):
        conv_state_shape = cache_params.shape.conv
        temporal_state_shape = cache_params.shape.temporal
        conv_dtype = cache_params.dtype.conv
        ssm_dtype = cache_params.dtype.temporal
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        num_mamba_layers = len(cache_params.layers)

        self.size = size
        self.device = device

        # for disagg with nvlink
        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
            maybe_init_custom_mem_pool(device=self.device)
        )

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE), (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.enable_custom_mem_pool
            else nullcontext()
        ):
            conv_state = [
                torch.zeros(
                    size=(num_mamba_layers, size + 1) + conv_shape,
                    dtype=conv_dtype,
                    device=device,
                )
                for conv_shape in conv_state_shape
            ]

            if _is_cpu and _cpu_has_amx_support:
                from sglang.srt.layers.amx_utils import _init_amx_conv_state

                # CPU uses a different layout of conv_state for kernel optimization
                conv_state = _init_amx_conv_state(conv_state)

            temporal_state = torch.zeros(
                size=(num_mamba_layers, size + 1) + temporal_state_shape,
                dtype=ssm_dtype,
                device=device,
            )
            if speculative_num_draft_tokens is not None:
                # Cache intermediate SSM states per draft token during target verify
                # Shape: [num_layers, size + 1, speculative_num_draft_tokens, HV, K, V]
                intermediate_ssm_state_cache = torch.zeros(
                    size=(
                        num_mamba_layers,
                        spec_state_size + 1,
                        speculative_num_draft_tokens,
                        temporal_state_shape[0],
                        temporal_state_shape[1],
                        temporal_state_shape[2],
                    ),
                    dtype=ssm_dtype,
                    device="cuda",
                )
                # Cache intermediate conv windows (last K-1 inputs) per draft token during target verify
                # Shape: [num_layers, size + 1, speculative_num_draft_tokens, dim, K-1]
                intermediate_conv_window_cache = [
                    torch.zeros(
                        size=(
                            num_mamba_layers,
                            spec_state_size + 1,
                            speculative_num_draft_tokens,
                            conv_shape[0],
                            conv_shape[1],
                        ),
                        dtype=conv_dtype,
                        device="cuda",
                    )
                    for conv_shape in conv_state_shape
                ]
                self.mamba_cache = self.SpeculativeState(
                    conv=conv_state,
                    temporal=temporal_state,
                    intermediate_ssm=intermediate_ssm_state_cache,
                    intermediate_conv_window=intermediate_conv_window_cache,
                )
                logger.info(
                    f"Mamba Cache is allocated. "
                    f"max_mamba_cache_size: {size}, "
                    f"conv_state size: {get_tensor_size_bytes(conv_state) / GB:.2f}GB, "
                    f"ssm_state size: {get_tensor_size_bytes(temporal_state) / GB:.2f}GB "
                    f"intermediate_ssm_state_cache size: {get_tensor_size_bytes(intermediate_ssm_state_cache) / GB:.2f}GB "
                    f"intermediate_conv_window_cache size: {get_tensor_size_bytes(intermediate_conv_window_cache) / GB:.2f}GB "
                )
            else:
                self.mamba_cache = self.State(conv=conv_state, temporal=temporal_state)
                logger.info(
                    f"Mamba Cache is allocated. "
                    f"max_mamba_cache_size: {size}, "
                    f"conv_state size: {get_tensor_size_bytes(conv_state) / GB:.2f}GB, "
                    f"ssm_state size: {get_tensor_size_bytes(temporal_state) / GB:.2f}GB "
                )
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            self.free_slots = torch.arange(
                1, self.size + 1, dtype=torch.int64, device=self.device
            )
            self.mem_usage = self.mamba_cache.mem_usage_bytes() / GB
            self.num_mamba_layers = num_mamba_layers

    def get_speculative_mamba2_params_all_layers(self) -> SpeculativeState:
        assert isinstance(self.mamba_cache, self.SpeculativeState)
        return self.mamba_cache

    def mamba2_layer_cache(self, layer_id: int):
        return self.mamba_cache.at_layer_idx(layer_id)

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        # clear at alloc time, fill allocated slots with zeros
        for i in range(len(self.mamba_cache.conv)):
            self.mamba_cache.conv[i][:, select_index] = 0
        self.mamba_cache.temporal[:, select_index] = 0

        return select_index

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        self.free_slots = torch.cat((self.free_slots, free_index))

    def clear(self):
        self.free_slots = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )

    def copy_from(self, src_index: torch.Tensor, dst_index: torch.Tensor):
        for i in range(len(self.mamba_cache.conv)):
            self.mamba_cache.conv[i][:, dst_index] = self.mamba_cache.conv[i][
                :, src_index
            ]
        self.mamba_cache.temporal[:, dst_index] = self.mamba_cache.temporal[
            :, src_index
        ]
        return

    def fork_from(self, src_index: torch.Tensor) -> Optional[torch.Tensor]:
        dst_index = self.alloc(1)
        if dst_index == None:
            return None
        self.copy_from(src_index, dst_index)
        return dst_index

    def get_contiguous_buf_infos(self):
        """
        Get buffer info for RDMA registration.
        Only returns conv and temporal state buffers, excluding intermediate buffers
        used for speculative decoding (intermediate_ssm, intermediate_conv_window).
        """
        state_tensors = []
        for field in vars(self.mamba_cache):
            # Skip intermediate buffers used only for speculative decoding
            # These buffers have different size (spec_state_size + 1) and should not be transferred
            if field in ("intermediate_ssm", "intermediate_conv_window"):
                continue
            value = getattr(self.mamba_cache, field)
            if isinstance(value, list):
                state_tensors.extend(value)
            else:
                state_tensors.append(value)
        data_ptrs, data_lens, item_lens = [], [], []

        for _, state_tensor in enumerate(state_tensors):
            data_ptrs += [
                state_tensor[i].data_ptr() for i in range(self.num_mamba_layers)
            ]
            data_lens += [state_tensor[i].nbytes for i in range(self.num_mamba_layers)]
            item_lens += [
                state_tensor[i][0].nbytes for i in range(self.num_mamba_layers)
            ]
        return data_ptrs, data_lens, item_lens

    def get_state_dim_per_tensor(self):
        """Get the sliceable dimension size for each state tensor.

        For mamba state, the layout is:
        - conv_state: [num_layers, size+1, conv_dim/tp, conv_kernel-1]
        - temporal_state: [num_layers, size+1, num_heads/tp, head_dim, state_size]

        The 3rd dimension (index 2) is the one that gets sliced by TP.
        Returns the size of this dimension for each tensor (repeated for each layer).
        """
        state_tensors = []
        for field in vars(self.mamba_cache):
            value = getattr(self.mamba_cache, field)
            if isinstance(value, list):
                state_tensors.extend(value)
            else:
                state_tensors.append(value)

        dim_per_tensor = []
        for state_tensor in state_tensors:
            # state_tensor shape: [num_layers, size+1, sliceable_dim, ...]
            # The sliceable dimension is at index 2 (after num_layers and size)
            sliceable_dim = state_tensor.shape[2]
            # Repeat for each layer since we have per-layer data_ptrs
            dim_per_tensor += [sliceable_dim] * self.num_mamba_layers
        return dim_per_tensor


class HybridReqToTokenPool(ReqToTokenPool):
    """A memory pool that maps a request to its token locations."""

    def __init__(
        self,
        *,
        size: int,
        mamba_size: int,
        mamba_spec_state_size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        cache_params: BaseLinearStateParams,
        enable_mamba_extra_buffer: bool,
        speculative_num_draft_tokens: int = None,
    ):
        super().__init__(
            size=size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
        )
        self.mamba_ping_pong_track_buffer_size = (
            2 if speculative_num_draft_tokens is None else 1
        )
        self.enable_mamba_extra_buffer = enable_mamba_extra_buffer
        self.enable_memory_saver = enable_memory_saver
        self._init_mamba_pool(
            size=mamba_size,
            mamba_spec_state_size=mamba_spec_state_size,
            cache_params=cache_params,
            device=device,
            enable_mamba_extra_buffer=enable_mamba_extra_buffer,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )

    def _init_mamba_pool(
        self,
        size: int,
        mamba_spec_state_size: int,
        cache_params: BaseLinearStateParams,
        device: str,
        enable_mamba_extra_buffer: bool,
        speculative_num_draft_tokens: int = None,
    ):
        self.mamba_pool = MambaPool(
            size=size,
            spec_state_size=mamba_spec_state_size,
            cache_params=cache_params,
            device=device,
            enable_memory_saver=self.enable_memory_saver,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )
        self.mamba_map = {layer_id: i for i, layer_id in enumerate(cache_params.layers)}

        self.device = device
        self.req_index_to_mamba_index_mapping: torch.Tensor = torch.zeros(
            size, dtype=torch.int32, device=self.device
        )
        if enable_mamba_extra_buffer:
            self.req_index_to_mamba_ping_pong_track_buffer_mapping: torch.Tensor = (
                torch.zeros(
                    (size, self.mamba_ping_pong_track_buffer_size),
                    dtype=torch.int32,
                    device=self.device,
                )
            )

    # For chunk prefill req, we do not need to allocate mamba cache,
    # We could use allocated mamba cache instead.
    def alloc(self, reqs: List["Req"]) -> Optional[List[int]]:
        select_index = super().alloc(reqs)
        if select_index is None:
            return None

        mamba_index = []
        mamba_ping_pong_track_buffer_list = []
        for req in reqs:
            mid = None
            if req.mamba_pool_idx is not None:  # for radix cache
                mid = req.mamba_pool_idx
            else:
                mid = self.mamba_pool.alloc(1)
                assert (
                    mid is not None
                ), f"Not enough space for mamba cache, try to increase --mamba-full-memory-ratio or --max-mamba-cache-size. {mid=}, {self.mamba_pool.size=}, {self.mamba_pool.available_size()=}, {len(reqs)=}"
                mid = mid[0]
                req.mamba_pool_idx = mid
            mamba_index.append(mid)
            if self.enable_mamba_extra_buffer:
                if req.mamba_ping_pong_track_buffer is None:
                    req.mamba_ping_pong_track_buffer = self.mamba_pool.alloc(
                        self.mamba_ping_pong_track_buffer_size
                    )
                    assert (
                        req.mamba_ping_pong_track_buffer is not None
                    ), "Not enough space for mamba ping pong idx, try to increase --mamba-full-memory-ratio."
                    req.mamba_next_track_idx = 0
                mamba_ping_pong_track_buffer_list.append(
                    req.mamba_ping_pong_track_buffer.tolist()
                )
        assert len(select_index) == len(
            mamba_index
        ), f"Not enough space for mamba cache, try to increase --mamba-full-memory-ratio or --max-mamba-cache-size."
        if self.enable_mamba_extra_buffer:
            assert len(select_index) == len(
                mamba_ping_pong_track_buffer_list
            ), f"Not enough space for mamba ping pong idx, try to increase --mamba-full-memory-ratio."
        self.req_index_to_mamba_index_mapping[select_index] = torch.tensor(
            mamba_index, dtype=torch.int32, device=self.device
        )
        if self.enable_mamba_extra_buffer:
            self.req_index_to_mamba_ping_pong_track_buffer_mapping[select_index] = (
                torch.tensor(
                    mamba_ping_pong_track_buffer_list,
                    dtype=torch.int32,
                    device=self.device,
                )
            )
        return select_index

    def get_mamba_indices(self, req_indices: torch.Tensor) -> torch.Tensor:
        return self.req_index_to_mamba_index_mapping[req_indices]

    def mamba2_layer_cache(self, layer_id: int):
        assert layer_id in self.mamba_map
        return self.mamba_pool.mamba2_layer_cache(self.mamba_map[layer_id])

    def get_speculative_mamba2_params_all_layers(self) -> MambaPool.SpeculativeState:
        return self.mamba_pool.get_speculative_mamba2_params_all_layers()

    def get_mamba_ping_pong_other_idx(self, mamba_next_track_idx: int) -> int:
        if self.mamba_ping_pong_track_buffer_size == 2:
            return 1 - mamba_next_track_idx
        else:
            return mamba_next_track_idx

    def free_mamba_cache(
        self, req: "Req", mamba_ping_pong_track_buffer_to_keep: Optional[int] = None
    ):
        mamba_index = req.mamba_pool_idx
        assert mamba_index is not None, "double free? mamba_index is None"
        self.mamba_pool.free(mamba_index.unsqueeze(0))
        req.mamba_pool_idx = None

        if self.enable_mamba_extra_buffer:
            mamba_ping_pong_track_buffer_to_free = (
                self.req_index_to_mamba_ping_pong_track_buffer_mapping[req.req_pool_idx]
            )
            if mamba_ping_pong_track_buffer_to_keep is not None:
                assert mamba_ping_pong_track_buffer_to_keep in [
                    0,
                    1,
                ], f"mamba_ping_pong_track_buffer_to_keep must be 0 or 1, {mamba_ping_pong_track_buffer_to_keep=}"
                idx_to_free = list(range(self.mamba_ping_pong_track_buffer_size))
                idx_to_free.remove(mamba_ping_pong_track_buffer_to_keep)
                mamba_ping_pong_track_buffer_to_free = (
                    mamba_ping_pong_track_buffer_to_free[idx_to_free]
                )
            self.mamba_pool.free(mamba_ping_pong_track_buffer_to_free)

    def clear(self):
        logger.info("Reset HybridReqToTokenPool")
        super().clear()
        self.mamba_pool.clear()
        self.req_index_to_mamba_index_mapping.zero_()
        if self.enable_mamba_extra_buffer:
            self.req_index_to_mamba_ping_pong_track_buffer_mapping.zero_()


class KVCache(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: Union[torch.dtype, str],
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        model_dtype: Optional[
            torch.dtype
        ] = None,  # to dequantize the kv cache to model_dtype
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        if model_dtype is not None:
            self.model_dtype = model_dtype
        elif dtype in ("int4", "int8"):
            raise ValueError(f"model_dtype is required for int4 or int8 kv cache")

        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn, "int4", "int8"):
            # NOTE: Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.layer_num = layer_num
        self.start_layer = start_layer or 0
        self.end_layer = end_layer or layer_num - 1
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        self.mem_usage = 0

        # used for chunked cpu-offloading
        self.cpu_offloading_chunk_size = 8192

        # default state for optional layer-wise transfer control
        self.layer_transfer_counter = None

        # for disagg with nvlink
        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
            maybe_init_custom_mem_pool(device=self.device)
        )

    def _finalize_allocation_log(self, num_tokens: int):
        """Common logging and mem_usage computation for KV cache allocation.
        Supports both tuple (K, V) size returns and single KV size returns.
        """
        kv_size_bytes = self.get_kv_size_bytes()
        if isinstance(kv_size_bytes, tuple):
            k_size, v_size = kv_size_bytes
            k_size_GB = k_size / GB
            v_size_GB = v_size / GB
            logger.info(
                f"KV Cache is allocated. #tokens: {num_tokens}, K size: {k_size_GB:.2f} GB, V size: {v_size_GB:.2f} GB"
            )
            self.mem_usage = k_size_GB + v_size_GB
        else:
            kv_size_GB = kv_size_bytes / GB
            logger.info(
                f"KV Cache is allocated. #tokens: {num_tokens}, KV size: {kv_size_GB:.2f} GB"
            )
            self.mem_usage = kv_size_GB

    @abc.abstractmethod
    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    @abc.abstractmethod
    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        forward_batch: Optional[Any] = None,
    ) -> None:
        raise NotImplementedError()

    def register_layer_transfer_counter(self, layer_transfer_counter: LayerDoneCounter):
        self.layer_transfer_counter = layer_transfer_counter

    def get_cpu_copy(self, indices):
        raise NotImplementedError()

    def load_cpu_copy(self, kv_cache_cpu, indices):
        raise NotImplementedError()

    def maybe_get_custom_mem_pool(self):
        return self.custom_mem_pool


class MHATokenToKVPool(KVCache):

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        v_head_dim: Optional[int] = None,
        swa_head_num: Optional[int] = None,
        swa_head_dim: Optional[int] = None,
        swa_v_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
        enable_kv_cache_copy: bool = False,
        model_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
            model_dtype,
        )
        self.head_num = swa_head_num if swa_head_num is not None else head_num
        self.head_dim = swa_head_dim if swa_head_dim is not None else head_dim
        self.v_head_dim = (
            swa_v_head_dim
            if swa_v_head_dim is not None
            else v_head_dim if v_head_dim is not None else head_dim
        )

        self._create_buffers()

        self.device_module = torch.get_device_module(self.device)
        self.alt_stream = (
            self.device_module.Stream() if _is_cuda and enable_alt_stream else None
        )

        if enable_kv_cache_copy:
            self._init_kv_copy_and_warmup()
        else:
            self._kv_copy_config = None

        self._finalize_allocation_log(size)

        # for store_cache JIT kernel
        self.row_dim = self.head_num * self.head_dim
        self.same_kv_dim = self.head_dim == self.v_head_dim
        self.history_fake_quant_cfg = _load_history_fake_quant_config()
        self.history_fake_quant_enabled = (
            self.history_fake_quant_cfg.enabled
            and self.dtype not in ("int4", "int8")
            and self.store_dtype == self.dtype
        )
        self._history_quantized_upto: Optional[np.ndarray] = None
        self._history_seen_req_epoch: Optional[np.ndarray] = None
        self._history_debug_counters = {
            "k_rows": np.zeros(self.layer_num, dtype=np.int64),
            "v_rows": np.zeros(self.layer_num, dtype=np.int64),
            "calls": 0,
        }
        self._kitty_rotations: dict[int, dict[str, torch.Tensor]] = {}
        if self.history_fake_quant_cfg.enabled and not self.history_fake_quant_enabled:
            logger.warning(
                "Disabling Kitty history fake quant for dtype=%s store_dtype=%s. "
                "Prototype currently only supports single-dtype floating KV pools.",
                self.dtype,
                self.store_dtype,
            )
        elif self.history_fake_quant_enabled:
            cfg = self.history_fake_quant_cfg
            if cfg.k_rotation_path or cfg.v_rotation_path:
                self._kitty_rotations = _load_kitty_rotations(
                    cfg.k_rotation_path, cfg.v_rotation_path
                )
            logger.info(
                "Kitty history fake quant enabled: sink=%d recent=%d bits=%d clip=%.4f "
                "k_clip=%.4f v_clip=%.4f k_grp=%d v_grp=%d k_method=%s v_method=%s "
                "k_rot=%s v_rot=%s debug=%s",
                cfg.sink_length,
                cfg.recent_window_length,
                cfg.history_quant_bits,
                cfg.clip_ratio,
                cfg.k_clip_ratio,
                cfg.v_clip_ratio,
                cfg.k_head_group_size,
                cfg.v_head_group_size,
                cfg.k_quant_method,
                cfg.v_quant_method,
                bool(cfg.k_rotation_path),
                bool(cfg.v_rotation_path),
                cfg.debug,
            )

    def _init_kv_copy_and_warmup(self):
        # Heuristics for KV copy tiling
        _KV_COPY_STRIDE_THRESHOLD_LARGE = 8192
        _KV_COPY_STRIDE_THRESHOLD_MEDIUM = 4096
        _KV_COPY_TILE_SIZE_LARGE = 512
        _KV_COPY_TILE_SIZE_MEDIUM = 256
        _KV_COPY_TILE_SIZE_SMALL = 128
        _KV_COPY_NUM_WARPS_LARGE_TILE = 8
        _KV_COPY_NUM_WARPS_SMALL_TILE = 4

        stride_bytes = int(self.data_strides[0].item())
        if stride_bytes >= _KV_COPY_STRIDE_THRESHOLD_LARGE:
            bytes_per_tile = _KV_COPY_TILE_SIZE_LARGE
        elif stride_bytes >= _KV_COPY_STRIDE_THRESHOLD_MEDIUM:
            bytes_per_tile = _KV_COPY_TILE_SIZE_MEDIUM
        else:
            bytes_per_tile = _KV_COPY_TILE_SIZE_SMALL

        # Calculate num_locs_upper to avoid large Triton specialization (e.g. 8192)
        chunk_upper = 128 if bytes_per_tile >= _KV_COPY_TILE_SIZE_LARGE else 256

        self._kv_copy_config = {
            "bytes_per_tile": bytes_per_tile,
            "byte_tiles": (stride_bytes + bytes_per_tile - 1) // bytes_per_tile,
            "num_warps": (
                _KV_COPY_NUM_WARPS_SMALL_TILE
                if bytes_per_tile <= _KV_COPY_TILE_SIZE_MEDIUM
                else _KV_COPY_NUM_WARPS_LARGE_TILE
            ),
            "num_locs_upper": chunk_upper,
        }

        dummy_loc = torch.zeros(chunk_upper, dtype=torch.int64, device=self.device)
        grid = (self.data_ptrs.numel(), self._kv_copy_config["byte_tiles"])

        copy_all_layer_kv_cache_tiled[grid](
            self.data_ptrs,
            self.data_strides,
            dummy_loc,
            dummy_loc,
            1,
            chunk_upper,
            BYTES_PER_TILE=self._kv_copy_config["bytes_per_tile"],
            num_warps=self._kv_copy_config["num_warps"],
            num_stages=2,
        )

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                if self.dtype == "int4":
                    assert (
                        self.head_dim % 2 == 0
                    ), f"head_dim: {self.head_dim}, kv cache dtype: int4"
                    self.k_buffer = [
                        torch.zeros(
                            (
                                self.size + self.page_size,
                                self.head_num,
                                self.head_dim // 2,
                            ),
                            dtype=self.store_dtype,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                    self.v_buffer = [
                        torch.zeros(
                            (
                                self.size + self.page_size,
                                self.head_num,
                                self.head_dim // 2,
                            ),
                            dtype=self.store_dtype,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                    # Scales and zeros: [cache_size, num_heads, 2] where dim=0 is scale, dim=1 is zero
                    self.k_scales_zeros = [
                        torch.zeros(
                            (self.size + self.page_size, self.head_num, 2),
                            dtype=torch.float32,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                    self.v_scales_zeros = [
                        torch.zeros(
                            (self.size + self.page_size, self.head_num, 2),
                            dtype=torch.float32,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                elif self.dtype == "int8":
                    self.k_buffer = [
                        torch.zeros(
                            (self.size + self.page_size, self.head_num, self.head_dim),
                            dtype=self.store_dtype,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                    self.v_buffer = [
                        torch.zeros(
                            (self.size + self.page_size, self.head_num, self.head_dim),
                            dtype=self.store_dtype,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                    # Scales and zeros: [cache_size, num_heads, 2]
                    self.k_scales_zeros = [
                        torch.zeros(
                            (self.size + self.page_size, self.head_num, 2),
                            dtype=torch.float32,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                    self.v_scales_zeros = [
                        torch.zeros(
                            (self.size + self.page_size, self.head_num, 2),
                            dtype=torch.float32,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                else:
                    # [size, head_num, head_dim] for each layer
                    # The padded slot 0 is used for writing dummy outputs from padded tokens.
                    self.k_buffer = [
                        torch.zeros(
                            (self.size + self.page_size, self.head_num, self.head_dim),
                            dtype=self.store_dtype,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                    self.v_buffer = [
                        torch.zeros(
                            (self.size + self.page_size, self.head_num, self.head_dim),
                            dtype=self.store_dtype,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]

        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
        self.data_strides = torch.tensor(
            [
                np.prod(x.shape[1:]) * x.dtype.itemsize
                for x in self.k_buffer + self.v_buffer
            ],
            device=self.device,
        )

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer

    def get_kv_size_bytes(self):
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        k_size_bytes = 0
        for k_cache in self.k_buffer:
            k_size_bytes += get_tensor_size_bytes(k_cache)
        v_size_bytes = 0
        for v_cache in self.v_buffer:
            v_size_bytes += get_tensor_size_bytes(v_cache)
        return k_size_bytes, v_size_bytes

    # for disagg
    def get_contiguous_buf_infos(self):
        # layer_num x [seq_len, head_num, head_dim]
        # layer_num x [page_num, page_size, head_num, head_dim]
        kv_data_ptrs = [
            self._get_key_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_data_lens = [
            self._get_key_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_item_lens = [
            self._get_key_buffer(i)[0].nbytes * self.page_size
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i)[0].nbytes * self.page_size
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_cpu_copy(self, indices):
        torch.cuda.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu = self.k_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                v_cpu = self.v_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                kv_cache_cpu[-1].append([k_cpu, v_cpu])
        torch.cuda.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices):
        torch.cuda.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu, v_cpu = (
                    kv_cache_cpu[layer_id][i // chunk_size][0],
                    kv_cache_cpu[layer_id][i // chunk_size][1],
                )
                assert k_cpu.shape[0] == v_cpu.shape[0] == len(chunk_indices)
                k_chunk = k_cpu.to(self.k_buffer[0].device, non_blocking=True)
                v_chunk = v_cpu.to(self.v_buffer[0].device, non_blocking=True)
                self.k_buffer[layer_id][chunk_indices] = k_chunk
                self.v_buffer[layer_id][chunk_indices] = v_chunk
        torch.cuda.synchronize()

    def _get_key_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.dtype in ("int4", "int8"):
            return self.k_buffer[layer_id - self.start_layer]
        elif self.store_dtype != self.dtype:
            return self.k_buffer[layer_id - self.start_layer].view(self.dtype)

        return self.k_buffer[layer_id - self.start_layer]

    def get_key_buffer(self, layer_id: int):
        # note: get_key_buffer is hooked with synchronization for layer-wise KV cache loading
        # it is supposed to be used only by attention backend not for information purpose
        # same applies to get_value_buffer and get_kv_buffer
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_key_buffer(layer_id)

    def _get_value_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.dtype in ("int4", "int8"):
            return self.v_buffer[layer_id - self.start_layer]
        elif self.store_dtype != self.dtype:
            return self.v_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.v_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def get_raw_key_buffer(self, layer_id: int):
        """Get raw quantized K buffer without dequantization (for INT4/INT8)."""
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.k_buffer[layer_id - self.start_layer]

    def get_raw_value_buffer(self, layer_id: int):
        """Get raw quantized V buffer without dequantization (for INT4/INT8)."""
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.v_buffer[layer_id - self.start_layer]

    def get_key_scales_zeros(self, layer_id: int):
        """Get scales and zeros for K (for INT4/INT8 quantization)."""
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.k_scales_zeros[layer_id - self.start_layer]

    def get_value_scales_zeros(self, layer_id: int):
        """Get scales and zeros for V (for INT4/INT8 quantization)."""
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.v_scales_zeros[layer_id - self.start_layer]

    def get_raw_kv_buffer(self, layer_id: int):
        """
        Get raw quantized KV buffer with scales/zeros for efficient dequantization.

        Returns a dict containing:
        - k_buffer: Raw quantized K buffer
        - v_buffer: Raw quantized V buffer
        - k_scales_zeros: Scales and zeros for K (if quantized)
        - v_scales_zeros: Scales and zeros for V (if quantized)
        - dtype: KV cache dtype string
        """
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        result = {
            "k_buffer": self.k_buffer[layer_id - self.start_layer],
            "v_buffer": self.v_buffer[layer_id - self.start_layer],
            "dtype": self.dtype,
        }

        if self.dtype in ("int4", "int8"):
            result["k_scales_zeros"] = self.k_scales_zeros[layer_id - self.start_layer]
            result["v_scales_zeros"] = self.v_scales_zeros[layer_id - self.start_layer]
        else:
            result["k_scales_zeros"] = None
            result["v_scales_zeros"] = None

        return result

    def _ensure_history_fake_quant_state(self, req_to_token_pool: ReqToTokenPool) -> None:
        req_pool_size = req_to_token_pool.size
        if (
            self._history_quantized_upto is not None
            and self._history_quantized_upto.shape[1] == req_pool_size
        ):
            return

        baseline = self.history_fake_quant_cfg.sink_length - 1
        self._history_quantized_upto = np.full(
            (self.layer_num, req_pool_size),
            baseline,
            dtype=np.int32,
        )
        self._history_seen_req_epoch = np.full(
            (self.layer_num, req_pool_size),
            -1,
            dtype=np.int64,
        )

    def _get_kitty_rotation(
        self, layer_id: int, key: str, device: torch.device, dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        ld = self._kitty_rotations.get(layer_id)
        if ld is None:
            return None
        R = ld.get(key)
        if R is None:
            return None
        return R.to(device=device, dtype=dtype)

    def _fake_quant_rows_inplace(
        self,
        cache: torch.Tensor,
        loc: torch.Tensor,
        R: Optional[torch.Tensor],
        clip_ratio: float = 0.0,
        quant_method: str = "asym",
        head_group_size: int = 0,
        _mse_layer_id: Optional[int] = None,
        _mse_kv: Optional[str] = None,
    ) -> None:
        """Rotate → clip → fake-quant → inv-rotate a batch of KV rows in-place."""
        cfg = self.history_fake_quant_cfg
        bits = cfg.history_quant_bits
        # Pre-quant snapshot for MSE measurement (only when SGLANG_KITTY_DUMP_MSE=1).
        pre_rows = (cache[loc].clone()
                    if _KITTY_MSE_DUMP_ENABLED and _mse_layer_id is not None else None)
        rows = cache[loc]
        if R is not None:
            rows = (rows.float() @ R).to(rows.dtype)
        if clip_ratio > 0:
            rows = _clip_quantile(rows, clip_ratio)
        if quant_method == "nf2":
            if head_group_size > 0 and head_group_size < rows.shape[-1]:
                # NF2 per group: reshape, apply, reshape back
                sh = rows.shape
                D = sh[-1]
                rows = rows.reshape(*sh[:-1], D // head_group_size, head_group_size)
                rows = _simulate_nf2_quantize_dequantize(rows)
                rows = rows.reshape(sh)
            else:
                rows = _simulate_nf2_quantize_dequantize(rows)
        elif head_group_size > 0 and head_group_size < rows.shape[-1]:
            rows = _simulate_group_quantize_dequantize(rows, head_group_size, bits)
        else:
            rows = _simulate_bits_quantize_dequantize(rows, bits)
        if R is not None:
            rows = (rows.float() @ R.T).to(rows.dtype)
        cache[loc] = rows
        if pre_rows is not None:
            with torch.no_grad():
                num = (pre_rows.float() - rows.float()).square().sum().item()
                den = pre_rows.float().square().sum().item()
                cnt = int(pre_rows.numel())
            slot = _KITTY_MSE_STATE["K" if _mse_kv == "k" else "V"][int(_mse_layer_id)]
            slot[0] += num
            slot[1] += den
            slot[2] += cnt
            _kitty_mse_periodic_dump()

    def _maybe_apply_history_fake_quant(
        self,
        *,
        layer: Optional[RadixAttention],
        storage_layer_id: int,
        rotation_layer_id: int,
        forward_batch: Optional[Any],
    ) -> None:
        if (
            not self.history_fake_quant_enabled
            or layer is None
            or forward_batch is None
            or getattr(layer, "is_cross_attention", False)
        ):
            return

        req_to_token_pool = getattr(forward_batch, "req_to_token_pool", None)
        req_pool_indices = getattr(forward_batch, "req_pool_indices", None)
        seq_lens = getattr(forward_batch, "seq_lens", None)
        if req_to_token_pool is None or req_pool_indices is None or seq_lens is None:
            return

        self._ensure_history_fake_quant_state(req_to_token_pool)
        cfg = self.history_fake_quant_cfg
        layer_idx = storage_layer_id - self.start_layer

        # --- CPU-only watermark check: collect locs that need quantization ---
        # Use cached CPU copies when available from the same forward_batch.
        _cache = getattr(forward_batch, "_hfq_cpu_cache", None)
        if _cache is None:
            _cache = {
                "req_pool": req_pool_indices.cpu().numpy(),
                "seq_lens": seq_lens.cpu().numpy(),
            }
            try:
                forward_batch._hfq_cpu_cache = _cache
            except AttributeError:
                pass

        req_pool_cpu = _cache["req_pool"]
        seq_lens_cpu = _cache["seq_lens"]

        all_locs = []
        for batch_idx in range(len(req_pool_cpu)):
            req_pool_idx = int(req_pool_cpu[batch_idx])
            req_epoch = int(req_to_token_pool.req_epoch[req_pool_idx])
            if self._history_seen_req_epoch[layer_idx, req_pool_idx] != req_epoch:
                self._history_quantized_upto[layer_idx, req_pool_idx] = cfg.sink_length - 1
                self._history_seen_req_epoch[layer_idx, req_pool_idx] = req_epoch

            seq_len = int(seq_lens_cpu[batch_idx])
            history_end = seq_len - cfg.recent_window_length - 1
            start_pos = max(
                cfg.sink_length,
                int(self._history_quantized_upto[layer_idx, req_pool_idx]) + 1,
            )
            if history_end < start_pos:
                continue

            all_locs.append((req_pool_idx, start_pos, history_end))

        if not all_locs:
            return

        # --- GPU path: batch all locs into one tensor, one R/W pass ---
        loc_parts = []
        for req_pool_idx, start_pos, history_end in all_locs:
            part = req_to_token_pool.req_to_token[
                req_pool_idx, start_pos : history_end + 1
            ]
            loc_parts.append(part)

        loc = torch.cat(loc_parts).to(device=self.device, dtype=torch.long)
        loc = loc[loc > 0]

        if loc.numel() == 0:
            for req_pool_idx, _, history_end in all_locs:
                self._history_quantized_upto[layer_idx, req_pool_idx] = history_end
            return

        Rk = self._get_kitty_rotation(rotation_layer_id, "k", self.device, torch.float32)
        Rv = self._get_kitty_rotation(rotation_layer_id, "v", self.device, torch.float32)

        self._fake_quant_rows_inplace(
            self.k_buffer[layer_idx], loc, Rk,
            clip_ratio=cfg.get_k_clip_ratio(),
            quant_method=cfg.k_quant_method,
            head_group_size=cfg.k_head_group_size,
            _mse_layer_id=storage_layer_id, _mse_kv="k",
        )
        self._fake_quant_rows_inplace(
            self.v_buffer[layer_idx], loc, Rv,
            clip_ratio=cfg.get_v_clip_ratio(),
            quant_method=cfg.v_quant_method,
            head_group_size=cfg.v_head_group_size,
            _mse_layer_id=storage_layer_id, _mse_kv="v",
        )

        n = int(loc.numel())
        self._history_debug_counters["k_rows"][layer_idx] += n
        self._history_debug_counters["v_rows"][layer_idx] += n
        self._history_debug_counters["calls"] += 1

        for req_pool_idx, _, history_end in all_locs:
            self._history_quantized_upto[layer_idx, req_pool_idx] = history_end

    def get_history_fake_quant_debug_state(self) -> dict[str, Any]:
        if self._history_quantized_upto is None:
            watermarks = None
        else:
            watermarks = self._history_quantized_upto.copy()

        if self._history_seen_req_epoch is None:
            req_epochs = None
        else:
            req_epochs = self._history_seen_req_epoch.copy()

        return {
            "enabled": self.history_fake_quant_enabled,
            "config": dataclasses.asdict(self.history_fake_quant_cfg),
            "watermarks": watermarks,
            "req_epochs": req_epochs,
            "k_rows": self._history_debug_counters["k_rows"].copy(),
            "v_rows": self._history_debug_counters["v_rows"].copy(),
            "calls": int(self._history_debug_counters["calls"]),
        }

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
        forward_batch: Optional[Any] = None,
    ):
        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        rotation_layer_id = layer.layer_id if layer is not None else layer_id

        if self.dtype in ("int4", "int8"):
            # Use Triton kernels for efficient quantization and direct cache write
            if self.dtype == "int4":
                hadamard_order = _hadamard_order
                if _hadamard_enabled:
                    assert (
                        cache_k.shape[-1] % hadamard_order == 0
                    ), f"head_dim must be divisible by {hadamard_order}"
                    orig_shape = cache_k.shape
                    cache_k = cache_k.view(
                        *orig_shape[:-1],
                        orig_shape[-1] // hadamard_order,
                        hadamard_order,
                    )
                    cache_k = hadamard_transform(cache_k / math.sqrt(hadamard_order))
                    cache_k = cache_k.view(orig_shape)
                if layer._use_post_hadamard_qk_rotation:
                    cache_k = maybe_apply_k_rotation(
                        cache_k,
                        layer_id=layer_id,
                        num_q_heads=layer.tp_q_head_num,
                        num_kv_heads=layer.tp_k_head_num,
                        head_dim=layer.qk_head_dim,
                    )
                if _hadamard_v_enabled:
                    orig_shape = cache_v.shape
                    cache_v = cache_v.view(
                        *orig_shape[:-1],
                        orig_shape[-1] // hadamard_order,
                        hadamard_order,
                    )
                    cache_v = hadamard_transform(
                        cache_v / math.sqrt(hadamard_order)
                    )
                    cache_v = cache_v.view(orig_shape)
                cache_v = maybe_apply_v_rotation(
                    cache_v,
                    layer_id=layer_id,
                    num_kv_heads=layer.tp_k_head_num,
                    head_dim=layer.v_head_dim,
                )
                if _kitty_boost_ratio > 0:
                    cache_k = _kitty_fake_quant(cache_k, _kitty_boost_ratio, _kv_clip_ratio)
                    cache_v = _kitty_fake_quant(cache_v, 0, _kv_clip_ratio)
                elif _kv_clip_ratio > 0:
                    cache_k = _clip_quantile(cache_k, _kv_clip_ratio)
                    cache_v = _clip_quantile(cache_v, _kv_clip_ratio)
                quantized_set_kv_int4_triton(
                    cache_k,
                    cache_v,
                    loc,
                    self.k_buffer[layer_id - self.start_layer],
                    self.v_buffer[layer_id - self.start_layer],
                    self.k_scales_zeros[layer_id - self.start_layer],
                    self.v_scales_zeros[layer_id - self.start_layer],
                    max_quant_val=_kv_max_quant_val,
                )

            elif self.dtype == "int8":

                # Quantize and write directly to cache buffers using Triton kernel
                quantized_set_kv_int8_triton(
                    cache_k,
                    cache_v,
                    loc,
                    self.k_buffer[layer_id - self.start_layer],
                    self.v_buffer[layer_id - self.start_layer],
                    self.k_scales_zeros[layer_id - self.start_layer],
                    self.v_scales_zeros[layer_id - self.start_layer],
                )

            # Early return - INT4/INT8 quantization is complete
            return

        if _quant_sim_layers and layer_id in _quant_sim_layers:
            # Simulate int4+H+QR quantization noise on this layer's K cache.
            # IMPORTANT: SGLANG_Q_ROTATION_PATH must NOT be set so that
            # radix_attention does not apply QR to Q/K. This simulation
            # handles everything independently via SGLANG_QUANT_SIM_ROTATION_PATH.
            #
            # Math: K_stored = H^{-1} R^{-1} dequant(int4(R H K))
            # During attention: score = Q^T @ K_stored = Q^T H R^T dequant(int4(R H K))
            # This equals the real int4 pipeline: (R H Q)^T dequant(int4(R H K))
            cache_k = cache_k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
            hadamard_order = _hadamard_order
            orig_shape = cache_k.shape
            # Forward: Hadamard
            cache_k = cache_k.view(*orig_shape[:-1], orig_shape[-1] // hadamard_order, hadamard_order)
            cache_k = hadamard_transform(cache_k / math.sqrt(hadamard_order))
            cache_k = cache_k.view(orig_shape)
            # Forward: QR rotation (using self-contained rotation, not _Q_ROTATION_MANAGER)
            has_rotation = layer_id in _quant_sim_rotations
            if has_rotation:
                cache_k = _sim_apply_rotation(cache_k, layer_id)
            # Simulate int4 quantize → dequantize
            cache_k = _simulate_int4_quantize_dequantize(cache_k)
            # Inverse: QR rotation
            if has_rotation:
                cache_k = _sim_apply_inverse_rotation(cache_k, layer_id)
            # Inverse: Hadamard (H is its own inverse when normalized)
            cache_k = cache_k.view(*orig_shape[:-1], orig_shape[-1] // hadamard_order, hadamard_order)
            cache_k = hadamard_transform(cache_k / math.sqrt(hadamard_order))
            cache_k = cache_k.view(-1, layer.tp_k_head_num * layer.qk_head_dim)

        if cache_k.dtype != self.dtype:  # fp8, fp4 kv cache
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        _set_kv_buffer_impl(
            cache_k,
            cache_v,
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
            loc,
            row_dim=self.row_dim,
            store_dtype=self.store_dtype,
            device_module=self.device_module,
            alt_stream=self.alt_stream,
            same_kv_dim=self.same_kv_dim,
        )
        self._maybe_apply_history_fake_quant(
            layer=layer,
            storage_layer_id=layer_id,
            rotation_layer_id=rotation_layer_id,
            forward_batch=forward_batch,
        )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        if envs.SGLANG_NATIVE_MOVE_KV_CACHE.get():
            move_kv_cache_native(self.k_buffer, self.v_buffer, tgt_loc, src_loc)
            return

        N = tgt_loc.numel()
        if N == 0:
            return

        assert (
            self._kv_copy_config is not None
        ), "KV copy not initialized. Set enable_kv_cache_copy=True in __init__"

        cfg = self._kv_copy_config
        cap = int(cfg.get("num_locs_upper", 256))
        grid = (self.data_ptrs.numel(), cfg["byte_tiles"])

        if N <= cap:
            upper = next_power_of_2(N)
            copy_all_layer_kv_cache_tiled[grid](
                self.data_ptrs,
                self.data_strides,
                tgt_loc,
                src_loc,
                N,
                upper,
                BYTES_PER_TILE=cfg["bytes_per_tile"],
                num_warps=cfg["num_warps"],
                num_stages=2,
            )
            return

        # Huge N: chunk, but each chunk's upper is still pow2(<= cap)
        for start in range(0, N, cap):
            end = min(start + cap, N)
            chunk_len = end - start
            upper = next_power_of_2(chunk_len)
            copy_all_layer_kv_cache_tiled[grid](
                self.data_ptrs,
                self.data_strides,
                tgt_loc[start:end],
                src_loc[start:end],
                chunk_len,
                upper,
                BYTES_PER_TILE=cfg["bytes_per_tile"],
                num_warps=cfg["num_warps"],
                num_stages=2,
            )


class MHATokenToKVPoolFP4(MHATokenToKVPool):

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                # [size, head_num, head_dim] for each layer
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                m = self.size + self.page_size
                n = self.head_num
                k = self.head_dim

                scale_block_size = 16
                self.store_dtype = torch.uint8
                self.k_buffer = [
                    torch.zeros(
                        (m, n, k // 2),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_buffer = [
                    torch.zeros(
                        (m, n, k // 2),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

                self.k_scale_buffer = [
                    torch.zeros(
                        (m, (n * k) // scale_block_size),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_scale_buffer = [
                    torch.zeros(
                        (m, (n * k) // scale_block_size),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer
        del self.k_scale_buffer
        del self.v_scale_buffer

    def _get_key_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            cache_k_nope_fp4 = self.k_buffer[layer_id - self.start_layer].view(
                torch.uint8
            )
            cache_k_nope_fp4_sf = self.k_scale_buffer[layer_id - self.start_layer]

            from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil

            cache_k_nope_fp4_dequant = KVFP4QuantizeUtil.batched_dequantize(
                cache_k_nope_fp4, cache_k_nope_fp4_sf
            )
            return cache_k_nope_fp4_dequant
        return self.k_buffer[layer_id - self.start_layer]

    def _get_value_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            cache_v_nope_fp4 = self.v_buffer[layer_id - self.start_layer].view(
                torch.uint8
            )
            cache_v_nope_fp4_sf = self.v_scale_buffer[layer_id - self.start_layer]

            from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil

            cache_v_nope_fp4_dequant = KVFP4QuantizeUtil.batched_dequantize(
                cache_v_nope_fp4, cache_v_nope_fp4_sf
            )
            return cache_v_nope_fp4_dequant
        return self.v_buffer[layer_id - self.start_layer]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
        forward_batch: Optional[Any] = None,
    ):
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)

            from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil

            cache_k, cache_k_fp4_sf = KVFP4QuantizeUtil.batched_quantize(cache_k)
            cache_v, cache_v_fp4_sf = KVFP4QuantizeUtil.batched_quantize(cache_v)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

            cache_k_fp4_sf = cache_k_fp4_sf.view(self.store_dtype)
            cache_v_fp4_sf = cache_v_fp4_sf.view(self.store_dtype)

        if get_is_capture_mode() and self.alt_stream is not None:
            # Overlap the copy of K and V cache for small batch size
            current_stream = self.device_module.current_stream()
            self.alt_stream.wait_stream(current_stream)
            self.k_buffer[layer_id - self.start_layer][loc] = cache_k

            self.k_scale_buffer[layer_id - self.start_layer][loc] = cache_k_fp4_sf
            with self.device_module.stream(self.alt_stream):
                self.v_buffer[layer_id - self.start_layer][loc] = cache_v

                self.v_scale_buffer[layer_id - self.start_layer][loc] = cache_v_fp4_sf
            current_stream.wait_stream(self.alt_stream)
        else:
            self.k_buffer[layer_id - self.start_layer][loc] = cache_k
            self.v_buffer[layer_id - self.start_layer][loc] = cache_v

            self.k_scale_buffer[layer_id - self.start_layer][loc] = cache_k_fp4_sf
            self.v_scale_buffer[layer_id - self.start_layer][loc] = cache_v_fp4_sf


class HybridLinearKVPool(KVCache):
    """KV cache with separate pools for full and linear attention layers."""

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        page_size: int,
        head_num: int,
        head_dim: int,
        full_attention_layer_ids: List[int],
        enable_kvcache_transpose: bool,
        device: str,
        mamba_pool: MambaPool,
        enable_memory_saver: bool = False,
        # TODO: refactor mla related args
        use_mla: bool = False,
        kv_lora_rank: int = None,
        qk_rope_head_dim: int = None,
    ):
        self.size = size
        self.dtype = dtype
        self.device = device
        self.full_layer_nums = len(full_attention_layer_ids)
        self.page_size = page_size
        # TODO support pp?
        self.start_layer = 0
        self.head_num = head_num
        self.head_dim = head_dim
        self.mamba_pool = mamba_pool
        # TODO MHATransposedTokenToKVPool if enable_kvcache_transpose is True
        assert not enable_kvcache_transpose
        self.use_mla = use_mla
        if not use_mla:

            TokenToKVPoolClass = MHATokenToKVPool

            if _is_npu:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMHATokenToKVPool,
                )

                TokenToKVPoolClass = NPUMHATokenToKVPool

            self.full_kv_pool = TokenToKVPoolClass(
                size=size,
                page_size=self.page_size,
                dtype=dtype,
                head_num=head_num,
                head_dim=head_dim,
                layer_num=self.full_layer_nums,
                device=device,
                enable_memory_saver=enable_memory_saver,
            )
        else:

            TokenToKVPoolClass = MLATokenToKVPool

            if _is_npu:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMLATokenToKVPool,
                )

                TokenToKVPoolClass = NPUMLATokenToKVPool

            self.full_kv_pool = TokenToKVPoolClass(
                size=size,
                page_size=self.page_size,
                dtype=dtype,
                layer_num=self.full_layer_nums,
                device=device,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                enable_memory_saver=enable_memory_saver,
            )
        self.full_attention_layer_id_mapping = {
            id: i for i, id in enumerate(full_attention_layer_ids)
        }
        if use_mla:
            self.mem_usage = self.get_kv_size_bytes() / GB
        else:
            k_size, v_size = self.get_kv_size_bytes()
            self.mem_usage = (k_size + v_size) / GB

    def get_kv_size_bytes(self):
        return self.full_kv_pool.get_kv_size_bytes()

    def get_contiguous_buf_infos(self):
        return self.full_kv_pool.get_contiguous_buf_infos()

    def get_state_buf_infos(self):
        mamba_data_ptrs, mamba_data_lens, mamba_item_lens = (
            self.mamba_pool.get_contiguous_buf_infos()
        )
        return mamba_data_ptrs, mamba_data_lens, mamba_item_lens

    def get_state_dim_per_tensor(self):
        """Get the sliceable dimension size for each mamba state tensor."""
        return self.mamba_pool.get_state_dim_per_tensor()

    def maybe_get_custom_mem_pool(self):
        return self.full_kv_pool.maybe_get_custom_mem_pool()

    def _transfer_full_attention_id(self, layer_id: int):
        if layer_id not in self.full_attention_layer_id_mapping:
            raise ValueError(
                f"{layer_id=} not in full attention layers: {self.full_attention_layer_id_mapping.keys()}"
            )
        return self.full_attention_layer_id_mapping[layer_id]

    def get_key_buffer(self, layer_id: int):
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_key_buffer(layer_id)

    def get_value_buffer(self, layer_id: int):
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_kv_buffer(layer_id)

    @contextmanager
    def _transfer_id_context(self, layer: RadixAttention):

        @contextmanager
        def _patch_layer_id(layer):
            original_layer_id = layer.layer_id
            layer.layer_id = self._transfer_full_attention_id(layer.layer_id)
            try:
                yield
            finally:
                layer.layer_id = original_layer_id

        with _patch_layer_id(layer):
            yield

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        forward_batch: Optional[Any] = None,
    ):
        layer_id = self._transfer_full_attention_id(layer.layer_id)
        if not self.use_mla:
            self.full_kv_pool.set_kv_buffer(
                layer,
                loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=layer_id,
                forward_batch=forward_batch,
            )
        else:
            with self._transfer_id_context(layer):
                self.full_kv_pool.set_kv_buffer(
                    layer,
                    loc,
                    cache_k,
                    cache_v,
                    forward_batch=forward_batch,
                )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        self.full_kv_pool.move_kv_cache(tgt_loc, src_loc)

    def get_v_head_dim(self):
        return self.full_kv_pool.get_value_buffer(0).shape[-1]

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        assert self.use_mla, "set_mla_kv_buffer called when use_mla is False"
        with self._transfer_id_context(layer):
            self.full_kv_pool.set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope)

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        assert self.use_mla, "get_mla_kv_buffer called when use_mla is False"
        with self._transfer_id_context(layer):
            return self.full_kv_pool.get_mla_kv_buffer(layer, loc, dst_dtype)


class MLATokenToKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        use_nsa: bool = False,
        override_kv_cache_dim: Optional[int] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )

        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.use_nsa = use_nsa
        self.nsa_kv_cache_store_fp8 = use_nsa and dtype == torch.float8_e4m3fn
        assert not (
            self.nsa_kv_cache_store_fp8 and override_kv_cache_dim is None
        ), "override_kv_cache_dim must be provided when using NSA with FP8 kv cache storage"
        self.kv_cache_dim = (
            override_kv_cache_dim
            if self.use_nsa and self.nsa_kv_cache_store_fp8
            else (kv_lora_rank + qk_rope_head_dim)
        )

        self._create_buffers()

        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.kv_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        if not use_nsa:
            # NSA will allocate indexer KV cache later and then log the total size
            self._finalize_allocation_log(size)

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                self.kv_buffer = [
                    torch.zeros(
                        (self.size + self.page_size, 1, self.kv_cache_dim),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def _clear_buffers(self):
        del self.kv_buffer

    def get_kv_size_bytes(self):
        assert hasattr(self, "kv_buffer")
        kv_size_bytes = 0
        for kv_cache in self.kv_buffer:
            kv_size_bytes += get_tensor_size_bytes(kv_cache)
        return kv_size_bytes

    # for disagg
    def get_contiguous_buf_infos(self):
        # MLA has only one kv_buffer, so only the information of this buffer needs to be returned.
        kv_data_ptrs = [self.kv_buffer[i].data_ptr() for i in range(self.layer_num)]
        kv_data_lens = [self.kv_buffer[i].nbytes for i in range(self.layer_num)]
        kv_item_lens = [
            self.kv_buffer[i][0].nbytes * self.page_size for i in range(self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id - self.start_layer].view(self.dtype)

        return self.kv_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id - self.start_layer][
                ..., : self.kv_lora_rank
            ].view(self.dtype)
        return self.kv_buffer[layer_id - self.start_layer][..., : self.kv_lora_rank]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        forward_batch: Optional[Any] = None,
    ):
        layer_id = layer.layer_id
        assert not (self.use_nsa and self.nsa_kv_cache_store_fp8)
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)

        if self.store_dtype != self.dtype:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k.view(
                self.store_dtype
            )
        else:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        layer_id = layer.layer_id

        if self.use_nsa and self.nsa_kv_cache_store_fp8:
            # OPTIMIZATION: Quantize k_nope and k_rope separately to avoid concat overhead
            # This also enables reuse of set_mla_kv_buffer_triton two-tensor write path
            # quantize_k_cache_separate returns (nope_part, rope_part) as uint8 bytes
            cache_k_nope_fp8, cache_k_rope_fp8 = quantize_k_cache_separate(
                cache_k_nope, cache_k_rope
            )

            # Reuse existing two-tensor write kernel (works with FP8 byte layout)
            # cache_k_nope_fp8: (num_tokens, 1, 528) uint8 [nope_fp8(512) | scales(16)]
            # cache_k_rope_fp8: (num_tokens, 1, 128) uint8 [rope_bf16_bytes(128)]
            set_mla_kv_buffer_triton(
                self.kv_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope_fp8,
                cache_k_rope_fp8,
            )
        else:
            if cache_k_nope.dtype != self.dtype:
                cache_k_nope = cache_k_nope.to(self.dtype)
                cache_k_rope = cache_k_rope.to(self.dtype)
            if self.store_dtype != self.dtype:
                cache_k_nope = cache_k_nope.view(self.store_dtype)
                cache_k_rope = cache_k_rope.view(self.store_dtype)

            set_mla_kv_buffer_triton(
                self.kv_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope,
                cache_k_rope,
            )

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        # get k nope and k rope from the kv buffer, and optionally cast them to dst_dtype.
        layer_id = layer.layer_id
        kv_buffer = self.get_key_buffer(layer_id)
        dst_dtype = dst_dtype or self.dtype
        cache_k_nope = torch.empty(
            (loc.shape[0], 1, self.kv_lora_rank),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )
        cache_k_rope = torch.empty(
            (loc.shape[0], 1, self.qk_rope_head_dim),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )
        get_mla_kv_buffer_triton(kv_buffer, loc, cache_k_nope, cache_k_rope)
        return cache_k_nope, cache_k_rope

    def get_cpu_copy(self, indices):
        torch.cuda.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                kv_cpu = self.kv_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                kv_cache_cpu[-1].append(kv_cpu)
        torch.cuda.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices):
        torch.cuda.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                kv_cpu = kv_cache_cpu[layer_id][i // chunk_size]
                assert kv_cpu.shape[0] == len(chunk_indices)
                kv_chunk = kv_cpu.to(self.kv_buffer[0].device, non_blocking=True)
                self.kv_buffer[layer_id][chunk_indices] = kv_chunk
        torch.cuda.synchronize()


class MLATokenToKVPoolFP4(MLATokenToKVPool):

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                m = self.size + self.page_size
                n = 1  # head_num
                k = self.kv_cache_dim  # head_dim

                scale_block_size = 16
                self.store_dtype = torch.uint8

                self.kv_buffer = [
                    torch.zeros(
                        (m, n, k // 2),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

                self.kv_scale_buffer = [
                    torch.zeros(
                        (m, k // scale_block_size),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def _clear_buffers(self):
        del self.kv_buffer
        del self.kv_scale_buffer

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            cache_k_nope_fp4 = self.kv_buffer[layer_id - self.start_layer].view(
                torch.uint8
            )
            cache_k_nope_fp4_sf = self.kv_scale_buffer[layer_id - self.start_layer]

            from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil

            cache_k_nope_fp4_dequant = KVFP4QuantizeUtil.batched_dequantize(
                cache_k_nope_fp4, cache_k_nope_fp4_sf
            )
            return cache_k_nope_fp4_dequant

        return self.kv_buffer[layer_id - self.start_layer]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        forward_batch: Optional[Any] = None,
    ):
        layer_id = layer.layer_id
        assert not (self.use_nsa and self.nsa_kv_cache_store_fp8)
        if cache_k.dtype != self.dtype:
            from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil

            cache_k_fp4, cache_k_fp4_sf = KVFP4QuantizeUtil.batched_quantize(cache_k)

        if self.store_dtype != self.dtype:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k_fp4.view(
                self.store_dtype
            )
            self.kv_scale_buffer[layer_id - self.start_layer][loc] = (
                cache_k_fp4_sf.view(self.store_dtype)
            )
        else:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        layer_id = layer.layer_id

        if self.use_nsa and self.nsa_kv_cache_store_fp8:
            # original cache_k: (num_tokens, num_heads 1, hidden 576); we unsqueeze the page_size=1 dim here
            # TODO no need to cat
            cache_k = torch.cat([cache_k_nope, cache_k_rope], dim=-1)
            cache_k = quantize_k_cache(cache_k.unsqueeze(1)).squeeze(1)
            cache_k = cache_k.view(self.store_dtype)
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k
        else:
            if cache_k_nope.dtype != self.dtype:
                from sglang.srt.layers.quantization.kvfp4_tensor import (
                    KVFP4QuantizeUtil,
                )

                cache_k_nope_fp4, cache_k_nope_fp4_sf = (
                    KVFP4QuantizeUtil.batched_quantize(cache_k_nope)
                )
                cache_k_rope_fp4, cache_k_rope_fp4_sf = (
                    KVFP4QuantizeUtil.batched_quantize(cache_k_rope)
                )

            if self.store_dtype != self.dtype:
                cache_k_nope = cache_k_nope.view(self.store_dtype)
                cache_k_rope = cache_k_rope.view(self.store_dtype)

            set_mla_kv_buffer_triton(
                self.kv_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope_fp4,
                cache_k_rope_fp4,
            )
            set_mla_kv_scale_buffer_triton(
                self.kv_scale_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope_fp4_sf,
                cache_k_rope_fp4_sf,
            )


class NSATokenToKVPool(MLATokenToKVPool):
    quant_block_size = 128
    index_k_with_scale_buffer_dtype = torch.uint8
    rope_storage_dtype = torch.bfloat16  # rope is always stored in bf16

    def __init__(
        self,
        size: int,
        page_size: int,
        kv_lora_rank: int,
        dtype: torch.dtype,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        index_head_dim: int,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        assert (
            kv_lora_rank % self.quant_block_size == 0
        ), f"kv_lora_rank {kv_lora_rank} must be multiple of quant_block_size {self.quant_block_size}"

        # Calculate override_kv_cache_dim for FP8 storage:
        # kv_lora_rank + scale storage (kv_lora_rank // quant_block_size * 4 bytes) + rope dimension storage
        # Note: rope dimension is stored in original dtype (bf16), not quantized to fp8
        override_dim = (
            kv_lora_rank
            + kv_lora_rank // self.quant_block_size * 4
            + qk_rope_head_dim * self.rope_storage_dtype.itemsize
        )

        super().__init__(
            size,
            page_size,
            dtype,
            kv_lora_rank,
            qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
            use_nsa=True,
            override_kv_cache_dim=override_dim,
        )
        # self.index_k_dtype = torch.float8_e4m3fn
        # self.index_k_scale_dtype = torch.float32
        self.index_head_dim = index_head_dim
        # num head == 1 and head dim == 128 for index_k in NSA
        assert index_head_dim == 128

        if _is_hip:
            assert self.page_size == 1
        else:
            assert self.page_size == 64
        with (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.custom_mem_pool
            else nullcontext()
        ):
            self.index_k_with_scale_buffer = [
                torch.zeros(
                    # Layout:
                    #     ref: test_attention.py :: kv_cache_cast_to_fp8
                    #     shape: (num_pages, page_size 64 * head_dim 128 + page_size 64 * fp32_nbytes 4)
                    #     data: for page i,
                    #         * buf[i, :page_size * head_dim] for fp8 data
                    #         * buf[i, page_size * head_dim:].view(float32) for scale
                    (
                        (size + page_size + 1) // self.page_size,
                        self.page_size
                        * (
                            index_head_dim + index_head_dim // self.quant_block_size * 4
                        ),
                    ),
                    dtype=self.index_k_with_scale_buffer_dtype,
                    device=device,
                )
                for _ in range(layer_num)
            ]
        self._finalize_allocation_log(size)

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.index_k_with_scale_buffer[layer_id - self.start_layer]

    def get_index_k_continuous(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        return index_buf_accessor.GetK.execute(
            self, buf, seq_len=seq_len, page_indices=page_indices
        )

    def get_index_k_scale_continuous(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        return index_buf_accessor.GetS.execute(
            self, buf, seq_len=seq_len, page_indices=page_indices
        )

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len_tensor: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
    ):
        """
        Fused method to get both index K and scale data in a single call using Triton.
        More efficient than calling get_index_k_continuous and get_index_k_scale_continuous separately.

        :param layer_id: Layer index
        :param seq_len: Sequence length
        :param page_indices: Page indices tensor
        :return: tuple of (k_fp8, k_scale) where
                 k_fp8: (seq_len, index_head_dim), uint8
                 k_scale: (seq_len, 4), uint8
        """
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        return index_buf_accessor.GetKAndS.execute(
            self,
            buf,
            page_indices=page_indices,
            seq_len_tensor=seq_len_tensor,
            seq_len_sum=seq_len_sum,
            max_seq_len=max_seq_len,
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        index_buf_accessor.SetKAndS.execute(
            pool=self, buf=buf, loc=loc, index_k=index_k, index_k_scale=index_k_scale
        )

    def get_state_buf_infos(self):
        data_ptrs = [
            self.index_k_with_scale_buffer[i].data_ptr() for i in range(self.layer_num)
        ]
        data_lens = [
            self.index_k_with_scale_buffer[i].nbytes for i in range(self.layer_num)
        ]
        item_lens = [
            self.index_k_with_scale_buffer[i][0].nbytes for i in range(self.layer_num)
        ]
        return data_ptrs, data_lens, item_lens

    def get_kv_size_bytes(self):
        kv_size_bytes = super().get_kv_size_bytes()
        for index_k_cache in self.index_k_with_scale_buffer:
            kv_size_bytes += get_tensor_size_bytes(index_k_cache)
        return kv_size_bytes


class DoubleSparseTokenToKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        heavy_channel_num: int,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                # [size, head_num, head_dim] for each layer
                self.k_buffer = [
                    torch.zeros(
                        (size + page_size, head_num, head_dim),
                        dtype=dtype,
                        device=device,
                    )
                    for _ in range(layer_num)
                ]
                self.v_buffer = [
                    torch.zeros(
                        (size + page_size, head_num, head_dim),
                        dtype=dtype,
                        device=device,
                    )
                    for _ in range(layer_num)
                ]

                # [size, head_num, heavy_channel_num] for each layer
                self.label_buffer = [
                    torch.zeros(
                        (size + 1, head_num, heavy_channel_num),
                        dtype=dtype,
                        device=device,
                    )
                    for _ in range(layer_num)
                ]

    def get_key_buffer(self, layer_id: int):
        return self.k_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        return self.v_buffer[layer_id - self.start_layer]

    def get_label_buffer(self, layer_id: int):
        return self.label_buffer[layer_id - self.start_layer]

    def get_kv_buffer(self, layer_id: int):
        return (
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
        )

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        cache_label: torch.Tensor,
        forward_batch: Optional[Any] = None,
    ):
        # NOTE(Andy): ignore the dtype check
        layer_id = layer.layer_id
        self.k_buffer[layer_id - self.start_layer][loc] = cache_k
        self.v_buffer[layer_id - self.start_layer][loc] = cache_v
        self.label_buffer[layer_id - self.start_layer][loc] = cache_label


def move_kv_cache_native(
    k_buffer: List[torch.Tensor],
    v_buffer: List[torch.Tensor],
    tgt_loc: torch.Tensor,
    src_loc: torch.Tensor,
):
    if tgt_loc.numel() == 0:
        return

    tgt_loc_flat = tgt_loc.view(-1).long()
    src_loc_flat = src_loc.view(-1).long()
    for k_cache, v_cache in zip(k_buffer, v_buffer):
        k_cache[tgt_loc_flat] = k_cache[src_loc_flat]
        v_cache[tgt_loc_flat] = v_cache[src_loc_flat]


@triton.jit
def copy_all_layer_kv_cache_tiled(
    data_ptrs,
    strides,
    tgt_loc_ptr,
    src_loc_ptr,
    num_locs,
    num_locs_upper: tl.constexpr,
    BYTES_PER_TILE: tl.constexpr,
):
    """2D tiled kernel. Safe for in-place copy."""
    bid = tl.program_id(0)
    tid = tl.program_id(1)

    stride = tl.load(strides + bid)
    base_ptr = tl.load(data_ptrs + bid)
    base_ptr = tl.cast(base_ptr, tl.pointer_type(tl.uint8))

    byte_off = tid * BYTES_PER_TILE + tl.arange(0, BYTES_PER_TILE)
    mask_byte = byte_off < stride
    tl.multiple_of(byte_off, 16)

    loc_idx = tl.arange(0, num_locs_upper)
    mask_loc = loc_idx < num_locs

    src = tl.load(src_loc_ptr + loc_idx, mask=mask_loc, other=0)
    tgt = tl.load(tgt_loc_ptr + loc_idx, mask=mask_loc, other=0)

    src_ptr = base_ptr + src[:, None] * stride + byte_off[None, :]
    tgt_ptr = base_ptr + tgt[:, None] * stride + byte_off[None, :]

    mask = mask_loc[:, None] & mask_byte[None, :]
    vals = tl.load(src_ptr, mask=mask)
    tl.store(tgt_ptr, vals, mask=mask)
