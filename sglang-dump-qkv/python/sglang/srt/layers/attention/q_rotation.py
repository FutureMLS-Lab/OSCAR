from __future__ import annotations

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)
_hadamard_enabled = os.environ.get("HADAMARD", "0") in ("1", "true", "True")


def _parse_compute_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in ("float64", "fp64", "double"):
        return torch.float64
    if normalized in ("float32", "fp32", "single"):
        return torch.float32
    if normalized in ("bfloat16", "bf16"):
        return torch.bfloat16
    raise ValueError(
        "Unsupported SGLANG_Q_ROTATION_COMPUTE_DTYPE "
        f"'{name}'. Expected float64, float32, or bfloat16."
    )


class QRotationManager:
    def __init__(self) -> None:
        self.path = os.environ.get("SGLANG_Q_ROTATION_PATH")
        self.compute_dtype = _parse_compute_dtype(
            os.environ.get("SGLANG_Q_ROTATION_COMPUTE_DTYPE", "float32")
        )
        self.enabled = bool(self.path)
        self.grouping: Optional[str] = None
        self._cpu_layers: dict[int, torch.Tensor] = {}
        self._device_cache: dict[tuple[int, str], torch.Tensor] = {}
        self._loaded = False

    def _load_if_needed(self) -> None:
        if not self.enabled or self._loaded:
            return

        state = torch.load(self.path, map_location="cpu")
        layers = state.get("layers")
        if not isinstance(layers, dict) or not layers:
            raise ValueError(
                f"Invalid Q rotation file at {self.path}: missing non-empty 'layers'"
            )

        self.grouping = state.get("source_grouping", state.get("grouping", "layer"))
        for layer_id, layer_data in layers.items():
            rotation = layer_data["rotation"].to(dtype=self.compute_dtype)
            self._cpu_layers[int(layer_id)] = rotation.contiguous()

        self._loaded = True
        logger.info(
            "Loaded Q rotation from %s with grouping=%s, compute_dtype=%s, hadamard_enabled=%s",
            self.path,
            self.grouping,
            self.compute_dtype,
            _hadamard_enabled,
        )

    def get_rotation(self, layer_id: int, device: torch.device) -> tuple[Optional[torch.Tensor], str]:
        self._load_if_needed()
        rotation = self._cpu_layers.get(layer_id)
        if rotation is None:
            return None, self.grouping or "layer"

        cache_key = (layer_id, str(device))
        if cache_key not in self._device_cache:
            self._device_cache[cache_key] = rotation.to(device=device, copy=True)
        return self._device_cache[cache_key], self.grouping or "layer"


_Q_ROTATION_MANAGER = QRotationManager()


class VRotationManager:
    def __init__(self) -> None:
        self.path = os.environ.get("SGLANG_V_ROTATION_PATH")
        self.compute_dtype = _parse_compute_dtype(
            os.environ.get("SGLANG_V_ROTATION_COMPUTE_DTYPE", "float32")
        )
        self.enabled = bool(self.path)
        self.grouping: Optional[str] = None
        self._cpu_layers: dict[int, torch.Tensor] = {}
        self._device_cache: dict[tuple[int, str], torch.Tensor] = {}
        self._loaded = False

    def _load_if_needed(self) -> None:
        if not self.enabled or self._loaded:
            return

        state = torch.load(self.path, map_location="cpu")
        layers = state.get("layers")
        if not isinstance(layers, dict) or not layers:
            raise ValueError(
                f"Invalid V rotation file at {self.path}: missing non-empty 'layers'"
            )

        self.grouping = state.get("source_grouping", state.get("grouping", "layer"))
        for layer_id, layer_data in layers.items():
            rotation = layer_data["rotation"].to(dtype=self.compute_dtype)
            self._cpu_layers[int(layer_id)] = rotation.contiguous()

        self._loaded = True
        logger.info(
            "Loaded V rotation from %s with grouping=%s, compute_dtype=%s",
            self.path,
            self.grouping,
            self.compute_dtype,
        )

    def get_rotation(self, layer_id: int, device: torch.device) -> tuple[Optional[torch.Tensor], str]:
        self._load_if_needed()
        rotation = self._cpu_layers.get(layer_id)
        if rotation is None:
            return None, self.grouping or "layer"

        cache_key = (layer_id, str(device))
        if cache_key not in self._device_cache:
            self._device_cache[cache_key] = rotation.to(device=device, copy=True)
        return self._device_cache[cache_key], self.grouping or "layer"


_V_ROTATION_MANAGER = VRotationManager()


def should_apply_post_hadamard_qk_rotation(kv_cache_dtype: Optional[str]) -> bool:
    return bool(_Q_ROTATION_MANAGER.enabled and _hadamard_enabled and kv_cache_dtype == "int4")


def _apply_layer_rotation(x: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x, rotation)


def _apply_head_rotation(x: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    if rotation.shape[0] != x.shape[1]:
        raise ValueError(
            "Head-wise Q rotation expects rotation.shape[0] == num_heads, "
            f"got {rotation.shape[0]} and {x.shape[1]}"
        )
    return torch.einsum("thd,hdf->thf", x, rotation)


def _apply_q_kv_group_rotation(
    q: torch.Tensor,
    rotation: torch.Tensor,
    num_kv_heads: int,
) -> torch.Tensor:
    num_groups = rotation.shape[0]
    if num_groups != num_kv_heads:
        raise ValueError(
            "KV-group Q rotation expects rotation.shape[0] == num_kv_heads, "
            f"got {rotation.shape[0]} and {num_kv_heads}"
        )
    if q.shape[1] % num_groups != 0:
        raise ValueError(
            f"num_q_heads ({q.shape[1]}) must be divisible by num_groups ({num_groups})"
        )
    group_size = q.shape[1] // num_groups
    q_grouped = q.reshape(q.shape[0], num_groups, group_size, q.shape[-1])
    return torch.einsum("tghd,gdf->tghf", q_grouped, rotation).reshape_as(q)


def _apply_k_kv_group_rotation(
    k: torch.Tensor,
    rotation: torch.Tensor,
    num_kv_heads: int,
) -> torch.Tensor:
    num_groups = rotation.shape[0]
    if num_groups != num_kv_heads:
        raise ValueError(
            "KV-group Q rotation expects rotation.shape[0] == num_kv_heads, "
            f"got {rotation.shape[0]} and {num_kv_heads}"
        )
    if k.shape[1] != num_groups:
        raise ValueError(
            f"num_kv_heads ({k.shape[1]}) must match num_groups ({num_groups})"
        )
    return torch.einsum("tgd,gdf->tgf", k, rotation)


def _apply_rotation_to_q(
    q: torch.Tensor,
    rotation: torch.Tensor,
    grouping: str,
    num_kv_heads: int,
) -> torch.Tensor:
    if grouping == "layer":
        return _apply_layer_rotation(q, rotation)
    if grouping == "head":
        return _apply_head_rotation(q, rotation)
    if grouping == "kv_group":
        return _apply_q_kv_group_rotation(q, rotation, num_kv_heads)
    raise ValueError(
        f"Unsupported Q rotation grouping '{grouping}' in {_Q_ROTATION_MANAGER.path}"
    )


def _apply_rotation_to_k(
    k: torch.Tensor,
    rotation: torch.Tensor,
    grouping: str,
    num_kv_heads: int,
) -> torch.Tensor:
    if grouping == "layer":
        return _apply_layer_rotation(k, rotation)
    if grouping == "head":
        return _apply_head_rotation(k, rotation)
    if grouping == "kv_group":
        return _apply_k_kv_group_rotation(k, rotation, num_kv_heads)
    raise ValueError(
        f"Unsupported Q rotation grouping '{grouping}' in {_Q_ROTATION_MANAGER.path}"
    )


def _apply_inverse_rotation_to_k(
    k: torch.Tensor,
    rotation: torch.Tensor,
    grouping: str,
    num_kv_heads: int,
) -> torch.Tensor:
    if grouping == "layer":
        return torch.matmul(k, rotation.t())
    if grouping == "head":
        return torch.einsum("thd,hfd->thf", k, rotation)
    if grouping == "kv_group":
        return torch.einsum("tgd,gfd->tgf", k, rotation)
    raise ValueError(
        f"Unsupported Q rotation grouping '{grouping}' in {_Q_ROTATION_MANAGER.path}"
    )


def _reshape_for_rotation(
    x: torch.Tensor,
    num_heads: int,
    head_dim: int,
    rotation_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Size]:
    original_shape = x.shape
    x_work = x.reshape(-1, num_heads, head_dim).to(dtype=rotation_dtype)
    return x_work, original_shape


def _restore_rotated_tensor(x: torch.Tensor, original_shape: torch.Size, dtype: torch.dtype):
    return x.to(dtype=dtype).reshape(original_shape)


@torch._dynamo.disable()
def maybe_apply_q_rotation(
    q: torch.Tensor,
    *,
    layer_id: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    if not _Q_ROTATION_MANAGER.enabled:
        return q

    rotation, grouping = _Q_ROTATION_MANAGER.get_rotation(layer_id, q.device)
    if rotation is None:
        return q
    q_work, q_shape = _reshape_for_rotation(q, num_q_heads, head_dim, rotation.dtype)
    q_rotated = _apply_rotation_to_q(q_work, rotation, grouping, num_kv_heads)
    return _restore_rotated_tensor(q_rotated, q_shape, q.dtype)


@torch._dynamo.disable()
def maybe_apply_k_rotation(
    k: torch.Tensor,
    *,
    layer_id: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    if not _Q_ROTATION_MANAGER.enabled:
        return k

    rotation, grouping = _Q_ROTATION_MANAGER.get_rotation(layer_id, k.device)
    if rotation is None:
        return k
    k_work, k_shape = _reshape_for_rotation(k, num_kv_heads, head_dim, rotation.dtype)
    k_rotated = _apply_rotation_to_k(k_work, rotation, grouping, num_kv_heads)
    return _restore_rotated_tensor(k_rotated, k_shape, k.dtype)


@torch._dynamo.disable()
def apply_inverse_k_rotation(
    k: torch.Tensor,
    *,
    layer_id: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    if not _Q_ROTATION_MANAGER.enabled:
        return k
    rotation, grouping = _Q_ROTATION_MANAGER.get_rotation(layer_id, k.device)
    if rotation is None:
        return k
    k_work, k_shape = _reshape_for_rotation(k, num_kv_heads, head_dim, rotation.dtype)
    k_inv = _apply_inverse_rotation_to_k(k_work, rotation, grouping, num_kv_heads)
    return _restore_rotated_tensor(k_inv, k_shape, k.dtype)


@torch._dynamo.disable()
def maybe_apply_qk_rotation(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    *,
    layer_id: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if not _Q_ROTATION_MANAGER.enabled or k is None:
        return q, k

    q_rotated = maybe_apply_q_rotation(
        q,
        layer_id=layer_id,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    k_rotated = maybe_apply_k_rotation(
        k,
        layer_id=layer_id,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    return q_rotated, k_rotated


@torch._dynamo.disable()
def maybe_apply_v_rotation(
    v: torch.Tensor,
    *,
    layer_id: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Apply V rotation before quantization: V_rot = V @ R_V."""
    if not _V_ROTATION_MANAGER.enabled:
        return v

    rotation, grouping = _V_ROTATION_MANAGER.get_rotation(layer_id, v.device)
    if rotation is None:
        return v
    v_work, v_shape = _reshape_for_rotation(v, num_kv_heads, head_dim, rotation.dtype)
    v_rotated = _apply_rotation_to_k(v_work, rotation, grouping, num_kv_heads)
    return _restore_rotated_tensor(v_rotated, v_shape, v.dtype)


@torch._dynamo.disable()
def maybe_apply_inv_v_rotation(
    o: torch.Tensor,
    *,
    layer_id: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Apply inverse V rotation to attention output: O = O_rot @ R_V^T."""
    if not _V_ROTATION_MANAGER.enabled:
        return o

    rotation, grouping = _V_ROTATION_MANAGER.get_rotation(layer_id, o.device)
    if rotation is None:
        return o
    if grouping == "layer":
        o_work, o_shape = _reshape_for_rotation(o, num_q_heads, head_dim, rotation.dtype)
        o_inv = torch.matmul(o_work, rotation.t())
        return _restore_rotated_tensor(o_inv, o_shape, o.dtype)
    if grouping == "kv_group":
        o_work, o_shape = _reshape_for_rotation(o, num_q_heads, head_dim, rotation.dtype)
        group_size = num_q_heads // num_kv_heads
        o_grouped = o_work.reshape(o_work.shape[0], num_kv_heads, group_size, head_dim)
        o_inv = torch.einsum("tghd,gfd->tghf", o_grouped, rotation).reshape_as(o_work)
        return _restore_rotated_tensor(o_inv, o_shape, o.dtype)
    if grouping == "head":
        o_work, o_shape = _reshape_for_rotation(o, num_q_heads, head_dim, rotation.dtype)
        o_inv = torch.einsum("thd,hfd->thf", o_work, rotation)
        return _restore_rotated_tensor(o_inv, o_shape, o.dtype)
    raise ValueError(f"Unsupported V rotation grouping '{grouping}'")
