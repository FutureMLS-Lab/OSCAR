# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Vendored image + multimodal processor for ``google/gemma-4-*`` (``gemma4_unified``).

Why vendored: the unified vision pipeline (encoder-free; raw merged pixel patches
projected into LM space) ships its HF ``Gemma4UnifiedImageProcessor`` /
``Gemma4UnifiedProcessor`` only in transformers >= 5.10.  This serving env pins
transformers 5.9 (so the vendored ``Gemma4UnifiedConfig`` keeps precedence over the
native one for the text/INT2 path).  These classes reproduce the 5.10.1 image
processing math (torch + torchvision only) and a minimal processor that ties the
tokenizer + image processor and performs ``<|image|>`` placeholder expansion.

Source of the patchify math: transformers 5.10.1
``src/transformers/models/gemma4_unified/image_processing_gemma4_unified.py``.
"""

import math
from typing import List, Optional

import torch
from torchvision.transforms.v2 import functional as tvF

_SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)


def get_aspect_ratio_preserving_size(
    height: int,
    width: int,
    patch_size: int,
    max_patches: int,
    pooling_kernel_size: int,
) -> tuple[int, int]:
    """Largest aspect-ratio-preserving size that (1) yields <= ``max_patches``
    teacher patches and (2) is divisible by ``pooling_kernel_size * patch_size``."""
    total_px = height * width
    target_px = max_patches * (patch_size**2)
    factor = math.sqrt(target_px / total_px)
    ideal_height = factor * height
    ideal_width = factor * width
    side_mult = pooling_kernel_size * patch_size

    target_height = int(math.floor(ideal_height / side_mult)) * side_mult
    target_width = int(math.floor(ideal_width / side_mult)) * side_mult

    if target_height == 0 and target_width == 0:
        raise ValueError(
            "Attempting to resize to a 0 x 0 image. Resized height should be divisble "
            f"by `pooling_kernel_size * patch_size`={side_mult}."
        )

    max_side_length = (max_patches // pooling_kernel_size**2) * side_mult
    if target_height == 0:
        target_height = side_mult
        target_width = min(int(math.floor(width / height)) * side_mult, max_side_length)
    elif target_width == 0:
        target_width = side_mult
        target_height = min(
            int(math.floor(height / width)) * side_mult, max_side_length
        )

    if target_height * target_width > target_px:
        raise ValueError(
            f"Resizing [{height}x{width}] to [{target_height}x{target_width}] "
            f"but this exceeds {max_patches} patches with patch_size {patch_size}"
        )

    return target_height, target_width


def convert_image_to_patches(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(C, H, W) -> (num_patches_h * num_patches_w, patch_size*patch_size*C)."""
    num_channels, image_height, image_width = image.shape
    num_patches_height = image_height // patch_size
    num_patches_width = image_width // patch_size
    patched_image = image.reshape(
        num_channels, num_patches_height, patch_size, num_patches_width, patch_size
    )
    patched_image = patched_image.permute(1, 3, 2, 4, 0)
    patched_image = patched_image.reshape(num_patches_height * num_patches_width, -1)
    return patched_image


def pad_along_first_dim(
    image: torch.Tensor, positions: torch.Tensor, target_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad patches with zeros and positions with ``-1`` to ``target_length``."""
    current_length = image.shape[0]
    padding_length = target_length - current_length
    if padding_length > 0:
        padding = [0, 0] * (image.ndim - 1) + [0, padding_length]
        pos_padding = (0, 0, 0, padding_length)
        image = torch.nn.functional.pad(image, padding, mode="constant", value=0)
        positions = torch.nn.functional.pad(
            positions, pos_padding, mode="constant", value=-1
        )
    return image, positions


def patches_merge(
    patches: torch.Tensor,
    positions_xy: torch.Tensor,
    length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge k x k groups of teacher patches into ``length`` model patches."""
    patch_size = math.isqrt(patches.shape[-1] // 3)
    if patches.shape[-1] != patch_size * patch_size * 3:
        raise ValueError(
            f"Patch dimension {patches.shape[-1]} is not a valid `patch_size * patch_size * 3`"
        )

    k = math.isqrt(patches.shape[-2] // length)
    if k * k * length != patches.shape[-2]:
        raise ValueError(f"Cannot merge {patches.shape} to {length}")

    max_x = positions_xy[..., 0].max(dim=-1, keepdim=True)[0] + 1
    kernel_idxs = torch.div(positions_xy, k, rounding_mode="floor")
    num_patches_from_top_left = (
        k * k * kernel_idxs[..., 0] + k * max_x * kernel_idxs[..., 1]
    )

    position_within_kernel = torch.remainder(positions_xy, k)
    num_patches_from_top_left_of_kernel = (
        position_within_kernel[..., 0] + position_within_kernel[..., 1] * k
    )
    target_ordering = num_patches_from_top_left_of_kernel + num_patches_from_top_left

    perm = target_ordering.long().argsort(dim=-1)
    perm_expanded = perm.unsqueeze(-1).expand_as(patches)
    kernel_ordered_patches = patches.gather(-2, perm_expanded)

    batch_shape = patches.shape[:-2]

    kernel_ordered_patches = kernel_ordered_patches.reshape(
        *batch_shape, length, k * k, patch_size, patch_size, 3
    )
    kernel_ordered_patches = kernel_ordered_patches.reshape(
        *batch_shape, length, k, k, patch_size, patch_size, 3
    )
    kernel_ordered_patches = kernel_ordered_patches.permute(
        *range(len(batch_shape)), -6, -5, -3, -4, -2, -1
    )
    merged_patches = kernel_ordered_patches.reshape(
        *batch_shape, length, k * patch_size * k * patch_size * 3
    )

    perm_pos = perm.unsqueeze(-1).expand_as(positions_xy)
    kernel_ordered_positions = positions_xy.float().gather(-2, perm_pos.long())

    padding = (positions_xy == -1).all(dim=-1, keepdim=True)
    kernel_ordered_positions = (
        kernel_ordered_positions * (~padding).float() + positions_xy.float() * padding.float()
    )

    kernel_ordered_positions = kernel_ordered_positions.reshape(
        *batch_shape, length, k * k, 2
    )
    new_positions = torch.div(kernel_ordered_positions, k, rounding_mode="floor")
    new_positions = new_positions.min(dim=-2)[0].to(torch.long)

    return merged_patches, new_positions


class Gemma4UnifiedImageProcessor:
    """Standalone Gemma4-unified image processor.

    Reproduces the transformers 5.10.1 ``Gemma4UnifiedImageProcessor`` math without
    inheriting ``TorchvisionBackend`` (whose ``preprocess``/``_preprocess`` dispatch
    differs across the 5.9/5.10 boundary).  Outputs ``pixel_values``
    (merged raw patches), ``image_position_ids`` (-1 padded) and
    ``num_soft_tokens_per_image``.
    """

    model_input_names = [
        "pixel_values",
        "image_position_ids",
        "num_soft_tokens_per_image",
    ]

    def __init__(
        self,
        patch_size: int = 16,
        max_soft_tokens: int = 280,
        pooling_kernel_size: int = 3,
        do_resize: bool = True,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = False,
        image_mean=(0.0, 0.0, 0.0),
        image_std=(1.0, 1.0, 1.0),
        do_convert_rgb: bool = True,
        **kwargs,
    ):
        if max_soft_tokens not in _SUPPORTED_SOFT_TOKENS:
            raise ValueError(
                f"`max_soft_tokens` must be one of {_SUPPORTED_SOFT_TOKENS}, got {max_soft_tokens}."
            )
        self.patch_size = patch_size
        self.max_soft_tokens = max_soft_tokens
        self.pooling_kernel_size = pooling_kernel_size
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = list(image_mean)
        self.image_std = list(image_std)
        self.do_convert_rgb = do_convert_rgb

    @classmethod
    def from_dict(cls, cfg: dict, **kwargs):
        cfg = {**cfg, **kwargs}
        allowed = {
            "patch_size",
            "max_soft_tokens",
            "pooling_kernel_size",
            "do_resize",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "do_convert_rgb",
        }
        return cls(**{k: v for k, v in cfg.items() if k in allowed})

    @staticmethod
    def _to_chw_float(image) -> torch.Tensor:
        """Convert a PIL.Image / ndarray / tensor to a float32 (C, H, W) tensor."""
        if isinstance(image, torch.Tensor):
            t = image
            if t.ndim == 3 and t.shape[0] not in (1, 3):
                # (H, W, C) -> (C, H, W)
                t = t.permute(2, 0, 1)
        else:
            # PIL.Image or numpy array
            t = tvF.pil_to_tensor(image) if not hasattr(image, "shape") else torch.as_tensor(image)
            if t.ndim == 3 and t.shape[0] not in (1, 3):
                t = t.permute(2, 0, 1)
        return t.to(torch.float32)

    def _convert_rgb(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[0] == 1:
            image = image.expand(3, -1, -1)
        elif image.shape[0] == 4:
            image = image[:3]
        return image

    def aspect_ratio_preserving_resize(
        self, image: torch.Tensor, max_patches: int
    ) -> torch.Tensor:
        height, width = image.shape[-2], image.shape[-1]
        target_height, target_width = get_aspect_ratio_preserving_size(
            height=height,
            width=width,
            patch_size=self.patch_size,
            max_patches=max_patches,
            pooling_kernel_size=self.pooling_kernel_size,
        )
        if target_height == height and target_width == width:
            return image
        return tvF.resize(
            image,
            size=[target_height, target_width],
            interpolation=tvF.InterpolationMode.BICUBIC,
            antialias=True,
        )

    def __call__(self, images, return_tensors: Optional[str] = "pt", **kwargs):
        return self.preprocess(images, return_tensors=return_tensors, **kwargs)

    def preprocess(
        self,
        images,
        return_tensors: Optional[str] = "pt",
        max_soft_tokens: Optional[int] = None,
        **kwargs,
    ) -> dict:
        if not isinstance(images, (list, tuple)):
            images = [images]
        max_soft_tokens = max_soft_tokens or self.max_soft_tokens
        if max_soft_tokens not in _SUPPORTED_SOFT_TOKENS:
            raise ValueError(
                f"`max_soft_tokens` must be one of {_SUPPORTED_SOFT_TOKENS}, got {max_soft_tokens}."
            )
        max_patches = max_soft_tokens * self.pooling_kernel_size**2

        pixel_values: List[torch.Tensor] = []
        position_ids: List[torch.Tensor] = []
        num_soft_tokens_per_image: List[int] = []

        for image in images:
            img = self._to_chw_float(image)
            if self.do_convert_rgb:
                img = self._convert_rgb(img)

            if self.do_resize:
                img = self.aspect_ratio_preserving_resize(img, max_patches)

            if self.do_rescale:
                img = img * self.rescale_factor
            if self.do_normalize:
                mean = torch.tensor(self.image_mean, dtype=img.dtype).view(-1, 1, 1)
                std = torch.tensor(self.image_std, dtype=img.dtype).view(-1, 1, 1)
                img = (img - mean) / std

            teacher_patches = convert_image_to_patches(img, self.patch_size)

            patch_height = img.shape[-2] // self.patch_size
            patch_width = img.shape[-1] // self.patch_size
            patch_grid = torch.meshgrid(
                torch.arange(patch_width),
                torch.arange(patch_height),
                indexing="xy",
            )
            teacher_positions = torch.stack(patch_grid, dim=-1).reshape(
                teacher_patches.shape[0], 2
            )

            num_model_patches = teacher_patches.shape[0] // (
                self.pooling_kernel_size**2
            )
            merged_patches, merged_positions = patches_merge(
                teacher_patches.unsqueeze(0),
                teacher_positions.unsqueeze(0),
                num_model_patches,
            )
            merged_patches = merged_patches.squeeze(0)
            merged_positions = merged_positions.squeeze(0)
            num_soft_tokens_per_image.append(int(merged_patches.shape[0]))

            merged_patches, merged_positions = pad_along_first_dim(
                merged_patches, merged_positions, max_soft_tokens
            )
            pixel_values.append(merged_patches)
            position_ids.append(merged_positions)

        data = {
            "pixel_values": torch.stack(pixel_values, dim=0),
            "image_position_ids": torch.stack(position_ids, dim=0),
            "num_soft_tokens_per_image": num_soft_tokens_per_image,
        }
        return data


__all__ = ["Gemma4UnifiedImageProcessor", "get_aspect_ratio_preserving_size"]
