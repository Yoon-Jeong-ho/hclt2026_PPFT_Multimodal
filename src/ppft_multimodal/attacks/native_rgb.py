"""Exact inverse of native Qwen still-image patchification for attack targets."""
from __future__ import annotations

from collections.abc import Sequence

import torch


def native_patch_tensor_to_rgb(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor | Sequence[int],
    *,
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
    image_mean: Sequence[float],
    image_std: Sequence[float],
) -> torch.Tensor:
    """Invert Qwen2/3-VL native patchify and normalization exactly.

    Qwen's image processor duplicates still images along the temporal-patch
    dimension.  Equality is checked before selecting the first copy.
    """
    grid = [int(value) for value in torch.as_tensor(grid_thw).tolist()]
    if len(grid) != 3 or grid[0] != 1:
        raise ValueError(f"expected one still-image grid, got {grid}")
    _, grid_h, grid_w = grid
    channels = len(image_mean)
    expected = (grid_h * grid_w, channels * temporal_patch_size * patch_size * patch_size)
    if tuple(pixel_values.shape) != expected:
        raise ValueError(f"native pixel tensor shape {tuple(pixel_values.shape)} != {expected}")
    if grid_h % merge_size or grid_w % merge_size:
        raise ValueError("native image grid is not divisible by spatial merge size")
    patches = pixel_values.reshape(
        1,
        grid_h // merge_size,
        grid_w // merge_size,
        merge_size,
        merge_size,
        channels,
        temporal_patch_size,
        patch_size,
        patch_size,
    )
    if temporal_patch_size > 1:
        reference = patches[..., :1, :, :].expand_as(patches)
        if not torch.equal(patches, reference):
            raise ValueError("still-image temporal patches are not exact duplicates")
    image = patches[..., 0, :, :].permute(0, 5, 1, 3, 6, 2, 4, 7).reshape(
        1, channels, grid_h * patch_size, grid_w * patch_size
    )[0]
    mean = torch.as_tensor(image_mean, dtype=image.dtype, device=image.device)[:, None, None]
    std = torch.as_tensor(image_std, dtype=image.dtype, device=image.device)[:, None, None]
    return image * std + mean
