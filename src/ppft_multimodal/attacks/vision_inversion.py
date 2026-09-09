"""Compact image decoder for clean pre-merger visual representations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.layers(inputs))


class VisionInversionAttacker(nn.Module):
    """Restore flattened native-grid tokens then decode deterministic RGB [0,1]."""

    def __init__(self, representation_dim: int, channels: int = 128, blocks: int = 3, spatial_merge_size: int = 2):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.input_projection = nn.Conv2d(representation_dim, channels, kernel_size=1)
        self.residual = nn.Sequential(*(ResidualConvBlock(channels) for _ in range(blocks)))
        self.rgb_head = nn.Conv2d(channels, 3, kernel_size=3, padding=1)

    def forward(
        self,
        representation: torch.Tensor,
        image_grid_thw: torch.Tensor | tuple[int, int, int],
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        if representation.ndim == 2:
            representation = representation.unsqueeze(0)
        if representation.ndim != 3:
            raise ValueError("vision representation must have shape [batch,tokens,features]")
        grid = torch.as_tensor(image_grid_thw, device=representation.device)
        if grid.ndim == 2:
            if not torch.equal(grid, grid[0].expand_as(grid)):
                raise ValueError("batch together only identical native grid shapes")
            grid = grid[0]
        temporal, height, width = map(int, grid.tolist())
        expected_tokens = temporal * height * width
        if representation.shape[1] != expected_tokens:
            raise ValueError(f"native grid requires {expected_tokens} tokens, got {representation.shape[1]}")
        representation = representation.float()
        features = undo_qwen_merge_major_order(
            representation,
            (temporal, height, width),
            merge_size=self.spatial_merge_size,
        ).mean(dim=1)
        features = features.permute(0, 3, 1, 2).contiguous()
        decoded = self.rgb_head(self.residual(self.input_projection(features)))
        decoded = F.interpolate(decoded, size=target_size, mode="bilinear", align_corners=False)
        return decoded.sigmoid()


def undo_qwen_merge_major_order(
    representation: torch.Tensor,
    image_grid_thw: tuple[int, int, int],
    *,
    merge_size: int,
) -> torch.Tensor:
    """Undo official Qwen2-VL patchify's merge-major flattening to T,H,W.

    Official order is ``[T, H/m, W/m, m_h, m_w]`` rather than row-major
    ``[T,H,W]``. This inverse changes only the attacker's coordinate view; the
    victim representation and native token ordering remain untouched.
    """
    if representation.ndim != 3:
        raise ValueError("representation must have shape [batch,tokens,features]")
    temporal, height, width = image_grid_thw
    if merge_size < 1 or height % merge_size or width % merge_size:
        raise ValueError("native grid dimensions must be divisible by spatial_merge_size")
    expected = temporal * height * width
    if representation.shape[1] != expected:
        raise ValueError(f"native grid requires {expected} tokens, got {representation.shape[1]}")
    return (
        representation.reshape(
            representation.shape[0],
            temporal,
            height // merge_size,
            width // merge_size,
            merge_size,
            merge_size,
            representation.shape[-1],
        )
        .permute(0, 1, 2, 4, 3, 5, 6)
        .reshape(representation.shape[0], temporal, height, width, representation.shape[-1])
    )


def unique_images(records: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Deduplicate QA rows so an image contributes once to inversion training."""

    selected: dict[str, Mapping[str, Any]] = {}
    for record in records:
        identity = record.get("image_sha256") or record.get("image_id")
        if not identity:
            raise ValueError("vision inversion records require image_sha256 or image_id")
        selected.setdefault(str(identity), record)
    return list(selected.values())


def reconstruction_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and processor-space RGB target shapes differ")
    return F.mse_loss(prediction.float(), target.float())
