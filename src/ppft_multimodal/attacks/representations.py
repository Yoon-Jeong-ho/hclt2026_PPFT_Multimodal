"""Hash-bound metadata for private, local-only representation caches."""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

RepresentationKind = Literal["vision_pre_merger"]

def tensor_sha256(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()

@dataclass(frozen=True)
class RepresentationMetadata:
    victim_checkpoint_sha256: str
    example_id: str
    kind: RepresentationKind
    tensor_shape: tuple[int, ...]
    dtype: str
    feature_norm: float
    tensor_sha256: str
    model_revision: str
    processor_revision: str | None = None
    image_grid_thw: tuple[int, int, int] | None = None
    valid_token_count: int | None = None
    source: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        *,
        victim_checkpoint_sha256: str,
        example_id: str,
        kind: RepresentationKind,
        model_revision: str,
        processor_revision: str | None = None,
        image_grid_thw: tuple[int, int, int] | None = None,
        valid_token_count: int | None = None,
        source: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> RepresentationMetadata:
        if tensor.ndim < 2:
            raise ValueError("a transmitted representation must include token and feature axes")
        return cls(
            victim_checkpoint_sha256=victim_checkpoint_sha256,
            example_id=str(example_id),
            kind=kind,
            tensor_shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            feature_norm=float(tensor.detach().float().norm().item()),
            tensor_sha256=tensor_sha256(tensor),
            model_revision=model_revision,
            processor_revision=processor_revision,
            image_grid_thw=image_grid_thw,
            valid_token_count=valid_token_count,
            source=source,
            extra=dict(extra or {}),
        )

    def validate_tensor(self, tensor: torch.Tensor) -> None:
        if tuple(tensor.shape) != self.tensor_shape:
            raise ValueError("representation shape does not match sidecar")
        if str(tensor.dtype) != self.dtype:
            raise ValueError("representation dtype does not match sidecar")
        if tensor_sha256(tensor) != self.tensor_sha256:
            raise ValueError("representation content hash mismatch")
