from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .noise import NoiseStatistics, add_norm_preserving_noise


class VisionPreMergerNoiseHook:
    """Pre-hook the native visual merger without copying or changing it."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        epsilon: float | None = None,
        noise_scale: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> None:
        self.enabled = enabled
        self.epsilon = epsilon
        self.noise_scale = noise_scale
        self.generator = generator
        self.call_count = 0
        self.input_shape: tuple[int, ...] | None = None
        self.clean_input: torch.Tensor | None = None
        self.transmitted_input: torch.Tensor | None = None
        self.statistics: NoiseStatistics | None = None
        self.segment_generators: list[torch.Generator] | None = None
        self.segment_sizes: list[int] | None = None
        self.segment_clean_inputs: list[torch.Tensor] = []
        self.segment_transmitted_inputs: list[torch.Tensor] = []
        self.segment_statistics: list[NoiseStatistics] = []
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    def register(self, native_merger: nn.Module) -> VisionPreMergerNoiseHook:
        if self._handle is not None:
            raise RuntimeError("vision pre-merger hook is already registered")
        self._handle = native_merger.register_forward_pre_hook(self._pre_hook, with_kwargs=True)
        return self

    install = register

    def __enter__(self) -> VisionPreMergerNoiseHook:
        if self._handle is None:
            raise RuntimeError("install the vision hook before entering its context")
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.remove()

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _pre_hook(
        self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        del module
        if args and isinstance(args[0], torch.Tensor):
            tensor, in_args = args[0], True
        elif isinstance(kwargs.get("hidden_states"), torch.Tensor):
            tensor, in_args = kwargs["hidden_states"], False
        else:
            raise RuntimeError("native visual merger first hidden-state tensor was not found")
        self.call_count += 1
        self.input_shape = tuple(tensor.shape)
        self.clean_input = tensor.detach()
        if self.segment_sizes is not None:
            if sum(self.segment_sizes) != tensor.shape[0]:
                raise RuntimeError("vision noise segment sizes do not cover native merger input")
            if self.segment_generators is None or len(self.segment_generators) != len(self.segment_sizes):
                raise RuntimeError("vision noise segments require one deterministic generator per image")
            clean_chunks = list(torch.split(tensor, self.segment_sizes, dim=0))
            noisy_chunks: list[torch.Tensor] = []
            statistics: list[NoiseStatistics] = []
            for chunk, generator in zip(clean_chunks, self.segment_generators, strict=True):
                noisy_chunk, chunk_stats = add_norm_preserving_noise(
                    chunk,
                    epsilon=self.epsilon,
                    enabled=self.enabled,
                    noise_scale=self.noise_scale,
                    generator=generator,
                    return_statistics=True,
                )
                noisy_chunks.append(noisy_chunk)
                statistics.append(chunk_stats)
            noisy = torch.cat(noisy_chunks, dim=0)
            stats = NoiseStatistics(
                **{
                    field: torch.cat([getattr(item, field) for item in statistics], dim=0)
                    for field in NoiseStatistics.__dataclass_fields__
                }
            )
            self.segment_clean_inputs = [chunk.detach() for chunk in clean_chunks]
            self.segment_transmitted_inputs = [chunk.detach() for chunk in noisy_chunks]
            self.segment_statistics = statistics
        else:
            noisy, stats = add_norm_preserving_noise(
                tensor,
                epsilon=self.epsilon,
                enabled=self.enabled,
                noise_scale=self.noise_scale,
                generator=self.generator,
                return_statistics=True,
            )
            self.segment_clean_inputs = [tensor.detach()]
            self.segment_transmitted_inputs = [noisy.detach()]
            self.segment_statistics = [stats]
        self.transmitted_input = noisy.detach()
        self.statistics = stats
        if in_args:
            return (noisy, *args[1:]), kwargs
        return args, {**kwargs, "hidden_states": noisy}


VisionNoiseHook = VisionPreMergerNoiseHook
