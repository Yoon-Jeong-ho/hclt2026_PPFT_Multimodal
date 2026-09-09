from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, overload

import torch


@dataclass(frozen=True)
class NoiseStatistics:
    clean_norm: torch.Tensor
    sampled_noise_norm: torch.Tensor
    noisy_before_rescale_norm: torch.Tensor
    final_norm: torch.Tensor
    cosine: torch.Tensor


def _stats(clean: torch.Tensor, noise: torch.Tensor, before: torch.Tensor, final: torch.Tensor) -> NoiseStatistics:
    work_clean, work_noise, work_before, work_final = (x.float() for x in (clean, noise, before, final))
    return NoiseStatistics(
        clean_norm=work_clean.norm(dim=-1),
        sampled_noise_norm=work_noise.norm(dim=-1),
        noisy_before_rescale_norm=work_before.norm(dim=-1),
        final_norm=work_final.norm(dim=-1),
        cosine=torch.nn.functional.cosine_similarity(work_clean, work_final, dim=-1),
    )


@overload
def add_norm_preserving_noise(
    representations: torch.Tensor,
    epsilon: float | None = None,
    *,
    valid_mask: torch.Tensor | None = None,
    enabled: bool = True,
    noise_scale: float = 1.0,
    generator: torch.Generator | None = None,
    return_statistics: Literal[False] = False,
    return_stats: Literal[False, None] = None,
) -> torch.Tensor: ...


@overload
def add_norm_preserving_noise(
    representations: torch.Tensor,
    epsilon: float | None = None,
    *,
    valid_mask: torch.Tensor | None = None,
    enabled: bool = True,
    noise_scale: float = 1.0,
    generator: torch.Generator | None = None,
    return_statistics: Literal[True],
    return_stats: bool | None = None,
) -> tuple[torch.Tensor, NoiseStatistics]: ...


@overload
def add_norm_preserving_noise(
    representations: torch.Tensor,
    epsilon: float | None = None,
    *,
    valid_mask: torch.Tensor | None = None,
    enabled: bool = True,
    noise_scale: float = 1.0,
    generator: torch.Generator | None = None,
    return_statistics: bool = False,
    return_stats: Literal[True],
) -> tuple[torch.Tensor, NoiseStatistics]: ...


def add_norm_preserving_noise(
    representations: torch.Tensor,
    epsilon: float | None = None,
    *,
    valid_mask: torch.Tensor | None = None,
    enabled: bool = True,
    noise_scale: float = 1.0,
    generator: torch.Generator | None = None,
    return_statistics: bool = False,
    return_stats: bool | None = None,
) -> torch.Tensor | tuple[torch.Tensor, NoiseStatistics]:
    """Apply ACL PPFT isotropic L2-Laplace noise along the last dimension.

    Provenance: ACL_2025 ``program_code/model/model.py::_inject_noise``.
    Direction is uniform on the sphere and radius is Gamma(d, rate=epsilon),
    after which each valid vector is rescaled to its original L2 norm.
    """
    if return_stats is not None:
        if return_statistics and return_stats != return_statistics:
            raise ValueError("return_stats and return_statistics disagree")
        return_statistics = return_stats
    if not representations.is_floating_point() or representations.ndim < 2:
        raise ValueError("representations must be a floating-point tensor with a feature dimension")
    if noise_scale <= 0:
        raise ValueError("noise_scale must be positive")
    if not enabled:
        if epsilon is not None:
            raise ValueError("disabled noise must use epsilon=None, not epsilon=0")
        if return_statistics:
            zero = torch.zeros_like(representations)
            return representations, _stats(representations, zero, representations, representations)
        return representations
    if epsilon is None or epsilon <= 0:
        raise ValueError("enabled noise requires a positive epsilon")

    leading_shape = representations.shape[:-1]
    if valid_mask is None:
        mask = torch.ones(leading_shape, dtype=torch.bool, device=representations.device)
    else:
        if tuple(valid_mask.shape) != tuple(leading_shape):
            raise ValueError(f"valid_mask shape {tuple(valid_mask.shape)} must equal {tuple(leading_shape)}")
        mask = valid_mask.to(device=representations.device, dtype=torch.bool)

    flat = representations.reshape(-1, representations.shape[-1])
    flat_mask = mask.reshape(-1)
    selected = flat[flat_mask]
    output = flat.clone()
    all_noise = torch.zeros_like(flat)
    before = flat.clone()
    if selected.numel():
        # Sampling in fp32 avoids unsupported/unstable Gamma paths for bf16.
        sample = selected.float()
        direction = torch.randn(sample.shape, device=sample.device, dtype=sample.dtype, generator=generator)
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(sample.dtype).tiny)
        concentration = torch.full(
            (sample.shape[0], 1), float(sample.shape[-1]), device=sample.device, dtype=sample.dtype
        )
        radius = torch._standard_gamma(concentration, generator=generator) / float(epsilon)
        noise = direction * radius * noise_scale
        perturbed = sample + noise
        clean_norm = sample.norm(dim=-1, keepdim=True)
        perturbed_norm = perturbed.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(sample.dtype).tiny)
        final = torch.where(clean_norm > 0, perturbed / perturbed_norm * clean_norm, sample)
        output[flat_mask] = final.to(flat.dtype)
        all_noise[flat_mask] = noise.to(flat.dtype)
        before[flat_mask] = perturbed.to(flat.dtype)

    result = output.reshape_as(representations)
    if not torch.isfinite(result).all():
        raise FloatingPointError("noise application produced NaN or Inf")
    if return_statistics:
        return result, _stats(
            representations,
            all_noise.reshape_as(representations),
            before.reshape_as(representations),
            result,
        )
    return result
