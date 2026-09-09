"""Canonical Stage-2 noise profiles and strict epsilon resolution."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

PAPER_STAGE2_ARM_ORDER = (
    "no_noise",
    "text_75_image_2",
    "text_75_image_2p5",
)
SPLIT_ARM_ORDER = PAPER_STAGE2_ARM_ORDER

NOISE_PROFILES: dict[str, tuple[int | float | None, int | float | None]] = {
    "no_noise": (None, None),
    "text_75_image_2": (75.0, 2.0),
    "text_75_image_2p5": (75.0, 2.5),
}


def _positive_finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"noise.{field} must be a finite positive number")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0:
        raise ValueError(f"noise.{field} must be a finite positive number")
    return resolved


def resolve_noise_epsilons(noise: Mapping[str, Any]) -> tuple[float | None, float | None]:
    """Validate one noise mapping and return independent text/vision epsilons."""

    enabled = noise.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("noise.enabled must be a boolean")
    shared = noise.get("epsilon")
    text = noise.get("text_epsilon")
    vision = noise.get("vision_epsilon")

    if not enabled:
        if any(value is not None for value in (shared, text, vision)):
            raise ValueError("disabled noise requires all epsilons to be null")
        return None, None

    if shared is not None:
        if text is not None or vision is not None:
            raise ValueError("shared noise.epsilon cannot be mixed with modality epsilons")
        epsilon = _positive_finite_number(shared, field="epsilon")
        return epsilon, epsilon

    if text is None or vision is None:
        raise ValueError("enabled split noise requires both text_epsilon and vision_epsilon")
    return (
        _positive_finite_number(text, field="text_epsilon"),
        _positive_finite_number(vision, field="vision_epsilon"),
    )


def noise_profile_name(noise: Mapping[str, Any]) -> str:
    """Return the declared canonical profile name, rejecting undeclared pairs."""

    epsilons = resolve_noise_epsilons(noise)
    matches = [name for name, profile in NOISE_PROFILES.items() if profile == epsilons]
    if len(matches) != 1:
        raise ValueError(f"noise epsilons do not match a declared profile: {epsilons}")
    return matches[0]
