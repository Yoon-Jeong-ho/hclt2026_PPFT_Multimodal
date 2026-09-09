from __future__ import annotations

import math

import pytest

from ppft_multimodal.noise_profiles import (
    NOISE_PROFILES,
    PAPER_STAGE2_ARM_ORDER,
    noise_profile_name,
    resolve_noise_epsilons,
)


def test_declares_only_three_paper_profiles_in_fixed_order() -> None:
    assert PAPER_STAGE2_ARM_ORDER == ("no_noise", "text_75_image_2", "text_75_image_2p5")
    assert NOISE_PROFILES == {
        "no_noise": (None, None),
        "text_75_image_2": (75.0, 2.0),
        "text_75_image_2p5": (75.0, 2.5),
    }


def test_resolves_clean_and_split_noise() -> None:
    assert resolve_noise_epsilons(
        {"enabled": False, "epsilon": None, "text_epsilon": None, "vision_epsilon": None}
    ) == (None, None)
    split = {"enabled": True, "epsilon": None, "text_epsilon": 75.0, "vision_epsilon": 2.5}
    assert resolve_noise_epsilons(split) == (75.0, 2.5)
    assert noise_profile_name(split) == "text_75_image_2p5"


@pytest.mark.parametrize("noise", [
    {"enabled": True, "epsilon": True},
    {"enabled": True, "epsilon": "75"},
    {"enabled": True, "epsilon": math.inf},
    {"enabled": True, "epsilon": math.nan},
    {"enabled": True, "epsilon": None, "text_epsilon": 75.0},
    {"enabled": False, "epsilon": None, "vision_epsilon": 2.0},
])
def test_rejects_invalid_epsilon_shapes_and_values(noise: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        resolve_noise_epsilons(noise)


def test_undeclared_pair_fails_closed() -> None:
    with pytest.raises(ValueError, match="declared profile"):
        noise_profile_name(
            {"enabled": True, "epsilon": None, "text_epsilon": 75.0, "vision_epsilon": 3.0}
        )
