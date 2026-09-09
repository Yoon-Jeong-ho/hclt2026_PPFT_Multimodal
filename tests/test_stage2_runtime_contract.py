from pathlib import Path

import pytest

from ppft_multimodal.noise_profiles import resolve_noise_epsilons
from ppft_multimodal.training import stage1_decoder_tuning, stage2_invariant_fingerprint
from scripts.training_common import read_config, resolve_paper_stage2


def test_stage1_uses_full_decoder_tuning() -> None:
    assert stage1_decoder_tuning(read_config(Path("configs/stage1/paper.yaml"))) == "full"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("no_noise", (None, None)),
        ("text_75_image_2", (75.0, 2.0)),
        ("text_75_image_2p5", (75.0, 2.5)),
    ],
)
def test_paper_stage2_arms_resolve_as_one_family(name: str, expected: tuple[float | None, float | None]) -> None:
    config = resolve_paper_stage2(
        Path(f"configs/stage2/{name}.yaml").resolve(),
        qwen_path=Path("models/qwen35-08b"),
        kure_path=Path("models/kure-v1"),
        parent_checkpoint=Path("outputs/checkpoints/stage1/final"),
        max_updates=None,
    )
    assert resolve_noise_epsilons(config["noise"]) == expected


def test_stage2_arm_non_noise_settings_are_identical() -> None:
    configs = [read_config(path) for path in sorted(Path("configs/stage2").glob("*.yaml"))]
    assert len({stage2_invariant_fingerprint(config) for config in configs}) == 1
