from pathlib import Path

import pytest
import yaml

from ppft_multimodal.config import load_model_config

MODEL_CONFIG = Path("configs/model/qwen35_08b_kure.yaml")


def test_paper_model_config_is_exactly_pinned() -> None:
    config = load_model_config(MODEL_CONFIG)
    assert config.backbone.name == "Qwen/Qwen3.5-0.8B"
    assert config.backbone.revision == "2fc06364715b967f1860aea9cf38778875588b17"
    assert config.text_encoder.name == "nlpai-lab/KURE-v1"
    assert config.text_encoder.revision == "4ed4540949c70b7da2c74004a915e1f2d5e46e4f"
    assert config.text_encoder.pooling_k == 2
    assert config.runtime.max_target_length == 512
    assert config.lora.rank == 16
    assert config.lora.target_policy == "decoder_only_all_linear"


@pytest.mark.parametrize("name", ["Qwen/Qwen3.5-0.8B-Base", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-1B"])
def test_non_paper_backbones_are_rejected(tmp_path: Path, name: str) -> None:
    raw = yaml.safe_load(MODEL_CONFIG.read_text(encoding="utf-8"))
    raw["backbone"]["name"] = name
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_model_config(path)


def test_wrong_kure_revision_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(MODEL_CONFIG.read_text(encoding="utf-8"))
    raw["text_encoder"]["revision"] = "0" * 40
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="text encoder and revision must remain pinned"):
        load_model_config(path)
