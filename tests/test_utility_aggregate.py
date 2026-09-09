import pytest

from ppft_multimodal.evaluation.aggregate import (
    auxiliary_utility_aggregate,
    utility_aggregate,
)


def canonical_rows() -> list[dict[str, object]]:
    return [
        {"source": "Pri-DDX", "modality": "text", "relaxed_result": True},
        {"source": "Pri-NLICE", "modality": "text", "relaxed_result": False},
        {"source": "VQA-RAD", "modality": "image_text", "relaxed_result": True},
        {"source": "SLAKE", "modality": "image_text", "relaxed_result": False},
        {"source": "PathVQA", "modality": "image_text", "relaxed_result": False},
    ]


def test_main_utility_excludes_support_datasets_from_all_macros() -> None:
    rows = canonical_rows() + [
        {"source": "MedMCQA", "modality": "text", "relaxed_result": True},
        {"source": "PMC-VQA", "relaxed_result": True},
    ]

    result = utility_aggregate(rows)

    assert "MedMCQA" not in result
    assert "PMC-VQA" not in result
    assert result["text_macro"] == 0.5
    assert result["vision_macro"] == pytest.approx(1 / 3)
    assert result["overall_macro"] == pytest.approx(2 / 5)
    assert auxiliary_utility_aggregate(rows) == {"MedMCQA": 1.0, "PMC-VQA": 1.0}


def test_main_utility_rejects_missing_canonical_dataset() -> None:
    with pytest.raises(ValueError, match="PathVQA"):
        utility_aggregate(canonical_rows()[:-1])


def test_main_utility_rejects_string_boolean() -> None:
    rows = canonical_rows()
    rows[0]["relaxed_result"] = "false"
    with pytest.raises(TypeError, match="must be a boolean"):
        utility_aggregate(rows)
