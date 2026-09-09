from __future__ import annotations

import json
from pathlib import Path

import pytest

from ppft_multimodal.data.preparation import (
    balanced_epoch_indices,
    canonicalize_row,
    prepare_data,
)


def row(
    identifier: str,
    *,
    source: str = "support",
    modality: str = "text",
    private_input: str | None = None,
    visual_tokens: int = 16,
) -> dict:
    payload = {
        "example_id": identifier,
        "source_record_id": identifier,
        "source": source,
        "original_split": "train",
        "language": "en",
        "modality": modality,
        "private_input": private_input or f"Question {identifier} has enough English letters?",
        "final_answer": "yes",
        "target": "yes",
        "input_token_length": 10,
        "target_token_length": 2,
        "metadata": {},
    }
    if modality == "image_text":
        payload.update(
            {
                "image_path": f"/external/{identifier}.png",
                "image_sha256": identifier[0].lower() * 64,
                "metadata": {
                    "native_visual_tokens": visual_tokens,
                    "image_grid_thw": [1, 2, 2 * visual_tokens],
                },
            }
        )
    return payload


def write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")


def test_stage2_order_matches_paper_padding_rule() -> None:
    rows = [
        {"modality": "text", "repeat_factor": 4},
        {"modality": "image_text", "repeat_factor": 1},
        {"modality": "image_text", "repeat_factor": 1},
    ]
    first = balanced_epoch_indices(rows, seed=42, homogeneous_block_size=2, cost_bucket_size=0)
    second = balanced_epoch_indices(rows, seed=42, homogeneous_block_size=2, cost_bucket_size=0)
    assert first == second
    assert len(first) == 8
    assert sum(rows[index]["modality"] == "text" for index in first) == 4


def test_mcqa_requires_structured_answer_and_complete_rendering() -> None:
    valid = row("a")
    valid.update(
        {
            "private_input": (
                "Instruction: Choose the answer.\n\nQuestion: Which is correct?"
                "\n\nOptions:\nA. first\nB. second"
            ),
            "options": ["first", "second"],
            "gold_option_index": 1,
            "final_answer": "second",
            "target": "second",
        }
    )
    assert canonicalize_row(valid, stage="stage2", training=True)["semantic_id"]
    invalid = {**valid, "private_input": valid["private_input"].replace("B. second", "B. hidden")}
    with pytest.raises(ValueError, match="every option"):
        canonicalize_row(invalid, stage="stage2", training=True)


def test_publication_removes_leakage_duplicates_and_visual_outliers(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    evaluation = tmp_path / "evaluation.jsonl"
    output = tmp_path / "published"
    leaked = row("leaked", private_input="The same English medical question")
    duplicate = row("duplicate", private_input="A retained English support question")
    oversized = row("b", modality="image_text", visual_tokens=1025)
    image = row("c", source="PathVQA", modality="image_text")
    write(train, [leaked, duplicate, {**duplicate, "example_id": "duplicate-two"}, oversized, image])
    write(
        evaluation,
        [{**leaked, "example_id": "eval", "source_record_id": "eval", "original_split": "test"}],
    )

    report = prepare_data(
        stage="stage2",
        train_path=train,
        evaluation_path=evaluation,
        output_dir=output,
        homogeneous_block_size=2,
        cost_bucket_size=0,
        expected_train_rows=2,
        expected_evaluation_rows=1,
        expected_balanced_exposures=8,
    )

    assert report["excluded"] == {
        "duplicate:support": 1,
        "evaluation_semantic_overlap": 1,
        "train:visual_too_large": 1,
    }
    assert report["retained"]["train_modalities"] == {"image_text": 1, "text": 1}
    assert "private_input" not in json.dumps(report)
    assert not any(str(tmp_path) in json.dumps(value) for value in report["inputs"].values())
    published = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert [item["repeat_factor"] for item in published] == [1, 4]
