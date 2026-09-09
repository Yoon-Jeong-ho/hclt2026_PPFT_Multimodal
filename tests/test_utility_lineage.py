from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts import evaluate_utility


def prediction_rows() -> list[dict[str, object]]:
    return [
        {"example_id": f"{source}-{index}", "source": source, "relaxed_result": True,
         "evaluation_scope": "complete", "checkpoint_sha256": "a" * 64, "max_new_tokens": 512}
        for source, count in evaluate_utility.PAPER_COUNTS.items() for index in range(count)
    ]


@pytest.mark.parametrize("second_hash", ["b" * 64, ""])
def test_full_panel_rejects_mixed_or_missing_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, second_hash: str,
) -> None:
    rows = prediction_rows()
    rows[-1]["checkpoint_sha256"] = second_hash
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "metrics.json"
    monkeypatch.setattr(sys, "argv", ["evaluate_utility", str(predictions), "--output", str(output)])
    with pytest.raises(ValueError, match="checkpoint"):
        evaluate_utility.main()
    assert not output.exists()


def test_complete_utility_records_its_single_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("".join(json.dumps(row) + "\n" for row in prediction_rows()))
    output = tmp_path / "metrics.json"
    monkeypatch.setattr(sys, "argv", ["evaluate_utility", str(predictions), "--output", str(output)])
    evaluate_utility.main()
    report = json.loads(output.read_text())
    assert report["checkpoint_sha256"] == "a" * 64
    assert report["paper_scope_complete"] is True
    assert report["rows"] == 17734


def test_full_panel_rejects_nonpaper_generation_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = prediction_rows()
    for row in rows:
        row["max_new_tokens"] = 1
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "metrics.json"
    monkeypatch.setattr(sys, "argv", ["evaluate_utility", str(predictions), "--output", str(output)])
    with pytest.raises(ValueError, match="paper utility"):
        evaluate_utility.main()
    assert not output.exists()


@pytest.mark.parametrize("identifier", ["", "same"])
def test_training_manifest_rejects_empty_or_duplicate_ids(
    tmp_path: Path, identifier: str,
) -> None:
    from scripts.training_common import read_rows

    row = {"example_id": identifier, "source": "synthetic", "modality": "text",
           "private_input": "synthetic question", "target": "synthetic answer"}
    manifest = tmp_path / "train.jsonl"
    manifest.write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError, match="example_id"):
        read_rows(manifest)


@pytest.mark.parametrize("cap, expected_scope", [(1, "partial"), (512, "complete")])
def test_generation_records_cap_and_scope_without_loading_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cap: int, expected_scope: str,
) -> None:
    from types import SimpleNamespace

    from scripts import generate_predictions

    fake_batch = SimpleNamespace(to=lambda _device: None)
    fake_model = SimpleNamespace(
        prepare_batch=lambda **_kwargs: fake_batch,
        generate=lambda *_args, **_kwargs: [[7]],
        qwen_processor=SimpleNamespace(batch_decode=lambda *_args, **_kwargs: ["synthetic answer"]),
    )
    monkeypatch.setattr(generate_predictions, "load_victim", lambda *_args, **_kwargs: (fake_model, None))
    monkeypatch.setattr(generate_predictions, "hash_path", lambda _path: "c" * 64)
    manifest = tmp_path / "evaluation.jsonl"
    manifest.write_text(json.dumps({"example_id": "synthetic-id", "source": "PathVQA",
                                    "private_input": "synthetic question", "final_answer": "synthetic answer"}) + "\n")
    output = tmp_path / "predictions.jsonl"
    monkeypatch.setattr(sys, "argv", ["generate_predictions", "--config", "unused.yaml", "--checkpoint", "unused",
                                     "--evaluation", str(manifest), "--output", str(output),
                                     "--max-new-tokens", str(cap), "--device", "cpu"])
    generate_predictions.main()
    row = json.loads(output.read_text())
    summary = json.loads(output.with_suffix(".jsonl.summary.json").read_text())
    assert row["max_new_tokens"] == summary["max_new_tokens"] == cap
    assert row["evaluation_scope"] == summary["evaluation_scope"] == expected_scope
