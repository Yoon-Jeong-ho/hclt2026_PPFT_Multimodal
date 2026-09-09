from __future__ import annotations

import json
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .dedup import content_hash, file_sha256, semantic_hash, stable_hash
from .language import is_english_text

MAIN_MEDICAL_BENCHMARKS = frozenset(
    {"Pri-DDX", "Pri-NLICE", "VQA-RAD", "SLAKE", "PathVQA"}
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"input is empty: {path}")
    return rows


def _positive_int(row: Mapping[str, Any], name: str) -> int:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def canonicalize_row(row: Mapping[str, Any], *, stage: str, training: bool) -> dict[str, Any]:
    """Validate an adapted row and add the paper's stable identities.

    Dataset-specific parsing is deliberately outside this public boundary.  This
    function consumes canonical rows after source terms have been observed and
    before leakage removal, budget filtering, and sampling publication.
    """

    copied = json.loads(json.dumps(dict(row), ensure_ascii=False))
    for name in ("private_input", "final_answer", "target", "source", "original_split"):
        if not str(copied.get(name) or "").strip():
            raise ValueError(f"{name} must be non-empty")
    if copied.get("language") != "en":
        raise ValueError("paper manifests require language='en'")
    language_payload = [copied["private_input"], copied["final_answer"], copied["target"]]
    language_payload.extend(copied.get("options") or ())
    if not is_english_text("\n".join(map(str, language_payload))):
        raise ValueError("row fails the conservative English-script filter")

    options = copied.get("options")
    if options is not None:
        if not isinstance(options, list) or not options or any(not str(value).strip() for value in options):
            raise ValueError("options must be a non-empty list of strings")
        gold = copied.get("gold_option_index")
        if isinstance(gold, bool) or not isinstance(gold, int) or not 0 <= gold < len(options):
            raise ValueError("MCQA gold_option_index is outside structured options")
        final = str(copied["final_answer"]).strip().casefold()
        if final != str(options[gold]).strip().casefold():
            raise ValueError("MCQA final_answer does not match the gold option text")
        rendered = "\n".join(
            f"{chr(65 + index)}. {option}" for index, option in enumerate(options)
        )
        private_input = str(copied["private_input"])
        if (
            not private_input.startswith("Instruction: ")
            or "\n\nQuestion: " not in private_input
            or f"\n\nOptions:\n{rendered}" not in private_input
        ):
            raise ValueError("MCQA private_input must contain instruction, question, and every option")

    image_path = copied.get("image_path") or copied.get("image")
    inferred_modality = "image_text" if image_path else "text"
    modality = copied.get("modality", inferred_modality)
    if modality != inferred_modality or modality not in {"text", "image_text"}:
        raise ValueError("modality and flat image path fields disagree")
    copied["modality"] = modality
    if modality == "image_text":
        image_sha = str(copied.get("image_sha256") or "")
        if _SHA256.fullmatch(image_sha) is None:
            raise ValueError("image rows require a lowercase image_sha256")
        if not Path(str(image_path)).is_absolute():
            raise ValueError("image rows require an absolute external image_path")
        copied["image_path"] = str(image_path)
        copied["image"] = str(image_path)
        metadata = copied.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("image rows require metadata")
        visual_tokens = _positive_int(metadata, "native_visual_tokens")
        grid = metadata.get("image_grid_thw")
        if (
            not isinstance(grid, list)
            or len(grid) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in grid)
            or grid[0] != 1
            or grid[1] * grid[2] // 4 != visual_tokens
        ):
            raise ValueError("image_grid_thw is inconsistent with native_visual_tokens")

    input_length = _positive_int(copied, "input_token_length")
    target_length = _positive_int(copied, "target_token_length")
    copied["input_token_length"] = input_length
    copied["target_token_length"] = target_length
    copied["source_record_id"] = str(copied.get("source_record_id") or copied.get("example_id") or "")
    if not copied["source_record_id"]:
        raise ValueError("source_record_id or example_id is required")
    copied["example_id"] = str(
        copied.get("example_id")
        or stable_hash(
            {
                "source": copied["source"],
                "split": copied["original_split"],
                "record_id": copied["source_record_id"],
            }
        )
    )
    copied["semantic_id"] = semantic_hash(copied["private_input"], options)
    copied["content_id"] = content_hash(copied)
    copied["input_hash"] = stable_hash(copied["private_input"])
    copied["answer_hash"] = stable_hash(copied["final_answer"])
    copied["group_id"] = str(
        copied.get("group_id") or copied.get("image_sha256") or copied["example_id"]
    )
    copied["repeat_factor"] = (
        4 if stage == "stage2" and training and copied["source"] in MAIN_MEDICAL_BENCHMARKS else 1
    )
    return copied


def exclusion_reason(row: Mapping[str, Any]) -> str | None:
    if int(row["input_token_length"]) > 512:
        return "input_too_long"
    if int(row["target_token_length"]) > 512:
        return "target_too_long"
    if row["modality"] == "image_text":
        visual_tokens = int(row["metadata"]["native_visual_tokens"])
        if visual_tokens > 1024:
            return "visual_too_large"
    return None


def remove_train_eval_overlap(
    train: Sequence[Mapping[str, Any]], evaluation: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], Counter[str]]:
    evaluation_semantics = {str(row["semantic_id"]) for row in evaluation}
    evaluation_groups = {str(row["group_id"]) for row in evaluation if row.get("group_id")}
    kept: list[dict[str, Any]] = []
    removed: Counter[str] = Counter()
    for source_row in train:
        row = dict(source_row)
        if str(row["semantic_id"]) in evaluation_semantics:
            removed["evaluation_semantic_overlap"] += 1
        elif row.get("group_id") and str(row["group_id"]) in evaluation_groups:
            removed["evaluation_image_or_patient_group_overlap"] += 1
        else:
            kept.append(row)
    return kept, removed


def deduplicate_train(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    seen: set[tuple[str, str, str | None]] = set()
    kept: list[dict[str, Any]] = []
    removed: Counter[str] = Counter()
    for source_row in rows:
        row = dict(source_row)
        key = (
            str(row["source"]),
            str(row["semantic_id"]),
            str(row["image_sha256"]) if row["modality"] == "image_text" else None,
        )
        if key in seen:
            removed[str(row["source"])] += 1
        else:
            seen.add(key)
            kept.append(row)
    return kept, removed


def balanced_epoch_indices(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = 42,
    epoch: int = 0,
    homogeneous_block_size: int = 32,
    cost_bucket_size: int = 256,
) -> list[int]:
    """Delegate to the same ordering helper used by the training runtime."""

    from ppft_multimodal.training import deterministic_epoch_indices

    return deterministic_epoch_indices(
        rows,
        epoch=epoch,
        seed=seed,
        balance_modalities=True,
        homogeneous_block_size=homogeneous_block_size,
        cost_bucket_size=cost_bucket_size,
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def prepare_data(
    *,
    stage: str,
    train_path: Path,
    evaluation_path: Path,
    output_dir: Path,
    seed: int = 42,
    homogeneous_block_size: int = 32,
    cost_bucket_size: int = 256,
    expected_train_rows: int | None = None,
    expected_evaluation_rows: int | None = None,
    expected_balanced_exposures: int | None = None,
) -> dict[str, Any]:
    if stage not in {"stage1", "stage2"}:
        raise ValueError("stage must be stage1 or stage2")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    train = [canonicalize_row(row, stage=stage, training=True) for row in read_jsonl(train_path)]
    evaluation = [
        canonicalize_row(row, stage=stage, training=False) for row in read_jsonl(evaluation_path)
    ]
    train, overlap = remove_train_eval_overlap(train, evaluation)
    train, duplicates = deduplicate_train(train)

    exclusions: Counter[str] = Counter(overlap)
    exclusions.update({f"duplicate:{source}": count for source, count in duplicates.items()})
    retained_train: list[dict[str, Any]] = []
    retained_evaluation: list[dict[str, Any]] = []
    for row in train:
        reason = exclusion_reason(row)
        if reason:
            exclusions[f"train:{reason}"] += 1
        else:
            retained_train.append(row)
    for row in evaluation:
        reason = exclusion_reason(row)
        if reason:
            exclusions[f"evaluation:{reason}"] += 1
        else:
            retained_evaluation.append(row)

    if expected_train_rows is not None and len(retained_train) != expected_train_rows:
        raise ValueError(f"retained train rows {len(retained_train)} != expected {expected_train_rows}")
    if expected_evaluation_rows is not None and len(retained_evaluation) != expected_evaluation_rows:
        raise ValueError(
            f"retained evaluation rows {len(retained_evaluation)} != expected {expected_evaluation_rows}"
        )
    order = (
        balanced_epoch_indices(
            retained_train,
            seed=seed,
            homogeneous_block_size=homogeneous_block_size,
            cost_bucket_size=cost_bucket_size,
        )
        if stage == "stage2"
        else []
    )
    if expected_balanced_exposures is not None and len(order) != expected_balanced_exposures:
        raise ValueError(
            f"balanced exposures {len(order)} != expected {expected_balanced_exposures}"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        train_name = "train.jsonl"
        evaluation_name = "evaluation.jsonl"
        epoch_name = "epoch_indices.json"
        train_output = temporary / train_name
        evaluation_output = temporary / evaluation_name
        _write_jsonl(train_output, retained_train)
        _write_jsonl(evaluation_output, retained_evaluation)
        if stage == "stage2":
            (temporary / epoch_name).write_text(
                json.dumps(order, separators=(",", ":")) + "\n", encoding="utf-8"
            )
        modalities = Counter(str(row["modality"]) for row in retained_train)
        effective: Counter[str] = Counter()
        for row in retained_train:
            effective[str(row["modality"])] += int(row["repeat_factor"])
        report: dict[str, Any] = {
            "schema_version": 1,
            "status": "passed",
            "stage": stage,
            "seed": seed,
            "limits": {
                "kure_input_with_specials": 512,
                "qwen_target_with_eos": 512,
                "native_visual_tokens": 1024,
            },
            "retained": {
                "train_rows": len(retained_train),
                "evaluation_rows": len(retained_evaluation),
                "train_modalities": dict(sorted(modalities.items())),
                "effective_before_balance": dict(sorted(effective.items())),
                "balanced_exposures": len(order) if stage == "stage2" else None,
            },
            "excluded": dict(sorted(exclusions.items())),
            "inputs": {
                "train": {"sha256": file_sha256(train_path)},
                "evaluation": {"sha256": file_sha256(evaluation_path)},
            },
            "outputs": {
                train_name: {"sha256": file_sha256(train_output)},
                evaluation_name: {"sha256": file_sha256(evaluation_output)},
            },
            "preservation": {
                "whole_row_filtering": True,
                "text_truncation": False,
                "image_resize_override": False,
                "raw_payloads_embedded_in_report": False,
            },
        }
        if stage == "stage2":
            epoch_path = temporary / epoch_name
            report["outputs"][epoch_name] = {"sha256": file_sha256(epoch_path)}
            report["sampling"] = {
                "homogeneous_block_size": homogeneous_block_size,
                "cost_bucket_size": cost_bucket_size,
                "modality_ratio": {"text": 0.5, "image_text": 0.5},
            }
        (temporary / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(output_dir)
        return report
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
