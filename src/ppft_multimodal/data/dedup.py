from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .language import normalize_whitespace


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text(value: object) -> str:
    return normalize_whitespace(value).casefold()


def canonical_mcqa_stem(private_input: object, options: Sequence[object]) -> str:
    """Remove rendered option order while preserving the question and context."""

    text = str(private_input)
    rendered = "\n".join(f"{chr(65 + index)}. {option}" for index, option in enumerate(options))
    marker = f"\n\nOptions:\n{rendered}"
    if marker not in text:
        if "\n\nOptions:\n" in text or text.startswith("Instruction: "):
            raise ValueError("MCQA private_input does not match its structured options")
        return text
    head, suffix = text.rsplit(marker, 1)
    if suffix and not suffix.startswith("\n\nContext: "):
        raise ValueError("MCQA private_input has an unrecognized suffix after options")
    question_marker = "\n\nQuestion: "
    if question_marker in head:
        instruction, question = head.split(question_marker, 1)
        if not instruction.startswith("Instruction: ") or not question:
            raise ValueError("MCQA private_input has malformed instruction/question sections")
    else:
        question = head.removeprefix("Question: ")
    if not question:
        raise ValueError("MCQA question stem is empty")
    context = suffix.removeprefix("\n\nContext: ").strip() if suffix else ""
    return f"{question}\nContext: {context}" if context else question


def semantic_hash(
    private_input: object,
    options: Sequence[object] | None = None,
    *,
    order_sensitive: bool = False,
) -> str:
    if options:
        private_input = canonical_mcqa_stem(private_input, options)
    normalized_options = [normalized_text(option) for option in options or ()]
    if not order_sensitive:
        normalized_options.sort()
    return stable_hash(
        {"question": normalized_text(private_input), "options": normalized_options}
    )


def content_hash(row: Mapping[str, Any]) -> str:
    return stable_hash(
        {
            "semantic_id": str(row["semantic_id"]),
            "modality": str(row["modality"]),
            "image_sha256": str(row["image_sha256"]) if row.get("image_sha256") else None,
        }
    )
