from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .answer_format import format_answer_target
from .language import normalize_language, normalize_whitespace


@dataclass(frozen=True)
class CanonicalRecord:
    private_input: str
    final_answer: str
    source: str
    original_split: str
    language: str
    image: str | None = None
    reasoning: str | None = None
    options: tuple[str, ...] | None = None
    gold_option_index: int | None = None
    answer_aliases: tuple[str, ...] = ()
    source_record_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    target_style: Literal["legacy_qa", "assistant_response"] = "legacy_qa"

    def __post_init__(self) -> None:
        for name in ("private_input", "final_answer", "source", "original_split"):
            if not normalize_whitespace(getattr(self, name)):
                raise ValueError(f"{name} must be non-empty")
        object.__setattr__(self, "language", normalize_language(self.language, text=self.private_input))
        if self.options is None and self.gold_option_index is not None:
            raise ValueError("gold_option_index requires options")
        if self.options is not None:
            if not self.options or any(not normalize_whitespace(x) for x in self.options):
                raise ValueError("options must contain non-empty strings")
            if self.gold_option_index is None or not 0 <= self.gold_option_index < len(self.options):
                raise ValueError("gold_option_index is outside options")
            if (
                normalize_whitespace(self.options[self.gold_option_index]).casefold()
                != normalize_whitespace(self.final_answer).casefold()
            ):
                aliases = {normalize_whitespace(x).casefold() for x in self.answer_aliases}
                if normalize_whitespace(self.options[self.gold_option_index]).casefold() not in aliases:
                    raise ValueError("final answer/aliases do not contain the gold option text")

    @property
    def modality(self) -> Literal["text", "image_text"]:
        return "image_text" if self.image else "text"

    @property
    def target(self) -> str:
        if self.target_style == "assistant_response":
            return self.final_answer.replace("\r\n", "\n").replace("\r", "\n").strip()
        return format_answer_target(self.final_answer, language=self.language, reasoning=self.reasoning)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update({"modality": self.modality, "target": self.target})
        return payload


@dataclass(frozen=True)
class ManifestRecord:
    example_id: str
    semantic_id: str
    source: str
    original_split: str
    language: str
    modality: Literal["text", "image_text"]
    input_hash: str
    answer_hash: str
    filtering_result: Literal["included", "excluded"]
    repeat_factor: int = 1
    image_path: str | None = None
    image_sha256: str | None = None
    option_order_semantic_id: str | None = None
    has_reasoning: bool = False
    answer_type: str = "open"
    number_of_choices: int = 0
    input_token_length: int | None = None
    target_token_length: int | None = None
    exclusion_reason: str | None = None

    def __post_init__(self) -> None:
        if self.repeat_factor < 1:
            raise ValueError("repeat_factor must be >= 1")
        if self.filtering_result == "excluded" and not self.exclusion_reason:
            raise ValueError("excluded records require exclusion_reason")
        if self.modality == "image_text" and not self.image_path:
            raise ValueError("image_text records require image_path")

    @property
    def effective_examples(self) -> int:
        return self.repeat_factor if self.filtering_result == "included" else 0
