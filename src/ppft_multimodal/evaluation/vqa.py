from __future__ import annotations

import re
import string
from collections.abc import Iterable

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_ANSWER_FIRST = re.compile(r"^\s*the\s+answer\s+is\s+(.+)$", re.IGNORECASE)
_NUMBER_WORDS = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
_COMMA_BETWEEN_DIGITS = re.compile(r"(?<=\d),(?=\d)")
_PERIOD_NOT_DECIMAL = re.compile(r"(?<!\d)\.|\.(?!\d)")


def normalize_vqa_answer(value: str) -> str:
    """Apply official VQA punctuation, digit, and article normalization.

    The source annotator answers are retained with multiplicity; this function
    intentionally does not deduplicate them before consensus scoring.
    """

    normalized = value.replace("\n", " ").replace("\t", " ").strip().lower()
    normalized = _COMMA_BETWEEN_DIGITS.sub("", normalized)
    normalized = _PERIOD_NOT_DECIMAL.sub("", normalized)
    normalized = normalized.translate(
        {ord(mark): " " for mark in string.punctuation if mark != "."}
    )
    normalized = " ".join(_NUMBER_WORDS.get(token, token) for token in normalized.split())
    normalized = _ARTICLES.sub(" ", normalized)
    return " ".join(normalized.split())


def vqa_consensus_accuracy(prediction: str, annotator_answers: Iterable[str]) -> float:
    """Return official leave-one-annotator-out VQA consensus accuracy."""

    answers = [normalize_vqa_answer(answer) for answer in annotator_answers]
    answers = [answer for answer in answers if answer]
    if not answers:
        raise ValueError("VQA consensus scoring requires annotator answers")
    first_line = prediction.strip().splitlines()[0] if prediction.strip() else ""
    match = _ANSWER_FIRST.match(first_line)
    candidate = match.group(1).strip() if match else first_line
    if candidate.endswith("."):
        candidate = candidate[:-1]
    predicted = normalize_vqa_answer(candidate)
    scores = []
    for held_out in range(len(answers)):
        matching_others = sum(
            answer == predicted for index, answer in enumerate(answers) if index != held_out
        )
        scores.append(min(1.0, matching_others / 3.0))
    return sum(scores) / len(scores)


__all__ = ["normalize_vqa_answer", "vqa_consensus_accuracy"]
