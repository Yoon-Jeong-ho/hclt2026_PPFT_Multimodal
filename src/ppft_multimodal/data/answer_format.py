from __future__ import annotations

import re

from .language import normalize_language, normalize_whitespace

_LABEL = re.compile(r"^[A-Z]$")
_DECLARATIVE_KO = re.compile(r"(?:다|요|임|함|됨|있음|없음)[.!?]?$")


def _strip_terminal_punctuation(text: str) -> str:
    return text.rstrip().rstrip(".!?").rstrip()


def _english_answer(answer: str) -> str:
    return f"The answer is {_strip_terminal_punctuation(answer)}."


def _korean_answer(answer: str) -> str:
    clean = normalize_whitespace(answer)
    if _LABEL.fullmatch(clean.upper()):
        return f"정답은 {clean.upper()}다."
    if _DECLARATIVE_KO.search(clean):
        return f"정답은 {clean}"
    return f"정답은 {_strip_terminal_punctuation(clean)}이다."


def format_answer_target(
    final_answer: object,
    *,
    language: str | None,
    reasoning: object | None = None,
) -> str:
    """Render the answer-first target while preserving supplied reasoning verbatim-ish.

    The structured answer is authoritative.  Reasoning is normalized only for
    whitespace and is never mined for an answer (PPFT_MCQA provenance pattern).
    """

    answer = normalize_whitespace(final_answer)
    if not answer:
        raise ValueError("final_answer must be non-empty")
    lang = normalize_language(language, text=answer)
    prefix = _korean_answer(answer) if lang == "ko" else _english_answer(answer)
    rationale = normalize_whitespace(reasoning)
    if not rationale:
        return prefix
    label = "풀이" if lang == "ko" else "Reasoning"
    return f"{prefix}\n{label}: {rationale}"


def answer_prefix_matches(target: str, language: str) -> bool:
    expected = "정답은" if normalize_language(language) == "ko" else "The answer is"
    return normalize_whitespace(target).startswith(expected)
