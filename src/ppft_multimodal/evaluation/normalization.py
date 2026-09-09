from __future__ import annotations

import re
import string
import unicodedata

_SPACE = re.compile(r"\s+")
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_EDGE_PUNCT = re.compile(r"^[\W_]+|[\W_]+$")


def nfkc_casefold(value: object) -> str:
    return unicodedata.normalize("NFKC", "" if value is None else str(value)).casefold()


def normalize_medical_answer(value: object) -> str:
    """Common VQA normalization; aliases (including abbreviations) stay distinct."""

    text = _SPACE.sub(" ", nfkc_casefold(value).strip())
    previous = None
    while text != previous:
        previous = text
        text = _EDGE_PUNCT.sub("", text).strip()
    return _SPACE.sub(" ", text)


def normalize_answer(value: object) -> str:
    """Official SQuAD v1-style English normalization."""

    text = nfkc_casefold(value)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = _ARTICLES.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def valid_alias(value: object, *, medical: bool = True) -> bool:
    normalized = normalize_medical_answer(value) if medical else normalize_answer(value)
    return bool(normalized and any(ch.isalnum() for ch in normalized))
