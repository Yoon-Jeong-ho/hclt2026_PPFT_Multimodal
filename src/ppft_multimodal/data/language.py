from __future__ import annotations

import re
import unicodedata

_SPACE = re.compile(r"\s+")
_DISALLOWED_SCRIPT = re.compile(
    "[\u1100-\u11ff\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af"
    "\u0400-\u052f\u0600-\u06ff]"
)


def normalize_whitespace(value: object) -> str:
    """NFKC-normalize and collapse whitespace without changing letter case."""

    text = "" if value is None else str(value)
    return _SPACE.sub(" ", unicodedata.normalize("NFKC", text).strip())


def is_english_text(value: object) -> bool:
    """Apply the conservative script filter used for the paper's English runs."""

    text = str("" if value is None else value).strip()
    if not text or _DISALLOWED_SCRIPT.search(text):
        return False
    alphabetic = [character for character in text if character.isalpha()]
    if len(alphabetic) < 3:
        return False
    latin = sum("a" <= character.casefold() <= "z" for character in alphabetic)
    return latin / len(alphabetic) >= 0.80


def normalize_language(value: object, *, text: object | None = None) -> str:
    """Normalize the language tag used by canonical paper manifests."""

    language = normalize_whitespace(value).casefold().replace("_", "-")
    aliases = {"en": "en", "eng": "en", "english": "en"}
    if language in aliases:
        return aliases[language]
    if not language and text is not None and is_english_text(text):
        return "en"
    raise ValueError(f"unsupported language for the English paper corpus: {value!r}")
