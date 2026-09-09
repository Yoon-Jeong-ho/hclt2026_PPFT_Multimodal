from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

from ppft_multimodal.data.language import normalize_whitespace

from .normalization import normalize_answer, normalize_medical_answer, valid_alias
from .squad import relaxed_containment as squad_relaxed_containment


@dataclass(frozen=True)
class FirstMatchResult:
    predicted_index: int | None
    matched_text: str | None
    match_start: int | None
    match_kind: str | None
    ambiguous: bool
    valid: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _bounded_literal(text: str) -> re.Pattern[str]:
    escaped = re.escape(text)
    left = r"(?<!\w)" if text and text[0].isalnum() else ""
    right = r"(?!\w)" if text and text[-1].isalnum() else ""
    return re.compile(left + escaped + right, re.IGNORECASE)


def _label_patterns(label: str) -> list[tuple[re.Pattern[str], str]]:
    escaped = re.escape(label)
    return [
        (re.compile(rf"\({escaped}\)", re.IGNORECASE), "label"),
        (re.compile(rf"(?<!\w){escaped}[.:](?!\w)", re.IGNORECASE), "label"),
        # Korean answer-first targets render a label as ``정답은 B다.``.
        # Accept only the explicit copular ending; particles such as ``A는``
        # remain non-matches so a later distractor discussion is not promoted.
        (re.compile(rf"(?<![A-Za-z0-9_]){escaped}(?=이?다(?:[.!?\s]|$))"), "label"),
        # A bare lowercase "a" is normally an English article, not option A.
        # Bare labels therefore remain case-sensitive while explicit decorated
        # label forms are case-insensitive.
        (re.compile(rf"(?<!\w){escaped}(?!\w)"), "label"),
    ]


def relaxed_first_match(
    generation: str,
    options: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
    option_aliases: Sequence[Iterable[str]] | None = None,
    allow_labels: bool = True,
) -> FirstMatchResult:
    """Select the earliest explicit option mention.

    At one start position the longest exact span wins.  If equal-length spans
    still identify different options, the response is ambiguous and invalid.
    Label matching uses word boundaries, preventing matches inside words.  Set
    ``allow_labels=False`` when the evaluated contract requires generated
    option text rather than an A/B/C/D shortcut.
    """

    if not options:
        raise ValueError("options must be non-empty")
    if labels is None:
        labels = tuple(chr(65 + index) for index in range(len(options)))
    if len(labels) != len(options):
        raise ValueError("labels and options must have equal length")
    if option_aliases is not None and len(option_aliases) != len(options):
        raise ValueError("option_aliases and options must have equal length")

    # Search a single NFKC-normalized character stream so option and label
    # offsets are comparable.  Keep case intact here: decorated labels are
    # case-insensitive, while bare lowercase articles must not become option A.
    text = unicodedata.normalize("NFKC", normalize_whitespace(generation))
    candidates: list[tuple[int, int, int, str, str]] = []
    normalized_labels = {normalize_medical_answer(label) for label in labels}
    for index, (label, option) in enumerate(zip(labels, options, strict=True)):
        if allow_labels:
            for pattern, kind in _label_patterns(str(label)):
                for match in pattern.finditer(text):
                    candidates.append((match.start(), -(match.end() - match.start()), index, match.group(), kind))
        aliases = [(option, True), *((alias, False) for alias in (option_aliases[index] if option_aliases else ()))]
        seen: set[str] = set()
        for alias, is_canonical_option in aliases:
            normalized = normalize_medical_answer(alias)
            if not allow_labels and not is_canonical_option and normalized in normalized_labels:
                continue
            if normalized in seen or not valid_alias(normalized):
                continue
            seen.add(normalized)
            for match in _bounded_literal(normalized).finditer(text):
                candidates.append((match.start(), -(match.end() - match.start()), index, match.group(), "option_text"))
    if not candidates:
        return FirstMatchResult(None, None, None, None, False, False)
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    earliest = candidates[0][0]
    longest_negative = min(item[1] for item in candidates if item[0] == earliest)
    finalists = [item for item in candidates if item[0] == earliest and item[1] == longest_negative]
    indices = {item[2] for item in finalists}
    if len(indices) != 1:
        first = finalists[0]
        return FirstMatchResult(None, first[3], earliest, first[4], True, False)
    selected = finalists[0]
    return FirstMatchResult(selected[2], selected[3], selected[0], selected[4], False, True)


def score_mcqa(
    generation: str,
    options: Sequence[str],
    gold_index: int,
    *,
    labels: Sequence[str] | None = None,
    option_aliases: Sequence[Iterable[str]] | None = None,
    allow_labels: bool = False,
) -> dict[str, object]:
    if not 0 <= gold_index < len(options):
        raise ValueError("gold_index outside options")
    text_result = relaxed_first_match(
        generation,
        options,
        labels=labels,
        option_aliases=option_aliases,
        allow_labels=False,
    )
    legacy_result = relaxed_first_match(
        generation,
        options,
        labels=labels,
        option_aliases=option_aliases,
        allow_labels=True,
    )
    result = legacy_result if allow_labels else text_result
    first_line = generation.strip().splitlines()[0] if generation.strip() else ""
    strict = normalize_medical_answer(first_line) == normalize_medical_answer(options[gold_index])
    option_text_correct = (
        text_result.valid
        and text_result.match_kind == "option_text"
        and text_result.predicted_index == gold_index
    )
    legacy_first_label_correct = (
        legacy_result.valid
        and legacy_result.match_kind == "label"
        and legacy_result.predicted_index == gold_index
    )
    return {
        **result.to_dict(),
        "relaxed_correct": result.valid and result.predicted_index == gold_index,
        "strict_correct": strict,
        "legacy_relaxed_correct": legacy_result.valid and legacy_result.predicted_index == gold_index,
        "label_only_correct": legacy_first_label_correct and not option_text_correct,
        "option_text_correct": option_text_correct,
    }


def relaxed_containment(prediction: str, aliases: Iterable[str], *, medical: bool = True) -> bool:
    if not medical:
        return squad_relaxed_containment(prediction, aliases)
    normalized_prediction = normalize_medical_answer(prediction)
    for alias in aliases:
        normalized_alias = normalize_medical_answer(alias)
        if valid_alias(normalized_alias) and _bounded_literal(normalized_alias).search(normalized_prediction):
            return True
    return False


def first_contained_alias(
    prediction: str, aliases: Iterable[str], *, medical: bool = True
) -> str | None:
    """Return the provided alias whose normalized span occurs first.

    Ties at the same position prefer the longer normalized alias, matching the
    MCQA first-match rule rather than substituting the canonical final answer.
    """

    normalizer = normalize_medical_answer if medical else normalize_answer
    normalized_prediction = normalizer(prediction)
    candidates: list[tuple[int, int, int, str]] = []
    for index, alias in enumerate(aliases):
        value = str(alias)
        normalized = normalizer(value)
        if not valid_alias(normalized, medical=medical):
            continue
        match = _bounded_literal(normalized).search(normalized_prediction)
        if match is not None:
            candidates.append((match.start(), -len(normalized), index, value))
    return min(candidates)[3] if candidates else None
