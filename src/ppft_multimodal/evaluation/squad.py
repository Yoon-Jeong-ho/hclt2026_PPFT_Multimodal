from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from .normalization import normalize_answer, valid_alias


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not prediction_tokens or not gold_tokens:
        return float(prediction_tokens == gold_tokens)
    common = Counter(prediction_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def relaxed_containment(prediction: str, aliases: Iterable[str]) -> bool:
    normalized_prediction = normalize_answer(prediction)
    for alias in aliases:
        normalized_alias = normalize_answer(alias)
        if valid_alias(alias, medical=False) and normalized_alias in normalized_prediction:
            return True
    return False


def score_squad(prediction: str, aliases: Iterable[str]) -> dict[str, float | bool]:
    golds = [gold for gold in aliases if valid_alias(gold, medical=False)]
    if not golds:
        raise ValueError("at least one non-empty answer alias is required")
    return {
        "exact_match": max(exact_match(prediction, gold) for gold in golds),
        "token_f1": max(token_f1(prediction, gold) for gold in golds),
        "relaxed_correct": relaxed_containment(prediction, golds),
    }
