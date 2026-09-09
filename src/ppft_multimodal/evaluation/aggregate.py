from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

MAIN_UTILITY_TEXT_DATASETS = ("Pri-DDX", "Pri-NLICE")
MAIN_UTILITY_VISION_DATASETS = ("VQA-RAD", "SLAKE", "PathVQA")
MAIN_UTILITY_DATASETS = MAIN_UTILITY_TEXT_DATASETS + MAIN_UTILITY_VISION_DATASETS


def dataset_macro(scores: Mapping[str, float]) -> float:
    return sum(scores.values()) / len(scores) if scores else 0.0


def utility_aggregate(rows: Iterable[Mapping[str, object]]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        source = str(row["source"])
        if source not in MAIN_UTILITY_DATASETS:
            continue
        relaxed_result = row["relaxed_result"]
        if not isinstance(relaxed_result, bool):
            raise TypeError(f"relaxed_result must be a boolean for {source}; got {type(relaxed_result).__name__}")
        values[source].append(1.0 if relaxed_result else 0.0)
    missing = [source for source in MAIN_UTILITY_DATASETS if not values[source]]
    if missing:
        raise ValueError(f"main utility predictions are missing canonical datasets: {missing}")
    datasets = {source: sum(values[source]) / len(values[source]) for source in MAIN_UTILITY_DATASETS}
    text = {source: datasets[source] for source in MAIN_UTILITY_TEXT_DATASETS}
    vision = {source: datasets[source] for source in MAIN_UTILITY_VISION_DATASETS}
    return {
        **datasets,
        "text_macro": dataset_macro(text),
        "vision_macro": dataset_macro(vision),
        "overall_macro": dataset_macro(datasets),
    }


def auxiliary_utility_aggregate(rows: Iterable[Mapping[str, object]]) -> dict[str, float]:
    """Return per-dataset scores excluded from the five-dataset main utility table."""

    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        source = str(row["source"])
        if source in MAIN_UTILITY_DATASETS:
            continue
        relaxed_result = row["relaxed_result"]
        if not isinstance(relaxed_result, bool):
            raise TypeError(f"relaxed_result must be a boolean for {source}; got {type(relaxed_result).__name__}")
        values[source].append(1.0 if relaxed_result else 0.0)
    return {source: sum(values[source]) / len(values[source]) for source in sorted(values)}
