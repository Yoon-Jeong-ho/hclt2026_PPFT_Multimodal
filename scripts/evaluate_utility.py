#!/usr/bin/env python3
"""Aggregate answer-content accuracy for the five paper benchmarks."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ppft_multimodal.evaluation.aggregate import utility_aggregate

PAPER_COUNTS = {
    "PathVQA": 12949,
    "Pri-DDX": 1549,
    "Pri-NLICE": 650,
    "SLAKE": 2114,
    "VQA-RAD": 472,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true", help="development only; output is not a paper result")
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for path in args.predictions
        for line in path.open(encoding="utf-8")
        if line.strip()
    ]
    identifiers = [str(row["example_id"]) for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("prediction inputs contain duplicate example_id values")
    checkpoints = {row.get("checkpoint_sha256") for row in rows}
    if len(checkpoints) != 1:
        raise ValueError("predictions must come from exactly one checkpoint")
    checkpoint = next(iter(checkpoints))
    if not isinstance(checkpoint, str) or len(checkpoint) != 64 or any(
        character not in "0123456789abcdef" for character in checkpoint
    ):
        raise ValueError("predictions require a non-empty checkpoint SHA-256")
    counts = Counter(str(row["source"]) for row in rows)
    complete = dict(counts) == PAPER_COUNTS and all(
        row.get("evaluation_scope") == "complete" and row.get("max_new_tokens") == 512 for row in rows
    )
    if not complete and not args.allow_partial:
        raise ValueError(
            f"paper utility requires the exact 17,734-row five-source panel; observed {dict(sorted(counts.items()))}"
        )
    scores = utility_aggregate(rows)
    payload = {
        "accuracy": scores,
        "accuracy_percent": {name: round(value * 100.0, 2) for name, value in scores.items()},
        "rows": len(rows),
        "checkpoint_sha256": checkpoint,
        "source_counts": dict(sorted(counts.items())),
        "paper_scope_complete": complete,
        "benchmark_only": not complete,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
