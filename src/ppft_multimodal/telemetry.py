"""Distributed aggregation for compact training-window metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import torch


@dataclass
class TrainingWindowAccumulator:
    """Accumulate metrics over one complete gradient-accumulation window.

    Loss follows the distributed optimization objective: one scalar per
    rank-local microbatch. Token accuracy is weighted by supervised target
    tokens. Detached tensors remain on device until ``snapshot`` so logging
    does not synchronize every microbatch.
    """

    loss_sums: dict[str, torch.Tensor] = field(default_factory=dict)
    loss_counts: dict[str, int] = field(default_factory=dict)
    correct_tokens: torch.Tensor | None = None
    target_tokens: int = 0
    elapsed_seconds: float = 0.0
    processed_examples: int = 0
    microbatches: int = 0

    def add(
        self,
        *,
        loss: torch.Tensor,
        target_token_accuracy: torch.Tensor,
        target_token_count: int,
        rows: Sequence[Mapping[str, Any]],
        elapsed_seconds: float,
        processed_examples: int,
    ) -> None:
        if loss.numel() != 1 or target_token_accuracy.numel() != 1:
            raise ValueError("window telemetry requires scalar loss and accuracy")
        if not rows or target_token_count < 1 or elapsed_seconds < 0 or processed_examples < 1:
            raise ValueError("invalid gradient-window telemetry sample")
        detached_loss = loss.detach().float()
        keys = {"loss/overall"}
        keys.update(f"loss/modality/{row.get('modality', 'text')}" for row in rows)
        keys.update(f"loss/language/{row.get('language', 'unknown')}" for row in rows)
        for key in keys:
            if key in self.loss_sums:
                self.loss_sums[key] = self.loss_sums[key] + detached_loss
            else:
                self.loss_sums[key] = detached_loss.clone()
            self.loss_counts[key] = self.loss_counts.get(key, 0) + 1
        correct = target_token_accuracy.detach().float() * target_token_count
        self.correct_tokens = correct.clone() if self.correct_tokens is None else self.correct_tokens + correct
        self.target_tokens += target_token_count
        self.elapsed_seconds += elapsed_seconds
        self.processed_examples += processed_examples
        self.microbatches += 1

    def snapshot(self) -> dict[str, Any]:
        if not self.microbatches or self.correct_tokens is None:
            raise ValueError("cannot snapshot an empty gradient window")
        return {
            "loss_sums": {key: float(value.item()) for key, value in self.loss_sums.items()},
            "loss_counts": dict(self.loss_counts),
            "correct_tokens": float(self.correct_tokens.item()),
            "target_tokens": self.target_tokens,
            "elapsed_seconds": self.elapsed_seconds,
            "processed_examples": self.processed_examples,
            "microbatches": self.microbatches,
        }

    def reset(self) -> None:
        self.loss_sums.clear()
        self.loss_counts.clear()
        self.correct_tokens = None
        self.target_tokens = 0
        self.elapsed_seconds = 0.0
        self.processed_examples = 0
        self.microbatches = 0


def aggregate_training_window_snapshots(snapshots: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Combine one completed local window from every distributed rank."""

    if not snapshots:
        raise ValueError("at least one training-window snapshot is required")
    loss_sums: dict[str, float] = {}
    loss_counts: dict[str, int] = {}
    correct_tokens = 0.0
    target_tokens = 0
    processed_examples = 0
    elapsed_seconds: list[float] = []
    microbatches: list[int] = []
    for snapshot in snapshots:
        sums = cast(Mapping[str, Any], snapshot.get("loss_sums", {}))
        counts = cast(Mapping[str, Any], snapshot.get("loss_counts", {}))
        if set(sums) != set(counts) or "loss/overall" not in sums:
            raise ValueError("malformed training-window loss groups")
        for key, raw_sum in sums.items():
            count = int(counts[key])
            if count < 1:
                raise ValueError("training-window loss counts must be positive")
            loss_sums[key] = loss_sums.get(key, 0.0) + float(raw_sum)
            loss_counts[key] = loss_counts.get(key, 0) + count
        correct_tokens += float(snapshot["correct_tokens"])
        target_tokens += int(snapshot["target_tokens"])
        processed_examples += int(snapshot["processed_examples"])
        elapsed_seconds.append(float(snapshot["elapsed_seconds"]))
        microbatches.append(int(snapshot["microbatches"]))
    if target_tokens < 1 or processed_examples < 1 or any(value < 1 for value in microbatches):
        raise ValueError("malformed training-window totals")
    wall_seconds = max(elapsed_seconds)
    return {
        **{key: loss_sums[key] / loss_counts[key] for key in sorted(loss_sums)},
        "target_token_accuracy": correct_tokens / target_tokens,
        "throughput/examples_per_second": processed_examples / max(wall_seconds, 1e-9),
        "telemetry/window_examples": float(processed_examples),
        "telemetry/window_microbatches_per_rank": float(min(microbatches)),
    }
