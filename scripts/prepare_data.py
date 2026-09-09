#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ppft_multimodal.data.preparation import prepare_data


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Publish validated paper training/evaluation JSONL from canonical adapted rows."
    )
    result.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    result.add_argument("--train", type=Path, required=True)
    result.add_argument("--evaluation", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--homogeneous-block-size", type=int, default=32)
    result.add_argument("--cost-bucket-size", type=int, default=256)
    result.add_argument("--expect-train-rows", type=int)
    result.add_argument("--expect-evaluation-rows", type=int)
    result.add_argument("--expect-balanced-exposures", type=int)
    return result


def main() -> None:
    args = parser().parse_args()
    report = prepare_data(
        stage=args.stage,
        train_path=args.train,
        evaluation_path=args.evaluation,
        output_dir=args.output_dir,
        seed=args.seed,
        homogeneous_block_size=args.homogeneous_block_size,
        cost_bucket_size=args.cost_bucket_size,
        expected_train_rows=args.expect_train_rows,
        expected_evaluation_rows=args.expect_evaluation_rows,
        expected_balanced_exposures=args.expect_balanced_exposures,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
