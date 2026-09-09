#!/usr/bin/env python3
"""Train one of the three Stage-2 paper arms from the same Stage-1 parent."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.training_common import resolve_paper_stage2, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--qwen-path", type=Path, required=True)
    parser.add_argument("--kure-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--parent-checkpoint", type=Path)
    parser.add_argument("--max-updates", type=int)
    args = parser.parse_args()
    train(
        stage="stage2",
        config_path=args.config.resolve(),
        qwen_path=args.qwen_path,
        kure_path=args.kure_path,
        manifest_override=args.manifest,
        parent_checkpoint=args.parent_checkpoint,
        max_updates=args.max_updates,
    )


if __name__ == "__main__":
    main()

__all__ = ["main", "resolve_paper_stage2"]
