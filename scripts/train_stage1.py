#!/usr/bin/env python3
"""Train the clean Stage-1 alignment model from the paper."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.training_common import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage1/paper.yaml"))
    parser.add_argument("--qwen-path", type=Path, required=True)
    parser.add_argument("--kure-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--max-updates", type=int)
    args = parser.parse_args()
    train(
        stage="stage1",
        config_path=args.config.resolve(),
        qwen_path=args.qwen_path,
        kure_path=args.kure_path,
        manifest_override=args.manifest,
        max_updates=args.max_updates,
    )


if __name__ == "__main__":
    main()
