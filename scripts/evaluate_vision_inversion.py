#!/usr/bin/env python3
"""Evaluate the fixed clean-trained PathVQA CNN. No plotting or image export."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import torch

from ppft_multimodal.artifacts import sha256_file, write_json_atomic
from ppft_multimodal.attacks.cache import (
    load_cache_rows,
    load_yaml,
    validate_vision_population,
    verify_attacker_lineage,
)
from ppft_multimodal.attacks.vision_inversion import VisionInversionAttacker, reconstruction_mse
from ppft_multimodal.model.noise import add_norm_preserving_noise


def deterministic_noise_seed(example_id: str, epsilon: float, draw: int, seed: int = 42) -> int:
    """Original paper seed derivation; independent of processing order or GPU."""
    payload = f"{seed}\0{example_id}\0{epsilon:g}\0{draw}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def parse_grid(config: dict[str, Any]) -> tuple[list[float | None], int, int]:
    settings = config["evaluation"]
    grid = [None if item == "clean" else float(item) for item in settings["epsilons"]]
    if (not grid or grid[0] is not None or len(set(grid)) != len(grid)
            or any(item is not None and (not math.isfinite(item) or item <= 0) for item in grid)):
        raise ValueError("epsilon grid must start with clean and contain unique positive finite values")
    draws = int(settings["draws_per_noisy_epsilon"])
    if draws < 1:
        raise ValueError("at least one noisy draw is required")
    return grid, draws, int(settings["seed"])


@torch.no_grad()
def sweep(
    model: VisionInversionAttacker, rows: list[dict[str, Any]], *, epsilons: list[float | None],
    draws: int, seed: int, device: str, output: Path,
) -> list[dict[str, Any]]:
    """Mean pixels within each image, then equal-weight image and draw means."""
    if not rows:
        raise ValueError("evaluation requires held-out images")
    model.eval()
    results = []
    with (output / "draws.jsonl").open("x", encoding="utf-8") as handle:
        for epsilon in epsilons:
            draw_means = []
            for draw in range(1 if epsilon is None else draws):
                values = []
                for row in rows:
                    # The recorded experiment samples on CPU in fp32, then casts
                    # back to the cached bf16 dtype before decoder transfer.
                    clean = row["representation"].cpu()
                    noise_seed = None
                    if epsilon is None:
                        exposed = clean
                    else:
                        noise_seed = deterministic_noise_seed(str(row["example_id"]), epsilon, draw, seed)
                        exposed = add_norm_preserving_noise(
                            clean, epsilon=epsilon,
                            generator=torch.Generator(device="cpu").manual_seed(noise_seed),
                        )
                    prediction = model(exposed.to(device), row["image_grid_thw"], tuple(row["target_size"])).cpu()
                    mse = float(reconstruction_mse(prediction, row["target_rgb"].unsqueeze(0)))
                    if not math.isfinite(mse):
                        raise FloatingPointError("non-finite reconstruction error; no rows are silently dropped")
                    values.append(mse)
                    handle.write(json.dumps({"example_id": row["example_id"], "epsilon": epsilon,
                                             "draw": draw, "noise_seed": noise_seed, "mse": mse}) + "\n")
                draw_means.append(fmean(values))
            result = {"epsilon": "clean" if epsilon is None else epsilon, "images": len(rows),
                      "draws": len(draw_means), "mse": fmean(draw_means),
                      "draw_mean_sd": pstdev(draw_means)}
            results.append(result)
            print(json.dumps(result), flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/attack/pathvqa.yaml"))
    parser.add_argument("--cache-index", type=Path, required=True)
    parser.add_argument("--attacker-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = load_yaml(args.config)
    grid, draws, seed = parse_grid(config)
    rows = load_cache_rows(args.cache_index)
    victim_hash = validate_vision_population(rows)
    checkpoint = args.attacker_dir / "attacker.pt"
    metadata = verify_attacker_lineage(checkpoint, args.attacker_dir / "attacker_metadata.json",
                                       victim_checkpoint_sha256=victim_hash)
    if (metadata.get("complete_training") is not True or metadata.get("selected_epoch") != 10
            or metadata.get("cache_index_sha256") != sha256_file(args.cache_index)):
        raise ValueError("evaluation requires the completed epoch-10 attacker and its identical image cache")
    identity = metadata["identity"]
    model = VisionInversionAttacker(int(identity["representation_dim"]), channels=int(identity["channels"]),
                                    blocks=int(identity["residual_blocks"]),
                                    spatial_merge_size=int(identity["spatial_merge_size"]))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.to(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = sweep(model, [row for row in rows if row["split"] == "test"], epsilons=grid,
                    draws=draws, seed=seed, device=args.device, output=args.output_dir)
    with (args.output_dir / "metrics.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    write_json_atomic(args.output_dir / "summary.json", {
        "status": "completed", "metric": "image_macro_processor_space_rgb_mse",
        "sd_definition": "population SD of noise-draw means; not a confidence interval",
        "seed": seed, "results": results, "config_sha256": sha256_file(args.config),
        "cache_index_sha256": sha256_file(args.cache_index),
        "victim_checkpoint_sha256": victim_hash, "attacker_checkpoint_sha256": sha256_file(checkpoint),
    })


if __name__ == "__main__":
    main()
