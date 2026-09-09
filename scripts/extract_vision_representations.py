#!/usr/bin/env python3
"""Build private PathVQA attack caches from a clean Stage-2 checkpoint."""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from ppft_multimodal.artifacts import hash_path, sha256_file, write_json_atomic
from ppft_multimodal.attacks.cache import read_jsonl, safe_cache_name, write_cache_index, write_cache_sample
from ppft_multimodal.attacks.native_rgb import native_patch_tensor_to_rgb
from ppft_multimodal.attacks.representations import RepresentationMetadata

SPLIT_PRIORITY = {"train": 0, "validation": 1, "test": 2}


def source_split(row: Mapping[str, Any]) -> str:
    split = str(row.get("original_split", row.get("split", "train")))
    if split == "dev":
        split = "validation"
    if split not in SPLIT_PRIORITY:
        raise ValueError(f"unsupported source split: {split}")
    return split


def select_images(rows: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Original image-SHA deduplication: preserve test before validation before train."""
    selected: dict[str, dict[str, Any]] = {}
    path_hashes: dict[str, str] = {}
    candidates: Counter[str] = Counter()
    for original in rows:
        if original.get("source") != "PathVQA":
            continue
        row = dict(original)
        split = source_split(row)
        candidates[split] += 1
        value = row.get("image_path") or row.get("image")
        if not value:
            raise ValueError("PathVQA row lacks an image path")
        path = str(Path(value).resolve())
        if path not in path_hashes:
            path_hashes[path] = sha256_file(path)
        digest = path_hashes[path]
        if row.get("image_sha256") != digest:
            raise ValueError("declared image SHA does not match its bytes")
        row["image_path"] = path
        previous = selected.get(digest)
        if previous is None or SPLIT_PRIORITY[split] > SPLIT_PRIORITY[source_split(previous)]:
            selected[digest] = row
    ordered = sorted(selected.items(), key=lambda item: (SPLIT_PRIORITY[source_split(item[1])], item[0]))
    output = [row for _, row in ordered]
    counts = Counter(source_split(row) for row in output)
    if len({row["example_id"] for row in output}) != len(output):
        raise ValueError("distinct images cannot share an example ID")
    return output, {"candidate_counts": dict(candidates), "selected_counts": dict(counts),
                    "excluded_duplicate_count": sum(candidates.values()) - len(output)}


@torch.no_grad()
def extract(args: argparse.Namespace) -> Path:
    # Loaded lazily: --help and membership tests do not load model weights.
    from ppft_multimodal.runtime import load_victim

    selected, report = select_images(row for manifest in args.manifest for row in read_jsonl(manifest))
    if report["selected_counts"] != {"train": 2350, "validation": 831, "test": 857}:
        raise ValueError(f"paper PathVQA population mismatch: {report['selected_counts']}")
    if args.output_dir.exists():
        raise FileExistsError("use a fresh cache directory; incomplete caches are not silently reused")
    model, model_config = load_victim(args.stage2_config, args.checkpoint, args.device,
                                qwen_path=args.qwen_path, kure_path=args.kure_path, require_clean=True)
    pinned = model_config.as_dict()
    model.eval()
    victim_hash = hash_path(args.checkpoint)
    args.output_dir.mkdir(parents=True)
    write_json_atomic(args.output_dir / "preflight.json", {
        **report, "victim_checkpoint_sha256": victim_hash,
        "manifests_sha256": [sha256_file(path) for path in args.manifest],
    })
    processor = model.qwen_processor.image_processor
    index_rows = []
    for row in selected:
        if sha256_file(row["image_path"]) != row["image_sha256"]:
            raise ValueError("image changed after preflight")
        with Image.open(row["image_path"]) as image:
            batch = model.prepare_batch(private_texts=[str(row["private_input"])],
                                         private_options=[row.get("options")], targets=[None],
                                         images=[image.convert("RGB")], example_ids=[str(row["example_id"])])
        if batch.pixel_values is None or batch.image_grid_thw is None:
            raise RuntimeError("native processor produced no image tensors")
        grid = batch.image_grid_thw[0].detach().cpu()
        if int(grid.prod()) // int(processor.merge_size) ** 2 > 1024:
            raise ValueError("manifest contains an image beyond the 1024 post-merge token budget")
        target = native_patch_tensor_to_rgb(batch.pixel_values.detach().cpu(), grid,
                                            patch_size=int(processor.patch_size),
                                            temporal_patch_size=int(processor.temporal_patch_size),
                                            merge_size=int(processor.merge_size),
                                            image_mean=processor.image_mean, image_std=processor.image_std)
        clean, transmitted, _ = model._encode_images(batch.to(args.device))
        if len(clean) != 1 or not torch.equal(clean[0], transmitted[0]):
            raise RuntimeError("attack cache extraction requires the clean native representation")
        representation = clean[0].detach().cpu()
        split = source_split(row)
        metadata = RepresentationMetadata.from_tensor(
            representation, victim_checkpoint_sha256=victim_hash, example_id=str(row["example_id"]),
            kind="vision_pre_merger", model_revision=str(pinned["backbone"]["revision"]),
            processor_revision=str(pinned["backbone"]["revision"]),
            image_grid_thw=(int(grid[0]), int(grid[1]), int(grid[2])), source="PathVQA",
            valid_token_count=int(representation.shape[0]),
            extra={"source_split": split, "image_sha256": row["image_sha256"],
                   "spatial_merge_size": int(processor.merge_size), "victim_noise": "clean"},
        )
        payload = {"example_id": str(row["example_id"]), "representation": representation,
                   "metadata": metadata.__dict__, "split": split, "source": "PathVQA",
                   "target_rgb": target, "image_grid_thw": grid, "target_size": tuple(target.shape[-2:]),
                   "image_sha256": str(row["image_sha256"])}
        sample_dir = args.output_dir / split
        if (sample_dir / f"{safe_cache_name(str(row['example_id']))}.pt").exists():
            raise FileExistsError("duplicate cache output")
        index_rows.append(write_cache_sample(sample_dir, example_id=str(row["example_id"]), payload=payload,
                                             public_metadata={**metadata.__dict__, "split": split}))
    # Completion marker is written last, never for partial extraction.
    return write_cache_index(args.output_dir / "index.jsonl", index_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path, required=True,
                        help="Repeat for canonical TRAIN and pooled held-out JSONL; preserves original_split")
    parser.add_argument("--stage2-config", type=Path, default=Path("configs/stage2/no_noise.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--qwen-path", type=Path)
    parser.add_argument("--kure-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(extract(args))


if __name__ == "__main__":
    main()
