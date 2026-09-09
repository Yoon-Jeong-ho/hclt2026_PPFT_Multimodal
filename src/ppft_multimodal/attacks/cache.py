"""Local-only attack cache IO; raw inputs and targets must never be published."""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
import yaml

from ppft_multimodal.artifacts import sha256_file, write_json_atomic
from ppft_multimodal.attacks.representations import RepresentationMetadata


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

def load_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return value

def validate_attacker_output(output_dir: Path, resume: Path | None) -> None:
    """Resume only an incomplete run at its own atomic checkpoint."""
    if any((output_dir / name).exists() for name in ("attacker.pt", "attacker_metadata.json")):
        raise FileExistsError(f"refusing to overwrite completed attacker artifacts: {output_dir}")
    if resume is None:
        if output_dir.exists():
            raise FileExistsError(f"refusing to overwrite attacker output: {output_dir}")
    elif resume.resolve() != (output_dir / "resume.pt").resolve() or not resume.is_file():
        raise ValueError("resume must be the existing resume.pt in this incomplete output directory")

def safe_cache_name(example_id: str) -> str:
    return hashlib.sha256(example_id.encode()).hexdigest()[:24]

def write_cache_sample(
    cache_dir: str | Path,
    *,
    example_id: str,
    payload: Mapping[str, Any],
    public_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Write raw targets only under the ignored representation cache."""
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    stem = safe_cache_name(example_id)
    tensor_path = root / f"{stem}.pt"
    metadata_path = root / f"{stem}.metadata.json"
    torch.save(dict(payload), tensor_path)
    write_json_atomic(metadata_path, dict(public_metadata))
    return {
        "example_id": example_id,
        "cache_path": str(tensor_path.resolve()),
        "cache_sha256": sha256_file(tensor_path),
        "metadata_path": str(metadata_path.resolve()),
        "split": public_metadata.get("split"),
        "source": public_metadata.get("source"),
    }

def write_cache_index(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    return destination

def load_cache_rows(index_path: str | Path, *, split: str | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(index_path)
    selected: list[dict[str, Any]] = []
    for row in rows:
        if split is not None and row.get("split") != split:
            continue
        sidecar = json.loads(Path(row["metadata_path"]).read_text(encoding="utf-8"))
        if not row.get("cache_sha256") or sha256_file(row["cache_path"]) != row["cache_sha256"]:
            raise ValueError("representation cache payload hash mismatch (including reconstruction target)")
        payload = torch.load(row["cache_path"], map_location="cpu", weights_only=True)
        required = {"example_id", "kind", "tensor_shape", "dtype", "tensor_sha256", "source", "split"}
        missing = required - sidecar.keys()
        if missing:
            raise ValueError(f"representation sidecar missing fields: {sorted(missing)}")
        expected_kind = "vision_pre_merger"
        checks = {
            "example_id": row["example_id"],
            "kind": expected_kind,
            "source": row.get("source"),
            "split": row.get("split"),
        }
        for key, expected in checks.items():
            if key != "kind" and payload.get(key) != expected:
                raise ValueError(f"cache payload {key} mismatch")
            if sidecar.get(key) != expected:
                raise ValueError(f"representation sidecar {key} mismatch")
        metadata_payload = dict(sidecar)
        metadata_payload.pop("split")
        if isinstance(metadata_payload.get("tensor_shape"), list):
            metadata_payload["tensor_shape"] = tuple(metadata_payload["tensor_shape"])
        if isinstance(metadata_payload.get("image_grid_thw"), list):
            metadata_payload["image_grid_thw"] = tuple(metadata_payload["image_grid_thw"])
        metadata = RepresentationMetadata(**metadata_payload)
        metadata.validate_tensor(payload["representation"])
        cached_metadata = payload.get("metadata")
        if not isinstance(cached_metadata, Mapping) or cached_metadata.get("tensor_sha256") != metadata.tensor_sha256:
            raise ValueError("private cache metadata does not match the public sidecar")
        payload["metadata"] = sidecar
        payload.update({key: value for key, value in row.items() if key not in payload})
        selected.append(payload)
    return selected

def save_attacker_checkpoint(
    output_dir: str | Path,
    *,
    state_dict: Mapping[str, Any],
    victim_checkpoint_sha256: str,
    config: Mapping[str, Any],
    extra: Mapping[str, Any],
) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "attacker.pt"
    torch.save(dict(state_dict), checkpoint)
    metadata = output / "attacker_metadata.json"
    write_json_atomic(
        metadata,
        {
            "attacker_checkpoint_sha256": sha256_file(checkpoint),
            "victim_checkpoint_sha256": victim_checkpoint_sha256,
            "trained_on": "clean_representations_only",
            "noise_augmentation": False,
            "config": dict(config),
            **dict(extra),
        },
    )
    return checkpoint, metadata

def save_resume_checkpoint(path: str | Path, *, payload: Mapping[str, Any], victim_checkpoint_sha256: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, destination)
    write_json_atomic(
        destination.with_suffix(".metadata.json"),
        {
            "checkpoint_sha256": sha256_file(destination),
            "victim_checkpoint_sha256": victim_checkpoint_sha256,
        },
    )
    return destination

def load_resume_checkpoint(path: str | Path, *, victim_checkpoint_sha256: str) -> dict[str, Any]:
    source = Path(path)
    metadata = json.loads(source.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if metadata["checkpoint_sha256"] != sha256_file(source):
        raise ValueError("attack resume checkpoint hash mismatch")
    if metadata["victim_checkpoint_sha256"] != victim_checkpoint_sha256:
        raise ValueError("attack resume checkpoint victim mismatch")
    return torch.load(source, map_location="cpu", weights_only=True)

def verify_attacker_lineage(
    checkpoint: str | Path, metadata: str | Path, *, victim_checkpoint_sha256: str
) -> dict[str, Any]:
    payload = json.loads(Path(metadata).read_text(encoding="utf-8"))
    if payload["attacker_checkpoint_sha256"] != sha256_file(checkpoint):
        raise ValueError("attacker checkpoint hash mismatch")
    if payload["victim_checkpoint_sha256"] != victim_checkpoint_sha256:
        raise ValueError("attacker and representation victim hashes differ")
    if payload.get("trained_on") != "clean_representations_only" or payload.get("noise_augmentation") is not False:
        raise ValueError("privacy sweep requires a clean-only attacker")
    return payload


def validate_vision_population(rows: list[dict[str, Any]], *, paper_counts: bool = True) -> str:
    """Reject mixed victims, duplicate images, missing splits, or non-native targets."""
    from collections import Counter

    counts = Counter(str(row.get("split")) for row in rows)
    if paper_counts and counts != {"train": 2350, "validation": 831, "test": 857}:
        raise ValueError(f"paper PathVQA image population mismatch: {dict(counts)}")
    if not rows or set(counts) - {"train", "validation", "test"}:
        raise ValueError("empty or unsupported vision cache splits")
    hashes: set[str] = set()
    identities: set[str] = set()
    example_ids: set[str] = set()
    for row in rows:
        metadata = row["metadata"]
        identity = str(row.get("image_sha256", ""))
        example_id = str(row["example_id"])
        if not identity or identity in identities or example_id in example_ids:
            raise ValueError("missing/duplicate image or example identity (including across splits)")
        identities.add(identity)
        example_ids.add(example_id)
        if row["source"] != "PathVQA" or metadata.get("kind") != "vision_pre_merger":
            raise ValueError("only PathVQA native pre-merger vision caches are supported")
        hashes.add(str(metadata["victim_checkpoint_sha256"]))
        if metadata.get("extra", {}).get("spatial_merge_size") != 2:
            raise ValueError("cache must preserve the native spatial merge size 2")
        representation = row["representation"]
        target = row["target_rgb"]
        grid = torch.as_tensor(row["image_grid_thw"])
        if (representation.ndim != 2 or representation.shape[-1] != 768
                or grid.numel() != 3 or int(grid[0]) != 1
                or int(grid.prod()) != representation.shape[0]
                or not torch.isfinite(representation).all()):
            raise ValueError("invalid Qwen0.8B native representation/grid")
        if (target.ndim != 3 or target.shape[0] != 3
                or tuple(target.shape[-2:]) != tuple(row["target_size"])
                or not torch.isfinite(target).all() or target.min() < -1e-6 or target.max() > 1 + 1e-6):
            raise ValueError("invalid processor-space RGB target")
    if len(hashes) != 1 or not next(iter(hashes)):
        raise ValueError("vision cache mixes victim checkpoints")
    return next(iter(hashes))
