"""Small, deterministic artifact manifests and checkpoint lineage helpers.

Raw datasets, checkpoints, and representation tensors are intentionally not
handled as publishable artifacts here.  The functions in this module produce
compact metadata that can safely be versioned or attached to W&B runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_HASH_CHUNK_BYTES = 1024 * 1024


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(payload)


def iter_tree_files(
    root: str | os.PathLike[str], *, exclude_names: Iterable[str] = ()
) -> list[Path]:
    root_path = Path(root).resolve()
    excluded = set(exclude_names)
    return sorted(
        path
        for path in root_path.rglob("*")
        if path.is_file() and path.name not in excluded
    )


def sha256_tree(
    root: str | os.PathLike[str], *, exclude_names: Iterable[str] = ()
) -> str:
    """Hash relative paths and file hashes, independent of mtimes/permissions."""

    root_path = Path(root).resolve()
    entries = [
        {
            "path": path.relative_to(root_path).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in iter_tree_files(root_path, exclude_names=exclude_names)
    ]
    return canonical_json_sha256(entries)


def hash_path(path: str | os.PathLike[str]) -> str:
    target = Path(path)
    if target.is_file():
        return sha256_file(target)
    if target.is_dir():
        return sha256_tree(target)
    raise FileNotFoundError(target)


@dataclass(frozen=True)
class CheckpointLineage:
    checkpoint_path: str
    checkpoint_sha256: str
    parent_checkpoint_path: str | None
    parent_checkpoint_sha256: str | None
    git_sha: str
    config_sha256: str
    manifest_sha256: str
    created_at_utc: str

    @classmethod
    def capture(
        cls,
        checkpoint_path: str | os.PathLike[str],
        *,
        git_sha: str,
        config: Mapping[str, Any] | str | os.PathLike[str],
        manifest: Mapping[str, Any] | str | os.PathLike[str],
        parent_checkpoint_path: str | os.PathLike[str] | None = None,
    ) -> CheckpointLineage:
        def hash_value(value: Mapping[str, Any] | str | os.PathLike[str]) -> str:
            if isinstance(value, Mapping):
                return canonical_json_sha256(value)
            return hash_path(value)

        parent = Path(parent_checkpoint_path).resolve() if parent_checkpoint_path else None
        checkpoint = Path(checkpoint_path).resolve()
        return cls(
            checkpoint_path=str(checkpoint),
            checkpoint_sha256=hash_path(checkpoint),
            parent_checkpoint_path=str(parent) if parent else None,
            parent_checkpoint_sha256=hash_path(parent) if parent else None,
            git_sha=git_sha,
            config_sha256=hash_value(config),
            manifest_sha256=hash_value(manifest),
            created_at_utc=datetime.now(UTC).isoformat(),
        )

    def verify(self, *, verify_parent: bool = True) -> None:
        if hash_path(self.checkpoint_path) != self.checkpoint_sha256:
            raise ValueError("checkpoint hash mismatch")
        if verify_parent and self.parent_checkpoint_path:
            if hash_path(self.parent_checkpoint_path) != self.parent_checkpoint_sha256:
                raise ValueError("parent checkpoint hash mismatch")


def write_json_atomic(path: str | os.PathLike[str], value: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(value) if hasattr(value, "__dataclass_fields__") else value
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)
    return destination
