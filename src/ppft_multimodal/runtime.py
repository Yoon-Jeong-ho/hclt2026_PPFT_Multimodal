"""Portable, strict loading of paper checkpoints for evaluation and attacks."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml

from .config import ModelConfig, load_model_config
from .model.multimodal_ppft import MultimodalPPFT
from .training import (
    configure_stage1_from_config,
    configure_stage2_from_parent,
    load_checkpoint_strict,
    resolve_and_validate_stage2_arms,
)


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def _repo_path(config_path: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if path.exists():
        return path
    for parent in (config_path.parent, *config_path.parents):
        candidate = parent / path
        if candidate.exists() or (parent / "configs").is_dir():
            return candidate
    return path


def _model_paths(
    config: Mapping[str, Any],
    *,
    qwen_path: str | Path | None,
    kure_path: str | Path | None,
) -> tuple[Path, Path]:
    overrides = config.get("runtime_overrides", {})
    if not isinstance(overrides, Mapping):
        raise ValueError("runtime_overrides must be a mapping")
    qwen = qwen_path or overrides.get("qwen_path")
    kure = kure_path or overrides.get("kure_path")
    if qwen is None or kure is None:
        raise ValueError(
            "provide qwen_path and kure_path, or record both under runtime_overrides in the Stage-2 config"
        )
    return Path(qwen).expanduser().resolve(), Path(kure).expanduser().resolve()


def load_victim(
    stage2_config: str | Path,
    checkpoint: str | Path,
    device: str | torch.device,
    qwen_path: str | Path | None = None,
    kure_path: str | Path | None = None,
    require_clean: bool = True,
) -> tuple[MultimodalPPFT, ModelConfig]:
    """Load one Stage-2 paper arm and validate its complete checkpoint lineage.

    Paths may be supplied explicitly or through ``runtime_overrides``. The
    function reconstructs the same three-arm resolution used by training so
    the strict configuration hash is not weakened for evaluation.
    """

    config_path = Path(stage2_config).expanduser().resolve()
    requested = _read_yaml(config_path)
    qwen, kure = _model_paths(requested, qwen_path=qwen_path, kure_path=kure_path)
    arm_paths = sorted(config_path.parent.glob("*.yaml"))
    arms = [_read_yaml(path) for path in arm_paths]
    parent = _repo_path(config_path, requested["parent_checkpoint"]).resolve()
    resolved = resolve_and_validate_stage2_arms(
        arms,
        parent_checkpoint=parent,
        max_updates=None,
        runtime_overrides={"qwen_path": str(qwen), "kure_path": str(kure)},
        allow_split_noise=True,
    )
    matches = [item for item in resolved if item.get("run_name") == requested.get("run_name")]
    if len(matches) != 1:
        raise ValueError("requested config is not exactly one of the three paper arms")
    config = matches[0]
    if require_clean and bool(config.get("noise", {}).get("enabled")):
        raise ValueError("vision reconstruction uses the clean Stage-2 victim checkpoint")

    model_config_path = _repo_path(config_path, config["model_config"])
    model_config = load_model_config(model_config_path)
    identity = {
        "model": model_config.as_dict(),
        "qwen_path": str(qwen),
        "kure_path": str(kure),
    }
    parent_config_path = _repo_path(config_path, config["parent_config"])
    parent_config = _read_yaml(parent_config_path)
    parent_manifest = _repo_path(config_path, config["parent_manifest"])
    stage2_manifest = _repo_path(config_path, config["manifest"])

    model = MultimodalPPFT.from_pretrained(qwen_path=qwen, kure_path=kure)
    configure_stage1_from_config(model, parent_config)
    load_checkpoint_strict(
        parent,
        model=model,
        expected_config=parent_config,
        expected_manifest=parent_manifest,
        expected_model_identity=identity,
    )
    configure_stage2_from_parent(model, config=config, parent_config=parent_config)
    load_checkpoint_strict(
        Path(checkpoint).expanduser().resolve(),
        model=model,
        expected_config=config,
        expected_manifest=stage2_manifest,
        expected_model_identity=identity,
        expected_parent_checkpoint=parent,
    )
    noise = config["noise"]
    model.set_noise(
        enabled=bool(noise["enabled"]),
        text_epsilon=noise.get("text_epsilon"),
        vision_epsilon=noise.get("vision_epsilon"),
    )
    model = model.to(device).eval()
    if torch.device(device).type == "cuda":
        model.activate_kernel_runtime(mode="inference")
    return model, model_config
