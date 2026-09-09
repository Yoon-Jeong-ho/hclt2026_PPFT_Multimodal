from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from .contracts import (
    CAUSAL_CONV_BACKEND,
    GATED_DELTA_KERNEL,
    GATED_DELTA_KERNEL_REVISION,
    KERNELS_VERSION,
    LORA_TARGET_POLICY,
    NOISE_KIND,
    POOLING_K,
    TEXT_ENCODER,
    TEXT_ENCODER_REVISION,
    TILELANG_VERSION,
    assert_supported_backbone,
)


@dataclass(frozen=True)
class BackboneConfig:
    name: str
    revision: str
    trust_remote_code: bool = True
    profile: str | None = None


@dataclass(frozen=True)
class TextEncoderConfig:
    name: str
    revision: str
    trust_remote_code: bool = True
    max_length: int = 512
    pooling_k: int = POOLING_K


@dataclass(frozen=True)
class ProjectionConfig:
    kind: str = "linear"


@dataclass(frozen=True)
class NoiseConfig:
    kind: str = NOISE_KIND
    enabled: bool = False
    epsilon: float | None = None
    scale: float = 1.0


@dataclass(frozen=True)
class LoraConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_policy: str = LORA_TARGET_POLICY


@dataclass(frozen=True)
class RuntimeConfig:
    dtype: str = "bfloat16"
    assert_vision_decoder_dimension_match: bool = True
    max_target_length: int = 512
    use_filtered_fla_kernels: bool = True
    kernels_version: str = KERNELS_VERSION
    tilelang_version: str = TILELANG_VERSION
    gated_delta_kernel: str = GATED_DELTA_KERNEL
    gated_delta_kernel_revision: str = GATED_DELTA_KERNEL_REVISION
    causal_conv_backend: str = CAUSAL_CONV_BACKEND


@dataclass(frozen=True)
class ModelConfig:
    backbone: BackboneConfig
    text_encoder: TextEncoderConfig
    projection: ProjectionConfig
    noise: NoiseConfig
    lora: LoraConfig
    runtime: RuntimeConfig

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        # Preserve the legacy/default 2B identity exactly. The profile is an
        # explicit experimental opt-in and is serialized only when selected.
        if value["backbone"]["profile"] is None:
            del value["backbone"]["profile"]
        return value

    def digest(self) -> str:
        encoded = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def validate_model_config(config: ModelConfig) -> None:
    if not config.backbone.revision or not config.text_encoder.revision:
        raise ValueError("model revisions must be pinned")
    assert_supported_backbone(
        name=config.backbone.name,
        revision=config.backbone.revision,
        profile=config.backbone.profile,
    )
    expected_text_encoder = (TEXT_ENCODER, TEXT_ENCODER_REVISION)
    configured_text_encoder = (config.text_encoder.name, config.text_encoder.revision)
    if configured_text_encoder != expected_text_encoder:
        raise ValueError(
            f"text encoder and revision must remain pinned to {expected_text_encoder}, "
            f"got {configured_text_encoder}"
        )
    if config.text_encoder.pooling_k != POOLING_K:
        raise ValueError(f"text pooling_k is fixed at {POOLING_K}")
    if config.text_encoder.max_length < 1:
        raise ValueError("text encoder max_length must be positive")
    if config.projection.kind != "linear":
        raise ValueError("the legacy-parity text projection must remain linear")
    if config.noise.kind != NOISE_KIND:
        raise ValueError(f"noise kind must be {NOISE_KIND}")
    if config.noise.scale <= 0:
        raise ValueError("noise scale must be positive")
    if config.noise.enabled != (config.noise.epsilon is not None):
        raise ValueError("enabled noise requires epsilon; disabled noise requires epsilon=null")
    if config.noise.epsilon is not None and config.noise.epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if config.lora.rank < 1 or config.lora.alpha < 1 or not 0 <= config.lora.dropout < 1:
        raise ValueError("invalid LoRA rank, alpha, or dropout")
    if config.lora.target_policy != LORA_TARGET_POLICY:
        raise ValueError(f"LoRA policy must be {LORA_TARGET_POLICY}")
    if config.runtime.dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError("unsupported runtime dtype")
    if not config.runtime.assert_vision_decoder_dimension_match:
        raise ValueError("vision/decoder dimension assertion may not be disabled")
    if config.runtime.max_target_length < 2:
        raise ValueError("max_target_length must reserve at least one target token and EOS")
    if not config.runtime.use_filtered_fla_kernels:
        raise ValueError(
            "the filtered Qwen3.5 FLA runtime is required; "
            "the PyTorch DeltaNet fallback is not training-safe"
        )
    expected_kernels = (
        KERNELS_VERSION,
        TILELANG_VERSION,
        GATED_DELTA_KERNEL,
        GATED_DELTA_KERNEL_REVISION,
        CAUSAL_CONV_BACKEND,
    )
    configured_kernels = (
        config.runtime.kernels_version,
        config.runtime.tilelang_version,
        config.runtime.gated_delta_kernel,
        config.runtime.gated_delta_kernel_revision,
        config.runtime.causal_conv_backend,
    )
    if configured_kernels != expected_kernels:
        raise ValueError(f"kernel runtime must remain pinned to {expected_kernels}, got {configured_kernels}")


def load_model_config(path: str | Path) -> ModelConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("model config root must be a mapping")
    required = {"backbone", "text_encoder", "projection", "noise", "lora", "runtime"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"model config missing keys: {sorted(missing)}")
    config = ModelConfig(
        backbone=BackboneConfig(**raw["backbone"]),
        text_encoder=TextEncoderConfig(**raw["text_encoder"]),
        projection=ProjectionConfig(**raw["projection"]),
        noise=NoiseConfig(**raw["noise"]),
        lora=LoraConfig(**raw["lora"]),
        runtime=RuntimeConfig(**raw["runtime"]),
    )
    validate_model_config(config)
    return config
