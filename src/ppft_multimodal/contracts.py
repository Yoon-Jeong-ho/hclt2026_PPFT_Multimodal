from __future__ import annotations

POOLING_K = 2
PRIMARY_BACKBONE = "Qwen/Qwen3.5-0.8B"
PRIMARY_BACKBONE_PROFILE = "paper_qwen35_08b"
PRIMARY_BACKBONE_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
FORBIDDEN_BACKBONES = frozenset({"Qwen/Qwen3.5-2B-Base", "Qwen/Qwen3.5-0.8B-Base"})
TEXT_ENCODER = "nlpai-lab/KURE-v1"
TEXT_ENCODER_REVISION = "4ed4540949c70b7da2c74004a915e1f2d5e46e4f"
NOISE_KIND = "isotropic_l2_laplace"
STAGE2_EPSILONS = (150.0, 75.0)
LORA_TARGET_POLICY = "decoder_only_all_linear"
KERNELS_VERSION = "0.16.1"
TILELANG_VERSION = "0.1.14"
GATED_DELTA_KERNEL = "kernels-community/fla@v1"
GATED_DELTA_KERNEL_REVISION = "0747b00089cdc018687bec907ae569d82813e767"
CAUSAL_CONV_BACKEND = "transformers_pytorch_fallback"


def assert_supported_backbone(*, name: str, revision: str, profile: str | None) -> None:
    if name in FORBIDDEN_BACKBONES:
        raise ValueError(f"Base checkpoints are forbidden, got {name}")
    if profile in {None, PRIMARY_BACKBONE_PROFILE}:
        expected = (PRIMARY_BACKBONE, PRIMARY_BACKBONE_REVISION)
        if (name, revision) == expected:
            return
        raise ValueError(
            f"{PRIMARY_BACKBONE_PROFILE} requires exact backbone and revision {expected}, "
            f"got {(name, revision)}"
        )
    raise ValueError(
        f"unsupported backbone selection: profile={profile!r}, name={name!r}; "
        f"the paper backbone is {PRIMARY_BACKBONE}"
    )


def assert_runtime_dimensions(*, vision_out_dim: int, decoder_hidden_dim: int) -> None:
    if vision_out_dim <= 0 or decoder_hidden_dim <= 0:
        raise ValueError("runtime dimensions must be positive")
    if vision_out_dim != decoder_hidden_dim:
        raise ValueError(
            f"native vision output ({vision_out_dim}) must equal decoder hidden size ({decoder_hidden_dim})"
        )
