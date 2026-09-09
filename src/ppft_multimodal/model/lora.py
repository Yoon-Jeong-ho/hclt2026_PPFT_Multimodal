from __future__ import annotations

from collections.abc import Iterable

from torch import nn

_FORBIDDEN_NAMESPACES = ("vision", "visual", "merger", "projector", "lm_head", "embed_tokens")


def discover_decoder_linear_modules(
    decoder: nn.Module, *, forbidden_namespaces: Iterable[str] = _FORBIDDEN_NAMESPACES
) -> tuple[str, ...]:
    """Return exact decoder-relative paths for every eligible linear module.

    Exact paths cover Qwen full attention, linear attention, and MLP projections
    without relying on q_proj/v_proj-only assumptions. Passing the decoder
    module (not the whole multimodal model) is a required namespace boundary.
    """
    forbidden = tuple(forbidden_namespaces)
    targets = tuple(
        name
        for name, module in decoder.named_modules()
        if name and isinstance(module, nn.Linear) and not any(part in name.lower() for part in forbidden)
    )
    if not targets:
        raise ValueError("no decoder-only linear modules were discovered")
    return targets


def attach_decoder_lora(
    decoder: nn.Module,
    *,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: tuple[str, ...] | None = None,
) -> tuple[nn.Module, tuple[str, ...]]:
    if rank < 1 or alpha < 1 or not 0 <= dropout < 1:
        raise ValueError("invalid LoRA rank, alpha, or dropout")
    targets = target_modules or discover_decoder_linear_modules(decoder)
    from peft import LoraConfig, inject_adapter_in_model

    decoder.requires_grad_(False)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=list(targets),
    )
    # ``decoder`` is the Qwen language-model body, not a complete
    # ``PreTrainedModelForCausalLM``.  In-place adapter injection preserves its
    # native forward signature, while a PEFT task wrapper would expect methods
    # such as ``prepare_inputs_for_generation`` that this body does not own.
    wrapped = inject_adapter_in_model(config, decoder)
    unexpected = [
        name
        for name, parameter in wrapped.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    ]
    if unexpected:
        raise RuntimeError(f"decoder base parameters remained trainable: {unexpected[:10]}")
    return wrapped, targets


def lora_module_names(model: nn.Module) -> tuple[str, ...]:
    return tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad and "lora_" in name)
