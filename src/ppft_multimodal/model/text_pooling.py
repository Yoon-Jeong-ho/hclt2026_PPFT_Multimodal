from __future__ import annotations

import torch

from ..contracts import POOLING_K


def padding_aware_block_mean(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    pooling_k: int = POOLING_K,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool contiguous non-padding token blocks with the fixed PPFT width k=2.

    This adapts ``HybridEncDecModel._mean_pool_slots`` from ACL_2025 while
    removing its right-padding assumption: valid tokens are selected in their
    original order before non-overlapping blocks are formed. Output padding is
    zero and marked false in the returned mask.
    """
    if pooling_k != POOLING_K:
        raise ValueError(f"pooling_k is fixed at {POOLING_K}")
    if hidden_states.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("expected hidden_states [batch, sequence, hidden] and attention_mask [batch, sequence]")
    if hidden_states.shape[:2] != attention_mask.shape:
        raise ValueError("hidden_states and attention_mask leading dimensions must match")
    if not hidden_states.is_floating_point():
        raise ValueError("hidden_states must be floating point")

    rows: list[torch.Tensor] = []
    for hidden, mask in zip(hidden_states, attention_mask.to(torch.bool), strict=True):
        valid = hidden[mask]
        if valid.numel() == 0:
            raise ValueError("every example must have at least one valid text token")
        chunks = tuple(valid[start : start + POOLING_K].mean(dim=0) for start in range(0, len(valid), POOLING_K))
        rows.append(torch.stack(chunks))

    pooled = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True)
    lengths = torch.tensor([len(row) for row in rows], device=pooled.device)
    pooled_mask = torch.arange(pooled.shape[1], device=pooled.device).unsqueeze(0) < lengths.unsqueeze(1)
    return pooled, pooled_mask


def masked_block_mean(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    block_size: int = POOLING_K,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable integration API; ``block_size`` is contractually fixed to two."""
    return padding_aware_block_mean(hidden_states, attention_mask, pooling_k=block_size)
