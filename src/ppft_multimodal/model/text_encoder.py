from __future__ import annotations

from typing import Any

import torch
from torch import nn


class TokenLevelTextEncoder(nn.Module):
    """Expose only underlying-transformer token hidden states, never sentence pooling."""

    def __init__(self, transformer: nn.Module) -> None:
        super().__init__()
        self.transformer = transformer

    @property
    def hidden_size(self) -> int:
        config = getattr(self.transformer, "config", None)
        value = getattr(config, "hidden_size", None)
        if not isinstance(value, int) or value <= 0:
            raise ValueError("text encoder hidden size is unavailable from runtime config")
        return value

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden = getattr(outputs, "last_hidden_state", None)
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise RuntimeError("KURE underlying transformer did not return token-level last_hidden_state")
        return hidden


KURETextEncoder = TokenLevelTextEncoder
