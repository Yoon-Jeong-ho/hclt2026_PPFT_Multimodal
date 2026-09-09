from __future__ import annotations

from torch import nn


class TextProjection(nn.Linear):
    """Legacy-parity single linear KURE-to-decoder projection."""

    def __init__(self, encoder_hidden_dim: int, decoder_hidden_dim: int, *, bias: bool = True) -> None:
        if encoder_hidden_dim <= 0 or decoder_hidden_dim <= 0:
            raise ValueError("projection dimensions must be discovered positive runtime values")
        super().__init__(encoder_hidden_dim, decoder_hidden_dim, bias=bias)


TextProjector = TextProjection
