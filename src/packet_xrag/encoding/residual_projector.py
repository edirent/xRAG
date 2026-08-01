"""Residual adapter for a calibrated packet projector."""

from __future__ import annotations

import torch
from torch import nn


class ResidualPacketProjector(nn.Module):
    """Add a zero-initialized trainable residual to a frozen projector."""

    def __init__(
        self,
        base_projector: nn.Module,
        hidden_size: int = 4096,
        bottleneck_size: int = 1024,
    ) -> None:
        super().__init__()
        self.base_projector = base_projector
        for parameter in self.base_projector.parameters():
            parameter.requires_grad = False

        self.input_norm = nn.LayerNorm(hidden_size)
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, bottleneck_size),
            nn.GELU(),
            nn.Linear(bottleneck_size, hidden_size),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, retrieval_embeddings: torch.Tensor) -> torch.Tensor:
        base_output = self.base_projector(retrieval_embeddings)
        residual = self.adapter(self.input_norm(retrieval_embeddings))
        return base_output + residual
