"""STATIC-2-preserving residual fusion for extra retrieved evidence."""

from __future__ import annotations

import torch
import torch.nn as nn

from .set_fuser import QueryConditionedSetFuser


class ResidualSetFuser(nn.Module):
    def __init__(self, dimension=4096, latent_dim=512, output_slots=4, heads=8):
        super().__init__()
        self.output_slots = output_slots
        self.residual_fuser = QueryConditionedSetFuser(
            dimension, latent_dim, dimension, output_slots, heads, depth=1)
        self.gate = nn.Linear(dimension, 1)
        nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)

    def forward(self, query_embeddings, base_tokens, extra_packet_tokens, extra_mask):
        if base_tokens.shape[1:] != (self.output_slots, query_embeddings.shape[-1]):
            raise ValueError("base tokens must be STATIC-2's four K2 tokens")
        valid = extra_mask.any(dim=1) if extra_mask.shape[1] else torch.zeros(
            base_tokens.shape[0], device=base_tokens.device, dtype=torch.bool)
        if not bool(valid.any()):
            alpha = torch.zeros(base_tokens.shape[0], device=base_tokens.device,
                                dtype=base_tokens.dtype)
            return base_tokens, alpha
        residual = torch.zeros_like(base_tokens)
        residual[valid] = self.residual_fuser(query_embeddings[valid],
            extra_packet_tokens[valid], extra_mask[valid])
        alpha = torch.tanh(self.gate(query_embeddings)).squeeze(-1)
        alpha = alpha * valid.to(alpha.dtype)
        # The effective residual is exactly zero at initialization because alpha=0.
        return base_tokens + alpha[:, None, None] * residual, alpha
