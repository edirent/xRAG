"""Query-conditioned fixed-slot fuser over frozen packet representations."""

from __future__ import annotations

import torch
import torch.nn as nn


class QueryConditionedSetFuser(nn.Module):
    def __init__(self, input_dim=4096, latent_dim=512, output_dim=4096,
                 output_slots=4, heads=8, depth=1, query_conditioned=True):
        super().__init__()
        if depth != 1: raise ValueError("the locked initial search permits depth=1 only")
        self.input_dim = input_dim; self.output_dim = output_dim
        self.output_slots = output_slots; self.query_conditioned = query_conditioned
        self.input_projection = nn.Linear(input_dim, latent_dim)
        self.query_projection = nn.Linear(input_dim, latent_dim)
        self.latent_slots = nn.Parameter(torch.empty(output_slots, latent_dim))
        nn.init.normal_(self.latent_slots, std=.02)
        self.cross_attention = nn.MultiheadAttention(latent_dim, heads, batch_first=True)
        self.attention_norm = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, 4 * latent_dim),
                                 nn.GELU(), nn.Linear(4 * latent_dim, latent_dim))
        self.output_norm = nn.LayerNorm(latent_dim)
        self.output_projection = nn.Linear(latent_dim, output_dim)

    def forward(self, query_embeddings, packet_tokens, packet_mask):
        if packet_tokens.ndim != 4 or packet_mask.shape != packet_tokens.shape[:2]:
            raise ValueError("packet tokens/mask must be [B,P,K,D] and [B,P]")
        batch, packets, tokens, dimension = packet_tokens.shape
        if dimension != self.input_dim: raise ValueError("fuser input dimension mismatch")
        flattened = packet_tokens.reshape(batch, packets * tokens, dimension)
        token_mask = packet_mask[:, :, None].expand(-1, -1, tokens).reshape(batch, -1)
        if not bool(token_mask.any(dim=1).all()): raise ValueError("each set needs one valid packet")
        keys = self.input_projection(flattened)
        latents = self.latent_slots.unsqueeze(0).expand(batch, -1, -1)
        if self.query_conditioned:
            latents = latents + self.query_projection(query_embeddings).unsqueeze(1)
        attended, _ = self.cross_attention(latents, keys, keys,
                                           key_padding_mask=~token_mask, need_weights=False)
        latents = self.attention_norm(latents + attended)
        latents = latents + self.ffn(latents)
        return self.output_projection(self.output_norm(latents))

