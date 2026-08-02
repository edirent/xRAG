import copy

import torch
from torch import nn


class ResamplerBlock(nn.Module):
    """Cross-attend learned packet latents to an encoder token sequence."""

    def __init__(self, latent_size=512, num_heads=8, ffn_size=2048):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            latent_size, num_heads, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(latent_size)
        self.ffn = nn.Sequential(
            nn.Linear(latent_size, ffn_size),
            nn.GELU(),
            nn.Linear(ffn_size, latent_size),
        )
        self.ffn_norm = nn.LayerNorm(latent_size)

    def forward(self, latents, memory, memory_mask, need_weights=False):
        attended, weights = self.cross_attention(
            latents,
            memory,
            memory,
            key_padding_mask=~memory_mask.bool(),
            need_weights=need_weights,
            average_attn_weights=False,
        )
        latents = self.attention_norm(latents + attended)
        latents = self.ffn_norm(latents + self.ffn(latents))
        return latents, weights


class ResidualTokenStateResampler(nn.Module):
    """Learn token-level residuals over a frozen pooled K=2 projector."""

    tokens_per_packet = 2

    def __init__(
        self,
        pooled_k2_projector,
        encoder_hidden_size=4096,
        llm_hidden_size=4096,
        latent_size=512,
        num_latents=2,
        num_heads=8,
        ffn_size=2048,
    ):
        super().__init__()
        if num_latents != 2:
            raise ValueError("The locked protocol requires exactly two latents")
        self.encoder_hidden_size = encoder_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.latent_size = latent_size
        self.num_latents = num_latents
        self.pooled_k2_projector = copy.deepcopy(pooled_k2_projector)
        for parameter in self.pooled_k2_projector.parameters():
            parameter.requires_grad = False

        self.token_norm = nn.LayerNorm(encoder_hidden_size)
        self.memory_projection = nn.Linear(encoder_hidden_size, latent_size)
        self.latent_queries = nn.Parameter(torch.empty(num_latents, latent_size))
        nn.init.normal_(self.latent_queries, mean=0.0, std=0.02)
        self.block = ResamplerBlock(latent_size, num_heads, ffn_size)
        self.output_projection = nn.Linear(latent_size, llm_hidden_size)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        token_states,
        token_mask,
        pooled_embeddings,
        return_diagnostics=False,
    ):
        if token_states.ndim != 3:
            raise ValueError("token_states must have shape [packets, tokens, hidden]")
        if token_mask.shape != token_states.shape[:2]:
            raise ValueError("token_mask shape does not match token_states")
        if pooled_embeddings.shape != (token_states.shape[0], self.encoder_hidden_size):
            raise ValueError("pooled_embeddings shape mismatch")
        if not bool(token_mask.bool().any(dim=1).all()):
            raise ValueError("every packet must contain at least one non-padding token")

        base = self.pooled_k2_projector(pooled_embeddings)
        memory = self.memory_projection(self.token_norm(token_states))
        latents = self.latent_queries.unsqueeze(0).expand(token_states.shape[0], -1, -1)
        latents, weights = self.block(
            latents, memory, token_mask, need_weights=return_diagnostics
        )
        residual = self.output_projection(latents)
        output = base + residual
        if return_diagnostics:
            return output, {
                "base": base,
                "residual": residual,
                "latents": latents,
                "attention": weights,
            }
        return output

    def flattened(self, token_states, token_mask, pooled_embeddings):
        output = self(token_states, token_mask, pooled_embeddings)
        return output.reshape(-1, self.llm_hidden_size)


class PooledResidualResamplerControl(ResidualTokenStateResampler):
    """Parameter-matched control whose memory is one pooled SFR embedding."""

    def forward(
        self,
        token_states,
        token_mask,
        pooled_embeddings,
        return_diagnostics=False,
    ):
        del token_states, token_mask
        pooled_memory = pooled_embeddings.unsqueeze(1)
        pooled_mask = torch.ones(
            pooled_memory.shape[:2], dtype=torch.bool, device=pooled_memory.device
        )
        return super().forward(
            pooled_memory, pooled_mask, pooled_embeddings, return_diagnostics
        )
