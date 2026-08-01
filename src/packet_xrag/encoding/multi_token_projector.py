import copy

import torch
from torch import nn


class MultiTokenPacketProjector(nn.Module):
    """Keep the calibrated V1 token and learn zero-initialized extra tokens."""

    def __init__(self, base_projector, retriever_hidden_size=4096, llm_hidden_size=4096,
                 tokens_per_packet=2, residual_hidden_size=1024):
        super().__init__()
        if tokens_per_packet < 1:
            raise ValueError("tokens_per_packet must be at least 1")
        self.tokens_per_packet = tokens_per_packet
        self.retriever_hidden_size = retriever_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.base_projector = copy.deepcopy(base_projector)
        for parameter in self.base_projector.parameters():
            parameter.requires_grad = False
        self.extra_heads = nn.ModuleList()
        for _ in range(tokens_per_packet - 1):
            head = nn.Sequential(nn.LayerNorm(retriever_hidden_size),
                nn.Linear(retriever_hidden_size, residual_hidden_size), nn.GELU(),
                nn.Linear(residual_hidden_size, llm_hidden_size))
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            self.extra_heads.append(head)

    def forward(self, retrieval_embeddings):
        if retrieval_embeddings.ndim != 2:
            raise ValueError("Expected retrieval embeddings with shape [num_packets, hidden_size]")
        if retrieval_embeddings.shape[-1] != self.retriever_hidden_size:
            raise ValueError("retrieval embedding hidden size mismatch")
        first_token = self.base_projector(retrieval_embeddings).unsqueeze(1)
        extra_tokens = [head(retrieval_embeddings).unsqueeze(1) for head in self.extra_heads]
        return torch.cat([first_token, *extra_tokens], dim=1)

    def flattened(self, retrieval_embeddings):
        output = self(retrieval_embeddings)
        return output.reshape(-1, self.llm_hidden_size)
