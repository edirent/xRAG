"""Model A: state-marginalized static generator-utility predictor."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.packet_xrag.controller.static_scorer import SFR_DIM


PROJECTION_DIM = 512
BASE_FEATURE_DIM = PROJECTION_DIM * 4 + 5


class StaticUtilityPredictor(nn.Module):
    def __init__(self, input_dim=SFR_DIM, projection_dim=PROJECTION_DIM, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim; self.projection_dim = projection_dim
        self.query_projection = nn.Linear(input_dim, projection_dim)
        self.packet_projection = nn.Linear(input_dim, projection_dim)
        self.utility_head = nn.Sequential(
            nn.Linear(projection_dim * 4 + 5, 2048), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2048, 512), nn.GELU(), nn.Dropout(dropout), nn.Linear(512, 1),
        )

    def initialize_projections(self, static_state):
        self.query_projection.load_state_dict({
            key.removeprefix("query_projection."): value
            for key, value in static_state.items() if key.startswith("query_projection.")
        }, strict=True)
        self.packet_projection.load_state_dict({
            key.removeprefix("packet_projection."): value
            for key, value in static_state.items() if key.startswith("packet_projection.")
        }, strict=True)

    def project(self, query_embeddings, packet_embeddings):
        return self.query_projection(query_embeddings), self.packet_projection(packet_embeddings)

    def base_features(self, query_embeddings, packet_embeddings, position_features,
                      static_scores):
        query, packet = self.project(query_embeddings, packet_embeddings)
        projected_cosine = F.cosine_similarity(query.float(), packet.float(), dim=-1)
        original_cosine = F.cosine_similarity(
            query_embeddings.float(), packet_embeddings.float(), dim=-1
        )
        scalars = torch.cat([
            projected_cosine[:, None].to(query.dtype),
            original_cosine[:, None].to(query.dtype),
            static_scores.reshape(-1, 1).to(query.dtype),
            position_features.to(query.dtype),
        ], dim=-1)
        return torch.cat([query, packet, query * packet, torch.abs(query - packet), scalars], -1)

    def forward(self, query_embeddings, packet_embeddings, position_features, static_scores,
                **_):
        return self.utility_head(self.base_features(
            query_embeddings, packet_embeddings, position_features, static_scores
        )).squeeze(-1)
