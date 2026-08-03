"""Model C: candidate-specific residual beyond the global selected-set shift."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.packet_xrag.controller.state_shift_utility_predictor import (
    PROJECTION_DIM, StateShiftUtilityPredictor,
)


INTERACTION_SCALAR_DIM = 6
INTERACTION_FEATURE_DIM = PROJECTION_DIM * 7 + INTERACTION_SCALAR_DIM


class InteractionUtilityPredictor(nn.Module):
    def __init__(self, state_model: StateShiftUtilityPredictor | None = None, dropout=0.1):
        super().__init__()
        self.state_model = state_model or StateShiftUtilityPredictor()
        self.interaction_head = nn.Sequential(
            nn.Linear(INTERACTION_FEATURE_DIM, 1024), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(1024, 256), nn.GELU(), nn.Linear(256, 1),
        )
        nn.init.zeros_(self.interaction_head[-1].weight)
        nn.init.zeros_(self.interaction_head[-1].bias)

    @property
    def query_projection(self):
        return self.state_model.query_projection

    @property
    def packet_projection(self):
        return self.state_model.packet_projection

    def interaction_residual(self, packet_embeddings, selected_embeddings, selected_mask,
                             selected_mean, selected_max, relation_features):
        packet = self.packet_projection(packet_embeddings)
        projected_selected = self.packet_projection(selected_embeddings)
        normalized_packet = F.normalize(packet.float(), dim=-1)
        normalized_selected = F.normalize(projected_selected.float(), dim=-1)
        cosine = torch.einsum("bd,bmd->bm", normalized_packet, normalized_selected)
        mask = selected_mask.bool(); counts = mask.sum(1)
        cosine_sum = (cosine * mask).sum(1)
        cosine_mean = cosine_sum / counts.clamp_min(1)
        cosine_max = cosine.masked_fill(~mask, -torch.inf).max(1).values
        cosine_min = cosine.masked_fill(~mask, torch.inf).min(1).values
        empty = counts.eq(0)
        cosine_mean = torch.where(empty, torch.zeros_like(cosine_mean), cosine_mean)
        cosine_max = torch.where(empty, torch.zeros_like(cosine_max), cosine_max)
        cosine_min = torch.where(empty, torch.zeros_like(cosine_min), cosine_min)
        scalars = torch.cat([
            cosine_max[:, None], cosine_mean[:, None], cosine_min[:, None],
            relation_features.to(cosine.dtype),
        ], -1).to(packet.dtype)
        features = torch.cat([
            packet, selected_mean, selected_max,
            packet * selected_mean, torch.abs(packet - selected_mean),
            packet * selected_max, torch.abs(packet - selected_max), scalars,
        ], -1)
        return self.interaction_head(features).squeeze(-1)

    def forward(self, query_embeddings, packet_embeddings, position_features, static_scores,
                selected_embeddings, selected_mask, state_features, relation_features,
                return_components=False, **_):
        base_output, components = self.state_model(
            query_embeddings, packet_embeddings, position_features, static_scores,
            selected_embeddings, selected_mask, state_features, return_components=True,
        )
        residual = self.interaction_residual(
            packet_embeddings, selected_embeddings, selected_mask,
            components["selected_mean"], components["selected_max"], relation_features,
        )
        output = base_output + residual
        if return_components:
            return output, {**components, "interaction_residual": residual}
        return output
