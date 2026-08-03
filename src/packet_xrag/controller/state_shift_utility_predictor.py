"""Model B: static utility plus a candidate-independent selected-set shift."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.packet_xrag.controller.static_utility_predictor import (
    PROJECTION_DIM, StaticUtilityPredictor,
)


STATE_SCALAR_DIM = 7
STATE_FEATURE_DIM = PROJECTION_DIM * 5 + STATE_SCALAR_DIM


class StateShiftUtilityPredictor(nn.Module):
    def __init__(self, base_model: StaticUtilityPredictor | None = None, dropout=0.1):
        super().__init__()
        self.base_model = base_model or StaticUtilityPredictor()
        self.empty_mean = nn.Parameter(torch.zeros(PROJECTION_DIM))
        self.empty_max = nn.Parameter(torch.zeros(PROJECTION_DIM))
        self.state_shift_head = nn.Sequential(
            nn.Linear(STATE_FEATURE_DIM, 1024), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(1024, 256), nn.GELU(), nn.Linear(256, 1),
        )
        nn.init.zeros_(self.state_shift_head[-1].weight)
        nn.init.zeros_(self.state_shift_head[-1].bias)

    @property
    def query_projection(self):
        return self.base_model.query_projection

    @property
    def packet_projection(self):
        return self.base_model.packet_projection

    def summarize_selected(self, selected_embeddings, selected_mask):
        batch, maximum, _ = selected_embeddings.shape
        projected = self.packet_projection(selected_embeddings)
        mask = selected_mask.bool(); counts = mask.sum(1)
        safe_counts = counts.clamp_min(1).to(projected.dtype).unsqueeze(-1)
        mean = (projected * mask.unsqueeze(-1)).sum(1) / safe_counts
        masked = projected.masked_fill(~mask.unsqueeze(-1), torch.finfo(projected.dtype).min)
        maximum_values = masked.max(1).values
        empty = counts.eq(0)
        if bool(empty.any()):
            mean = torch.where(empty[:, None], self.empty_mean.to(mean.dtype), mean)
            maximum_values = torch.where(
                empty[:, None], self.empty_max.to(maximum_values.dtype), maximum_values
            )
        return mean, maximum_values, projected

    def state_shift(self, query_embeddings, selected_embeddings, selected_mask, state_features):
        query = self.query_projection(query_embeddings)
        selected_mean, selected_max, _ = self.summarize_selected(
            selected_embeddings, selected_mask
        )
        features = torch.cat([
            query, selected_mean, selected_max, query * selected_mean,
            torch.abs(query - selected_mean), state_features.to(query.dtype),
        ], -1)
        return self.state_shift_head(features).squeeze(-1), selected_mean, selected_max

    def forward(self, query_embeddings, packet_embeddings, position_features, static_scores,
                selected_embeddings, selected_mask, state_features, return_components=False,
                **_):
        base = self.base_model(
            query_embeddings, packet_embeddings, position_features, static_scores
        )
        shift, selected_mean, selected_max = self.state_shift(
            query_embeddings, selected_embeddings, selected_mask, state_features
        )
        output = base + shift
        if return_components:
            return output, {"base": base, "state_shift": shift,
                            "selected_mean": selected_mean, "selected_max": selected_max}
        return output

