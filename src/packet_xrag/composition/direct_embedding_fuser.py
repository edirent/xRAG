"""Direct fixed-slot fusion over frozen SFR pooled embeddings."""

from __future__ import annotations

from .set_fuser import QueryConditionedSetFuser


class DirectEmbeddingFuser(QueryConditionedSetFuser):
    def forward(self, query_embeddings, packet_embeddings, packet_mask):
        if packet_embeddings.ndim != 3:
            raise ValueError("direct SFR inputs must be [B,P,4096]")
        return super().forward(query_embeddings, packet_embeddings.unsqueeze(2), packet_mask)

