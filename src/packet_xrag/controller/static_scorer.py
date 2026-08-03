"""Static single-packet relevance scorer for frozen SFR embeddings."""

import torch
import torch.nn as nn
import torch.nn.functional as F


SFR_DIM = 4096
PROJECTION_DIM = 512
FEATURE_DIM = PROJECTION_DIM * 4 + 4


def packet_position_features(packets, device=None, dtype=torch.float32):
    """Return normalized sentence position and document length per packet."""
    doc_lengths = {}
    for packet in packets:
        doc_id = int(packet["doc_id"])
        doc_lengths[doc_id] = max(doc_lengths.get(doc_id, 0), int(packet["sentence_id"]) + 1)
    maximum_doc_length = max(doc_lengths.values())
    values = []
    for packet in packets:
        length = doc_lengths[int(packet["doc_id"])]
        position = int(packet["sentence_id"]) / max(length - 1, 1)
        values.append((position, length / maximum_doc_length))
    return torch.tensor(values, device=device, dtype=dtype)


def multi_positive_listwise_loss(scores, gold_mask):
    """-log(sum(exp(gold scores)) / sum(exp(all candidate scores)))."""
    scores = scores.flatten()
    gold_mask = gold_mask.flatten().bool()
    if scores.numel() == 0 or scores.shape != gold_mask.shape:
        raise ValueError("scores and gold mask must be non-empty and aligned")
    if not gold_mask.any():
        raise ValueError("listwise loss requires at least one positive")
    return torch.logsumexp(scores, dim=0) - torch.logsumexp(scores[gold_mask], dim=0)


def negative_analysis_labels(packets, gold_ids, topk_ranking):
    """Assign one deterministic diagnostic category to each non-gold packet."""
    gold = set(gold_ids)
    top6 = set(topk_ranking[:6]) - gold
    gold_packets = [packets[index] for index in gold_ids]
    support_docs = {int(packet["doc_id"]) for packet in gold_packets}
    labels = {}
    for index, packet in enumerate(packets):
        if index in gold:
            continue
        same_doc_gold = [gold_packet for gold_packet in gold_packets
                         if int(gold_packet["doc_id"]) == int(packet["doc_id"])]
        neighbor = any(abs(int(packet["sentence_id"]) -
                           int(gold_packet["sentence_id"])) == 1
                       for gold_packet in same_doc_gold)
        if neighbor:
            label = "gold-neighbor sentence"
        elif index in top6:
            label = "TOPK top-6 non-gold"
        elif int(packet["doc_id"]) in support_docs:
            label = "same-support-document non-gold"
        elif not packet.get("is_supporting", False):
            label = "random distractor"
        else:
            label = "other candidate"
        labels[index] = label
    return labels


class StaticPacketScorer(nn.Module):
    def __init__(self, input_dim=SFR_DIM, projection_dim=PROJECTION_DIM, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.query_projection = nn.Linear(input_dim, projection_dim)
        self.packet_projection = nn.Linear(input_dim, projection_dim)
        feature_dim = projection_dim * 4 + 4
        self.scorer = nn.Sequential(
            nn.Linear(feature_dim, 2048),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

    def score_pairs(self, query_embeddings, packet_embeddings, position_features):
        if query_embeddings.ndim != 2 or query_embeddings.shape[1] != self.input_dim:
            raise ValueError("paired query embeddings have invalid shape")
        if packet_embeddings.ndim != 2 or packet_embeddings.shape[1] != self.input_dim:
            raise ValueError("packet embeddings have invalid shape")
        if len(query_embeddings) != len(packet_embeddings):
            raise ValueError("query and packet pairs must be aligned")
        if position_features.shape != (len(packet_embeddings), 2):
            raise ValueError("position features must be [num_packets, 2]")
        query = self.query_projection(query_embeddings)
        packet = self.packet_projection(packet_embeddings)
        projected_cosine = F.cosine_similarity(query.float(), packet.float(), dim=-1)
        original_cosine = F.cosine_similarity(
            query_embeddings.float(), packet_embeddings.float(), dim=-1
        )
        scalar = torch.cat([
            projected_cosine[:, None].to(query.dtype),
            original_cosine[:, None].to(query.dtype),
            position_features.to(query.dtype),
        ], dim=-1)
        features = torch.cat([
            query, packet, query * packet, torch.abs(query - packet), scalar
        ], dim=-1)
        return self.scorer(features).squeeze(-1)

    def forward(self, query_embedding, packet_embeddings, position_features):
        if query_embedding.shape != (self.input_dim,):
            raise ValueError("query embedding has invalid shape")
        query_embeddings = query_embedding.unsqueeze(0).expand(len(packet_embeddings), -1)
        return self.score_pairs(query_embeddings, packet_embeddings, position_features)

    def score_record(self, record, device=None):
        parameter = next(self.parameters())
        device = device or parameter.device
        query = record["query_embedding"].to(device=device, dtype=parameter.dtype)
        packets = record["packet_embeddings"].to(device=device, dtype=parameter.dtype)
        positions = packet_position_features(
            record["packets"], device=device, dtype=parameter.dtype
        )
        return self(query, packets, positions)

    def score_records(self, records, device=None):
        """Score a question batch in one flattened candidate-pair forward."""
        if not records:
            raise ValueError("record batch must be non-empty")
        parameter = next(self.parameters())
        device = device or parameter.device
        counts = [record["packet_count"] for record in records]
        stacked_queries = torch.stack([record["query_embedding"] for record in records])
        queries = torch.repeat_interleave(
            stacked_queries, torch.tensor(counts, device=stacked_queries.device), dim=0,
        ).to(device=device, dtype=parameter.dtype)
        packets = torch.cat([record["packet_embeddings"] for record in records]).to(
            device=device, dtype=parameter.dtype
        )
        positions = torch.cat([
            packet_position_features(record["packets"]) for record in records
        ]).to(device=device, dtype=parameter.dtype)
        scores = self.score_pairs(queries, packets, positions)
        return list(scores.split(counts))
