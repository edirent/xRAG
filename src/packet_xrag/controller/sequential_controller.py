"""Set-conditioned fixed-k sequential packet controller."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.packet_xrag.controller.state_encoder import SelectedSetStateEncoder
from src.packet_xrag.controller.static_scorer import SFR_DIM


FEATURE_DIM = 512 * 8 + 7


def multi_positive_action_loss(scores, positive_mask):
    scores = scores.flatten()
    positive_mask = positive_mask.flatten().bool()
    if not positive_mask.any() or scores.shape != positive_mask.shape:
        raise ValueError("action loss requires aligned scores and a positive action")
    return torch.logsumexp(scores, 0) - torch.logsumexp(scores[positive_mask], 0)


class SequentialPacketController(nn.Module):
    def __init__(self, input_dim=SFR_DIM, projection_dim=512, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.query_projection = nn.Linear(input_dim, projection_dim)
        self.packet_projection = nn.Linear(input_dim, projection_dim)
        self.state_encoder = SelectedSetStateEncoder(projection_dim)
        self.candidate_mlp = nn.Sequential(
            nn.Linear(projection_dim * 8 + 7, 1024), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(1024, 256), nn.GELU(), nn.Linear(256, 1),
        )

    def initialize_from_static(self, state_dict):
        self.query_projection.load_state_dict({
            key.removeprefix("query_projection."): value for key, value in state_dict.items()
            if key.startswith("query_projection.")
        })
        self.packet_projection.load_state_dict({
            key.removeprefix("packet_projection."): value for key, value in state_dict.items()
            if key.startswith("packet_projection.")
        })

    def forward(self, query_embedding, packet_embeddings, packets, selected_indices):
        selected = list(selected_indices)
        selected_set = set(selected)
        remaining = [index for index in range(len(packets)) if index not in selected_set]
        if not remaining:
            raise ValueError("sequential state has no remaining candidates")
        query = self.query_projection(query_embedding)
        projected = self.packet_projection(packet_embeddings)
        candidate = projected[remaining]
        q = query.unsqueeze(0).expand(len(remaining), -1)
        selected_mean, selected_max = self.state_encoder(projected, selected)
        mean_state = selected_mean.unsqueeze(0).expand(len(remaining), -1)
        max_state = selected_max.unsqueeze(0).expand(len(remaining), -1)
        query_cos = F.cosine_similarity(q.float(), candidate.float(), dim=-1)
        if selected:
            selected_projected = projected[selected]
            similarities = F.normalize(candidate.float(), dim=-1) @ F.normalize(
                selected_projected.float(), dim=-1
            ).T
            max_selected_cos = similarities.max(dim=1).values
            mean_selected_cos = similarities.mean(dim=1)
        else:
            max_selected_cos = torch.zeros(len(remaining), device=candidate.device)
            mean_selected_cos = torch.zeros(len(remaining), device=candidate.device)
        selected_docs = [int(packets[index]["doc_id"]) for index in selected]
        scalars = []
        for index in remaining:
            packet = packets[index]
            same_doc_count = sum(doc == int(packet["doc_id"]) for doc in selected_docs)
            adjacent = any(
                int(packets[chosen]["doc_id"]) == int(packet["doc_id"]) and
                abs(int(packets[chosen]["sentence_id"]) - int(packet["sentence_id"])) == 1
                for chosen in selected
            )
            scalars.append((len(selected) / 6, same_doc_count / 6,
                            float(same_doc_count > 0), float(adjacent)))
        scalar_tensor = torch.tensor(scalars, device=candidate.device, dtype=candidate.dtype)
        cosine_tensor = torch.stack([
            query_cos, max_selected_cos, mean_selected_cos
        ], dim=-1).to(candidate.dtype)
        features = torch.cat([
            q, candidate, q * candidate, torch.abs(q - candidate),
            mean_state, max_state, candidate * mean_state,
            torch.abs(candidate - mean_state), cosine_tensor, scalar_tensor,
        ], dim=-1)
        return remaining, self.candidate_mlp(features).squeeze(-1)

    def score_record(self, record, selected_indices, device=None):
        parameter = next(self.parameters())
        device = device or parameter.device
        return self(
            record["query_embedding"].to(device=device, dtype=parameter.dtype),
            record["packet_embeddings"].to(device=device, dtype=parameter.dtype),
            record["packets"], selected_indices,
        )


@torch.inference_mode()
def greedy_rollout(controller, record, max_budget=6, device=None):
    selected = []
    controller.eval()
    for _ in range(min(max_budget, record["packet_count"])):
        remaining, scores = controller.score_record(record, selected, device)
        chosen_local = min(
            range(len(remaining)), key=lambda i: (-float(scores[i]), remaining[i])
        )
        selected.append(remaining[chosen_local])
    return selected
