"""Locked regression, within-state ranking, and sign losses for utility prediction."""

from __future__ import annotations

import hashlib
import random
from itertools import combinations

import torch
import torch.nn.functional as F


HUBER_DELTA = 0.1
PAIR_MIN_RAW_GAP = 0.05
MAX_PAIRS_PER_STATE = 16


def deterministic_pair_indices(raw_targets: torch.Tensor, sample_id: str, state_id: str,
                               epoch: int = 0, max_pairs: int = MAX_PAIRS_PER_STATE):
    values = raw_targets.detach().cpu().tolist()
    eligible = [(left, right) for left, right in combinations(range(len(values)), 2)
                if abs(values[left] - values[right]) >= PAIR_MIN_RAW_GAP]
    if len(eligible) > max_pairs:
        digest = hashlib.sha256(
            f"{sample_id}:{state_id}:{epoch}:utility-pairs".encode()
        ).hexdigest()
        random.Random(int(digest[:16], 16)).shuffle(eligible)
        eligible = eligible[:max_pairs]
    return eligible


def pairwise_ranking_loss(predictions: torch.Tensor, raw_targets: torch.Tensor,
                          pairs) -> torch.Tensor:
    if not pairs:
        return predictions.sum() * 0.0
    left = torch.tensor([pair[0] for pair in pairs], device=predictions.device)
    right = torch.tensor([pair[1] for pair in pairs], device=predictions.device)
    direction = torch.sign(raw_targets[left] - raw_targets[right])
    margin = direction * (predictions[left] - predictions[right])
    return F.softplus(-margin).mean()


def sign_classification_loss(predictions: torch.Tensor, raw_targets: torch.Tensor):
    mask = raw_targets.abs() > 0.02
    if not bool(mask.any()):
        return predictions.sum() * 0.0
    labels = raw_targets[mask].gt(0).to(predictions.dtype)
    return F.binary_cross_entropy_with_logits(predictions[mask] / 0.1, labels)


def utility_prediction_loss(predictions: torch.Tensor, normalized_targets: torch.Tensor,
                            raw_targets: torch.Tensor, state_slices, epoch: int = 0):
    regression = F.huber_loss(
        predictions, normalized_targets, delta=HUBER_DELTA, reduction="mean"
    )
    ranking_terms = []
    for start, stop, sample_id, state_id in state_slices:
        pairs = deterministic_pair_indices(
            raw_targets[start:stop], sample_id, state_id, epoch
        )
        ranking_terms.append(pairwise_ranking_loss(
            predictions[start:stop], raw_targets[start:stop], pairs
        ))
    ranking = (torch.stack(ranking_terms).mean() if ranking_terms
               else predictions.sum() * 0.0)
    sign = sign_classification_loss(predictions, raw_targets)
    total = regression + 0.5 * ranking + 0.25 * sign
    return total, {"regression": regression, "ranking": ranking, "sign": sign}

