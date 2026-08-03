"""Feature construction and dynamic marginal-utility STOP rollout."""

from __future__ import annotations

from collections import Counter

import torch

from src.packet_xrag.controller.static_scorer import packet_position_features


MIN_PACKETS = 1
MAX_PACKETS = 6
THRESHOLD_GRID = (0.00, 0.02, 0.05, 0.10)


def state_scalar_features(record, selected_ids, static_scores):
    selected = list(selected_ids)
    if not selected:
        return [0.0] * 7
    selected_static = [float(static_scores[index]) for index in selected]
    selected_cosine = [float(record["topk_scores"][index]) for index in selected]
    docs = [int(record["packets"][index]["doc_id"]) for index in selected]
    pair_count = len(docs) * (len(docs) - 1) // 2
    same_pairs = sum(docs[left] == docs[right] for left in range(len(docs))
                     for right in range(left + 1, len(docs)))
    return [
        len(selected) / 6.0,
        sum(selected_static) / len(selected_static), max(selected_static),
        sum(selected_cosine) / len(selected_cosine), max(selected_cosine),
        len(set(docs)) / 6.0, same_pairs / pair_count if pair_count else 0.0,
    ]


def candidate_relation_features(record, candidate_id, selected_ids):
    if not selected_ids:
        return [0.0, 0.0, 0.0]
    candidate = record["packets"][candidate_id]
    selected = [record["packets"][index] for index in selected_ids]
    same_doc = [packet for packet in selected
                if int(packet["doc_id"]) == int(candidate["doc_id"])]
    adjacent = any(abs(int(packet["sentence_id"]) - int(candidate["sentence_id"])) == 1
                   for packet in same_doc)
    return [float(bool(same_doc)), len(same_doc) / 6.0, float(adjacent)]


def make_prediction_batch(record, candidate_ids, selected_ids, static_scores, device,
                          dtype=torch.float32):
    candidate_ids = list(candidate_ids); selected_ids = list(selected_ids)
    if set(candidate_ids) & set(selected_ids):
        raise ValueError("selected packets must be masked from candidate batch")
    count = len(candidate_ids)
    if count == 0:
        raise ValueError("prediction batch requires candidates")
    query = record["query_embedding"].expand(count, -1).to(device=device, dtype=dtype)
    packets = record["packet_embeddings"][candidate_ids].to(device=device, dtype=dtype)
    positions = packet_position_features(record["packets"])[candidate_ids].to(
        device=device, dtype=dtype
    )
    static = torch.tensor([static_scores[index] for index in candidate_ids],
                          device=device, dtype=dtype)
    maximum = max(1, len(selected_ids))
    selected = torch.zeros(count, maximum, record["packet_embeddings"].shape[1],
                           device=device, dtype=dtype)
    mask = torch.zeros(count, maximum, device=device, dtype=torch.bool)
    if selected_ids:
        values = record["packet_embeddings"][selected_ids].to(device=device, dtype=dtype)
        selected[:, :len(selected_ids)] = values.unsqueeze(0)
        mask[:, :len(selected_ids)] = True
    state_values = state_scalar_features(record, selected_ids, static_scores)
    state = torch.tensor(state_values, device=device, dtype=dtype).expand(count, -1)
    relation = torch.tensor([
        candidate_relation_features(record, candidate_id, selected_ids)
        for candidate_id in candidate_ids
    ], device=device, dtype=dtype)
    return {
        "query_embeddings": query, "packet_embeddings": packets,
        "position_features": positions, "static_scores": static,
        "selected_embeddings": selected, "selected_mask": mask,
        "state_features": state, "relation_features": relation,
    }


@torch.inference_mode()
def score_candidates(model, record, candidate_ids, selected_ids, static_scores,
                     clip_value, device):
    parameter = next(model.parameters())
    batch = make_prediction_batch(
        record, candidate_ids, selected_ids, static_scores, device, parameter.dtype
    )
    model.eval()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=device.type == "cuda"):
        normalized = model(**batch)
    raw = normalized.float().cpu() * clip_value
    return {packet_id: float(value) for packet_id, value in zip(candidate_ids, raw)}


@torch.inference_mode()
def rollout_utility_policy(model, record, static_scores, clip_value, tau, device,
                           min_packets=MIN_PACKETS, max_packets=MAX_PACKETS):
    if tau not in THRESHOLD_GRID:
        raise ValueError("STOP threshold is outside the preregistered grid")
    selected, actions = [], []
    stop = {"reason": None, "stop_utility": 0.0,
            "highest_remaining_packet_id": None,
            "highest_remaining_score": None}
    while True:
        remaining = [index for index in range(record["packet_count"])
                     if index not in set(selected)]
        if not remaining:
            stop["reason"] = "no_remaining_candidates"; break
        scores = score_candidates(
            model, record, remaining, selected, static_scores, clip_value, device
        )
        best = min(remaining, key=lambda packet_id: (-scores[packet_id], packet_id))
        maximum = scores[best]
        if len(selected) >= min_packets and maximum <= tau:
            stop.update({"reason": "threshold", "highest_remaining_packet_id": best,
                         "highest_remaining_score": maximum})
            break
        selected.append(best)
        actions.append({"step": len(actions), "packet_id": best,
                        "predicted_delta": maximum,
                        "selected_packet_ids_before": selected[:-1]})
        if len(selected) >= max_packets:
            remaining = [index for index in range(record["packet_count"])
                         if index not in set(selected)]
            if remaining:
                scores = score_candidates(
                    model, record, remaining, selected, static_scores, clip_value, device
                )
                best = min(remaining, key=lambda packet_id: (-scores[packet_id], packet_id))
                stop.update({"reason": "max_packets", "highest_remaining_packet_id": best,
                             "highest_remaining_score": scores[best]})
            else:
                stop["reason"] = "max_packets_no_remaining"
            break
    if not min_packets <= len(selected) <= max_packets:
        raise RuntimeError("rollout violated packet-count constraints")
    return {"selected_packet_ids": selected, "actions": actions, "stop": stop,
            "tau": tau}


def length_distribution(selected_groups):
    counts = Counter(len(selected) for selected in selected_groups)
    return {str(length): counts[length] for length in range(1, 7)}
