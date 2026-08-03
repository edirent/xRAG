import pytest
import torch
from types import SimpleNamespace

from src.packet_xrag.controller.utility_evaluation import cached_label_mechanism_audit
from src.packet_xrag.controller.utility_rollout import (
    make_prediction_batch, rollout_utility_policy,
)


class PacketScoreModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, packet_embeddings, selected_mask, **kwargs):
        return packet_embeddings[:, 0] - selected_mask.sum(1) * .2


def record(scores):
    count = len(scores)
    return {
        "packet_count": count, "query_embedding": torch.zeros(2),
        "packet_embeddings": torch.tensor([[score, 0.0] for score in scores]),
        "topk_scores": list(scores),
        "packets": [{"packet_id": i, "doc_id": i // 2, "sentence_id": i % 2}
                    for i in range(count)],
    }


def test_rollout_enforces_minimum_threshold_stop_and_selected_mask():
    current = record([.05, .04, .03]); model = PacketScoreModel().eval()
    result = rollout_utility_policy(
        model, current, [.05, .04, .03], 1.0, .10, torch.device("cpu")
    )
    assert result["selected_packet_ids"] == [0]
    assert result["stop"]["reason"] == "threshold"
    assert result["stop"]["stop_utility"] == 0.0
    with pytest.raises(ValueError, match="masked"):
        make_prediction_batch(current, [0, 1], [1], [.05, .04, .03], torch.device("cpu"))


def test_rollout_never_exceeds_six_packets_and_stop_utility_is_zero():
    current = record([1.0 - index * .01 for index in range(10)])
    result = rollout_utility_policy(
        PacketScoreModel().eval(), current, current["topk_scores"], 1.0, 0.0,
        torch.device("cpu"),
    )
    assert len(result["selected_packet_ids"]) == 5  # shift reaches predicted utility <= 0
    assert result["stop"]["highest_remaining_score"] <= 0.0
    assert result["stop"]["stop_utility"] == 0.0


def test_mechanism_audit_does_not_reorder_state_packet_ids():
    labels = SimpleNamespace(rows=[{
        "sample_id": "s", "selected_packet_ids": [1, 4],
        "candidate_packet_id": 3, "delta_utility": 0.1,
    }])
    policies = {"s": {
        "actions": [{"selected_packet_ids_before": [4, 1], "packet_id": 3}],
        "selected_packet_ids": [4, 1, 3],
        "stop": {"highest_remaining_packet_id": None},
    }}
    audit = cached_label_mechanism_audit(policies, labels)
    assert audit["selected_actions_audited"] == 0
    assert audit["utility_not_available_actions"] == 1
