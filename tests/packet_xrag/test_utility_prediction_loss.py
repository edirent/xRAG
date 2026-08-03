import torch

from src.packet_xrag.controller.utility_prediction_loss import (
    deterministic_pair_indices, pairwise_ranking_loss, sign_classification_loss,
    utility_prediction_loss,
)


def test_pairwise_loss_rewards_correct_direction_and_gap_filtering():
    target = torch.tensor([.2, -.2, .18])
    pairs = deterministic_pair_indices(target, "s", "state")
    assert (0, 1) in pairs and (0, 2) not in pairs
    correct = pairwise_ranking_loss(torch.tensor([1.0, -1.0, .5]), target, [(0, 1)])
    wrong = pairwise_ranking_loss(torch.tensor([-1.0, 1.0, .5]), target, [(0, 1)])
    assert correct < wrong


def test_sign_loss_ignores_near_zero_and_total_weights_are_fixed():
    prediction = torch.tensor([.3, -.3, 100.0], requires_grad=True)
    raw = torch.tensor([.2, -.2, .01]); normalized = raw.clone()
    sign = sign_classification_loss(prediction, raw)
    sign.backward(retain_graph=True)
    assert prediction.grad[2] == 0
    total, parts = utility_prediction_loss(
        prediction, normalized, raw, [(0, 3, "s", "state")]
    )
    expected = parts["regression"] + .5 * parts["ranking"] + .25 * parts["sign"]
    assert torch.equal(total, expected)

