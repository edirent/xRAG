import math

import pytest
import torch

from src.packet_xrag.controller.static_scorer import multi_positive_listwise_loss


def test_multi_positive_listwise_loss_matches_definition():
    scores = torch.tensor([0.2, 1.1, -0.4, 0.7], requires_grad=True)
    gold = torch.tensor([False, True, False, True])
    loss = multi_positive_listwise_loss(scores, gold)
    expected = torch.logsumexp(scores, 0) - torch.logsumexp(scores[gold], 0)
    assert torch.allclose(loss, expected)
    loss.backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()


def test_loss_rewards_joint_positive_mass_and_is_shift_invariant():
    gold = torch.tensor([True, True, False])
    baseline = multi_positive_listwise_loss(torch.tensor([0.0, 0.0, 0.0]), gold)
    improved = multi_positive_listwise_loss(torch.tensor([1.0, 1.0, 0.0]), gold)
    shifted = multi_positive_listwise_loss(torch.tensor([6.0, 6.0, 5.0]), gold)
    assert improved < baseline
    assert torch.allclose(improved, shifted)
    assert baseline.item() == pytest.approx(math.log(3 / 2))


def test_loss_rejects_question_without_gold_packet():
    with pytest.raises(ValueError, match="at least one positive"):
        multi_positive_listwise_loss(torch.zeros(2), torch.zeros(2, dtype=torch.bool))
