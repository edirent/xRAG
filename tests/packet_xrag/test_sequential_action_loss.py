import torch

from src.packet_xrag.controller.sequential_controller import multi_positive_action_loss


def test_action_loss_uses_all_unselected_positive_actions():
    scores = torch.tensor([0., 1., 2.], requires_grad=True)
    mask = torch.tensor([True, False, True])
    loss = multi_positive_action_loss(scores, mask)
    expected = torch.logsumexp(scores, 0) - torch.logsumexp(scores[mask], 0)
    assert torch.allclose(loss, expected)
    loss.backward()
    assert scores.grad is not None
