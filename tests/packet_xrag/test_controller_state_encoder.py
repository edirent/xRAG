import torch

from src.packet_xrag.controller.state_encoder import SelectedSetStateEncoder


def test_empty_and_nonempty_selected_state():
    encoder = SelectedSetStateEncoder(4)
    packets = torch.tensor([[1., 2., 3., 4.], [4., 3., 2., 1.]])
    empty_mean, empty_max = encoder(packets, [])
    mean, maximum = encoder(packets, [0, 1])
    assert empty_mean.requires_grad and empty_max.requires_grad
    assert torch.allclose(mean, torch.tensor([2.5, 2.5, 2.5, 2.5]))
    assert torch.equal(maximum, torch.tensor([4., 3., 3., 4.]))
    assert not torch.equal(empty_mean, mean)
