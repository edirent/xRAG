import copy

import torch
from torch import nn

from src.packet_xrag.encoding.residual_projector import ResidualPacketProjector


def make_base(hidden_size=16):
    return nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, hidden_size),
    )


def test_zero_init_output_equals_base_and_shape_is_unchanged():
    torch.manual_seed(7)
    base = make_base()
    projector = ResidualPacketProjector(base, hidden_size=16, bottleneck_size=4)
    inputs = torch.randn(2, 3, 16)

    with torch.no_grad():
        base_output = base(inputs)
        residual_output = projector(inputs)

    assert residual_output.shape == base_output.shape == inputs.shape
    assert torch.allclose(base_output, residual_output, atol=1e-5, rtol=1e-5)


def test_only_norm_and_adapter_are_trainable():
    projector = ResidualPacketProjector(make_base(), hidden_size=16, bottleneck_size=4)

    assert all(
        not parameter.requires_grad
        for parameter in projector.base_projector.parameters()
    )
    assert all(parameter.requires_grad for parameter in projector.input_norm.parameters())
    assert all(parameter.requires_grad for parameter in projector.adapter.parameters())


def test_state_dict_can_be_saved_and_reloaded(tmp_path):
    torch.manual_seed(11)
    projector = ResidualPacketProjector(make_base(), hidden_size=16, bottleneck_size=4)
    inputs = torch.randn(5, 16)
    with torch.no_grad():
        projector.adapter[-1].weight.normal_()
        expected = projector(inputs)

    checkpoint = tmp_path / "projector.pt"
    torch.save(projector.state_dict(), checkpoint)
    restored = ResidualPacketProjector(
        copy.deepcopy(projector.base_projector), hidden_size=16, bottleneck_size=4
    )
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))

    with torch.no_grad():
        actual = restored(inputs)
    assert torch.equal(expected, actual)
    assert all(
        not parameter.requires_grad
        for parameter in restored.base_projector.parameters()
    )
