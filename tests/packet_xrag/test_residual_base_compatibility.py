import pytest
import torch

from scripts.packet_xrag.composition_training_common import static2_k2_base
from src.packet_xrag.composition.residual_set_fuser import ResidualSetFuser


def test_static2_base_replicates_singleton_packet_deterministically():
    packet = torch.arange(2 * 4096, dtype=torch.float32).reshape(1, 2, 4096)
    base = static2_k2_base(packet)
    assert base.shape == (4, 4096)
    assert torch.equal(base[:2], base[2:])


def test_static2_base_preserves_first_two_packets():
    packets = torch.arange(3 * 2 * 4096, dtype=torch.float32).reshape(3, 2, 4096)
    assert torch.equal(static2_k2_base(packets), packets[:2].reshape(4, 4096))
    with pytest.raises(ValueError, match="at least one"):
        static2_k2_base(torch.empty(0, 2, 4096))


def test_residual_fuser_mixed_batch_leaves_no_extra_row_at_base():
    fuser = ResidualSetFuser(dimension=8, latent_dim=8, output_slots=4, heads=2)
    with torch.no_grad():
        fuser.gate.bias.fill_(1.0)
    query = torch.randn(2, 8); base = torch.randn(2, 4, 8)
    extras = torch.randn(2, 2, 2, 8)
    mask = torch.tensor([[False, False], [True, True]])
    output, alpha = fuser(query, base, extras, mask)
    assert torch.equal(output[0], base[0])
    assert alpha[0].item() == 0.0
    assert not torch.equal(output[1], base[1])
    output.sum().backward()
    assert fuser.gate.bias.grad is not None
    assert fuser.residual_fuser.output_projection.weight.grad is not None
