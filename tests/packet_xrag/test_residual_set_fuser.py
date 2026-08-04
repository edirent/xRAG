import torch

from src.packet_xrag.composition.residual_set_fuser import ResidualSetFuser


def test_residual_zero_gate_exactly_reproduces_static2():
    torch.manual_seed(3); model = ResidualSetFuser(dimension=16, latent_dim=8,
                                                   output_slots=4, heads=2).eval()
    query = torch.randn(2, 16); base = torch.randn(2, 4, 16)
    extras = torch.randn(2, 4, 2, 16); mask = torch.ones(2, 4, dtype=torch.bool)
    output, alpha = model(query, base, extras, mask)
    assert torch.equal(output, base)
    assert torch.equal(alpha, torch.zeros_like(alpha))
