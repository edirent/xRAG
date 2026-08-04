import torch

from src.packet_xrag.composition.set_fuser import QueryConditionedSetFuser


def test_set_fuser_has_fixed_output_slots_and_masks_padding():
    torch.manual_seed(1); model = QueryConditionedSetFuser(input_dim=16, latent_dim=8,
                                                           output_dim=16, output_slots=4,
                                                           heads=2).eval()
    query = torch.randn(2, 16); packets = torch.randn(2, 6, 2, 16)
    mask = torch.tensor([[1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    first = model(query, packets, mask)
    packets[0, 2:] = torch.randn_like(packets[0, 2:]) * 100
    second = model(query, packets, mask)
    assert first.shape == (2, 4, 16)
    assert torch.allclose(first[0], second[0], atol=1e-6)


def test_set_fuser_is_permutation_invariant_without_positions():
    torch.manual_seed(2); model = QueryConditionedSetFuser(input_dim=16, latent_dim=8,
                                                           output_dim=16, heads=2).eval()
    query = torch.randn(1, 16); packets = torch.randn(1, 4, 2, 16)
    mask = torch.ones(1, 4, dtype=torch.bool); order = [2, 0, 3, 1]
    assert torch.allclose(model(query, packets, mask),
                          model(query, packets[:, order], mask[:, order]), atol=1e-5)
