import io

import torch
from torch import nn

from src.packet_xrag.encoding.token_state_resampler import (
    PooledResidualResamplerControl, ResidualTokenStateResampler,
)


class DummyK2(nn.Module):
    tokens_per_packet = 2

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(7, 22)

    def forward(self, pooled):
        return self.linear(pooled).view(-1, 2, 11)


def make_model():
    return ResidualTokenStateResampler(
        DummyK2(), 7, 11, latent_size=8, num_latents=2, num_heads=2, ffn_size=16
    )


def inputs():
    states = torch.randn(3, 6, 7)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    pooled = torch.randn(3, 7)
    return states, mask, pooled


def test_shape_zero_init_equivalence_and_freezing():
    model = make_model()
    states, mask, pooled = inputs()
    output = model(states, mask, pooled)
    assert output.shape == (3, 2, 11)
    assert torch.equal(output, model.pooled_k2_projector(pooled))
    assert not any(p.requires_grad for p in model.pooled_k2_projector.parameters())
    assert all(p.requires_grad for n, p in model.named_parameters() if not n.startswith("pooled_k2_projector."))


def test_padding_invariance_and_nonpadding_sensitivity():
    torch.manual_seed(4)
    model = make_model()
    with torch.no_grad():
        model.output_projection.weight.normal_(std=0.1)
    states, mask, pooled = inputs()
    baseline = model(states, mask, pooled)
    changed_padding = states.clone(); changed_padding[~mask] += 1000
    assert torch.allclose(baseline, model(changed_padding, mask, pooled), atol=1e-5, rtol=1e-5)
    changed_token = states.clone(); changed_token[0, 0, 0] += 1
    assert not torch.allclose(baseline[0], model(changed_token, mask, pooled)[0])


def test_checkpoint_round_trip_is_deterministic():
    model = make_model().eval()
    states, mask, pooled = inputs()
    before = model(states, mask, pooled)
    buffer = io.BytesIO(); torch.save(model.state_dict(), buffer); buffer.seek(0)
    loaded = make_model().eval(); loaded.load_state_dict(torch.load(buffer, weights_only=True))
    assert torch.equal(before, loaded(states, mask, pooled))


def test_pooled_control_is_parameter_matched_and_ignores_token_states():
    token_model = make_model()
    control = PooledResidualResamplerControl(
        DummyK2(), 7, 11, latent_size=8, num_latents=2, num_heads=2, ffn_size=16
    )
    assert sum(p.numel() for p in token_model.parameters()) == sum(p.numel() for p in control.parameters())
    states, mask, pooled = inputs()
    with torch.no_grad(): control.output_projection.weight.normal_(std=0.1)
    first = control(states, mask, pooled)
    assert torch.equal(first, control(states + 1000, ~mask, pooled))
