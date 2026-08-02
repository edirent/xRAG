import torch
from torch import nn

from src.packet_xrag.encoding.token_state_resampler import ResidualTokenStateResampler
from src.packet_xrag.modeling.token_state_xrag import prepare_token_state_inputs_embeds


class DummyK2(nn.Module):
    tokens_per_packet = 2
    def __init__(self):
        super().__init__(); self.linear = nn.Linear(3, 8)
    def forward(self, pooled):
        return self.linear(pooled).view(-1, 2, 4)


class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module(); self.model.embed_tokens = nn.Embedding(20, 4)
        self.xrag_token_id = 19
        self.projector = ResidualTokenStateResampler(
            DummyK2(), 3, 4, latent_size=4, num_heads=2, ffn_size=8
        )


def test_two_tokens_per_packet_and_packet_major_order():
    model = Dummy()
    with torch.no_grad(): model.projector.output_projection.weight.normal_(std=0.1)
    states = torch.randn(2, 3, 3); mask = torch.ones(2, 3, dtype=torch.bool)
    pooled = torch.randn(2, 3); ids = torch.tensor([[1, 19, 19, 19, 19, 2]])
    result = prepare_token_state_inputs_embeds(model, ids, states, mask, pooled)
    expected = model.projector(states, mask, pooled).reshape(4, 4)
    assert torch.equal(result[0, 1:5], expected)
    assert torch.equal(expected[1], model.projector(states, mask, pooled)[0, 1])
    assert torch.equal(expected[2], model.projector(states, mask, pooled)[1, 0])


def test_slot_mismatch_fails():
    model = Dummy()
    try:
        prepare_token_state_inputs_embeds(
            model, torch.tensor([[19, 19]]), torch.randn(2, 2, 3),
            torch.ones(2, 2, dtype=torch.bool), torch.randn(2, 3)
        )
        assert False
    except ValueError:
        pass
