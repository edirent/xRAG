import torch
from torch import nn

from src.packet_xrag.encoding.token_state_resampler import ResidualTokenStateResampler


class K2(nn.Module):
    tokens_per_packet = 2
    def __init__(self):
        super().__init__(); self.projection = nn.Linear(5, 14)
    def forward(self, pooled):
        return self.projection(pooled).view(-1, 2, 7)


def make():
    return ResidualTokenStateResampler(K2(), 5, 7, 8, 2, 2, 16)


def test_strict_checkpoint_loading_and_zero_equivalence(tmp_path):
    before = make(); path = tmp_path / "resampler.pt"
    torch.save(before.state_dict(), path)
    after = make(); after.load_state_dict(torch.load(path, weights_only=True), strict=True)
    states = torch.randn(2, 4, 5); mask = torch.ones(2, 4, dtype=torch.bool); pooled = torch.randn(2, 5)
    assert torch.equal(before(states, mask, pooled), after(states, mask, pooled))
    assert torch.equal(after(states, mask, pooled), after.pooled_k2_projector(pooled))
