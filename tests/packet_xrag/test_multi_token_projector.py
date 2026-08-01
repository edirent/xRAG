import io

import torch
from torch import nn

from src.packet_xrag.encoding.multi_token_projector import MultiTokenPacketProjector


def make_projector(k):
    return MultiTokenPacketProjector(nn.Linear(7, 11), 7, 11, k, 5)


def test_shapes_first_token_zero_extras_and_freezing():
    x = torch.randn(3, 7)
    for k in (1, 2, 4):
        projector = make_projector(k)
        output = projector(x)
        assert output.shape == (3, k, 11)
        assert torch.allclose(output[:, 0], projector.base_projector(x), atol=1e-5, rtol=1e-5)
        if k > 1:
            assert torch.allclose(output[:, 1:], torch.zeros_like(output[:, 1:]), atol=1e-6, rtol=0)
        assert not any(p.requires_grad for p in projector.base_projector.parameters())
        assert all(p.requires_grad for p in projector.extra_heads.parameters())


def test_state_dict_round_trip():
    x = torch.randn(3, 7)
    before = make_projector(4)
    buffer = io.BytesIO(); torch.save(before.state_dict(), buffer); buffer.seek(0)
    after = make_projector(4); after.load_state_dict(torch.load(buffer, weights_only=True))
    assert torch.equal(before(x), after(x))


def test_invalid_shape_and_k():
    with torch.no_grad():
        try:
            make_projector(0)
            assert False
        except ValueError:
            pass
    try:
        make_projector(2)(torch.randn(7))
        assert False
    except ValueError:
        pass
