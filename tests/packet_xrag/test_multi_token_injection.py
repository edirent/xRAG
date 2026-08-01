import torch
from torch import nn

from src.packet_xrag.encoding.multi_token_projector import MultiTokenPacketProjector
from src.packet_xrag.modeling.multi_token_xrag import prepare_multi_token_inputs_embeds


class Dummy(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.model = nn.Module(); self.model.embed_tokens = nn.Embedding(20, 4)
        self.retriever_hidden_size = 3; self.xrag_token_id = 19
        self.projector = MultiTokenPacketProjector(nn.Linear(3, 4, bias=False), 3, 4, k, 2)


def test_token_counts_and_packet_major_flatten_order():
    packets = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    for k in (1, 2, 4):
        model = Dummy(k)
        with torch.no_grad():
            for head_index, head in enumerate(model.projector.extra_heads, 1):
                head[-1].bias.fill_(head_index)
        input_ids = torch.tensor([[1] + [19] * (2 * k) + [2]])
        result = prepare_multi_token_inputs_embeds(model, input_ids, packets)
        expected = model.projector(packets).reshape(2 * k, 4)
        assert result.shape == (1, 2 * k + 2, 4)
        assert torch.equal(result[0, 1:-1], expected)
        assert torch.equal(expected[k - 1], model.projector(packets)[0, k - 1])
        assert torch.equal(expected[k], model.projector(packets)[1, 0])


def test_misaligned_xrag_count_fails():
    model = Dummy(2)
    try:
        prepare_multi_token_inputs_embeds(model, torch.tensor([[19, 19]]), torch.randn(2, 3))
        assert False
    except AssertionError:
        pass
