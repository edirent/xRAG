import torch

from src.packet_xrag.composition.fused_xrag_injection import replace_xrag_placeholders


class FakeModel:
    class Body:
        @staticmethod
        def embed_tokens(input_ids):
            return torch.nn.functional.one_hot(input_ids, 8).float()
    model = Body()


def test_fused_tokens_replace_exact_placeholders():
    ids = torch.tensor([[1, 7, 7, 2], [7, 3, 7, 4]])
    fused = torch.randn(2, 2, 8)
    result = replace_xrag_placeholders(FakeModel(), ids, 7, fused)
    assert torch.equal(result[ids == 7], fused.reshape(-1, 8))
