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


def test_packet_breadth_does_not_expand_llm_context_after_fusion():
    # The upstream fuser may consume N=12 (24 K2 tokens), but injection sees only M=4.
    ids = torch.tensor([[1, 7, 7, 7, 7, 2]])
    fused_from_twelve_packets = torch.randn(1, 4, 8)
    result = replace_xrag_placeholders(FakeModel(), ids, 7, fused_from_twelve_packets)
    assert result.shape[1] == ids.shape[1]
    assert ids.eq(7).sum().item() == 4
