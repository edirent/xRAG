import torch

from scripts.packet_xrag.run_k2_selector_benchmark import (
    mmr_selection,
    stable_random_indices,
    topk_selection,
)


def test_topk_sorting_tie_break_and_budget():
    relevance = torch.tensor([0.2, 0.8, 0.8, -0.1])
    assert topk_selection(relevance, 1) == [1]
    assert topk_selection(relevance, 3) == [1, 2, 0]
    assert topk_selection(relevance, 20) == [1, 2, 0, 3]


def test_mmr_first_relevance_then_novelty_and_ties():
    relevance = torch.tensor([0.9, 0.8, 0.8])
    packets = torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
    packets = torch.nn.functional.normalize(packets, dim=-1)
    selected, details = mmr_selection(relevance, packets, 3, 0.5)
    assert selected[0] == 0
    assert selected[1] == 2
    assert details[0]["mmr_score_at_selection"] == float(relevance[0])
    assert details[2]["max_similarity_to_selected"] == 0.0
    tied, _ = mmr_selection(torch.tensor([0.5, 0.5]), torch.eye(2), 1, 0.5)
    assert tied == [0]


def test_random_is_reproducible_distinct_and_without_replacement():
    first = stable_random_indices(20, 6, "sample", 13)
    assert first == stable_random_indices(20, 6, "sample", 13)
    assert first != stable_random_indices(20, 6, "sample", 37)
    assert len(first) == len(set(first)) == 6
    assert stable_random_indices(3, 6, "sample", 13) == [0, 1, 2]

