import torch

from scripts.packet_xrag.train_sequential_controller import trajectory_states
from src.packet_xrag.controller.sequential_controller import (
    FEATURE_DIM,
    SequentialPacketController,
    greedy_rollout,
)


def record():
    return {
        "query_embedding": torch.randn(8), "packet_embeddings": torch.randn(4, 8),
        "packet_count": 4, "gold_packet_ids": [1, 3],
        "packets": [{"doc_id": i // 2, "sentence_id": i % 2} for i in range(4)],
    }


def test_selected_candidates_are_masked_and_state_changes_logits():
    model = SequentialPacketController(input_dim=8, projection_dim=4, dropout=0).eval()
    item = record()
    remaining_empty, empty_scores = model.score_record(item, [])
    remaining, scores = model.score_record(item, [0])
    assert remaining_empty == [0, 1, 2, 3]
    assert remaining == [1, 2, 3]
    assert not torch.allclose(empty_scores[1:], scores)
    assert FEATURE_DIM == 4103


def test_greedy_rollout_has_no_duplicates_and_respects_budget():
    model = SequentialPacketController(input_dim=8, projection_dim=4, dropout=0).eval()
    selected = greedy_rollout(model, record(), 3)
    assert len(selected) == len(set(selected)) == 3


def test_trajectory_mixture_is_locked_and_never_covers_all_gold():
    item=record();item["sample_id"]="sample";item["topk_ranking"]=[0,1,2,3]
    states=trajectory_states(item,[2,0,3,1],epoch=1)
    categories=[category for category,_ in states]
    assert categories.count("gold")==4
    assert categories.count("topk")==2 and categories.count("static")==2
    assert categories.count("same_document")==1 and categories.count("random_error")==1
    assert all(not set(selected)>=set(item["gold_packet_ids"]) for _,selected in states)
