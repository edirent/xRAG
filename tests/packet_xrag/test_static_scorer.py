import torch

from scripts.packet_xrag.train_static_scorer import choose_budget, choose_checkpoint
from src.packet_xrag.controller.static_scorer import (
    FEATURE_DIM,
    StaticPacketScorer,
    negative_analysis_labels,
    packet_position_features,
)


def test_static_scorer_architecture_and_output_shape():
    model = StaticPacketScorer(input_dim=8, projection_dim=4, dropout=0.1)
    assert model.query_projection.in_features == 8
    assert model.query_projection.out_features == 4
    assert model.packet_projection.out_features == 4
    assert model.scorer[0].in_features == 20
    assert model.scorer[0].out_features == 2048
    assert model.scorer[3].out_features == 512
    scores = model(torch.randn(8), torch.randn(3, 8), torch.rand(3, 2))
    assert scores.shape == (3,)
    assert FEATURE_DIM == 2052


def test_packet_position_features_are_normalized_per_document():
    packets = [
        {"doc_id": 0, "sentence_id": 0}, {"doc_id": 0, "sentence_id": 1},
        {"doc_id": 0, "sentence_id": 2}, {"doc_id": 1, "sentence_id": 0},
    ]
    features = packet_position_features(packets)
    assert torch.allclose(features[:, 0], torch.tensor([0.0, 0.5, 1.0, 0.0]))
    assert torch.allclose(features[:, 1], torch.tensor([1.0, 1.0, 1.0, 1 / 3]))


def test_negative_analysis_labels_are_exhaustive_and_deterministic():
    packets = [
        {"doc_id": 0, "sentence_id": 0, "is_supporting": True},
        {"doc_id": 0, "sentence_id": 1, "is_supporting": False},
        {"doc_id": 0, "sentence_id": 3, "is_supporting": False},
        {"doc_id": 1, "sentence_id": 0, "is_supporting": False},
    ]
    labels = negative_analysis_labels(packets, [0], [0, 3, 2, 1])
    assert labels == {
        1: "gold-neighbor sentence", 2: "TOPK top-6 non-gold",
        3: "TOPK top-6 non-gold",
    }


def test_internal_dev_selection_prefers_smaller_budget_within_point_25_then_loss():
    metrics = {
        "STATIC_1": {"short_f1": 60.0}, "STATIC_2": {"short_f1": 61.0},
        "STATIC_3": {"short_f1": 61.20}, "STATIC_4": {"short_f1": 61.30},
    }
    assert choose_budget(metrics) == 3
    history = [
        {"epoch": 1, "selected_short_f1": 61.30, "selected_budget": 4,
         "validation_listwise_loss": 0.5},
        {"epoch": 2, "selected_short_f1": 61.20, "selected_budget": 3,
         "validation_listwise_loss": 0.6},
        {"epoch": 3, "selected_short_f1": 61.20, "selected_budget": 3,
         "validation_listwise_loss": 0.4},
    ]
    assert choose_checkpoint(history)["epoch"] == 3


def test_flattened_question_batch_matches_individual_scoring():
    model = StaticPacketScorer(input_dim=8, projection_dim=4, dropout=0.0).eval()
    records = []
    for index, count in enumerate((2, 3)):
        records.append({
            "query_embedding": torch.randn(8),
            "packet_embeddings": torch.randn(count, 8),
            "packet_count": count,
            "packets": [{"doc_id": 0, "sentence_id": packet_id}
                        for packet_id in range(count)],
        })
    individual = [model.score_record(record) for record in records]
    flattened = model.score_records(records)
    for expected, actual in zip(individual, flattened):
        assert torch.allclose(expected, actual, atol=1e-6)
