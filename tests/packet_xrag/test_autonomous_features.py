import pytest

from src.packet_xrag.controller.autonomous_features import (
    average_precision, binary_auc, candidate_metrics, repetition_fraction,
    stop_metrics,
)


def test_binary_metrics_rank_perfect_classifier():
    scores = [.1, .8, .2, .9]
    labels = [False, True, False, True]
    assert binary_auc(scores, labels) == 1.0
    assert average_precision(scores, labels) == 1.0
    assert stop_metrics(scores, labels)["balanced_accuracy"] == 1.0


def test_repetition_fraction_handles_empty_and_duplicates():
    assert repetition_fraction([]) == 0.0
    assert repetition_fraction([1, 1, 2, 2]) == .5


def test_candidate_metrics_use_state_local_choice_and_regret():
    rows = [
        {"sample_id": "a", "selected_packet_ids": [], "candidate_packet_id": 0,
         "delta_utility": -.1},
        {"sample_id": "a", "selected_packet_ids": [], "candidate_packet_id": 1,
         "delta_utility": .3},
    ]
    metrics = candidate_metrics([0.0, 1.0], rows)
    assert metrics["best_action_top1_accuracy"] == 1.0
    assert metrics["best_action_top3_recall"] == 1.0
    assert metrics["pairwise_ranking_accuracy"] == 1.0
    assert metrics["teacher_policy_regret"] == 0.0
    assert metrics["within_state_spearman"] == pytest.approx(1.0)
