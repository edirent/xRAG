import pytest

from scripts.packet_xrag.build_controller_data import preflight_gold_mappings
from src.packet_xrag.controller.feature_cache import (
    assert_no_overlap,
    make_candidate_packets,
    normalized_question,
    overlap_audit,
    split_records,
)


def record(index, question=None):
    sample = {"id": str(index), "question": question or f"Question {index}"}
    return sample, [], []


def test_split_is_deterministic_disjoint_and_complete():
    source = [record(index) for index in range(5000)]
    train_a, dev_a = split_records(source)
    train_b, dev_b = split_records(source)
    train_ids = [item[0]["id"] for item in train_a]
    dev_ids = [item[0]["id"] for item in dev_a]
    assert train_ids == [item[0]["id"] for item in train_b]
    assert dev_ids == [item[0]["id"] for item in dev_b]
    assert len(train_ids) == 4500 and len(dev_ids) == 500
    assert set(train_ids).isdisjoint(dev_ids)
    assert set(train_ids) | set(dev_ids) == {str(index) for index in range(5000)}


def test_question_hash_overlap_detects_exact_and_normalized_duplicates():
    named = {
        "train": [record(1, "Who wrote Hamlet?")],
        "dev": [record(2, "Different")],
        "benchmark": [record(3, " WHO wrote—Hamlet ? ")],
    }
    overlaps = overlap_audit(named)
    assert not overlaps["train__benchmark"]["exact_question_overlap"]
    assert overlaps["train__benchmark"]["normalized_question_overlap"]
    with pytest.raises(RuntimeError, match="data isolation failed"):
        assert_no_overlap(overlaps)
    assert normalized_question(" WHO wrote—Hamlet ? ") == "whowrotehamlet"


def test_packet_ids_and_gold_mapping_are_deterministic_and_complete():
    sample = {
        "context": {"title": ["A", "B"], "sentences": [[" one ", "two"], ["three"]]},
        "supporting_facts": {"title": ["A", "B"], "sent_id": [1, 0]},
    }
    packets_a, gold_a = make_candidate_packets(sample)
    packets_b, gold_b = make_candidate_packets(sample)
    assert packets_a == packets_b
    assert [packet["packet_id"] for packet in packets_a] == [0, 1, 2]
    assert gold_a == gold_b == [1, 2]
    assert {packet["packet_id"] for packet in packets_a if packet["is_supporting"]} == set(gold_a)


def test_gold_mapping_failure_is_hard_error():
    sample = {
        "context": {"title": ["A"], "sentences": [["one"]]},
        "supporting_facts": {"title": ["missing"], "sent_id": [0]},
    }
    with pytest.raises(ValueError, match="gold mapping failed"):
        make_candidate_packets(sample)


def test_gold_mapping_preflight_reports_split_index_and_sample_id():
    sample = {
        "id": "broken-id", "question": "broken question",
        "context": {"title": ["A"], "sentences": [["one"]]},
        "supporting_facts": {"title": ["A"], "sent_id": [2]},
    }
    with pytest.raises(RuntimeError, match=r"MANDATORY STOP.*broken-id"):
        preflight_gold_mappings({"train": [(sample, [], [])]})
