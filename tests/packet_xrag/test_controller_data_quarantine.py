import json

import pytest

from scripts.packet_xrag.build_controller_data import preflight_gold_mappings
from src.packet_xrag.controller.feature_cache import (
    filter_quarantined_records,
    load_effective_split_ids,
    ordered_ids_sha256,
    quarantine_fraction,
)


def write_inputs(tmp_path):
    original_ids = ["good-a", "malformed", "good-b"]
    split = tmp_path / "controller_train_ids.json"
    split.write_text(json.dumps({
        "sample_count": 3,
        "ordered_sample_ids": original_ids,
        "sha256": ordered_ids_sha256(original_ids),
    }, sort_keys=True) + "\n")
    quarantine = tmp_path / "controller_quarantine.json"
    quarantine.write_text(json.dumps({
        "schema_version": 1,
        "entries": [{
            "sample_id": "malformed", "split": "controller_train",
            "reason": "supporting_fact_sentence_out_of_range", "title": "T",
            "requested_sentence_id": 2, "available_sentence_ids": [0, 1],
            "action": "exclude_from_all_controller_training_and_label_generation",
        }],
    }, sort_keys=True) + "\n")
    return split, quarantine


def record(sample_id):
    return ({"id": sample_id, "question": sample_id}, [], [])


def test_effective_train_excludes_quarantine_deterministically_without_mutating_original(tmp_path):
    split, quarantine = write_inputs(tmp_path)
    original_bytes = split.read_bytes()
    first = load_effective_split_ids(split, quarantine)
    second = load_effective_split_ids(split, quarantine)
    assert first == second == ["good-a", "good-b"]
    assert split.read_bytes() == original_bytes
    assert "malformed" not in first


def test_central_filter_keeps_order_and_blocks_quarantine_from_all_training_caches(tmp_path):
    _, quarantine = write_inputs(tmp_path)
    records = [record("good-a"), record("malformed"), record("good-b")]
    effective = filter_quarantined_records(records, quarantine)
    effective_ids = [item[0]["id"] for item in effective]
    assert effective_ids == ["good-a", "good-b"]
    packet_cache_ids = set(effective_ids)
    feature_cache_ids = set(effective_ids)
    assert "malformed" not in packet_cache_ids
    assert "malformed" not in feature_cache_ids


def test_quarantine_fraction_and_strict_threshold():
    assert quarantine_fraction(4500, 1) == pytest.approx(1 / 4500)
    assert quarantine_fraction(4500, 1) < 0.001
    assert quarantine_fraction(1000, 1) >= 0.001


def test_evaluation_mapping_failure_is_never_silently_quarantined():
    sample = {
        "id": "bad-eval", "question": "bad evaluation annotation",
        "context": {"title": ["T"], "sentences": [["zero", "one"]]},
        "supporting_facts": {"title": ["T"], "sent_id": [2]},
    }
    with pytest.raises(RuntimeError, match=r"MANDATORY STOP.*internal_dev.*bad-eval"):
        preflight_gold_mappings({"internal_dev": [(sample, [], [])]})
