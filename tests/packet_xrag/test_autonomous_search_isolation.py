import pytest

from src.packet_xrag.controller.autonomous_search import (
    FORBIDDEN_INFERENCE_KEYS, assert_no_inference_leakage, assert_only_search_dev,
    fixed_probe_subset, split_search_ids,
)


def test_search_split_is_deterministic_disjoint_and_locked_size():
    ids = [f"s{index}" for index in range(500)]
    first = split_search_ids(ids); second = split_search_ids(ids)
    assert first == second
    assert len(first[0]) == 350 and len(first[1]) == 150
    assert not set(first[0]) & set(first[1])
    assert set(first[0] + first[1]) == set(ids)
    subset = fixed_probe_subset(first[0])
    assert len(subset) == 150 and set(subset) <= set(first[0])


def test_non_search_dev_ids_are_rejected():
    with pytest.raises(RuntimeError, match="non-SEARCH_DEV"):
        assert_only_search_dev(["dev", "shadow"], ["dev"])


@pytest.mark.parametrize("key", sorted(FORBIDDEN_INFERENCE_KEYS))
def test_gold_derived_inference_keys_are_rejected(key):
    with pytest.raises(RuntimeError, match="leakage"):
        assert_no_inference_leakage({"features": [{key: 1.0}]})


def test_provisional_answer_is_allowed():
    assert assert_no_inference_leakage({"provisional_answer": "Paris", "entropy": 0.2})
