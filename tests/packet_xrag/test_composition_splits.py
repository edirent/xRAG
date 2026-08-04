import pytest

from src.packet_xrag.composition.protocol import (
    QUARANTINED_ID, diagnostic_ids, ordered_ids_sha256, split_composition_ids,
)


def locked_ids():
    import json
    return json.load(open("cache/controller/splits/controller_effective_train_ids.json"))["ordered_sample_ids"]


def test_composition_split_is_deterministic_disjoint_and_quarantined():
    first = split_composition_ids(locked_ids()); second = split_composition_ids(locked_ids())
    assert first == second
    train, dev, shadow = first
    assert [len(train), len(dev), len(shadow)] == [3999, 250, 250]
    assert not set(train) & set(dev) and not set(train) & set(shadow) and not set(dev) & set(shadow)
    assert QUARANTINED_ID not in set(train + dev + shadow)


def test_diagnostic_subset_is_locked_to_dev():
    _, dev, _ = split_composition_ids(locked_ids())
    subset = diagnostic_ids(dev)
    assert len(subset) == 150 and set(subset) <= set(dev)
    assert subset == diagnostic_ids(dev)


def test_wrong_effective_hash_is_rejected():
    values = locked_ids(); values[0], values[1] = values[1], values[0]
    with pytest.raises(ValueError, match="hash mismatch"):
        split_composition_ids(values)
