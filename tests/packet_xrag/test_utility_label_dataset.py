import json

import pytest
import torch

from src.packet_xrag.controller.utility_label_dataset import (
    LABEL_FORMAT, ShardedUtilityLabelDataset, build_full_state_pool,
    normalize_delta, utility_target_statistics,
)


def record():
    return {
        "gold_packet_ids": [4, 1], "topk_ranking": [3, 2, 0, 5, 4, 1],
        "packets": [{"packet_id": index} for index in range(6)],
    }


def test_full_state_pool_preserves_order_and_applies_five_state_cap():
    states, omitted = build_full_state_pool(record(), [1, 3, 0, 5, 2, 4], oracle_packet_id=2)
    assert len(states) == 5
    assert omitted == ["S_STATIC1_five_state_cap"]
    sufficient = next(state for state in states if "S_SUFFICIENT" in state["state_source_tags"])
    assert sufficient["selected_packet_ids"] == [4, 1]
    tags = {tag for state in states for tag in state["state_source_tags"]}
    assert {"S0_EMPTY", "S_WRONG1", "S_SUFFICIENT", "S_ORACLE1"}.issubset(tags)


def test_oracle_state_is_not_created_for_nonpositive_s0_utility():
    states, _ = build_full_state_pool(record(), [4, 3, 0, 5, 1, 2], oracle_packet_id=None)
    assert all("S_ORACLE1" not in state["state_source_tags"] for state in states)


def test_target_statistics_use_train_p99_absolute_and_minimum():
    stats = utility_target_statistics([-0.01, 0.02, 0.03])
    assert stats["utility_clip_value"] == 0.1
    values = torch.tensor([-2.0, 0.5, 3.0])
    assert torch.equal(normalize_delta(values, 2.0), torch.tensor([-1.0, .25, 1.0]))


def test_sharded_reader_rejects_incomplete_manifest(tmp_path):
    root = tmp_path / "labels"; root.mkdir()
    (root / "train_manifest.json").write_text(json.dumps({
        "format": LABEL_FORMAT, "completion_status": "in_progress",
    }))
    with pytest.raises(ValueError):
        ShardedUtilityLabelDataset(root, "train")
