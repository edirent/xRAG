import json

import pytest

from src.packet_xrag.composition.protocol import (
    assert_evaluation_lock, assert_final_100_authorized,
)


def test_evaluation_lock_allows_only_remaining_runs(tmp_path):
    path = tmp_path / "lock.json"
    path.write_text(json.dumps({"split": "SHADOW", "runs": 0, "maximum_runs": 2}))
    assert assert_evaluation_lock(path, "SHADOW", 2)["runs"] == 0
    path.write_text(json.dumps({"split": "SHADOW", "runs": 2, "maximum_runs": 2}))
    with pytest.raises(RuntimeError, match="exhausted"):
        assert_evaluation_lock(path, "SHADOW", 2)


def test_benchmark_lock_is_single_run(tmp_path):
    path = tmp_path / "benchmark_lock.json"
    path.write_text(json.dumps({"split": "BENCHMARK_500", "runs": 0, "maximum_runs": 1}))
    assert_evaluation_lock(path, "BENCHMARK_500", 1)
    path.write_text(json.dumps({"split": "BENCHMARK_500", "runs": 1, "maximum_runs": 1}))
    with pytest.raises(RuntimeError, match="exhausted"):
        assert_evaluation_lock(path, "BENCHMARK_500", 1)


def test_final_100_requires_separate_explicit_authorization(tmp_path):
    path = tmp_path / "final_authorization.json"
    with pytest.raises(RuntimeError, match="separate explicit authorization"):
        assert_final_100_authorized(path)
    path.write_text(json.dumps({"explicit_final_100_authorization": False}))
    with pytest.raises(RuntimeError, match="separate explicit authorization"):
        assert_final_100_authorized(path)
