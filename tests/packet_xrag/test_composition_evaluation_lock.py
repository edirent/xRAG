import json

import pytest

from src.packet_xrag.composition.protocol import assert_evaluation_lock


def test_evaluation_lock_allows_only_remaining_runs(tmp_path):
    path = tmp_path / "lock.json"
    path.write_text(json.dumps({"split": "SHADOW", "runs": 0, "maximum_runs": 2}))
    assert assert_evaluation_lock(path, "SHADOW", 2)["runs"] == 0
    path.write_text(json.dumps({"split": "SHADOW", "runs": 2, "maximum_runs": 2}))
    with pytest.raises(RuntimeError, match="exhausted"):
        assert_evaluation_lock(path, "SHADOW", 2)
