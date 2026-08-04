import hashlib
import json

import pytest

from src.packet_xrag.generalization.protocol import (
    assert_inference_view, assert_manifest_immutable, consume_single_run_lock,
    deterministic_partition, split_audit,
)


def test_generalization_split_is_deterministic_and_disjoint():
    train = [f"t{i}" for i in range(5000)]; validation = [f"v{i}" for i in range(600)]
    first = deterministic_partition(train, validation)
    assert first == deterministic_partition(train, validation)
    assert [len(first[name]) for name in ("train", "dev", "shadow", "benchmark")] == [3999, 250, 250, 500]
    assert not set(first["train"]) & set(first["dev"])


def test_split_audit_rejects_question_overlap():
    def sample(sid, question):
        return {"id": sid, "question": question, "answers": [sid],
                "documents": [{"document_id": sid}]}
    with pytest.raises(RuntimeError, match="split leakage"):
        split_audit({"train": [sample("1", "Same question?")],
                     "dev": [sample("2", "same QUESTION")]})


def test_support_label_leakage_audit():
    assert assert_inference_view({"query_embedding": 1, "packet_embeddings": 2})
    with pytest.raises(RuntimeError, match="leaked"):
        assert_inference_view({"packet_embeddings": 2, "contains_answer": False})


def test_final_suite_lock_and_manifest_immutability(tmp_path):
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"split": "FINAL", "runs": 0, "maximum_runs": 1}))
    consume_single_run_lock(lock, "FINAL")
    with pytest.raises(RuntimeError, match="not pristine"):
        consume_single_run_lock(lock, "FINAL")
    manifest = tmp_path / "manifest.json"; manifest.write_text("{}")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert assert_manifest_immutable(manifest, digest)
    manifest.write_text('{"changed": true}')
    with pytest.raises(RuntimeError, match="changed"):
        assert_manifest_immutable(manifest, digest)
