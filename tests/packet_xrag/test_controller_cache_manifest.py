import json

import pytest
import torch

from src.packet_xrag.controller.feature_cache import (
    ControllerFeatureCache,
    ControllerFeatureWriter,
    FORMAT,
    validate_complete_manifest,
)


def required_fields():
    return {
        "effective_split_hash": "split", "quarantine_hash": "quarantine",
        "source_dataset_identifier": "dataset", "packet_construction_version": "packets",
        "sfr_checkpoint_identifier": "sfr", "query_encoding_template_hash": "query",
        "number_of_samples": 1, "number_of_packets": 1, "creation_command": "pytest",
    }


def test_incomplete_manifest_is_rejected(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "format": FORMAT, "completion_status": "in_progress", "hidden_size": 4,
    }))
    with pytest.raises(ValueError, match="incomplete"):
        ControllerFeatureCache(tmp_path)


def test_complete_manifest_missing_provenance_is_rejected():
    with pytest.raises(ValueError, match="missing required fields"):
        validate_complete_manifest({"completion_status": "complete"})


def test_complete_manifest_is_accepted(tmp_path):
    writer = ControllerFeatureWriter(tmp_path, hidden_size=4)
    sample = {"id": "one", "question": "q", "answer": "a"}
    packets = [{
        "packet_id": 0, "doc_id": 0, "title": "T", "sentence_id": 0,
        "text": "p", "encoder_text": "[T] p", "is_supporting": True,
    }]
    writer.add(sample, packets, [0], torch.tensor([[1., 0., 0., 0.], [1., 1., 0., 0.]]))
    writer.close(required_fields())
    cache = ControllerFeatureCache(tmp_path)
    assert len(cache) == 1
    assert cache.manifest["completion_status"] == "complete"
