import torch

from scripts.packet_xrag.run_k2_selector_benchmark import mmr_selection
from src.packet_xrag.controller.feature_cache import (
    ControllerFeatureCache,
    ControllerFeatureWriter,
    ranking_features,
)


def sample():
    return {"id": "s1", "question": "question", "answer": "answer"}


def packets():
    return [
        {"packet_id": 0, "doc_id": 0, "title": "a", "sentence_id": 0,
         "text": "one", "encoder_text": "[a] one", "is_supporting": True},
        {"packet_id": 1, "doc_id": 0, "title": "a", "sentence_id": 1,
         "text": "two", "encoder_text": "[a] two", "is_supporting": False},
    ]


def complete_fields():
    return {
        "effective_split_hash": "split", "quarantine_hash": "quarantine",
        "source_dataset_identifier": "dataset", "packet_construction_version": "packets",
        "sfr_checkpoint_identifier": "sfr", "query_encoding_template_hash": "query",
        "number_of_samples": 1, "number_of_packets": 2, "creation_command": "test",
    }


def test_embedding_shapes_round_trip_and_topk_consistency(tmp_path):
    embeddings = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.2, 0.8, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0],
    ], dtype=torch.bfloat16)
    writer = ControllerFeatureWriter(tmp_path, hidden_size=4)
    record = writer.add(sample(), packets(), [0], embeddings)
    writer.close(complete_fields())
    cache = ControllerFeatureCache(tmp_path)
    item = cache[0]
    assert len(cache) == 1
    assert item["query_embedding"].shape == (4,)
    assert item["packet_embeddings"].shape == (2, 4)
    assert torch.equal(item["query_embedding"], embeddings[0])
    assert torch.equal(item["packet_embeddings"], embeddings[1:])
    relevance, topk, mmr = ranking_features(embeddings)
    assert record["topk_ranking"] == topk == [1, 0]
    assert record["mmr_ranking"] == mmr
    assert torch.allclose(torch.tensor(record["topk_scores"]), relevance)


def test_topk_tie_breaks_by_packet_id():
    embeddings = torch.tensor([[1.0, 0.0], [1.0, 1.0], [1.0, -1.0]])
    _, topk, _ = ranking_features(embeddings)
    assert topk == [0, 1]


def test_cached_mmr_ranking_matches_frozen_benchmark_reference():
    generator = torch.Generator().manual_seed(20260803)
    embeddings = torch.randn(18, 32, generator=generator)
    relevance, _, actual = ranking_features(embeddings)
    normalized_packets = torch.nn.functional.normalize(embeddings[1:].float(), dim=-1)
    expected, _ = mmr_selection(relevance, normalized_packets, len(normalized_packets), 0.5)
    assert actual == expected
