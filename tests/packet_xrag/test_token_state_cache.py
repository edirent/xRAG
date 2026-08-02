import torch

from src.packet_xrag.encoding.token_state_cache import (
    TokenStateCache,
    TokenStateShardWriter,
    packet_key,
)


def test_sharded_cache_round_trip_and_determinism(tmp_path):
    writer = TokenStateShardWriter(tmp_path, hidden_size=7, target_bytes=100)
    expected = {}
    for index, length in enumerate((3, 5, 2)):
        key = packet_key(f"packet {index}")
        hidden = torch.arange(length * 7).view(length, 7).to(torch.bfloat16) + index
        pooled = hidden[-1].clone()
        ids = torch.arange(length, dtype=torch.int32) + index
        mask = torch.ones(length, dtype=torch.bool)
        metadata = {"text": f"packet {index}", "title_span": [1, 2], "sentence_span": [2, length]}
        writer.add(key, hidden, pooled, ids, mask, metadata)
        expected[key] = (hidden, pooled, ids, mask)
    manifest = writer.close({"split_hash": "locked"})
    assert len(manifest["shards"]) > 1
    cache = TokenStateCache(tmp_path, max_open_shards=1)
    for key, values in expected.items():
        first, second = cache.get(key), cache.get(key)
        for field, value in zip(("hidden", "pooled", "input_ids", "mask"), values):
            assert torch.equal(first[field], value)
            assert torch.equal(first[field], second[field])

