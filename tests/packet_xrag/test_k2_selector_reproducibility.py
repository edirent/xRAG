import hashlib

import torch

from scripts.packet_xrag.run_k2_selector_benchmark import (
    EXPECTED_SPLIT_HASH,
    mmr_selection,
    stable_random_indices,
    topk_selection,
)
from scripts.packet_xrag.bootstrap_k2_selector_benchmark import paired_bootstrap


def test_locked_split_hash_definition_is_reproducible():
    ids = ["a", "b", "c"]
    assert hashlib.sha256("".join(ids).encode()).hexdigest() == hashlib.sha256(b"abc").hexdigest()
    assert len(EXPECTED_SPLIT_HASH) == 64


def test_all_selectors_are_deterministic():
    relevance = torch.tensor([0.7, 0.2, 0.9, 0.3])
    packets = torch.nn.functional.normalize(torch.tensor([[1., 0.], [0., 1.], [0.7, 0.7], [-1., 0.]]), dim=-1)
    assert topk_selection(relevance, 3) == topk_selection(relevance, 3)
    assert mmr_selection(relevance, packets, 3, 0.5) == mmr_selection(relevance, packets, 3, 0.5)
    assert stable_random_indices(4, 3, "x", 73) == stable_random_indices(4, 3, "x", 73)


def test_paired_bootstrap_resamples_samples_and_is_seeded():
    ids=["a","b","c","d"]
    left={sid:value for sid,value in zip(ids,[1.0,0.0,1.0,0.5])}
    right={sid:value for sid,value in zip(ids,[0.0,0.0,0.5,0.5])}
    first=paired_bootstrap(left,right,ids,1000,42)
    assert first==paired_bootstrap(left,right,ids,1000,42)
    assert first["delta"]==37.5
    try:
        paired_bootstrap(left,{"a":0.0},ids,1000,42)
        assert False
    except ValueError:
        pass
