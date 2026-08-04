import hashlib

import pytest
import torch

from src.packet_xrag.generalization.protocol import (
    load_checkpoint_state, register_checkpoint_owner,
)


def save_payload(path, value):
    torch.save({"state_dict": {"weight": torch.tensor([value])}}, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_zero_shot_checkpoint_loading_is_hash_checked(tmp_path):
    checkpoint = tmp_path / "hotpot.pt"; digest = save_payload(checkpoint, 1.0)
    payload = load_checkpoint_state(checkpoint, digest)
    assert payload["state_dict"]["weight"].item() == 1.0
    with pytest.raises(RuntimeError, match="hash mismatch"):
        load_checkpoint_state(checkpoint, "0" * 64)


def test_dataset_specific_checkpoint_loading_isolated_from_zero_shot(tmp_path):
    hotpot = tmp_path / "hotpot.pt"; hotpot_hash = save_payload(hotpot, 1.0)
    dataset = tmp_path / "dataset.pt"; dataset_hash = save_payload(dataset, 2.0)
    assert hotpot_hash != dataset_hash
    assert load_checkpoint_state(dataset, dataset_hash)["state_dict"]["weight"].item() == 2.0
    assert load_checkpoint_state(hotpot, hotpot_hash)["state_dict"]["weight"].item() == 1.0


def test_cross_dataset_checkpoint_owner_collision_is_rejected():
    owners = {}; register_checkpoint_owner(owners, "digest-a", "2wiki")
    register_checkpoint_owner(owners, "digest-b", "musique")
    with pytest.raises(RuntimeError, match="reused across"):
        register_checkpoint_owner(owners, "digest-a", "triviaqa")
