import pytest
import torch

from src.packet_xrag.controller.utility_checkpoint import (
    construct_utility_model, load_utility_checkpoint,
)


@pytest.mark.parametrize("model_type", ["A", "B", "C"])
def test_utility_checkpoint_round_trip_is_strict(tmp_path, model_type):
    model = construct_utility_model(model_type)
    path = tmp_path / f"{model_type}.pt"
    torch.save(model.state_dict(), path)
    loaded = load_utility_checkpoint(model_type, path)
    for name, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name])


def test_unknown_utility_model_type_is_rejected():
    with pytest.raises(ValueError, match="unknown"):
        construct_utility_model("D")
