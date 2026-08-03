import torch

from src.packet_xrag.controller.state_shift_utility_predictor import StateShiftUtilityPredictor
from src.packet_xrag.controller.static_utility_predictor import StaticUtilityPredictor


def batch(selected_count=2, batch=4):
    return {
        "query_embeddings": torch.randn(batch, 8),
        "packet_embeddings": torch.randn(batch, 8),
        "position_features": torch.rand(batch, 2),
        "static_scores": torch.rand(batch),
        "selected_embeddings": torch.randn(batch, max(1, selected_count), 8),
        "selected_mask": torch.tensor([[True] * selected_count + [False] * (max(1, selected_count) - selected_count)] * batch),
        "state_features": torch.rand(batch, 7),
    }


def model():
    return StateShiftUtilityPredictor(StaticUtilityPredictor(input_dim=8, projection_dim=512)).eval()


def test_model_b_zero_init_is_elementwise_equal_to_model_a():
    current = model(); values = batch()
    assert torch.equal(current(**values), current.base_model(**values))


def test_model_b_shift_is_identical_for_candidates_in_same_state():
    current = model(); values = batch(batch=3)
    values["query_embeddings"][:] = values["query_embeddings"][0]
    values["selected_embeddings"][:] = values["selected_embeddings"][0]
    values["selected_mask"][:] = values["selected_mask"][0]
    values["state_features"][:] = values["state_features"][0]
    with torch.no_grad(): current.state_shift_head[-1].bias.fill_(0.7)
    _, components = current(**values, return_components=True)
    assert torch.allclose(components["state_shift"], torch.full((3,), .7))


def test_state_change_affects_nonzero_shift_and_empty_is_deterministic():
    current = model()
    with torch.no_grad(): current.state_shift_head[-1].weight.normal_()
    first = batch(selected_count=0, batch=1)
    assert torch.equal(current(**first), current(**first))
    second = {key: value.clone() for key, value in first.items()}
    second["selected_embeddings"] = torch.randn(1, 1, 8)
    second["selected_mask"] = torch.ones(1, 1, dtype=torch.bool)
    second["state_features"][:, 0] = 1 / 6
    assert not torch.equal(current(**first), current(**second))

