import torch

from src.packet_xrag.controller.interaction_utility_predictor import InteractionUtilityPredictor
from src.packet_xrag.controller.state_shift_utility_predictor import StateShiftUtilityPredictor
from src.packet_xrag.controller.static_utility_predictor import StaticUtilityPredictor


def batch():
    return {
        "query_embeddings": torch.randn(2, 8), "packet_embeddings": torch.randn(2, 8),
        "position_features": torch.rand(2, 2), "static_scores": torch.rand(2),
        "selected_embeddings": torch.randn(2, 2, 8),
        "selected_mask": torch.ones(2, 2, dtype=torch.bool),
        "state_features": torch.rand(2, 7), "relation_features": torch.zeros(2, 3),
    }


def model():
    base = StaticUtilityPredictor(input_dim=8, projection_dim=512)
    return InteractionUtilityPredictor(StateShiftUtilityPredictor(base)).eval()


def test_model_c_zero_init_is_elementwise_equal_to_model_b():
    current = model(); values = batch()
    assert torch.equal(current(**values), current.state_model(**values))


def test_candidate_state_relation_changes_nonzero_interaction_residual():
    current = model(); values = batch()
    values["query_embeddings"][:] = values["query_embeddings"][0]
    values["packet_embeddings"][:] = values["packet_embeddings"][0]
    values["selected_embeddings"][:] = values["selected_embeddings"][0]
    values["selected_mask"][:] = values["selected_mask"][0]
    values["state_features"][:] = values["state_features"][0]
    values["relation_features"][1] = torch.tensor([1.0, .5, 1.0])
    with torch.no_grad(): current.interaction_head[-1].weight.normal_()
    _, components = current(**values, return_components=True)
    assert components["interaction_residual"][0] != components["interaction_residual"][1]

