import torch

from src.packet_xrag.controller.static_utility_predictor import (
    BASE_FEATURE_DIM, StaticUtilityPredictor,
)


def inputs(batch=3):
    return {
        "query_embeddings": torch.randn(batch, 8),
        "packet_embeddings": torch.randn(batch, 8),
        "position_features": torch.rand(batch, 2),
        "static_scores": torch.rand(batch),
    }


def test_model_a_architecture_and_selected_set_independence():
    model = StaticUtilityPredictor(input_dim=8, projection_dim=4).eval()
    values = inputs()
    first = model(**values, selected_embeddings=torch.randn(3, 2, 8))
    second = model(**values, selected_embeddings=torch.randn(3, 5, 8))
    assert torch.equal(first, second)
    assert model.utility_head[0].in_features == 4 * 4 + 5
    assert BASE_FEATURE_DIM == 2053


def test_model_a_projection_initialization_uses_only_static_projections():
    model = StaticUtilityPredictor(input_dim=8, projection_dim=4)
    state = {
        "query_projection.weight": torch.randn(4, 8),
        "query_projection.bias": torch.randn(4),
        "packet_projection.weight": torch.randn(4, 8),
        "packet_projection.bias": torch.randn(4),
        "scorer.0.weight": torch.randn(2, 2),
    }
    model.initialize_projections(state)
    assert torch.equal(model.query_projection.weight, state["query_projection.weight"])
    assert torch.equal(model.packet_projection.bias, state["packet_projection.bias"])

