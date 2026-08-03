"""Strict construction and loading of nested utility predictor checkpoints."""

from pathlib import Path

import torch

from src.packet_xrag.controller.interaction_utility_predictor import InteractionUtilityPredictor
from src.packet_xrag.controller.state_shift_utility_predictor import StateShiftUtilityPredictor
from src.packet_xrag.controller.static_utility_predictor import StaticUtilityPredictor


def construct_utility_model(model_type):
    name = model_type.upper()
    if name in {"A", "MODEL_A", "A_STATIC_UTILITY"}:
        return StaticUtilityPredictor()
    if name in {"B", "MODEL_B", "B_STATE_SHIFT"}:
        return StateShiftUtilityPredictor()
    if name in {"C", "MODEL_C", "C_FULL_INTERACTION"}:
        return InteractionUtilityPredictor()
    raise ValueError(f"unknown utility model type: {model_type}")


def load_utility_checkpoint(model_type, checkpoint, device="cpu"):
    model = construct_utility_model(model_type)
    state = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model

