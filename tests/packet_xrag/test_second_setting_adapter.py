import torch

from src.packet_xrag.generalization.second_setting_adapter import (
    BASE_REDUCTION, OUTPUT_M, k4_residual_inputs,
)


def test_k4_static2_base_and_extra_token_shapes_are_fixed():
    first = torch.arange(5 * 4 * 8, dtype=torch.float32).view(5, 4, 8)
    second = torch.ones(2, 4, 8)
    base, extras, mask = k4_residual_inputs([first, second], torch.device("cpu"))
    assert base.shape == (2, OUTPUT_M, 8)
    assert extras.shape == (2, 3, 4, 8)
    assert mask.tolist() == [[True, True, True], [False, False, False]]
    assert torch.equal(base[0], first[:2].mean(dim=0))
    assert BASE_REDUCTION == "slotwise mean of STATIC rank-1 and rank-2 K4 tokens"


def test_k4_adapter_rejects_single_packet_base():
    try:
        k4_residual_inputs([torch.zeros(1, 4, 8)], torch.device("cpu"))
    except ValueError as error:
        assert "rank 1-2" in str(error)
    else:
        raise AssertionError("single-packet K4 base was accepted")
