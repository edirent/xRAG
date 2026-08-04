import torch

from src.packet_xrag.composition.composition_losses import (
    distribution_invariance_kl, slot_cosine_consistency, text_teacher_kl,
)


def test_composition_losses_are_zero_for_identical_inputs():
    logits = torch.randn(2, 3, 7); slots = torch.randn(2, 4, 8)
    assert text_teacher_kl(logits, logits, torch.ones(2, 3)).abs() < 1e-6
    assert distribution_invariance_kl(logits, logits).abs() < 1e-6
    assert slot_cosine_consistency(slots, slots).abs() < 1e-6
