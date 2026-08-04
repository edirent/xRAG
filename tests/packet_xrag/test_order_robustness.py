from src.packet_xrag.composition.residual_set_fuser import ResidualSetFuser
from src.packet_xrag.generalization.order_robustness import (
    architecture_order_audit, permute_extras,
)


def record():
    return {"sample_id": "x", "packets": [
        {"doc_id": 1, "sentence_id": 0}, {"doc_id": 0, "sentence_id": 2},
        {"doc_id": 1, "sentence_id": 1}, {"doc_id": 0, "sentence_id": 1},
        {"doc_id": 0, "sentence_id": 0}, {"doc_id": 1, "sentence_id": 2}]}


def test_order_permutations_preserve_static_base_and_are_deterministic():
    selected = [0, 1, 2, 3, 4, 5]
    for variant in ("reverse", "document", "random_0", "random_1", "random_2"):
        first = permute_extras(record(), selected, variant)
        assert first == permute_extras(record(), selected, variant)
        assert first[:2] == selected[:2]
        assert set(first[2:]) == set(selected[2:])


def test_order_robust_variant_exact_config_has_no_position_embedding():
    audit = architecture_order_audit(ResidualSetFuser(dimension=16, latent_dim=8,
                                                       output_slots=4, heads=2))
    assert audit["attention_without_position_encoding"]
    assert not audit["forbidden_parameter_names"]
