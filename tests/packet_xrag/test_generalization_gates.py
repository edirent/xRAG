from src.packet_xrag.generalization.dataset_gates import generalization_gate, shadow_gate


def metric(value):
    return {"short_f1": value}


def test_shadow_gate_b_requires_stability_and_composition_gain():
    metrics = {"STATIC_2": metric(50), "INDEPENDENT_STATIC_6": metric(45),
               "INDEPENDENT_STATIC_12": metric(40), "DATASET_FUSER_6": metric(49.5),
               "DATASET_FUSER_12": metric(49)}
    gate = shadow_gate(metrics)
    assert gate["S-B"] and gate["passed"]
    metrics["DATASET_FUSER_12"] = metric(47)
    assert not shadow_gate(metrics)["S-B"]


def test_generalization_mandatory_stop_when_all_composition_gains_small():
    metrics = {name: {"STATIC_2": metric(50), "INDEPENDENT_STATIC_6": metric(48),
        "INDEPENDENT_STATIC_12": metric(45), "DATASET_FUSER_6": metric(49),
        "DATASET_FUSER_12": metric(49), "shadow_gate_passed": False}
        for name in ("a", "b", "c")}
    assert generalization_gate(metrics)["mandatory_stop"]
