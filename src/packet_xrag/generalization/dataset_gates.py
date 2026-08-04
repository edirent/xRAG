"""Preregistered Shadow and cross-dataset generalization gates."""


def shadow_gate(metrics):
    static2 = metrics["STATIC_2"]["short_f1"]
    independent6 = metrics["INDEPENDENT_STATIC_6"]["short_f1"]
    fuser6 = metrics["DATASET_FUSER_6"]["short_f1"]
    fuser12 = metrics["DATASET_FUSER_12"]["short_f1"]
    gate_a = fuser6 - static2 >= 1.0 and fuser6 - independent6 >= 4.0
    gate_b = (fuser6 >= static2 - .75 and fuser6 - independent6 >= 4.0 and
              fuser12 >= fuser6 - 1.0)
    gate_c = (independent6 - metrics.get("INDEPENDENT_STATIC_12", metrics[
        "INDEPENDENT_STATIC_6"])["short_f1"] >= 2.0 and
        fuser12 - metrics.get("INDEPENDENT_STATIC_12", metrics[
            "INDEPENDENT_STATIC_6"])["short_f1"] >= 8.0 and
        fuser12 >= static2 - 1.5)
    return {"S-A": gate_a, "S-B": gate_b, "S-C": gate_c,
            "passed": gate_a or gate_b or gate_c,
            "fuser6_minus_static2": fuser6 - static2,
            "fuser6_minus_independent6": fuser6 - independent6,
            "fuser12_minus_fuser6": fuser12 - fuser6}


def generalization_gate(dataset_metrics, hotpot_strong=True):
    a_eligible, b_eligible, beats_static = 0, 0, 0
    for metrics in dataset_metrics.values():
        static2 = metrics["STATIC_2"]["short_f1"]
        f6 = metrics["DATASET_FUSER_6"]["short_f1"]
        i6 = metrics["INDEPENDENT_STATIC_6"]["short_f1"]
        f12 = metrics["DATASET_FUSER_12"]["short_f1"]
        i12 = metrics["INDEPENDENT_STATIC_12"]["short_f1"]
        a_eligible += int(f6 - i6 >= 4 and f6 >= static2 - .75)
        b_eligible += int(f12 - i12 >= 8 and f12 >= static2 - 1.5)
        beats_static += int(f6 > static2)
    gate_a = a_eligible >= 2 and beats_static >= 1
    gate_b = b_eligible >= 2
    passing = sum(bool(metrics.get("shadow_gate_passed"))
                  for metrics in dataset_metrics.values())
    gate_c = passing == 1 and hotpot_strong
    all_below_independent = all(metrics["DATASET_FUSER_6"]["short_f1"] -
        metrics["INDEPENDENT_STATIC_6"]["short_f1"] < 3 for metrics in dataset_metrics.values())
    all_below_static = all(metrics["DATASET_FUSER_6"]["short_f1"] <
        metrics["STATIC_2"]["short_f1"] - 2 for metrics in dataset_metrics.values())
    return {"A": gate_a, "B": gate_b, "C": gate_c,
            "mandatory_stop": all_below_independent or all_below_static,
            "datasets_meeting_A_pair": a_eligible, "datasets_meeting_B_pair": b_eligible,
            "datasets_beating_STATIC2": beats_static,
            "datasets_passing_shadow": passing}
