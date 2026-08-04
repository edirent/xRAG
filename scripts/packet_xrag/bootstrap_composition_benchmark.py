#!/usr/bin/env python
"""Apply frozen bootstrap comparisons and final composition gates to benchmark-500."""

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import mean


def paired_bootstrap(left, right, samples=10000, seed=42):
    right_by_id = {row["sample_id"]: row["short_f1"] for row in right}
    differences = [row["short_f1"] - right_by_id[row["sample_id"]] for row in left]
    if len(differences) != 500 or len(right_by_id) != 500:
        raise RuntimeError("bootstrap requires 500 aligned sample IDs")
    rng = random.Random(seed); draws = []
    for _ in range(samples):
        draws.append(100 * mean(differences[rng.randrange(500)] for _ in range(500)))
    draws.sort()
    return {"samples": samples, "seed": seed, "sample_unit": "sample_id",
            "delta": 100 * mean(differences), "ci95_lower": draws[250],
            "ci95_upper": draws[9750], "p_delta_gt_0": sum(value > 0 for value in draws) / samples}


def degradation(metrics, prefix, stress_names):
    clean = metrics[f"{prefix}_N6"]["short_f1"]
    stressed = mean(metrics[f"{prefix}_{name}"]["short_f1"] for name in stress_names)
    return clean - stressed


def reduction(independent, fused):
    if independent <= 0:
        return 0.0 if fused >= independent else float("-inf")
    return (independent - fused) / independent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="cache/composition")
    args = parser.parse_args(argv); root = Path(args.root); benchmark = root / "benchmark"
    output = benchmark / "bootstrap.json"
    if output.exists():
        raise RuntimeError("refusing to overwrite formal composition bootstrap")
    result = json.loads((benchmark / "results.json").read_text())
    if result["status"] != "COMPLETE_FROZEN_NO_SELECTION" or result["benchmark_runs"] != 1:
        raise RuntimeError("formal benchmark must complete exactly once before bootstrap")
    grouped = {}
    for line in (benchmark / "predictions.jsonl").read_text().splitlines():
        row = json.loads(line); grouped.setdefault(row["configuration"], []).append(row)
    comparisons = {
        "Fuser - STATIC_2": paired_bootstrap(grouped["FUSER_N6"], grouped["STATIC_2"]),
        "Fuser - TOPK_3": paired_bootstrap(grouped["FUSER_N6"], grouped["TOPK_3"]),
        "Fuser - Independent same-breadth": paired_bootstrap(
            grouped["FUSER_N6"], grouped["INDEPENDENT_N6"]),
        "Fuser-N12 - Independent-N12": paired_bootstrap(
            grouped["FUSER_N12"], grouped["INDEPENDENT_N12"]),
    }
    metrics = result["metrics"]
    independent_degradation = {
        "order": degradation(metrics, "INDEPENDENT", ("REVERSE", "RANDOM")),
        "duplicate": degradation(metrics, "INDEPENDENT",
            ("GOLD_DUPLICATE_X2", "NONGOLD_DUPLICATE_X2")),
        "distractor": degradation(metrics, "INDEPENDENT", ("DISTRACTOR_X4",)),
    }
    fuser_degradation = {
        "order": degradation(metrics, "FUSER", ("REVERSE", "RANDOM")),
        "duplicate": degradation(metrics, "FUSER",
            ("GOLD_DUPLICATE_X2", "NONGOLD_DUPLICATE_X2")),
        "distractor": degradation(metrics, "FUSER", ("DISTRACTOR_X4",)),
    }
    reductions = {name: reduction(independent_degradation[name], fuser_degradation[name])
                  for name in independent_degradation}
    fuser = metrics["FUSER_N6"]["short_f1"]; static2 = metrics["STATIC_2"]["short_f1"]
    independent = metrics["INDEPENDENT_N6"]["short_f1"]
    ci = comparisons["Fuser - STATIC_2"]
    no_high_breadth_collapse = any(
        metrics[f"FUSER_{name}"]["short_f1"] >= static2 - 1 and
        metrics[f"FUSER_{name}"]["short_f1"] - metrics[
            "ALL" if name == "ALL" else "INDEPENDENT_N12"]["short_f1"] >= 10
        for name in ("N12", "ALL"))
    mechanism_a = (sum(value >= .5 for value in reductions.values()) >= 1 or
                   no_high_breadth_collapse)
    gate_a = (fuser - static2 >= 3 and ci["ci95_lower"] > 0 and
              fuser - independent >= 5 and metrics["FUSER_N6"]["output_fused_tokens"] <= 4 and
              mechanism_a)
    gate_b = (fuser >= static2 - .5 and 6 >= 6 and
              metrics["FUSER_N6"]["output_fused_tokens"] <= 4 and
              fuser - independent >= 5 and sum(value >= .4 for value in reductions.values()) >= 2)
    clear_mechanism = no_high_breadth_collapse or sum(value >= .4 for value in reductions.values()) >= 1
    gate_c = (((fuser - static2 >= 1.5 and ci["ci95_lower"] > 0) or
               (fuser >= static2 - 1 and fuser - independent >= 4)) and clear_mechanism)
    selected_gate = "A" if gate_a else "B" if gate_b else "C" if gate_c else "FAIL"
    status = "PASS" if selected_gate != "FAIL" else "MANDATORY_STOP_FINAL_BENCHMARK_GATE_FAILED"
    final = {"status": status, "selected_final_gate": selected_gate,
             "comparisons": comparisons, "independent_degradation": independent_degradation,
             "fuser_degradation": fuser_degradation, "robustness_reductions": reductions,
             "no_high_breadth_collapse": no_high_breadth_collapse,
             "gates": {"A": gate_a, "B": gate_b, "C": gate_c},
             "benchmark_runs": 1, "thresholds_changed": False,
             "final_100_accessed": False, "final_100_runs": 0}
    output.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    with (benchmark / "bootstrap.csv").open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["comparison", "delta", "ci95_lower",
                                                       "ci95_upper", "p_delta_gt_0"])
        for name, values in comparisons.items():
            writer.writerow([name, values["delta"], values["ci95_lower"],
                             values["ci95_upper"], values["p_delta_gt_0"]])
    lines = ["# Frozen Composition Benchmark Decision", "",
             f"- Status: {status}", f"- Selected final gate: {selected_gate}",
             f"- Fuser N6 - STATIC2: {fuser - static2:.4f} F1",
             f"- Fuser N6 - independent N6: {fuser - independent:.4f} F1",
             f"- 95% CI lower vs STATIC2: {ci['ci95_lower']:.4f}",
             f"- High-breadth collapse repaired: {no_high_breadth_collapse}",
             f"- Robustness reductions: {json.dumps(reductions, sort_keys=True)}", "",
             "Benchmark was evaluated once with a frozen candidate; final-100 was not accessed.", ""]
    (benchmark / "decision.md").write_text("\n".join(lines))
    ledger_path = root / "experiment_ledger.json"; ledger = json.loads(ledger_path.read_text())
    ledger["stage8_benchmark"].update({"status": status, "selected_final_gate": selected_gate})
    ledger["final_status"] = status
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
