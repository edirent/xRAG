#!/usr/bin/env python
"""Evaluate preregistered internal-dev Model-B and Model-C gates."""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", choices=("B", "C"), required=True)
    parser.add_argument("--root", default="cache/controller/utility_predictor")
    return parser.parse_args(argv)


def selected_rows(model_dir):
    frozen = json.loads((model_dir / "frozen_selection.json").read_text())
    rows = [json.loads(line) for line in
            (model_dir / "internal_dev_grid_predictions.jsonl").read_text().splitlines()]
    return frozen, [row for row in rows if row["configuration"] == frozen["configuration"]]


def paired(left, right, draws=10_000, seed=42):
    left = {row["sample_id"]: row for row in left}; right = {row["sample_id"]: row for row in right}
    if set(left) != set(right) or len(left) != 500: raise RuntimeError("gate sample alignment failed")
    ids = sorted(left)
    deltas = torch.tensor([100 * (left[sid]["short_f1"] - right[sid]["short_f1"])
                           for sid in ids], dtype=torch.float64)
    indices = torch.randint(500, (draws, 500), generator=torch.Generator().manual_seed(seed))
    values = deltas[indices].mean(1); q = torch.tensor([.025, .975], dtype=torch.float64)
    return {"short_f1_delta": float(deltas.mean()),
            "ci95": [float(value) for value in torch.quantile(values, q)],
            "p_delta_gt_0": float((values > 0).double().mean())}


def main(argv=None):
    args = parse_args(argv); root = Path(args.root)
    if args.gate == "B":
        left_name, right_name = "model_b", "model_a"
    else:
        left_name, right_name = "model_c", "model_b"
    left_frozen, left_rows = selected_rows(root / left_name)
    right_frozen, right_rows = selected_rows(root / right_name)
    bootstrap = paired(left_rows, right_rows)
    left_metrics, right_metrics = left_frozen["metrics"], right_frozen["metrics"]
    left_mech, right_mech = left_frozen["utility_mechanism"], right_frozen["utility_mechanism"]
    f1_delta = bootstrap["short_f1_delta"]
    packet_reduction = ((right_metrics["avg_packets"] - left_metrics["avg_packets"])
                        / right_metrics["avg_packets"])
    harmful_reduction = None
    if right_mech["harmful_addition_rate"] not in (None, 0):
        harmful_reduction = ((right_mech["harmful_addition_rate"] - left_mech["harmful_addition_rate"])
                             / right_mech["harmful_addition_rate"])
    missed_increase = ((left_mech["missed_positive_utility_at_stop"] or 0) -
                       (right_mech["missed_positive_utility_at_stop"] or 0))
    if args.gate == "B":
        conditions = {
            "quality": f1_delta >= 2 and bootstrap["ci95"][0] > 0,
            "cost": abs(f1_delta) <= .5 and packet_reduction >= .15,
            "mechanism": harmful_reduction is not None and harmful_reduction >= .25 and missed_increase <= .05,
        }
        passed = any(conditions.values()); result = "B-PASS" if passed else "B-FAIL"
    else:
        conditions = {
            "quality": f1_delta >= 1 and bootstrap["ci95"][0] > 0,
            "cost": abs(f1_delta) <= .5 and packet_reduction >= .10,
            "mechanism": harmful_reduction is not None and harmful_reduction >= .15 and f1_delta >= -.5,
        }
        passed = any(conditions.values()); result = "INTERACTION-PASS" if passed else "INTERACTION-NO-GAIN"
    payload = {"gate": args.gate, "result": result, "passed": passed,
               "left": left_name, "right": right_name, "bootstrap": bootstrap,
               "packet_reduction": packet_reduction, "harmful_addition_reduction": harmful_reduction,
               "missed_positive_stop_increase": missed_increase, "conditions": conditions,
               "left_selection": left_frozen, "right_selection": right_frozen,
               "audit_required": args.gate == "B" and not passed,
               "benchmark_used": False, "final_100_accessed": False, "final_100_runs": 0}
    path = root / f"model_{args.gate.lower()}" / f"model_{args.gate.lower()}_gate.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__": main()
