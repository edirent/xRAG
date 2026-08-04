#!/usr/bin/env python
"""Preregister the two bounded C1 objective configurations for full training."""

import json
from pathlib import Path


def main():
    root = Path("cache/composition"); path = root / "experiment_ledger.json"
    ledger = json.loads(path.read_text()); probes = ledger["stage3_architecture_probes"]
    if probes["promoted_branches"] != ["C1"] or "full_runs" in ledger:
        raise RuntimeError("full runs require the unique promoted C1 branch")
    ledger["full_runs"] = [
        {"experiment_id": "C1_O1", "branch": "C1", "architecture": "Residual STATIC2 + extra-evidence fusion",
         "input_selector": "STATIC", "input_breadths": [2, 4, 6], "output_token_budget": 4,
         "training_objective": "O1 answer NLL", "previous_failure_addressed": "probe success validation",
         "expected_evidence": "N6 quality without breadth degradation",
         "falsification_condition": "fails COMPOSITION_DEV gate", "estimated_gpu_cost": "6 epochs",
         "promotion_gate": "Stage-12 D-A/B/C", "status": "preregistered"},
        {"experiment_id": "C1_O1_O3_O4", "branch": "C1",
         "architecture": "Residual STATIC2 + extra-evidence fusion",
         "input_selector": "STATIC", "input_breadths": [2, 4, 6], "output_token_budget": 4,
         "training_objective": "O1 answer NLL + O3 distractor slot invariance + O4 duplicate slot invariance",
         "previous_failure_addressed": "duplicate/random distractor caused 22-24 F1 degradation",
         "expected_evidence": "retain C1 quality while reducing dominant stress degradation",
         "falsification_condition": "fails COMPOSITION_DEV gate", "estimated_gpu_cost": "6 epochs",
         "promotion_gate": "Stage-12 D-A/B/C", "status": "preregistered"},
    ]
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "registered", "full_runs": [r["experiment_id"]
                                                               for r in ledger["full_runs"]]}, indent=2))


if __name__ == "__main__": main()
