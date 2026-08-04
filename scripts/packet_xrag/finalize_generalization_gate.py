#!/usr/bin/env python
"""Aggregate the three frozen SHADOW results into the preregistered global gate."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from src.packet_xrag.generalization.dataset_gates import generalization_gate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="cache/generalization")
    args = parser.parse_args(argv); root = Path(args.root)
    output = root / "generalization_gate.json"
    if output.exists(): raise RuntimeError("refusing to overwrite global generalization gate")
    datasets = {}
    for name in ("2wiki", "musique", "triviaqa"):
        report = json.loads((root / name / "shadow/results.json").read_text())
        datasets[name] = {**report["metrics"], "shadow_gate_passed": report["gate"]["passed"]}
    gate = generalization_gate(datasets, hotpot_strong=True)
    if gate["mandatory_stop"]:
        status, second = "GENERALIZATION_FAIL", False
    elif gate["A"] or gate["B"]:
        status, second = "GENERALIZATION_PASS", True
    elif gate["C"]:
        status, second = "LIMITED_GENERALIZATION", True
    else:
        status, second = "GENERALIZATION_INCONCLUSIVE_STOP", False
    report = {"status": status, "gate": gate,
        "dataset_shadow_gates": {name: values["shadow_gate_passed"]
                                 for name, values in datasets.items()},
        "second_setting_authorized": second,
        "final100_authorized": False,
        "decision_source": "three single-use SHADOW suites",
        "benchmark_used_for_gate": False, "final100_accessed": False}
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    ledger_path = root / "experiment_ledger.json"; ledger = json.loads(ledger_path.read_text())
    ledger["generalization_gate"] = {"status": status, "second_setting_authorized": second}
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
