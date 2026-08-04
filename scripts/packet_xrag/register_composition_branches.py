#!/usr/bin/env python
"""Preregister the four bounded architecture probes selected by Stage-1 evidence."""

import json
from pathlib import Path


def branch(experiment_id, family, hypothesis, inputs, previous, cost, gate):
    return {"experiment_id": experiment_id, "family": family, "hypothesis": hypothesis,
            "model_inputs": inputs, "output_token_budget": 4,
            "training_objective": "O1 gold-answer NLL only",
            "previous_failure_addressed": previous, "expected_evidence": gate,
            "falsification_condition": "fails every Stage-3 promotion condition",
            "estimated_gpu_cost": cost, "promotion_gate": gate, "status": "preregistered",
            "actual_metrics": None, "decision": None, "reason": None,
            "checkpoint_hash": None}


def main():
    root = Path("cache/composition"); path = root / "experiment_ledger.json"
    ledger = json.loads(path.read_text())
    diagnostic = ledger["stage1_composition_gap"]
    if diagnostic["status"] != "PASS" or ledger["branches"]:
        raise RuntimeError("composition branches require one passed, unregistered diagnostic")
    # Stage 1 was one locked DEV generation suite with five diagnostic sub-experiments.
    if ledger["usage"]["composition_dev_generation"] != 5:
        raise RuntimeError("unexpected pre-registration DEV accounting")
    ledger["usage"]["composition_dev_generation"] = 1
    ledger["accounting_note"] = (
        "The 34 Stage-1 configurations are one locked diagnostic generation suite; "
        "D1-D5 count as five diagnostic/probe runs, not five independent DEV selection evaluations.")
    branches = [
        branch("A1", "A", "Query-conditioned attention can jointly compose TOPK K2 packet tokens.",
               "query SFR + TOPK N×2 frozen K2 tokens", "independent K2 concatenation collapse",
               "1 epoch over first 1,000; N=2/4/6 evaluation", "breadth robustness or +3 vs independent N6"),
        branch("B1", "B", "STATIC top-N plus fixed-slot fusion combines reliable selection and broad evidence.",
               "query SFR + STATIC top-N frozen K2 tokens", "STATIC2 strong but broad concat brittle",
               "1 epoch over first 1,000; N=2/4/6 evaluation", "quality, robustness, or interference gate"),
        branch("C1", "C", "Zero-gated residual fusion can preserve STATIC2 and safely use ranks 3-6.",
               "STATIC2 K2 base + STATIC rank3..N K2 residual", "avoid random-representation regression",
               "1 epoch over first 1,000; N=2/4/6 evaluation", "exact init and trained F1 >= STATIC2"),
        branch("D1", "D", "Direct SFR-set fusion avoids irreversible non-compositional K2 projection bias.",
               "query SFR + STATIC top-N pooled SFR embeddings", "test whether K2 inputs are the bottleneck",
               "1 epoch over first 1,000; N=2/4/6 evaluation", "quality, robustness, or interference gate"),
    ]
    ledger["branches"] = branches; ledger["usage"]["hypothesis_branches"] = 4
    ledger["family_decisions"] = {
        "A": "KEEP: directly tests query-conditioned K2 set composition.",
        "B": "KEEP: highest priority because STATIC2 is 69.19 F1 on diagnostic-150.",
        "C": "KEEP: preferred safety architecture with exact STATIC2 initialization.",
        "D": "KEEP: isolates K2 projection as a possible composition bottleneck.",
        "E": "EXCLUDE: order degradation <=1.09 F1; document hierarchy is not the dominant signal.",
        "F": "DEFER: a fully joint bridge is more complex than A-D and unnecessary before simpler probes pass.",
    }
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    lines = ["# Composition Experiment Ledger", "",
             "- Stage-1 composition gap: PASS (4/5)",
             "- Registered architecture probes: A1, B1, C1, D1", ""]
    for item in branches:
        lines += [f"## {item['experiment_id']} — Family {item['family']}", "",
                  f"- Hypothesis: {item['hypothesis']}", f"- Inputs: {item['model_inputs']}",
                  f"- Output M: {item['output_token_budget']}",
                  f"- Objective: {item['training_objective']}", f"- Gate: {item['promotion_gate']}", ""]
    (root / "experiment_ledger.md").write_text("\n".join(lines))
    print(json.dumps({"status": "registered", "branches": [b["experiment_id"] for b in branches],
                      "family_decisions": ledger["family_decisions"]}, indent=2))


if __name__ == "__main__": main()
