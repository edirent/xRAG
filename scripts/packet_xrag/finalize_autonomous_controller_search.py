#!/usr/bin/env python
"""Seal an autonomous search that reached a protocol-defined terminal decision."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from src.packet_xrag.controller.autonomous_search import (
    assert_no_inference_leakage, load_search_split, read_jsonl,
)
from scripts.packet_xrag.run_autonomous_probe_rollouts import summary as rollout_summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="cache/controller/autonomous_search")
    args = parser.parse_args(argv); root = Path(args.root)
    ledger_path = root / "experiment_ledger.json"; ledger = json.loads(ledger_path.read_text())
    stage1 = json.loads((root / "stage1/feasibility_probe_results.json").read_text())
    stage2 = json.loads((root / "stage2/probe_rollout_results.json").read_text())
    seal = json.loads((root / "search_shadow_seal.json").read_text())
    checkpoint = json.loads((root / "checkpoint_audit.json").read_text())
    load_search_split(root)
    if stage2["status"] != "MANDATORY_STOP_NO_STAGE2_BRANCH" or stage2["retained_branches"]:
        raise RuntimeError("finalizer only supports the reached Stage-2 mandatory stop")
    if ledger["usage"] != {"benchmark_evaluations": 0, "feasibility_probes": 8,
                            "full_training_runs": 0, "hypothesis_branches": 6,
                            "search_dev_generation": 4,
                            "search_shadow_evaluations": 0}:
        raise RuntimeError("experiment budget ledger mismatch at mandatory stop")
    if seal["status"] != "SEALED" or seal["evaluation_runs"] != 0:
        raise RuntimeError("SEARCH_SHADOW seal changed before mandatory stop")
    if checkpoint["final_100_accessed"] or checkpoint["final_100_runs"]:
        raise RuntimeError("final 100 access detected")
    # Metrics-only refresh keeps the sealed report complete without new generation.
    predictions = read_jsonl(root / "stage2/probe_rollout_predictions.jsonl")
    for branch_id in ("B1", "B5"):
        metrics = rollout_summary([row for row in predictions
                                   if row["configuration"] == branch_id])
        stage2["branches"][branch_id]["metrics"] = metrics
        stage2["branches"][branch_id]["cost"].update({
            "average_prompt_tokens_processed": metrics["avg_prompt_tokens"],
            "average_soft_tokens_processed": metrics["avg_soft_tokens"],
        })
    probe_ids = set(json.loads((root / "splits/probe_subset_ids.json").read_text())
                    ["ordered_sample_ids"])
    baseline_rows = read_jsonl(root / "stage0/search_dev_baselines.jsonl")
    for name in ("TOPK_3", "STATIC_2"):
        stage2["baselines"][name] = rollout_summary([
            row for row in baseline_rows if row["configuration"] == name and
            row["sample_id"] in probe_ids])
    (root / "stage2/probe_rollout_results.json").write_text(
        json.dumps(stage2, indent=2, sort_keys=True) + "\n")
    for filename in ("stage1/generator_state_features.jsonl",
                     "stage1/candidate_intervention_features.jsonl"):
        for row in read_jsonl(root / filename):
            assert_no_inference_leakage(row, filename)
    family_disposition = {
        "A Generator-State STOP": "kept through Stage 1 as B1/B5; rejected by Stage-2 rollout gate",
        "B Answer-conditioned candidate": "rejected at Stage 1; within-state Spearman 0.1273",
        "C Candidate intervention": "rejected at Stage 1 after same-pool reference correction",
        "D Sparse text relation": "rejected at Stage 1; within-state Spearman -0.0120",
        "E Confidence/rule STOP": "tested jointly with B1; rejected by Stage-2 rollout gate",
        "F Hybrid": "rejected at Stage 1; no same-pool promotion gate passed",
    }
    decision = {"status": "MANDATORY_STOP", "stage_reached": 2,
                "reason": "No promoted feature family passed the fixed 150-sample policy rollout gate.",
                "deployable_controller_found": False,
                "stage1_promoted": stage1["promoted_branches"],
                "stage2_retained": [], "family_disposition": family_disposition,
                "usage": ledger["usage"], "search_shadow_status": "SEALED_UNACCESSED",
                "benchmark_status": "NOT_RUN", "final_100_status": "NOT_ACCESSED",
                "frozen_representation_unchanged": True}
    (root / "final_decision.json").write_text(json.dumps(decision, indent=2,
                                                         sort_keys=True) + "\n")
    lines = ["# PacketRAG Autonomous Controller Search — Final Decision", "",
             "## Decision", "", "**MANDATORY STOP at Stage 2.**", "",
             "Neither promoted STOP branch passed the fixed 150-sample real-generation rollout gate.", "",
             "| Policy | Short F1 | Δ vs STATIC-2 | Avg packets | Decision |",
             "|---|---:|---:|---:|---|",
             f"| STATIC-2 | {stage2['baselines']['STATIC_2']['short_f1']:.3f} | — | 2.000 | baseline |"]
    for branch_id in ("B1", "B5"):
        result = stage2["branches"][branch_id]
        lines.append(f"| {branch_id} | {result['metrics']['short_f1']:.3f} | "
                     f"{result['delta_vs_static2']['short_f1']:+.3f} | "
                     f"{result['metrics']['avg_packets']:.3f} | reject |")
    lines += ["", "## Isolation and budgets", "",
              "- SEARCH_SHADOW: sealed, 0 evaluations",
              "- Benchmark-500: 0 evaluations",
              "- Final-100: not accessed, 0 runs",
              "- Full training: 0 runs (forbidden after the Stage-2 stop)",
              "- SEARCH_DEV generation evaluations: 4 / 8",
              "- Feasibility probes: 8 / 8", "", "## Family disposition", ""]
    lines.extend(f"- {name}: {reason}" for name, reason in family_disposition.items())
    lines += ["", "The frozen V1/K2 representation, SFR, STATIC scorer, and generator were not modified.", ""]
    (root / "final_decision.md").write_text("\n".join(lines))
    ledger["terminal_decision"] = decision
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    ledger_lines = ["# Autonomous Controller Experiment Ledger", "",
                    "- Terminal status: MANDATORY STOP at Stage 2",
                    "- SEARCH_SHADOW / benchmark / final 100 runs: 0 / 0 / 0", ""]
    for branch in ledger["branches"]:
        ledger_lines += [f"## {branch['experiment_id']} — Family {branch['family']}", "",
                         f"- Hypothesis: {branch['hypothesis']}",
                         f"- Result: {branch['result']}", f"- Reason: {branch['reason']}", ""]
    (root / "experiment_ledger.md").write_text("\n".join(ledger_lines))
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__": main()
