#!/usr/bin/env python
"""Paired bootstrap and frozen Stage-1 gate for the static scorer."""

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag.bootstrap_k2_selector_benchmark import paired_bootstrap
from scripts.packet_xrag.token_resampler_common import EXPECTED_SPLIT_HASH


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-input", default="cache/controller/static/benchmark_predictions.jsonl")
    parser.add_argument("--baseline-input", default="cache/results/k2_selector_benchmark_full500.jsonl")
    parser.add_argument("--training-config", default="cache/controller/static/best_short_f1/training_config.json")
    parser.add_argument("--num-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-output", default="cache/controller/static/static_bootstrap.json")
    parser.add_argument("--csv-output", default="cache/controller/static/static_bootstrap.csv")
    parser.add_argument("--markdown-output", default="cache/controller/static/static_bootstrap.md")
    parser.add_argument("--decision-output", default="cache/controller/static/static_decision.md")
    return parser.parse_args(argv)


def load_static(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = sorted({row["sample_id"] for row in rows})
    configurations = {row["configuration"] for row in rows}
    if len(rows) != 3000 or len(ids) != 500 or configurations != {
            f"STATIC_{budget}" for budget in range(1, 7)}:
        raise RuntimeError("unexpected STATIC benchmark cardinality")
    by = {name: {} for name in configurations}
    for row in rows:
        if row["sample_id"] in by[row["configuration"]]:
            raise RuntimeError("duplicate STATIC sample/configuration")
        by[row["configuration"]][row["sample_id"]] = row["short_f1"]
    return ids, by


def load_baselines(path, ids):
    wanted = ({f"TOPK_{budget}" for budget in range(1, 7)} |
              {f"MMR_{budget}" for budget in range(1, 7)} |
              {f"RANDOM_{budget}_SEED{seed}" for budget in range(1, 7)
               for seed in (13, 37, 73)})
    by = {name: {} for name in wanted}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row["configuration"] in wanted:
            by[row["configuration"]][row["sample_id"]] = row["short_f1"]
    for name, values in by.items():
        if set(values) != set(ids):
            raise RuntimeError(f"baseline sample mismatch: {name}")
    return by


def main(argv=None):
    args = parse_args(argv)
    if args.num_bootstrap != 10000 or args.seed != 42:
        raise RuntimeError("static bootstrap protocol is locked to 10,000/42")
    ids, static = load_static(args.static_input)
    baseline = load_baselines(args.baseline_input, ids)
    config = json.loads(Path(args.training_config).read_text())
    selected_budget = int(config["selected_budget"])
    comparisons = {}
    for budget in range(1, 7):
        left = static[f"STATIC_{budget}"]
        comparisons[f"STATIC_{budget} vs TOPK_{budget}"] = paired_bootstrap(
            left, baseline[f"TOPK_{budget}"], ids, args.num_bootstrap, args.seed
        )
        comparisons[f"STATIC_{budget} vs MMR_{budget}"] = paired_bootstrap(
            left, baseline[f"MMR_{budget}"], ids, args.num_bootstrap, args.seed
        )
        random_mean = {
            sid: sum(baseline[f"RANDOM_{budget}_SEED{seed}"][sid]
                     for seed in (13, 37, 73)) / 3
            for sid in ids
        }
        comparisons[f"STATIC_{budget} vs RANDOM_{budget}_MEAN"] = paired_bootstrap(
            left, random_mean, ids, args.num_bootstrap, args.seed
        )
    gate_comparison = paired_bootstrap(
        static[f"STATIC_{selected_budget}"], baseline["TOPK_3"], ids,
        args.num_bootstrap, args.seed,
    )
    gate_a = gate_comparison["delta"] >= 1.0 and gate_comparison["ci95_lower"] > 0
    payload = {
        "stage": "static_scorer", "split_hash": EXPECTED_SPLIT_HASH,
        "samples": 500, "resamples": args.num_bootstrap, "seed": args.seed,
        "internal_dev_selected_budget": selected_budget,
        "internal_dev_selected_epoch": config["selected_epoch"],
        "comparisons": comparisons,
        "gate_comparison": {
            "name": f"STATIC_{selected_budget} vs TOPK_3", **gate_comparison,
        },
        "static_gate": "A" if gate_a else "B",
        "static_passed": gate_a,
        "stage_2_authorized": True,
        "note": ("Static scorer passes; use it to initialize sequential controller."
                 if gate_a else
                 "Static learned relevance did not reliably improve over cosine TOPK."),
        "benchmark_runs": 1, "final_100_accessed": False, "final_100_runs": 0,
    }
    Path(args.json_output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    fields = ["Comparison", "Delta", "CI Low", "CI High", "P(delta>0)"]
    with Path(args.csv_output).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, result in comparisons.items():
            writer.writerow({
                "Comparison": name, "Delta": result["delta"],
                "CI Low": result["ci95_lower"], "CI High": result["ci95_upper"],
                "P(delta>0)": result["p_delta_gt_0"],
            })
    lines = ["# Static Scorer Paired Bootstrap", "",
             f"Samples: 500; resamples: {args.num_bootstrap}; seed: {args.seed}.", ""]
    for name, result in comparisons.items():
        lines.append(
            f"- {name}: {result['delta']:.6f}, 95% CI "
            f"[{result['ci95_lower']:.6f}, {result['ci95_upper']:.6f}]."
        )
    Path(args.markdown_output).write_text("\n".join(lines) + "\n")
    decision = [
        "# Static Scorer Decision", "",
        f"- Frozen internal-dev checkpoint epoch: {config['selected_epoch']}",
        f"- Frozen internal-dev packet budget: {selected_budget}",
        f"- Gate comparison: STATIC_{selected_budget} - TOPK_3 = "
        f"{gate_comparison['delta']:.6f} F1",
        f"- 95% CI: [{gate_comparison['ci95_lower']:.6f}, "
        f"{gate_comparison['ci95_upper']:.6f}]",
        f"- Static gate: {'A (PASS)' if gate_a else 'B (did not reliably improve)'}",
        "- Stage 2 authorized: yes",
        f"- Note: {payload['note']}",
        "- Benchmark runs: 1", "- Final 100 accessed: no", "- Final 100 runs: 0",
    ]
    Path(args.decision_output).write_text("\n".join(decision) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
