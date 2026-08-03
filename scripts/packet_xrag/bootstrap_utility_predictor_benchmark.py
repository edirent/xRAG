#!/usr/bin/env python
"""Locked 10,000-draw paired bootstrap for utility predictor benchmark ablations."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", default="cache/controller/utility_predictor/benchmark")
    return parser.parse_args(argv)


def compare(left, right, samples=10_000, seed=42):
    left_by_id = {row["sample_id"]: row for row in left}
    right_by_id = {row["sample_id"]: row for row in right}
    if set(left_by_id) != set(right_by_id) or len(left_by_id) != 500:
        raise RuntimeError("paired benchmark alignment requires the same 500 sample IDs")
    ids = sorted(left_by_id)
    f1 = torch.tensor([100 * (left_by_id[sid]["short_f1"] - right_by_id[sid]["short_f1"])
                       for sid in ids], dtype=torch.float64)
    packets = torch.tensor([left_by_id[sid]["num_packets"] - right_by_id[sid]["num_packets"]
                            for sid in ids], dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(ids), (samples, len(ids)), generator=generator)
    f1_boot = f1[indices].mean(1); packet_boot = packets[indices].mean(1)
    quantiles = torch.tensor([.025, .975], dtype=torch.float64)
    right_packets = sum(row["num_packets"] for row in right) / len(right)
    return {"short_f1_delta": float(f1.mean()),
            "short_f1_ci95": [float(value) for value in torch.quantile(f1_boot, quantiles)],
            "p_delta_gt_0": float((f1_boot > 0).double().mean()),
            "p_delta_ge_1": float((f1_boot >= 1).double().mean()),
            "p_delta_ge_2": float((f1_boot >= 2).double().mean()),
            "p_delta_ge_3": float((f1_boot >= 3).double().mean()),
            "avg_packet_delta": float(packets.mean()),
            "avg_packet_delta_ci95": [float(value) for value in torch.quantile(packet_boot, quantiles)],
            "relative_packet_change": float(packets.mean()) / right_packets}


def main(argv=None):
    args = parse_args(argv); root = Path(args.benchmark_dir)
    rows = [json.loads(line) for line in (root / "predictions.jsonl").read_text().splitlines()]
    grouped = defaultdict(list)
    for row in rows: grouped[row["configuration"]].append(row)
    manifest = json.loads((root / "benchmark_manifest.json").read_text())
    final_name = manifest["final_configuration_alias"]
    names = {"A": "MODEL_A_UTILITY_STOP", "B": "MODEL_B_STATE_SHIFT_STOP",
             "C": "MODEL_C_INTERACTION_STOP", "FINAL": final_name}
    specs = [("A", "STATIC_2"), ("A", "TOPK_3"), ("B", "A"),
             ("B", "STATIC_2"), ("B", "TOPK_3"), ("C", "B"),
             ("C", "STATIC_2"), ("C", "TOPK_3"), ("FINAL", "STATIC_2"),
             ("FINAL", "TOPK_3"), ("FINAL", "XRAG_ORACLE")]
    results = []
    for left_key, right_key in specs:
        left = names.get(left_key, left_key); right = names.get(right_key, right_key)
        results.append({"comparison": f"{left_key} - {right_key}", "left": left,
                        "right": right, **compare(grouped[left], grouped[right])})
    payload = {"bootstrap_samples": 10_000, "seed": 42, "sample_unit": "sample_id",
               "sample_count": 500, "comparisons": results,
               "mechanism_decomposition": {"static_generator_utility": "A - STATIC_2",
                                           "state_sufficiency": "B - A",
                                           "candidate_state_interaction": "C - B"},
               "final_100_accessed": False, "final_100_runs": 0}
    (root / "bootstrap_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    fields = ["comparison", "short_f1_delta", "short_f1_ci95", "p_delta_gt_0",
              "p_delta_ge_1", "p_delta_ge_2", "p_delta_ge_3", "avg_packet_delta",
              "avg_packet_delta_ci95", "relative_packet_change"]
    with (root / "bootstrap_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in results: writer.writerow({key: row[key] for key in fields})
    lines = ["# Utility Predictor Benchmark Bootstrap", ""]
    for row in results:
        lines.append(f"- {row['comparison']}: F1 {row['short_f1_delta']:.4f} "
                     f"CI [{row['short_f1_ci95'][0]:.4f}, {row['short_f1_ci95'][1]:.4f}]; "
                     f"packets {row['avg_packet_delta']:.4f}")
    lines.extend(["", "- Final 100 accessed: No", "- Final 100 runs: 0", ""])
    (root / "bootstrap_summary.md").write_text("\n".join(lines))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__": main()
