#!/usr/bin/env python
"""Paired bootstrap comparisons for the 200-sample utility oracle audit."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.packet_xrag.controller.generator_utility import paired_bootstrap


COMPARISONS = (
    ("STATE_UTILITY_STOP", "STATIC_UTILITY_STOP"),
    ("STATE_UTILITY_STOP", "STATIC_UTILITY_FIXED2"),
    ("STATE_UTILITY_STOP", "STATIC_2"),
    ("STATE_UTILITY_STOP", "TOPK_3"),
    ("STATIC_UTILITY_STOP", "STATIC_2"),
    ("STATIC_UTILITY_STOP", "TOPK_3"),
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--utility-dir", default="cache/controller/utility_feasibility")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.bootstrap_samples != 10_000 or args.seed != 42:
        raise RuntimeError("locked bootstrap requires 10,000 draws and seed 42")
    root = Path(args.utility_dir)
    rows = [json.loads(line) for line in
            (root / "oracle_generation_predictions.jsonl").read_text().splitlines()]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["configuration"]].append(row)
    results = []
    for left, right in COMPARISONS:
        result = paired_bootstrap(
            grouped[left], grouped[right], args.bootstrap_samples, args.seed
        )
        results.append({"left": left, "right": right, **result})
    payload = {
        "sample_unit": "sample_id", "sample_count": 200,
        "bootstrap_samples": args.bootstrap_samples, "seed": args.seed,
        "comparisons": results, "final_100_accessed": False, "final_100_runs": 0,
    }
    (root / "oracle_bootstrap.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    fields = ["left", "right", "short_f1_delta", "short_f1_ci95",
              "p_delta_gt_0", "p_delta_ge_1", "p_delta_ge_2",
              "avg_packet_delta", "avg_packet_delta_ci95"]
    with (root / "oracle_bootstrap.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in results:
            writer.writerow({key: row[key] for key in fields})
    lines = ["# Utility Oracle Paired Bootstrap", ""]
    for row in results:
        lines.append(
            f"- {row['left']} - {row['right']}: F1 {row['short_f1_delta']:.4f} "
            f"(95% CI {row['short_f1_ci95'][0]:.4f}, {row['short_f1_ci95'][1]:.4f}); "
            f"packets {row['avg_packet_delta']:.4f}"
        )
    lines.extend(["", "- Final 100 accessed: No", "- Final 100 runs: 0", ""])
    (root / "oracle_bootstrap.md").write_text("\n".join(lines))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
