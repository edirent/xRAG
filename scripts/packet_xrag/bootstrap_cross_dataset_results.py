#!/usr/bin/env python
"""Compute the frozen 10,000-draw paired bootstrap for a dataset benchmark."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


COMPARISONS = (
    ("DATASET_FUSER_6", "STATIC_2", "Fuser N6 - STATIC2"),
    ("DATASET_FUSER_6", "INDEPENDENT_STATIC_6", "Fuser N6 - Independent N6"),
    ("DATASET_FUSER_12", "INDEPENDENT_STATIC_12", "Fuser N12 - Independent N12"),
    ("DATASET_FUSER_ALL", "INDEPENDENT_STATIC_ALL", "Fuser ALL - Independent ALL"),
)


def paired_bootstrap(candidate, baseline, draws=10000, seed=42):
    baseline_by_id = {row["sample_id"]: row["short_f1"] for row in baseline}
    if set(baseline_by_id) != {row["sample_id"] for row in candidate}:
        raise RuntimeError("paired bootstrap sample IDs do not match")
    differences = np.asarray([row["short_f1"] - baseline_by_id[row["sample_id"]]
                              for row in candidate], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = differences[rng.integers(0, len(differences), size=(draws, len(differences)))]
    values = 100 * sampled.mean(axis=1)
    return {"draws": draws, "seed": seed, "sample_unit": "sample_id",
        "samples": len(differences), "delta": 100 * float(differences.mean()),
        "ci95_lower": float(np.quantile(values, .025)),
        "ci95_upper": float(np.quantile(values, .975)),
        "p_delta_gt_0": float((values > 0).mean())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("2wiki", "musique", "triviaqa"), required=True)
    parser.add_argument("--root", default="cache/generalization")
    args = parser.parse_args(argv)
    benchmark = Path(args.root) / args.dataset / "benchmark"
    output = benchmark / "bootstrap.json"
    if output.exists():
        # Recover only the deterministic CSV projection when an earlier process
        # completed JSON atomically but failed while formatting the CSV.
        report = json.loads(output.read_text())
        expected = {"dataset": args.dataset, "benchmark_suite_runs": 1,
                    "draws": 10000, "seed": 42}
        if any(report.get(key) != value for key, value in expected.items()):
            raise RuntimeError("existing frozen bootstrap JSON is incompatible")
    else:
        results = json.loads((benchmark / "results.json").read_text())
        if results["status"] != "complete" or results["benchmark_suite_runs"] != 1:
            raise RuntimeError("benchmark is not a single complete frozen suite")
        rows = [json.loads(line) for line in
                (benchmark / "predictions.jsonl").read_text().splitlines()]
        by_config = {}
        for row in rows:
            by_config.setdefault(row["configuration"], []).append(row)
        report = {"dataset": args.dataset, "benchmark_suite_runs": 1,
                  "draws": 10000, "seed": 42, "comparisons": {}}
        for candidate, baseline, label in COMPARISONS:
            report["comparisons"][label] = paired_bootstrap(by_config[candidate],
                                                             by_config[baseline])
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with (benchmark / "bootstrap.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("dataset", "comparison", "delta",
            "ci95_lower", "ci95_upper", "p_delta_gt_0", "samples", "draws", "seed"))
        writer.writeheader()
        for label, values in report["comparisons"].items():
            writer.writerow({key: value for key, value in
                {"dataset": args.dataset, "comparison": label, **values}.items()
                if key in writer.fieldnames})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
