#!/usr/bin/env python
"""Paired bootstrap and fixed-rule selection for packet representation ablations."""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag.run_packet_representation_ablation import VARIANTS

BASELINE = "V1_TITLE_SENTENCE"


def percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def load_paired(path):
    records = {variant: {} for variant in VARIANTS}
    rows = []
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            variant, sample_id = row["variant"], str(row["sample_id"])
            if variant not in records:
                raise ValueError(f"unknown variant at line {line_number}: {variant}")
            if sample_id in records[variant]:
                raise ValueError(f"duplicate {variant}/{sample_id}")
            value = float(row["short_f1"])
            if not 0 <= value <= 1:
                raise ValueError(f"invalid Short F1 at line {line_number}")
            records[variant][sample_id] = value
            rows.append(row)
    baseline_ids = set(records[BASELINE])
    if len(baseline_ids) != 500:
        raise ValueError(f"expected 500 baseline samples, got {len(baseline_ids)}")
    for variant in VARIANTS:
        if set(records[variant]) != baseline_ids:
            raise ValueError(f"sample_id set differs for {variant}")
    return records, rows


def paired_bootstrap(records, num_bootstrap=10_000, seed=42):
    sample_ids = sorted(records[BASELINE])
    values = {variant: [records[variant][sid] for sid in sample_ids] for variant in VARIANTS}
    rng = random.Random(seed)
    draws = {variant: [] for variant in VARIANTS}
    delta_draws = {variant: [] for variant in VARIANTS[1:]}
    for _ in range(num_bootstrap):
        indices = [rng.randrange(len(sample_ids)) for _ in sample_ids]
        sampled = {variant: mean(values[variant][i] for i in indices) for variant in VARIANTS}
        for variant in VARIANTS:
            draws[variant].append(sampled[variant])
        for variant in VARIANTS[1:]:
            delta_draws[variant].append(sampled[variant] - sampled[BASELINE])
    result = {"num_samples": 500, "num_bootstrap": num_bootstrap, "seed": seed,
        "baseline": {"variant": BASELINE, "short_f1": 100 * mean(values[BASELINE]),
            "ci95_low": 100 * percentile(draws[BASELINE], .025), "ci95_high": 100 * percentile(draws[BASELINE], .975)},
        "comparisons": {}}
    for variant in VARIANTS[1:]:
        delta = [values[variant][i] - values[BASELINE][i] for i in range(len(sample_ids))]
        result["comparisons"][variant] = {"short_f1": 100 * mean(values[variant]),
            "short_f1_ci95": [100 * percentile(draws[variant], .025), 100 * percentile(draws[variant], .975)],
            "delta_vs_v1": 100 * mean(delta),
            "delta_ci95": [100 * percentile(delta_draws[variant], .025), 100 * percentile(delta_draws[variant], .975)],
            "probability_positive": mean(value > 0 for value in delta_draws[variant]),
            "probability_at_least_1": mean(100 * value >= 1 for value in delta_draws[variant]),
            "probability_at_least_3": mean(100 * value >= 3 for value in delta_draws[variant])}
    return result


def aggregate_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)
    result = {}
    for variant, items in grouped.items():
        chunks = [chunk for row in items for chunk in row["selected_chunks"]]
        result[variant] = {"Short EM": 100 * mean(row["short_em"] for row in items),
            "Short F1": 100 * mean(row["short_f1"] for row in items),
            "Avg chunks": mean(row["num_selected_chunks"] for row in items),
            "Avg used tokens/sample": mean(row["total_used_retriever_tokens"] for row in items),
            "Truncation rate": 100 * mean(chunk["was_truncated"] for chunk in chunks)}
    return result


def tier_for(score, delta, ci_low):
    if score >= 66 and delta >= 3 and ci_low > 0:
        return "Tier 1"
    if score < 66 and delta >= 1 and ci_low > 0:
        return "Tier 2"
    if delta <= 0:
        return "Tier 4"
    return "Tier 3"


def select_best(stats, bootstrap):
    candidates = []
    for variant in VARIANTS[1:]:
        comparison = bootstrap["comparisons"][variant]
        tier = tier_for(stats[variant]["Short F1"], comparison["delta_vs_v1"], comparison["delta_ci95"][0])
        candidates.append((variant, tier))
    eligible = [(v, t) for v, t in candidates if t in {"Tier 1", "Tier 2"}]
    pool = eligible or candidates
    pool.sort(key=lambda item: (-stats[item[0]]["Short F1"], stats[item[0]]["Avg chunks"], stats[item[0]]["Avg used tokens/sample"]))
    return pool[0], dict(candidates)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="cache/results/packet_representation_ablation_full500.jsonl")
    parser.add_argument("--num-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-output", default="cache/results/packet_representation_ablation_bootstrap.json")
    parser.add_argument("--csv-output", default="cache/results/packet_representation_ablation_bootstrap.csv")
    parser.add_argument("--markdown-output", default="cache/results/packet_representation_ablation_bootstrap.md")
    parser.add_argument("--selection-output", default="cache/results/packet_representation_selection_summary.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    records, rows = load_paired(args.input)
    result = paired_bootstrap(records, args.num_bootstrap, args.seed)
    stats = aggregate_rows(rows)
    for variant in VARIANTS:
        point = result["baseline"]["short_f1"] if variant == BASELINE else result["comparisons"][variant]["short_f1"]
        if abs(point - stats[variant]["Short F1"]) > 1e-10:
            raise AssertionError(f"bootstrap/summary point estimate mismatch for {variant}")
    Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_output).write_text(json.dumps(result, indent=2) + "\n")
    comparison_fields = ["Variant", "Short F1", "F1 CI Low", "F1 CI High", "Delta vs V1", "Delta CI Low", "Delta CI High", "P(Delta>0)", "P(Delta>=1)", "P(Delta>=3)"]
    with Path(args.csv_output).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=comparison_fields); writer.writeheader()
        for variant in VARIANTS[1:]:
            c = result["comparisons"][variant]
            writer.writerow(dict(zip(comparison_fields, [variant, c["short_f1"], *c["short_f1_ci95"], c["delta_vs_v1"], *c["delta_ci95"], c["probability_positive"], c["probability_at_least_1"], c["probability_at_least_3"]])))
    lines = ["# Packet Representation Paired Bootstrap", "", f"- Samples: 500; resamples: {args.num_bootstrap}; seed: {args.seed}.",
        f"- V1 Short F1: {result['baseline']['short_f1']:.4f} (95% CI {result['baseline']['ci95_low']:.4f} to {result['baseline']['ci95_high']:.4f}).", ""]
    for variant in VARIANTS[1:]:
        c = result["comparisons"][variant]
        lines.append(f"- {variant}: {c['short_f1']:.4f}; delta {c['delta_vs_v1']:.4f} (95% CI {c['delta_ci95'][0]:.4f} to {c['delta_ci95'][1]:.4f}); P(delta>0)={c['probability_positive']:.4f}.")
    Path(args.markdown_output).write_text("\n".join(lines) + "\n")
    selection_fields = ["Variant", "Short EM", "Short F1", "Delta vs V1", "Delta CI Low", "Delta CI High", "P(Delta>0)", "Avg chunks", "Avg used tokens/sample", "Truncation rate"]
    selection_rows = []
    for variant in VARIANTS:
        c = {"delta_vs_v1": 0.0, "delta_ci95": [0.0, 0.0], "probability_positive": 0.0} if variant == BASELINE else result["comparisons"][variant]
        selection_rows.append({"Variant": variant, **stats[variant], "Delta vs V1": c["delta_vs_v1"], "Delta CI Low": c["delta_ci95"][0], "Delta CI High": c["delta_ci95"][1], "P(Delta>0)": c["probability_positive"]})
    selection_rows.sort(key=lambda row: (-row["Short F1"], row["Avg chunks"], row["Avg used tokens/sample"]))
    with Path(args.selection_output).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=selection_fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(selection_rows)
    (best, tier), tiers = select_best(stats, result)
    print(json.dumps({"bootstrap": result, "best_variant": best, "tier": tier, "tiers": tiers, "stats": stats}, indent=2))


if __name__ == "__main__":
    main()
