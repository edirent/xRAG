#!/usr/bin/env python3
"""Paired bootstrap analysis for the held-out projector gate."""

import argparse
import json
import random
from pathlib import Path
from statistics import mean


REQUIRED_SELECTOR_METHODS = ("XRAG_ORACLE", "TEXT_ORACLE")
PROJECTOR_METHOD = "XRAG_ORACLE"


def load_unique(path: Path, methods: tuple[str, ...]) -> dict[str, dict[str, float]]:
    records = {method: {} for method in methods}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            method = row.get("method")
            if method not in records:
                continue
            sample_id = str(row["sample_id"])
            if sample_id in records[method]:
                raise ValueError(f"duplicate {method}/{sample_id} in {path}:{line_number}")
            value = float(row["short_f1"])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"short_f1 outside [0, 1] in {path}:{line_number}: {value}")
            records[method][sample_id] = value
    return records


def percentile(values: list[float], percentage: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def interval(point: float, draws: list[float]) -> dict[str, float]:
    return {
        "point_estimate": 100.0 * point,
        "ci95_low": 100.0 * percentile(draws, 2.5),
        "ci95_high": 100.0 * percentile(draws, 97.5),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-results", type=Path, required=True)
    parser.add_argument("--projector-results", type=Path, required=True)
    parser.add_argument("--num-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_bootstrap <= 0:
        raise ValueError("--num-bootstrap must be positive")

    selector = load_unique(args.selector_results, REQUIRED_SELECTOR_METHODS)
    projector = load_unique(args.projector_results, (PROJECTOR_METHOD,))
    before_ids = set(selector["XRAG_ORACLE"])
    text_ids = set(selector["TEXT_ORACLE"])
    after_ids = set(projector["XRAG_ORACLE"])
    if len(before_ids) != 100 or len(after_ids) != 100:
        raise ValueError(
            f"expected 100 before and after samples, got {len(before_ids)} and {len(after_ids)}"
        )
    if before_ids != after_ids or before_ids != text_ids:
        raise ValueError("sample_id sets for before XRAG, after XRAG, and TEXT do not match")

    sample_ids = sorted(before_ids)
    before = [selector["XRAG_ORACLE"][sample_id] for sample_id in sample_ids]
    after = [projector["XRAG_ORACLE"][sample_id] for sample_id in sample_ids]
    text = [selector["TEXT_ORACLE"][sample_id] for sample_id in sample_ids]
    before_gap = [t - b for t, b in zip(text, before)]
    after_gap = [t - a for t, a in zip(text, after)]

    rng = random.Random(args.seed)
    draws = {name: [] for name in ("before", "after", "improvement", "before_gap", "after_gap")}
    for _ in range(args.num_bootstrap):
        indices = [rng.randrange(len(sample_ids)) for _ in sample_ids]
        before_draw = mean(before[index] for index in indices)
        after_draw = mean(after[index] for index in indices)
        draws["before"].append(before_draw)
        draws["after"].append(after_draw)
        draws["improvement"].append(after_draw - before_draw)
        draws["before_gap"].append(mean(before_gap[index] for index in indices))
        draws["after_gap"].append(mean(after_gap[index] for index in indices))

    before_point = mean(before)
    after_point = mean(after)
    result = {
        "num_samples": len(sample_ids),
        "num_bootstrap": args.num_bootstrap,
        "seed": args.seed,
        "before_xrag_oracle_f1": interval(before_point, draws["before"]),
        "after_xrag_oracle_f1": interval(after_point, draws["after"]),
        "improvement": {
            **interval(after_point - before_point, draws["improvement"]),
            "probability_positive": mean(value > 0.0 for value in draws["improvement"]),
        },
        "before_representation_gap": interval(mean(before_gap), draws["before_gap"]),
        "after_representation_gap": interval(mean(after_gap), draws["after_gap"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    improvement = result["improvement"]
    stable = improvement["ci95_low"] > 0.0
    after_interval = result["after_xrag_oracle_f1"]
    gate_in_interval = after_interval["ci95_low"] <= 55.0 <= after_interval["ci95_high"]
    gap_change = result["before_representation_gap"]["point_estimate"] - result["after_representation_gap"]["point_estimate"]
    markdown = f"""# Projector gate paired bootstrap

- Samples: {len(sample_ids)} paired examples; bootstrap draws: {args.num_bootstrap}; seed: {args.seed}.
- Before XRAG_ORACLE Short F1: {100 * before_point:.2f} (95% CI {after_or_before(result, 'before_xrag_oracle_f1')}).
- After XRAG_ORACLE Short F1: {100 * after_point:.2f} (95% CI {after_or_before(result, 'after_xrag_oracle_f1')}).
- Improvement: {improvement['point_estimate']:.2f} F1 (95% CI {improvement['ci95_low']:.2f} to {improvement['ci95_high']:.2f}); P(improvement > 0) = {improvement['probability_positive']:.4f}.
- The improvement is {'stable as positive at the 95% level' if stable else 'not conclusively positive at the 95% level'}.
- The 55 F1 gate is {'inside' if gate_in_interval else 'outside'} the after-score 95% bootstrap interval, so the 52.93-to-55 difference is {'within substantial sampling uncertainty' if gate_in_interval else 'not covered by that interval'}.
- Representation gap changed from {result['before_representation_gap']['point_estimate']:.2f} to {result['after_representation_gap']['point_estimate']:.2f} F1, a reduction of {gap_change:.2f} points. Its after-training 95% CI is {after_or_before(result, 'after_representation_gap')}.

These estimates are diagnostic only; the final 100 samples are not used for hyperparameter selection.
"""
    args.output.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(json.dumps(result, indent=2))


def after_or_before(result: dict, key: str) -> str:
    metric = result[key]
    return f"{metric['ci95_low']:.2f} to {metric['ci95_high']:.2f}"


if __name__ == "__main__":
    main()
