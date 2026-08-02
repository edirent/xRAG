#!/usr/bin/env python
"""Locked paired bootstrap comparisons for token-state resampler models."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag.token_resampler_common import EXPECTED_SPLIT_HASH


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-state", default="cache/resampler/token_state_depth1/best_short_f1/validation_predictions.jsonl")
    parser.add_argument("--pooled-control", default=None)
    parser.add_argument("--pooled-k2", default="cache/projector/multi_token_k2/best_short_f1/validation_predictions.jsonl")
    parser.add_argument("--pooled-k2-lora", default="cache/results/k2_lora_validation_predictions.jsonl")
    parser.add_argument("--pooled-k1", default="cache/results/packet_representation_ablation.jsonl")
    parser.add_argument("--num-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default="cache/results/token_resampler_bootstrap.json")
    parser.add_argument("--output-csv", default="cache/results/token_resampler_bootstrap.csv")
    parser.add_argument("--output-md", default="cache/results/token_resampler_bootstrap.md")
    return parser.parse_args()


def read_rows(path, variant=None):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if variant is not None:
        rows = [row for row in rows if row.get("variant") == variant]
    if len(rows) != 500:
        raise RuntimeError(f"expected 500 rows in {path}, got {len(rows)}")
    return {row["sample_id"]: float(row["short_f1"]) for row in rows}


def compare(name, left, right, ids, rng, count):
    left_scores = np.array([left[sid] for sid in ids]) * 100
    right_scores = np.array([right[sid] for sid in ids]) * 100
    differences = left_scores - right_scores
    sampled = np.empty(count, dtype=np.float64)
    for start in range(0, count, 1000):
        size = min(1000, count - start)
        indices = rng.integers(0, len(ids), size=(size, len(ids)))
        sampled[start:start + size] = differences[indices].mean(axis=1)
    point = float(differences.mean())
    return {
        "comparison": name, "delta_short_f1": point,
        "ci95_lower": float(np.quantile(sampled, 0.025)),
        "ci95_upper": float(np.quantile(sampled, 0.975)),
        "p_delta_gt_0": float(np.mean(sampled > 0)),
        "p_delta_ge_1": float(np.mean(sampled >= 1)),
        "p_delta_ge_2": float(np.mean(sampled >= 2)),
        "p_delta_ge_3": float(np.mean(sampled >= 3)),
    }


def main():
    args = parse_args(); assert args.num_bootstrap == 10000 and args.seed == 42
    split = json.loads(Path("cache/projector/packet_projector_calibration/data_split.json").read_text())
    ids = split["validation_sample_ids"]
    import hashlib
    if hashlib.sha256("".join(ids).encode()).hexdigest() != EXPECTED_SPLIT_HASH:
        raise RuntimeError("validation split mismatch")
    models = {
        "TokenState_D1": read_rows(args.token_state),
        "Pooled_K2": read_rows(args.pooled_k2),
        "Pooled_K2_LoRA": read_rows(args.pooled_k2_lora),
        "Pooled_K1": read_rows(args.pooled_k1, "V1_TITLE_SENTENCE"),
    }
    if args.pooled_control:
        models["PooledControl"] = read_rows(args.pooled_control)
    for name, scores in models.items():
        if set(scores) != set(ids): raise RuntimeError(f"ID mismatch: {name}")
    pairs = [
        ("TokenState_D1_vs_Pooled_K2", "TokenState_D1", "Pooled_K2"),
        ("TokenState_D1_vs_Pooled_K2_LoRA", "TokenState_D1", "Pooled_K2_LoRA"),
        ("TokenState_D1_vs_Pooled_K1", "TokenState_D1", "Pooled_K1"),
    ]
    if args.pooled_control:
        pairs += [
            ("TokenState_D1_vs_PooledControl", "TokenState_D1", "PooledControl"),
            ("PooledControl_vs_Pooled_K2", "PooledControl", "Pooled_K2"),
        ]
    rng = np.random.default_rng(args.seed)
    results = [compare(name, models[left], models[right], ids, rng, args.num_bootstrap)
               for name, left, right in pairs]
    payload = {"validation_split_hash": EXPECTED_SPLIT_HASH, "samples": 500,
               "resamples": args.num_bootstrap, "seed": args.seed, "comparisons": results}
    Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with Path(args.output_csv).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    lines = ["# Token Resampler Paired Bootstrap", ""]
    for row in results:
        lines.append(f"- {row['comparison']}: delta {row['delta_short_f1']:.6f}, 95% CI [{row['ci95_lower']:.6f}, {row['ci95_upper']:.6f}], P(delta>0)={row['p_delta_gt_0']:.4f}.")
    Path(args.output_md).write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__": main()
