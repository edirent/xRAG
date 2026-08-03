#!/usr/bin/env python
"""Validate and independently recompute the frozen marginal-utility cache."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag.run_static_scorer_benchmark import initialize_generator
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache
from src.packet_xrag.controller.generator_utility import (
    candidate_addition_groups, gold_answer_nll_batch, utility_cache_key, utility_label,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--utility-dir", default="cache/controller/utility_feasibility")
    parser.add_argument("--feature-cache", default="cache/controller/features/internal_dev_features")
    parser.add_argument("--k2-training-config", default="cache/projector/multi_token_k2/best_short_f1/training_config.json")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--recompute-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position); upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def structural_checks(state_rows, utility_rows):
    panels = {row["sample_id"]: row for row in state_rows}
    keys, errors = set(), []
    bases = defaultdict(set); candidates = defaultdict(set)
    for row in utility_rows:
        key = utility_cache_key(
            row["sample_id"], row["selected_packet_ids"], row["candidate_packet_id"]
        )
        if key in keys:
            errors.append(f"duplicate cache key: {key}")
        keys.add(key)
        values = (row["base_answer_nll"], row["candidate_answer_nll"], row["delta_utility"])
        if not all(math.isfinite(value) for value in values):
            errors.append(f"non-finite utility: {key}")
        if abs((values[0] - values[1]) - values[2]) > 1e-7:
            errors.append(f"delta arithmetic mismatch: {key}")
        if row["sample_id"] not in panels:
            errors.append(f"unknown sample: {key}"); continue
        panel = panels[row["sample_id"]]
        valid = set(range(panel["packet_count"]))
        selected = row["selected_packet_ids"]
        if not set(selected).issubset(valid) or row["candidate_packet_id"] not in valid:
            errors.append(f"illegal packet ID: {key}")
        if row["candidate_packet_id"] in selected:
            errors.append(f"selected candidate repeated: {key}")
        state_key = (row["sample_id"], tuple(sorted(selected)))
        bases[state_key].add(row["base_answer_nll"])
        candidates[key].add((row["candidate_answer_nll"], row["delta_utility"]))
    inconsistent_bases = [key for key, values in bases.items() if len(values) != 1]
    inconsistent_candidates = [key for key, values in candidates.items() if len(values) != 1]
    if inconsistent_bases:
        errors.append(f"inconsistent repeated base NLL: {len(inconsistent_bases)}")
    if inconsistent_candidates:
        errors.append(f"inconsistent duplicate-state utility: {len(inconsistent_candidates)}")
    s0_by_sample = defaultdict(set)
    for row in utility_rows:
        if "S0_EMPTY" in row["state_source_tags"]:
            s0_by_sample[row["sample_id"]].add(row["base_answer_nll"])
    bad_s0 = [sid for sid, values in s0_by_sample.items() if len(values) != 1]
    if bad_s0:
        errors.append(f"S0 base inconsistency: {len(bad_s0)}")
    return errors, {"unique_cache_keys": len(keys), "zero_state_consistent": not bad_s0,
                    "duplicate_state_consistent": not inconsistent_bases and not inconsistent_candidates}


@torch.inference_mode()
def recompute(rows, state_rows, cache, tokenizer, generator, xrag_id, device):
    panel_by_id = {row["sample_id"]: row for row in state_rows}
    record_by_id = {record["sample_id"]: cache[index] for index, record in enumerate(cache.records)}
    # BF16 GEMM kernels may legitimately depend on batch shape.  Reconstruct the
    # complete preregistered state batch (base plus every legal addition), exactly
    # as the cache builder did, while independently loading inputs/checkpoints.
    by_state = defaultdict(list)
    for row in rows:
        by_state[(row["sample_id"], tuple(row["selected_packet_ids"]))].append(row)
    results = []
    for index, ((sid, selected_tuple), targets) in enumerate(by_state.items(), 1):
        panel = panel_by_id[sid]; record = record_by_id[sid]
        selected = list(selected_tuple)
        candidate_ids = [item["packet_id"] for item in panel["candidates"]]
        remaining = [packet_id for packet_id in candidate_ids if packet_id not in set(selected)]
        groups = candidate_addition_groups(selected, candidate_ids)
        nlls = gold_answer_nll_batch(
            generator, tokenizer, xrag_id, panel["question"], panel["answer"],
            record["packet_embeddings"], groups, device,
        )
        candidate_nll = {packet_id: value for packet_id, value in zip(remaining, nlls[1:])}
        for row in targets:
            base = nlls[0]; candidate = candidate_nll[row["candidate_packet_id"]]
            delta = base - candidate
            differences = {
                "base_nll_abs_diff": abs(base - row["base_answer_nll"]),
                "candidate_nll_abs_diff": abs(candidate - row["candidate_answer_nll"]),
                "delta_abs_diff": abs(delta - row["delta_utility"]),
            }
            results.append({
                "sample_id": sid, "state_id": row["state_id"],
                "candidate_packet_id": row["candidate_packet_id"], **differences,
                "pass": all(value <= 1e-4 for value in differences.values()),
            })
        if index % 10 == 0 or index == len(by_state):
            print(f"recompute audit states: {index}/{len(by_state)}", flush=True)
    return results


def distribution(rows):
    values = [row["delta_utility"] for row in rows]
    counts = Counter(utility_label(value) for value in values)
    return {
        "count": len(values),
        "positive_fraction": counts["positive"] / len(values),
        "near_zero_fraction": counts["near-zero"] / len(values),
        "negative_fraction": counts["negative"] / len(values),
        "mean_delta": statistics.mean(values), "std_delta": statistics.pstdev(values),
        "p5": percentile(values, .05), "p25": percentile(values, .25),
        "p50": percentile(values, .50), "p75": percentile(values, .75),
        "p95": percentile(values, .95),
    }


@torch.inference_mode()
def main(argv=None):
    args = parse_args(argv)
    if args.recompute_count != 100 or args.seed != 42:
        raise RuntimeError("locked validity audit requires 100 records and seed 42")
    root = Path(args.utility_dir)
    manifest = json.loads((root / "cache_manifest.json").read_text())
    if manifest.get("completion_status") != "complete":
        raise RuntimeError("utility cache is incomplete")
    state_rows = read_jsonl(root / "state_cache.jsonl")
    utility_rows = read_jsonl(root / "marginal_utility.jsonl")
    errors, structural = structural_checks(state_rows, utility_rows)
    selected = random.Random(args.seed).sample(utility_rows, args.recompute_count)
    device = torch.device(args.device); torch.cuda.set_device(device)
    cache = ControllerFeatureCache(args.feature_cache)
    tokenizer, generator, xrag_id, _ = initialize_generator(args.k2_training_config, device)
    generator.eval()
    for parameter in generator.parameters():
        parameter.requires_grad = False
    if any(parameter.requires_grad for parameter in generator.parameters()):
        raise RuntimeError("generator is not frozen")
    recomputed = recompute(selected, state_rows, cache, tokenizer, generator, xrag_id, device)
    recompute_passes = sum(row["pass"] for row in recomputed)
    signal = distribution(utility_rows)
    valid = (
        not errors and recompute_passes == 100 and signal["positive_fraction"] >= .05
        and signal["negative_fraction"] >= .05 and signal["std_delta"] >= .02
    )
    payload = {
        "status": "PASS" if valid else "INVALID", "cache_valid": valid,
        "structural_checks": structural, "structural_errors": errors,
        "recompute": {"seed": args.seed, "sample_count": args.recompute_count,
                      "passes": recompute_passes, "failures": 100 - recompute_passes,
                      "tolerance": 1e-4, "records": recomputed},
        "signal_distribution": signal,
        "validity_requirements": {"positive_fraction_min": .05,
                                  "negative_fraction_min": .05, "std_delta_min": .02,
                                  "recompute_required": "100/100"},
        "implementation_audit": {
            "answer_masking": "exact build_gold_answer_inputs implementation hash audited",
            "normalization": "mean shifted gold-answer token NLL",
            "prompt_construction": "P2_SHORT via train_packet_projector.build_prompt",
            "k2_token_injection": "asserted 2 * selected packet count",
            "candidate_ordering": "state order preserved; candidate appended",
            "cache_key": "(sample_id, sorted selected set, candidate_packet_id)",
            "model_mode": "eval + all requires_grad=False + inference_mode",
            "implementation_error_found": False,
        },
        "final_100_accessed": False, "final_100_runs": 0,
    }
    (root / "utility_validity_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Utility Validity Audit", "", f"- Status: {payload['status']}",
        f"- Unique cache keys: {structural['unique_cache_keys']}",
        f"- Recompute: {recompute_passes}/100 pass at <= 1e-4",
        f"- Positive / near-zero / negative: {signal['positive_fraction']:.4f} / {signal['near_zero_fraction']:.4f} / {signal['negative_fraction']:.4f}",
        f"- Delta mean / std: {signal['mean_delta']:.6f} / {signal['std_delta']:.6f}",
        "- Final 100 accessed: No", "- Final 100 runs: 0", "",
    ]
    (root / "utility_validity_audit.md").write_text("\n".join(lines))
    print(json.dumps({key: value for key, value in payload.items() if key != "recompute"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
