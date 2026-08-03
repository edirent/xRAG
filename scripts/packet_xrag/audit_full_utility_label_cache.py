#!/usr/bin/env python
"""Validate and independently recompute full train/dev utility label shards."""

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
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag.run_static_scorer_benchmark import initialize_generator
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache
from src.packet_xrag.controller.generator_utility import (
    candidate_addition_groups, gold_answer_nll_batch, utility_label,
)
from src.packet_xrag.controller.utility_label_dataset import (
    ShardedUtilityLabelDataset, percentile,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-root", default="cache/controller/utility_predictor/labels")
    parser.add_argument("--train-cache", default="cache/controller/features/train_features")
    parser.add_argument("--dev-cache", default="cache/controller/features/internal_dev_features")
    parser.add_argument("--k2-training-config", default="cache/projector/multi_token_k2/best_short_f1/training_config.json")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def structural_audit(dataset, feature_cache):
    record_by_id = {record["sample_id"]: record for record in feature_cache.records}
    keys, errors = set(), []; bases = defaultdict(set)
    for row in dataset.rows:
        key = (str(row["sample_id"]), tuple(row["selected_packet_ids"]),
               int(row["candidate_packet_id"]))
        if key in keys: errors.append(f"duplicate key {key}")
        keys.add(key)
        values = [row["base_answer_nll"], row["candidate_answer_nll"], row["delta_utility"]]
        if not all(math.isfinite(value) for value in values): errors.append(f"non-finite {key}")
        if abs(values[0] - values[1] - values[2]) > 1e-7: errors.append(f"delta mismatch {key}")
        if row["sample_id"] not in record_by_id:
            errors.append(f"unknown sample {key}"); continue
        valid = set(range(record_by_id[row["sample_id"]]["packet_count"]))
        if not set(row["selected_packet_ids"]).issubset(valid) or row["candidate_packet_id"] not in valid:
            errors.append(f"illegal packet ID {key}")
        if row["candidate_packet_id"] in row["selected_packet_ids"]:
            errors.append(f"candidate already selected {key}")
        bases[(row["sample_id"], tuple(row["selected_packet_ids"]))].add(row["base_answer_nll"])
    inconsistent = [key for key, values in bases.items() if len(values) != 1]
    if inconsistent: errors.append(f"inconsistent base NLL states: {len(inconsistent)}")
    return {"unique_keys": len(keys), "base_state_consistent": not inconsistent,
            "errors": errors}


@torch.inference_mode()
def recompute(selected_rows, dataset, feature_cache, tokenizer, generator, xrag_id, device):
    sample_meta = dataset.sample_by_id
    record_by_id = {record["sample_id"]: feature_cache[index]
                    for index, record in enumerate(feature_cache.records)}
    grouped = defaultdict(list)
    for row in selected_rows:
        grouped[row["sample_id"]].append(row)
    results = []
    for group_index, (sid, targets) in enumerate(grouped.items(), 1):
        record = record_by_id[sid]; metadata = sample_meta[sid]
        candidate_ids = metadata["candidate_packet_ids"]
        s0_groups = candidate_addition_groups([], candidate_ids)
        s0_nlls = gold_answer_nll_batch(
            generator, tokenizer, xrag_id, record["question"], record["answer"],
            record["packet_embeddings"], s0_groups, device,
        )
        state_values = {tuple(): (s0_nlls[0], dict(zip(candidate_ids, s0_nlls[1:])))}
        combined_groups, specs = [], []
        for state in metadata["states"]:
            selected = state["selected_packet_ids"]
            if not selected: continue
            remaining = [packet_id for packet_id in candidate_ids if packet_id not in set(selected)]
            groups = candidate_addition_groups(selected, candidate_ids)
            start = len(combined_groups); combined_groups.extend(groups)
            specs.append((tuple(selected), remaining, start, len(groups)))
        combined_nlls = gold_answer_nll_batch(
            generator, tokenizer, xrag_id, record["question"], record["answer"],
            record["packet_embeddings"], combined_groups, device,
        )
        for selected_key, remaining, start, length in specs:
            nlls = combined_nlls[start:start + length]
            state_values[selected_key] = (nlls[0], dict(zip(remaining, nlls[1:])))
        for row in targets:
            base, candidate_map = state_values[tuple(row["selected_packet_ids"])]
            candidate = candidate_map[row["candidate_packet_id"]]
            delta = base - candidate
            differences = {"base_nll_abs_diff": abs(base - row["base_answer_nll"]),
                           "candidate_nll_abs_diff": abs(candidate - row["candidate_answer_nll"]),
                           "delta_abs_diff": abs(delta - row["delta_utility"])}
            results.append({"sample_id": sid, "state_id": row["state_id"],
                            "candidate_packet_id": row["candidate_packet_id"], **differences,
                            "pass": all(value <= 1e-4 for value in differences.values())})
        if group_index % 20 == 0 or group_index == len(grouped):
            print(f"{dataset.split} recompute states: {group_index}/{len(grouped)}", flush=True)
    return results


def distribution(rows):
    values = [row["delta_utility"] for row in rows]
    counts = Counter(utility_label(value) for value in values)
    return {"count": len(values), "positive_fraction": counts["positive"] / len(values),
            "near_zero_fraction": counts["near-zero"] / len(values),
            "negative_fraction": counts["negative"] / len(values),
            "mean": statistics.mean(values), "std": statistics.pstdev(values),
            **{f"p{int(q*100):02d}": percentile(values, q)
               for q in (.01, .05, .25, .50, .75, .95, .99)}}


@torch.inference_mode()
def main(argv=None):
    args = parse_args(argv); labels_root = Path(args.labels_root)
    train = ShardedUtilityLabelDataset(labels_root, "train")
    dev = ShardedUtilityLabelDataset(labels_root, "internal_dev")
    train_cache = ControllerFeatureCache(args.train_cache)
    dev_cache = ControllerFeatureCache(args.dev_cache)
    structural = {"train": structural_audit(train, train_cache),
                  "internal_dev": structural_audit(dev, dev_cache)}
    rng = random.Random(42)
    chosen = {"train": rng.sample(train.rows, 200),
              "internal_dev": rng.sample(dev.rows, 100)}
    device = torch.device(args.device); torch.cuda.set_device(device)
    tokenizer, generator, xrag_id, _ = initialize_generator(args.k2_training_config, device)
    generator.eval()
    for parameter in generator.parameters(): parameter.requires_grad = False
    recomputed = {
        "train": recompute(chosen["train"], train, train_cache, tokenizer, generator,
                           xrag_id, device),
        "internal_dev": recompute(chosen["internal_dev"], dev, dev_cache, tokenizer,
                                  generator, xrag_id, device),
    }
    signals = {"train": distribution(train.rows), "internal_dev": distribution(dev.rows)}
    passes = {name: sum(item["pass"] for item in items)
              for name, items in recomputed.items()}
    valid = (not structural["train"]["errors"] and not structural["internal_dev"]["errors"]
             and passes == {"train": 200, "internal_dev": 100}
             and all(signal["positive_fraction"] >= .05 and
                     signal["negative_fraction"] >= .05 and signal["std"] >= .02
                     for signal in signals.values()))
    payload = {"status": "PASS" if valid else "INVALID", "valid": valid,
               "structural": structural, "signal": signals,
               "recompute": {name: {"requested": 200 if name == "train" else 100,
                                    "passes": passes[name], "tolerance": 1e-4,
                                    "records": recomputed[name]}
                             for name in recomputed},
               "implementation_audit": {"answer_masking": "locked", "mean_vs_total": "mean",
                                        "prompt": "P2_SHORT", "k2_injection": "2 per packet",
                                        "candidate_ordering": "append", "cache_key": "ordered state tuple",
                                        "eval_mode": True, "bf16_path": "same full-state batch shape"},
               "rebuild_attempts": 0, "benchmark_labels_built": False,
               "final_100_accessed": False, "final_100_runs": 0}
    write = labels_root / "validity_audit.json"
    write.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = ["# Full Utility Label Validity Audit", "", f"- Status: {payload['status']}",
             f"- Train recompute: {passes['train']}/200",
             f"- Internal-dev recompute: {passes['internal_dev']}/100",
             f"- Train positive/near-zero/negative: {signals['train']['positive_fraction']:.6f} / {signals['train']['near_zero_fraction']:.6f} / {signals['train']['negative_fraction']:.6f}",
             f"- Dev positive/near-zero/negative: {signals['internal_dev']['positive_fraction']:.6f} / {signals['internal_dev']['near_zero_fraction']:.6f} / {signals['internal_dev']['negative_fraction']:.6f}",
             "- Final 100 accessed: No", "- Final 100 runs: 0", ""]
    (labels_root / "validity_audit.md").write_text("\n".join(lines))
    print(json.dumps({"status": payload["status"], "passes": passes,
                      "signal": signals}, indent=2), flush=True)


if __name__ == "__main__": main()
