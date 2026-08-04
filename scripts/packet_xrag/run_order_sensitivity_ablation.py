#!/usr/bin/env python
"""Evaluate physical extra-packet order while preserving STATIC rank-1/2 base."""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag.composition_training_common import (
    build_fuser, load_frozen_generator, load_frozen_k2_projector,
)
from scripts.packet_xrag.evaluate_full_composition_dev import evaluate_fuser, sha256, summarize
from scripts.packet_xrag.utility_predictor_training_common import load_static_score_cache
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache
from src.packet_xrag.generalization.order_robustness import architecture_order_audit, permute_extras
from src.packet_xrag.generalization.protocol import SEED, ordered_hash


VARIANTS = ("canonical", "reverse", "document", "random_0", "random_1", "random_2")


def split_metrics(rows_by_variant):
    summaries = {name: summarize(rows) for name, rows in rows_by_variant.items()}
    by_id = defaultdict(dict)
    for name, rows in rows_by_variant.items():
        for row in rows: by_id[row["sample_id"]][name] = row
    disagreements, ranges = [], []
    for values in by_id.values():
        predictions = [values[name]["short_prediction"] for name in VARIANTS]
        disagreements.append(float(len(set(predictions)) > 1))
        scores = [values[name]["short_f1"] for name in VARIANTS]
        ranges.append(max(scores) - min(scores))
    f1s = [summaries[name]["short_f1"] for name in VARIANTS]
    return {"variants": summaries, "f1_mean": mean(f1s), "f1_std": pstdev(f1s),
            "worst_order_f1": min(f1s), "prediction_disagreement": mean(disagreements),
            "empty_variation": max(value["empty"] for value in summaries.values()) -
                               min(value["empty"] for value in summaries.values()),
            "mean_per_sample_max_f1_range": 100 * mean(ranges)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="cache/generalization")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv); root = Path(args.root); output_dir = root / "order"
    if output_dir.exists(): raise RuntimeError("refusing to overwrite order diagnostic")
    cache = ControllerFeatureCache("cache/controller/features/internal_dev_features")
    materialized = [cache[index] for index in range(len(cache))]
    ids = [record["sample_id"] for record in materialized]
    shuffled = list(ids); random.Random(SEED).shuffle(shuffled)
    split_ids = {"ORDER_DEV": shuffled[:250], "ORDER_HOLDOUT": shuffled[250:]}
    output_dir.mkdir(parents=True)
    (output_dir / "split_manifest.json").write_text(json.dumps({name: {
        "sample_count": len(values), "ordered_ids_sha256": ordered_hash(values)}
        for name, values in split_ids.items()}, indent=2, sort_keys=True) + "\n")
    device = torch.device(args.device); torch.cuda.set_device(device)
    scores = load_static_score_cache(cache, "cache/controller/static/best_short_f1/scorer.pt",
        "cache/controller/utility_predictor/features/internal_dev_static_scores.pt", device)
    rankings = {record["sample_id"]: sorted(range(record["packet_count"]),
        key=lambda index: (-float(scores[record["sample_id"]][index]), index)) for record in materialized}
    tokenizer, generator, xrag_id, config = load_frozen_generator(device)
    k2 = load_frozen_k2_projector(config, device); fuser = build_fuser("C1").to(device)
    checkpoint = Path("cache/composition/full/C1_O1/epoch_6.pt")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    fuser.load_state_dict(payload["state_dict"], strict=True); fuser.eval()
    all_rows = []; results = {}
    by_id = {record["sample_id"]: record for record in materialized}
    for split_name, current_ids in split_ids.items():
        records = [by_id[sid] for sid in current_ids]; rows_by_variant = {}
        for variant in VARIANTS:
            groups = [permute_extras(record, rankings[record["sample_id"]][:6], variant)
                      for record in records]
            name = f"{split_name}_{variant.upper()}"
            rows = evaluate_fuser(name, fuser, records, groups, k2, tokenizer, generator,
                xrag_id, device, args.batch_size); rows_by_variant[variant] = rows; all_rows += rows
            print(json.dumps({name: summarize(rows)}), flush=True)
        results[split_name] = split_metrics(rows_by_variant)
    audit = architecture_order_audit(fuser)
    result = {"status": "COMPLETE", "checkpoint_sha256": sha256(checkpoint),
        "architecture_audit": audit, "base_policy": "STATIC rank 1-2 fixed",
        "physical_order_scope": "STATIC rank 3-6 only", "results": results,
        "variant_selected": "O2" if not audit["explicit_packet_order_embedding"] else "O1",
        "benchmark_regenerated": False, "final100_accessed": False}
    (output_dir / "original_results.json").write_text(json.dumps(result, indent=2,
                                                                   sort_keys=True) + "\n")
    with (output_dir / "original_predictions.jsonl").open("w") as stream:
        for row in all_rows: stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__": main()
