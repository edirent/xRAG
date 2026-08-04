#!/usr/bin/env python
"""Evaluate the sole O2 run and apply the preregistered adoption gate."""

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
from scripts.packet_xrag.evaluate_full_composition_dev import evaluate_fuser, summarize
from scripts.packet_xrag.train_full_composition import static_ranking
from scripts.packet_xrag.utility_predictor_training_common import load_static_score_cache
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache
from src.packet_xrag.generalization.order_robustness import permute_extras
from src.packet_xrag.generalization.protocol import SEED


VARIANTS = ("canonical", "reverse", "document", "random_0", "random_1", "random_2")


def aggregate(rows_by_variant):
    summaries = {name: summarize(rows) for name, rows in rows_by_variant.items()}
    by_id = defaultdict(dict)
    for name, rows in rows_by_variant.items():
        for row in rows: by_id[row["sample_id"]][name] = row
    f1s = [summaries[name]["short_f1"] for name in VARIANTS]
    return {"variants": summaries, "f1_mean": mean(f1s), "f1_std": pstdev(f1s),
            "worst_order_f1": min(f1s),
            "prediction_disagreement": mean(float(len({values[name]["short_prediction"]
                for name in VARIANTS}) > 1) for values in by_id.values()),
            "mean_per_sample_max_f1_range": 100 * mean(max(values[name]["short_f1"]
                for name in VARIANTS) - min(values[name]["short_f1"]
                for name in VARIANTS) for values in by_id.values())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="cache/generalization")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv); root = Path(args.root); order_root = root / "order"
    output = order_root / "O2_evaluation"
    if output.exists(): raise RuntimeError("refusing to overwrite O2 gate evaluation")
    selection = json.loads((order_root / "O2_training/selection.json").read_text())
    original = json.loads((order_root / "original_results.json").read_text())
    cache = ControllerFeatureCache("cache/controller/features/internal_dev_features")
    records = [cache[index] for index in range(len(cache))]
    ids = [record["sample_id"] for record in records]; random.Random(SEED).shuffle(ids)
    split_ids = {"ORDER_DEV": ids[:250], "ORDER_HOLDOUT": ids[250:]}
    device = torch.device(args.device); torch.cuda.set_device(device)
    scores = load_static_score_cache(cache, "cache/controller/static/best_short_f1/scorer.pt",
        "cache/controller/utility_predictor/features/internal_dev_static_scores.pt", device)
    rankings = {record["sample_id"]: static_ranking(record, scores[record["sample_id"]])
                for record in records}
    tokenizer, generator, xrag_id, config = load_frozen_generator(device)
    k2 = load_frozen_k2_projector(config, device); fuser = build_fuser("C1").to(device)
    payload = torch.load(selection["best_checkpoint"], map_location="cpu", weights_only=True)
    fuser.load_state_dict(payload["state_dict"], strict=True); fuser.eval()
    by_id = {record["sample_id"]: record for record in records}; results = {}; all_rows = []
    for split, current_ids in split_ids.items():
        current = [by_id[sid] for sid in current_ids]; rows_by_variant = {}
        for variant in VARIANTS:
            groups = [permute_extras(record, rankings[record["sample_id"]][:6], variant)
                      for record in current]
            rows = evaluate_fuser(f"O2_{split}_{variant}", fuser, current, groups, k2,
                tokenizer, generator, xrag_id, device, args.batch_size)
            rows_by_variant[variant] = rows; all_rows.extend(rows)
            print(json.dumps({f"{split}_{variant}": summarize(rows)}), flush=True)
        results[split] = aggregate(rows_by_variant)
    old = original["results"]["ORDER_HOLDOUT"]; new = results["ORDER_HOLDOUT"]
    old_canonical = old["variants"]["canonical"]["short_f1"]
    new_canonical = new["variants"]["canonical"]["short_f1"]
    old_degradation = old_canonical - old["worst_order_f1"]
    new_degradation = new_canonical - new["worst_order_f1"]
    canonical_gate = new_canonical >= old_canonical - .5
    absolute_gate = new["worst_order_f1"] >= old["worst_order_f1"] + 2.0
    reduction_gate = old_degradation > 0 and new_degradation <= .5 * old_degradation
    adopted = canonical_gate and (absolute_gate or reduction_gate)
    gate = {"canonical_noninferiority": canonical_gate,
            "worst_order_improvement_at_least_2": absolute_gate,
            "degradation_reduced_at_least_50_percent": reduction_gate,
            "original_canonical_f1": old_canonical, "variant_canonical_f1": new_canonical,
            "original_worst_f1": old["worst_order_f1"],
            "variant_worst_f1": new["worst_order_f1"],
            "original_degradation": old_degradation, "variant_degradation": new_degradation,
            "adopted": adopted,
            "policy": "O2" if adopted else "original canonical recipe"}
    report = {"status": "COMPLETE", "checkpoint_sha256": selection["best_checkpoint_sha256"],
              "results": results, "gate": gate, "benchmark_regenerated": False,
              "hotpot_primary_checkpoint_changed": False, "final100_accessed": False}
    output.mkdir(parents=True)
    (output / "results.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with (output / "predictions.jsonl").open("w") as stream:
        for row in all_rows: stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    ledger_path = root / "experiment_ledger.json"; ledger = json.loads(ledger_path.read_text())
    ledger["order_variant"] = {"status": "gate_complete", "variant": "O2",
        "checkpoint_sha256": selection["best_checkpoint_sha256"], "adopted": adopted,
        "frozen_policy": gate["policy"]}
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__": main()
