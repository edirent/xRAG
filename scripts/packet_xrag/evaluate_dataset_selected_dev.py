#!/usr/bin/env python
"""Evaluate the selected dataset fuser at N6/N12/ALL on DEV once."""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag.composition_training_common import (
    build_fuser, load_frozen_generator, load_frozen_k2_projector, make_fused_tokens,
)
from scripts.packet_xrag.utility_predictor_training_common import load_static_score_cache
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache, sha256_file
from src.packet_xrag.generalization.dataset_evaluation import (
    evaluate_fuser, make_c1_fused, summarize,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("2wiki", "musique", "triviaqa"), required=True)
    parser.add_argument("--root", default="cache/generalization")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv); root = Path(args.root); dataset_root = root / args.dataset
    output = dataset_root / "dev_selected"
    if output.exists(): raise RuntimeError("refusing to overwrite selected DEV suite")
    ledger_path = root / "experiment_ledger.json"; ledger = json.loads(ledger_path.read_text())
    usage = ledger["datasets"][args.dataset]
    if usage["dev_generation"] > 5:
        raise RuntimeError("no DEV generation budget remains for selected suite")
    selection = json.loads((dataset_root / "fuser/run_1_hotpot_init/selection.json").read_text())
    if selection["optimization_failure"]:
        raise RuntimeError("run-1 optimization failure requires the authorized run-2 path")
    if sha256_file(selection["best_checkpoint"]) != selection["best_checkpoint_sha256"]:
        raise RuntimeError("selected dataset fuser hash mismatch")
    static = json.loads((dataset_root / "static/selection.json").read_text())
    cache = ControllerFeatureCache(dataset_root / "features/dev")
    records = [cache[index] for index in range(len(cache))]
    device = torch.device(args.device); torch.cuda.set_device(device)
    scores = load_static_score_cache(cache, static["checkpoint"],
        dataset_root / "static/scores/dev.pt", device)
    rankings = {record["sample_id"]: sorted(range(record["packet_count"]),
        key=lambda index: (-float(scores[record["sample_id"]][index]), index))
        for record in records}
    tokenizer, generator, xrag_id, config = load_frozen_generator(device)
    k2 = load_frozen_k2_projector(config, device); fuser = build_fuser("C1").to(device)
    payload = torch.load(selection["best_checkpoint"], map_location="cpu", weights_only=True)
    fuser.load_state_dict(payload["state_dict"], strict=True); fuser.eval()
    metrics, all_rows = {}, []
    for breadth in (6, 12, None):
        label = "ALL" if breadth is None else str(breadth)
        groups = [rankings[record["sample_id"]][:
            record["packet_count"] if breadth is None else breadth] for record in records]
        rows = evaluate_fuser(f"DATASET_FUSER_{label}", fuser, records, groups,
            make_c1_fused, k2, tokenizer, generator, xrag_id, device, args.batch_size)
        metrics[f"DATASET_FUSER_{label}"] = summarize(rows); all_rows.extend(rows)
        print(json.dumps({f"DATASET_FUSER_{label}": summarize(rows)}), flush=True)
    baseline = json.loads((dataset_root / "dev_baselines/results.json").read_text())["metrics"]
    combined = {**baseline, **metrics}
    report = {"status": "complete", "dataset": args.dataset, "split": "DEV",
        "evaluation_suite_number": usage["dev_generation"] + 1, "metrics": combined,
        "selected_epoch": selection["best_epoch"],
        "checkpoint_sha256": selection["best_checkpoint_sha256"],
        "breadth_degradation_n6_to_n12": metrics["DATASET_FUSER_6"]["short_f1"] -
            metrics["DATASET_FUSER_12"]["short_f1"],
        "benchmark_accessed": False, "final100_accessed": False}
    output.mkdir(parents=True)
    (output / "results.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with (output / "predictions.jsonl").open("w") as stream:
        for row in all_rows: stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    usage["dev_generation"] += 1
    usage["selected_checkpoint_sha256"] = selection["best_checkpoint_sha256"]
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__": main()
