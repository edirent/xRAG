#!/usr/bin/env python
"""Consume a dataset's one SHADOW suite and apply its frozen transfer gate."""

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
    evaluate_fuser, evaluate_independent, make_c1_fused, summarize,
)
from src.packet_xrag.generalization.dataset_gates import shadow_gate
from src.packet_xrag.generalization.protocol import load_checkpoint_state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("2wiki", "musique", "triviaqa"), required=True)
    parser.add_argument("--root", default="cache/generalization")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv); root = Path(args.root); dataset_root = root / args.dataset
    output = dataset_root / "shadow"
    if output.exists(): raise RuntimeError("refusing to overwrite SHADOW suite")
    ledger_path = root / "experiment_ledger.json"; ledger = json.loads(ledger_path.read_text())
    usage = ledger["datasets"][args.dataset]
    if usage["shadow"] != 0 or not (dataset_root / "dev_selected/results.json").exists():
        raise RuntimeError("SHADOW lock/order is not pristine")
    selection = json.loads((dataset_root / "fuser/run_1_hotpot_init/selection.json").read_text())
    if sha256_file(selection["best_checkpoint"]) != selection["best_checkpoint_sha256"]:
        raise RuntimeError("frozen fuser hash mismatch")
    static = json.loads((dataset_root / "static/selection.json").read_text())
    cache = ControllerFeatureCache(dataset_root / "features/shadow")
    records = [cache[index] for index in range(len(cache))]
    device = torch.device(args.device); torch.cuda.set_device(device)
    scores = load_static_score_cache(cache, static["checkpoint"],
        dataset_root / "static/scores/shadow.pt", device)
    rankings = {record["sample_id"]: sorted(range(record["packet_count"]),
        key=lambda index: (-float(scores[record["sample_id"]][index]), index))
        for record in records}
    tokenizer, generator, xrag_id, config = load_frozen_generator(device)
    k2 = load_frozen_k2_projector(config, device); fuser = build_fuser("C1").to(device)
    payload = load_checkpoint_state(selection["best_checkpoint"],
                                    selection["best_checkpoint_sha256"])
    fuser.load_state_dict(payload["state_dict"], strict=True); fuser.eval()
    metrics, all_rows = {}, []
    for breadth, label in ((2, "STATIC_2"), (6, "INDEPENDENT_STATIC_6"),
                           (12, "INDEPENDENT_STATIC_12")):
        groups = [rankings[record["sample_id"]][:breadth] for record in records]
        rows = evaluate_independent(label, records, groups, k2, tokenizer, generator,
                                    xrag_id, device, args.batch_size)
        metrics[label] = summarize(rows); all_rows.extend(rows)
        print(json.dumps({label: metrics[label]}), flush=True)
    for breadth in (6, 12):
        label = f"DATASET_FUSER_{breadth}"
        groups = [rankings[record["sample_id"]][:breadth] for record in records]
        rows = evaluate_fuser(label, fuser, records, groups, make_c1_fused, k2,
            tokenizer, generator, xrag_id, device, args.batch_size)
        metrics[label] = summarize(rows); all_rows.extend(rows)
        print(json.dumps({label: metrics[label]}), flush=True)
    gate = shadow_gate(metrics)
    report = {"status": "complete", "dataset": args.dataset, "split": "SHADOW",
        "metrics": metrics, "gate": gate,
        "checkpoint_sha256": selection["best_checkpoint_sha256"],
        "benchmark_authorized": gate["passed"], "benchmark_accessed": False,
        "final100_accessed": False}
    output.mkdir(parents=True)
    (output / "results.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with (output / "predictions.jsonl").open("w") as stream:
        for row in all_rows: stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    usage["shadow"] = 1; usage["shadow_gate_passed"] = gate["passed"]
    usage["status"] = "benchmark_authorized" if gate["passed"] else "negative_transfer"
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__": main()
