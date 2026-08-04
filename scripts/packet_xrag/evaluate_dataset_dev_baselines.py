#!/usr/bin/env python
"""Run the single frozen DEV baseline/zero-shot generation suite for a dataset."""

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
    evaluate_fuser, evaluate_independent, evaluate_no_context, evaluate_text,
    make_c1_fused, summarize,
)


HOTPOT_FUSER = Path("cache/composition/full/C1_O1/epoch_6.pt")
HOTPOT_FUSER_SHA256 = "8f0f1161defb506b48dfac4249e2a395cb49cabad4ed75a3adc045ac02a9e6e3"


def clipped(ranking, record, count=None):
    count = record["packet_count"] if count is None else min(count, record["packet_count"])
    return list(ranking[:count])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("2wiki", "musique", "triviaqa"), required=True)
    parser.add_argument("--root", default="cache/generalization")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv); root = Path(args.root); dataset_root = root / args.dataset
    output = dataset_root / "dev_baselines"
    if output.exists(): raise RuntimeError("refusing to overwrite DEV baseline suite")
    ledger_path = root / "experiment_ledger.json"; ledger = json.loads(ledger_path.read_text())
    usage = ledger["datasets"][args.dataset]
    recovery = dataset_root / "dev_baselines_failure_1.json"
    if usage["dev_generation"] not in (0, 1):
        raise RuntimeError("DEV baseline suite budget/order mismatch")
    if usage["dev_generation"] == 1 and not recovery.exists():
        raise RuntimeError("DEV recovery requires a recorded incomplete first suite")
    suite_number = usage["dev_generation"] + 1
    if sha256_file(HOTPOT_FUSER) != HOTPOT_FUSER_SHA256:
        raise RuntimeError("frozen Hotpot fuser hash mismatch")
    static_selection = json.loads((dataset_root / "static/selection.json").read_text())
    if sha256_file(static_selection["checkpoint"]) != static_selection["checkpoint_sha256"]:
        raise RuntimeError("dataset STATIC hash mismatch")
    cache = ControllerFeatureCache(dataset_root / "features/dev")
    records = [cache[index] for index in range(len(cache))]
    device = torch.device(args.device); torch.cuda.set_device(device)
    static_scores = load_static_score_cache(cache, static_selection["checkpoint"],
        dataset_root / "static/scores/dev.pt", device)
    static = {record["sample_id"]: sorted(range(record["packet_count"]),
        key=lambda index: (-float(static_scores[record["sample_id"]][index]), index))
        for record in records}
    topk = {record["sample_id"]: record["topk_ranking"] for record in records}
    tokenizer, generator, xrag_id, config = load_frozen_generator(device)
    k2 = load_frozen_k2_projector(config, device)
    fuser = build_fuser("C1").to(device)
    payload = torch.load(HOTPOT_FUSER, map_location="cpu", weights_only=True)
    fuser.load_state_dict(payload["state_dict"], strict=True); fuser.eval()
    rows_by_config = {}
    def add(name, rows):
        rows_by_config[name] = rows
        print(json.dumps({name: summarize(rows)}), flush=True)
    add("NO_CONTEXT", evaluate_no_context(records, tokenizer, generator, device,
                                           args.batch_size))
    for breadth in (2, 6, None):
        label = "ALL" if breadth is None else str(breadth)
        groups = [clipped(topk[record["sample_id"]], record, breadth) for record in records]
        add(f"TEXT_TOP{label}", evaluate_text(f"TEXT_TOP{label}", records, groups,
            tokenizer, generator, device, max(1, args.batch_size // 2)))
    for breadth in (2, 3, 6, 12, None):
        label = "ALL" if breadth is None else str(breadth)
        groups = [clipped(topk[record["sample_id"]], record, breadth) for record in records]
        add(f"TOPK_{label}", evaluate_independent(f"TOPK_{label}", records, groups, k2,
            tokenizer, generator, xrag_id, device, args.batch_size))
    for breadth, label in ((2, "STATIC_2"), (6, "INDEPENDENT_STATIC_6"),
                           (12, "INDEPENDENT_STATIC_12"),
                           (None, "INDEPENDENT_STATIC_ALL")):
        groups = [clipped(static[record["sample_id"]], record, breadth) for record in records]
        add(label, evaluate_independent(label, records, groups, k2, tokenizer, generator,
                                        xrag_id, device, args.batch_size))
    groups = [clipped(static[record["sample_id"]], record, 6) for record in records]
    add("HOTPOT_ZERO_SHOT_FUSER_6", evaluate_fuser("HOTPOT_ZERO_SHOT_FUSER_6", fuser,
        records, groups, make_c1_fused, k2, tokenizer, generator, xrag_id, device,
        args.batch_size))
    metrics = {name: summarize(rows) for name, rows in rows_by_config.items()}
    report = {"status": "complete", "dataset": args.dataset, "split": "DEV",
        "evaluation_suite_number": suite_number, "metrics": metrics,
        "hotpot_fuser_checkpoint_sha256": HOTPOT_FUSER_SHA256,
        "dataset_static_checkpoint_sha256": static_selection["checkpoint_sha256"],
        "all_max_packets": 48, "benchmark_accessed": False, "final100_accessed": False}
    output.mkdir(parents=True)
    (output / "results.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with (output / "predictions.jsonl").open("w") as stream:
        for name in rows_by_config:
            for row in rows_by_config[name]: stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    usage["dev_generation"] = suite_number
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__": main()
