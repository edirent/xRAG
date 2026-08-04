#!/usr/bin/env python
"""Consume one authorized dataset BENCHMARK suite with every frozen baseline."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag.composition_training_common import (
    build_fuser, load_frozen_generator, load_frozen_k2_projector, make_fused_tokens,
)
from scripts.packet_xrag.evaluate_dataset_dev_baselines import (
    HOTPOT_FUSER, HOTPOT_FUSER_SHA256, clipped,
)
from scripts.packet_xrag.utility_predictor_training_common import load_static_score_cache
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache, sha256_file
from src.packet_xrag.generalization.dataset_evaluation import (
    evaluate_fuser, evaluate_independent, evaluate_no_context, evaluate_text,
    make_c1_fused, summarize,
)


def grouped_metrics(rows, key):
    groups = defaultdict(list)
    for row in rows: groups[str(row["metadata"].get(key, "unknown"))].append(row)
    return {name: {"samples": len(values),
                   "short_f1": 100 * mean(row["short_f1"] for row in values),
                   "short_em": 100 * mean(row["short_em"] for row in values),
                   "support_recall": mean(row["support_recall"] for row in values),
                   "full_support": mean(row["full_support"] for row in values)}
            for name, values in groups.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("2wiki", "musique", "triviaqa"), required=True)
    parser.add_argument("--root", default="cache/generalization")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv); root = Path(args.root); dataset_root = root / args.dataset
    output = dataset_root / "benchmark"
    if output.exists(): raise RuntimeError("refusing to overwrite BENCHMARK suite")
    ledger_path = root / "experiment_ledger.json"; ledger = json.loads(ledger_path.read_text())
    usage = ledger["datasets"][args.dataset]
    shadow = json.loads((dataset_root / "shadow/results.json").read_text())
    if not shadow["gate"]["passed"]: raise RuntimeError("SHADOW did not authorize BENCHMARK")
    if usage["benchmark"] != 0 or not usage["shadow_gate_passed"]:
        raise RuntimeError("BENCHMARK single-run lock is not pristine/authorized")
    # Consume before the first generation. A mid-suite failure is a mandatory stop,
    # never permission to generate this benchmark a second time.
    usage["benchmark"] = 1; usage["status"] = "benchmark_in_progress"
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    static = json.loads((dataset_root / "static/selection.json").read_text())
    selection = json.loads((dataset_root / "fuser/run_1_hotpot_init/selection.json").read_text())
    if sha256_file(selection["best_checkpoint"]) != selection["best_checkpoint_sha256"]:
        raise RuntimeError("frozen dataset fuser hash mismatch")
    if sha256_file(HOTPOT_FUSER) != HOTPOT_FUSER_SHA256:
        raise RuntimeError("frozen Hotpot fuser hash mismatch")
    cache = ControllerFeatureCache(dataset_root / "features/benchmark")
    records = [cache[index] for index in range(len(cache))]
    if args.dataset == "2wiki":
        for record in records:
            support_documents = {record["packets"][index]["doc_id"]
                                 for index in record["gold_packet_ids"]}
            record["metadata"]["support_scope"] = (
                "cross_document" if len(support_documents) > 1 else "same_document")
    device = torch.device(args.device); torch.cuda.set_device(device)
    scores = load_static_score_cache(cache, static["checkpoint"],
        dataset_root / "static/scores/benchmark.pt", device)
    static_rank = {record["sample_id"]: sorted(range(record["packet_count"]),
        key=lambda index: (-float(scores[record["sample_id"]][index]), index))
        for record in records}
    topk = {record["sample_id"]: record["topk_ranking"] for record in records}
    tokenizer, generator, xrag_id, config = load_frozen_generator(device)
    k2 = load_frozen_k2_projector(config, device)
    hotpot = build_fuser("C1").to(device)
    hotpot.load_state_dict(torch.load(HOTPOT_FUSER, map_location="cpu",
                                      weights_only=True)["state_dict"], strict=True)
    hotpot.eval(); fuser = build_fuser("C1").to(device)
    fuser.load_state_dict(torch.load(selection["best_checkpoint"], map_location="cpu",
                                     weights_only=True)["state_dict"], strict=True)
    fuser.eval(); rows_by_config = {}
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
        groups = [clipped(static_rank[record["sample_id"]], record, breadth) for record in records]
        add(label, evaluate_independent(label, records, groups, k2, tokenizer, generator,
                                        xrag_id, device, args.batch_size))
    groups6 = [clipped(static_rank[record["sample_id"]], record, 6) for record in records]
    add("HOTPOT_ZERO_SHOT_FUSER_6", evaluate_fuser("HOTPOT_ZERO_SHOT_FUSER_6", hotpot,
        records, groups6, make_c1_fused, k2, tokenizer, generator, xrag_id, device,
        args.batch_size))
    for breadth in (6, 12, None):
        label = "ALL" if breadth is None else str(breadth)
        groups = [clipped(static_rank[record["sample_id"]], record, breadth) for record in records]
        add(f"DATASET_FUSER_{label}", evaluate_fuser(f"DATASET_FUSER_{label}", fuser,
            records, groups, make_c1_fused, k2, tokenizer, generator, xrag_id, device,
            args.batch_size))
    metrics = {name: summarize(rows) for name, rows in rows_by_config.items()}
    specialized = {}
    if args.dataset == "musique":
        specialized["by_hop_count"] = {name: grouped_metrics(rows, "hop_count")
                                        for name, rows in rows_by_config.items()}
    elif args.dataset == "2wiki":
        specialized["by_question_type"] = {name: grouped_metrics(rows, "type")
                                            for name, rows in rows_by_config.items()}
        specialized["by_support_scope"] = {name: grouped_metrics(rows, "support_scope")
                                            for name, rows in rows_by_config.items()}
    else:
        ranks, mentions = [], []
        for record in records:
            gold = set(record["gold_packet_ids"]); ranking = static_rank[record["sample_id"]]
            ranks.append(next((index + 1 for index, packet in enumerate(ranking)
                               if packet in gold), None))
            mentions.append(len(gold))
        found = [value for value in ranks if value is not None]
        specialized["answer_packet_diagnostics"] = {
            "samples": len(records), "answer_containing_packet_recall": len(found) / len(records),
            "mean_first_answer_packet_rank": mean(found) if found else None,
            "mean_answer_mentions": mean(mentions),
            "mean_duplicate_answer_mentions": mean(max(0, value - 1) for value in mentions),
            "conflicting_entity_mentions": "not annotated by TriviaQA; not inferred from gold"}
    report = {"status": "complete", "dataset": args.dataset, "split": "BENCHMARK",
        "metrics": metrics, "specialized_metrics": specialized,
        "dataset_fuser_checkpoint_sha256": selection["best_checkpoint_sha256"],
        "dataset_static_checkpoint_sha256": static["checkpoint_sha256"],
        "hotpot_zero_shot_checkpoint_sha256": HOTPOT_FUSER_SHA256,
        "all_max_packets": 48, "benchmark_suite_runs": 1,
        "final100_accessed": False}
    output.mkdir(parents=True)
    (output / "results.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with (output / "predictions.jsonl").open("w") as stream:
        for name in rows_by_config:
            for row in rows_by_config[name]: stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    usage["status"] = "benchmark_complete"
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__": main()
