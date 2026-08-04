#!/usr/bin/env python
"""Run the fixed K4 evaluation for Hotpot or the passed MuSiQue transfer task."""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag.composition_training_common import build_fuser, load_frozen_generator
from scripts.packet_xrag.utility_predictor_training_common import load_static_score_cache
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache, sha256_file
from src.packet_xrag.generalization.dataset_evaluation import (
    evaluate_fuser, evaluate_independent, summarize,
)
from src.packet_xrag.generalization.second_setting_adapter import (
    K4_SHA256, load_frozen_k4_projector, make_k4_fused,
)


def heldout_data(task, root):
    if task == "hotpot":
        cache = ControllerFeatureCache("cache/controller/features/train_features")
        by_id = {record["sample_id"]: index for index, record in enumerate(cache.records)}
        ids = json.loads(Path("cache/composition/splits/composition_shadow_ids.json").read_text())[
            "ordered_sample_ids"]
        records = [cache[by_id[sample_id]] for sample_id in ids]
        checkpoint = "cache/controller/static/best_short_f1/scorer.pt"
        score_path = "cache/controller/utility_predictor/features/train_static_scores.pt"
        split = "COMPOSITION_SHADOW"
    else:
        dataset_root = root / "musique"
        cache = ControllerFeatureCache(dataset_root / "features/dev")
        records = [cache[index] for index in range(len(cache))]
        static = json.loads((dataset_root / "static/selection.json").read_text())
        checkpoint = static["checkpoint"]
        score_path = dataset_root / "static/scores/dev.pt"
        split = "DEV_FINAL_BUDGET_SLOT_6"
    return cache, records, checkpoint, score_path, split


def finalize_gate(second, ledger, ledger_path):
    reports = {task: json.loads((second / task / "evaluation/results.json").read_text())
               for task in ("hotpot", "musique")}
    task_gates = {}
    for task, report in reports.items():
        metrics = report["metrics"]
        sparse = metrics["K4_STATIC_2"]["short_f1"]
        independent = metrics["K4_INDEPENDENT_6"]["short_f1"]
        fused = metrics["K4_FUSER_6"]["short_f1"]
        task_gates[task] = {
            "gain_over_independent_n6": fused - independent,
            "delta_vs_sparse2": fused - sparse,
            "strong": fused - independent >= 4 and fused >= sparse - 1,
            "weak": fused - independent >= 2,
            "n12_stable": metrics["K4_FUSER_12"]["short_f1"] >= fused - 2,
            "all_stable": metrics["K4_FUSER_ALL"]["short_f1"] >= fused - 2,
        }
    strong_count = sum(value["strong"] for value in task_gates.values())
    both_weak = all(value["weak"] for value in task_gates.values())
    breadth_stable = all(value["n12_stable"] and value["all_stable"]
                         for value in task_gates.values())
    passed = strong_count >= 1 and both_weak and breadth_stable
    result = {"status": "PASS" if passed else "FAIL_REPRESENTATION_SPECIFIC_LIMITATION",
        "setting": "K4", "tasks": task_gates, "gate_passed": passed,
        "rule": "one strong, other at least +2 over independent; N12 and ALL within 2 F1 of N6",
        "full_training_runs": ledger["usage"]["second_setting_full_runs"],
        "k4_checkpoint_sha256": K4_SHA256, "output_M": 4,
        "benchmark_accessed": False, "final100_accessed": False}
    (second / "final_results.json").write_text(json.dumps(result, indent=2,
                                                            sort_keys=True) + "\n")
    ledger["second_setting"]["status"] = result["status"]
    ledger["second_setting"]["gate_passed"] = passed
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("hotpot", "musique"), required=True)
    parser.add_argument("--root", default="cache/generalization")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    root = Path(args.root); second = root / "second_setting"
    output = second / args.task / "evaluation"
    if output.exists():
        raise RuntimeError("refusing to overwrite K4 evaluation")
    selection = json.loads((second / args.task / "training/selection.json").read_text())
    if sha256_file(selection["checkpoint"]) != selection["checkpoint_sha256"]:
        raise RuntimeError("frozen K4 fuser hash mismatch")
    ledger_path = root / "experiment_ledger.json"
    ledger = json.loads(ledger_path.read_text())
    if args.task == "musique":
        usage = ledger["datasets"]["musique"]
        if usage["dev_generation"] != 5:
            raise RuntimeError("MuSiQue's single remaining DEV generation slot is not pristine")
        usage["dev_generation"] = 6
    ledger.setdefault("second_setting", {}).setdefault(args.task, {})[
        "evaluation_status"] = "in_progress"
    output.mkdir(parents=True)
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    cache, records, static_checkpoint, score_path, split = heldout_data(args.task, root)
    device = torch.device(args.device); torch.cuda.set_device(device)
    scores = load_static_score_cache(cache, static_checkpoint, score_path, device)
    rankings = {record["sample_id"]: sorted(range(record["packet_count"]),
        key=lambda index: (-float(scores[record["sample_id"]][index]), index))
        for record in records}
    tokenizer, generator, xrag_id, config = load_frozen_generator(device)
    k4 = load_frozen_k4_projector(config, device)
    fuser = build_fuser("C1").to(device)
    payload = torch.load(selection["checkpoint"], map_location="cpu", weights_only=True)
    fuser.load_state_dict(payload["state_dict"], strict=True); fuser.eval()
    rows_by_config = {}

    def add(name, rows):
        rows_by_config[name] = rows
        print(json.dumps({name: summarize(rows)}), flush=True)

    for breadth, name in ((2, "K4_STATIC_2"), (6, "K4_INDEPENDENT_6"),
                          (12, "K4_INDEPENDENT_12"), (None, "K4_INDEPENDENT_ALL")):
        groups = [rankings[record["sample_id"]][:
            record["packet_count"] if breadth is None else breadth] for record in records]
        add(name, evaluate_independent(name, records, groups, k4, tokenizer, generator,
            xrag_id, device, args.batch_size, tokens_per_packet=4))
    for breadth, name in ((6, "K4_FUSER_6"), (12, "K4_FUSER_12"),
                          (None, "K4_FUSER_ALL")):
        groups = [rankings[record["sample_id"]][:
            record["packet_count"] if breadth is None else breadth] for record in records]
        add(name, evaluate_fuser(name, fuser, records, groups, make_k4_fused, k4,
            tokenizer, generator, xrag_id, device, args.batch_size, tokens_per_packet=4))
    report = {"status": "complete", "setting": "K4", "task": args.task,
        "split": split, "samples": len(records),
        "metrics": {name: summarize(rows) for name, rows in rows_by_config.items()},
        "fuser_checkpoint_sha256": selection["checkpoint_sha256"],
        "k4_checkpoint_sha256": K4_SHA256, "output_M": 4,
        "evaluation_suite_runs": 1, "benchmark_accessed": False,
        "final100_accessed": False}
    (output / "results.json").write_text(json.dumps(report, indent=2,
                                                      sort_keys=True) + "\n")
    with (output / "predictions.jsonl").open("w") as stream:
        for rows in rows_by_config.values():
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    ledger["second_setting"][args.task]["evaluation_status"] = "complete"
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    final = None
    if all((second / task / "evaluation/results.json").exists()
           for task in ("hotpot", "musique")):
        final = finalize_gate(second, ledger, ledger_path)
    print(json.dumps({"evaluation": report, "final_gate": final}, indent=2), flush=True)


if __name__ == "__main__":
    main()
