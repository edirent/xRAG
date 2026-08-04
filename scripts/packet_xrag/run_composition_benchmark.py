#!/usr/bin/env python
"""Run the sole frozen benchmark-500 composition evaluation after Shadow passes."""

import argparse
import json
import random
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag.composition_training_common import (
    build_fuser, load_frozen_generator, load_frozen_k2_projector,
)
from scripts.packet_xrag.evaluate_full_composition_dev import (
    evaluate_fuser, evaluate_independent, sha256, summarize,
)
from scripts.packet_xrag.utility_predictor_training_common import load_static_score_cache
from src.packet_xrag.composition.protocol import EXPECTED_BENCHMARK_HASH, assert_evaluation_lock
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache


SEED = 20260804


def benchmark_stresses(records, rankings):
    output = {name: [] for name in ("REVERSE", "RANDOM", "GOLD_DUPLICATE_X2",
                                     "NONGOLD_DUPLICATE_X2", "DISTRACTOR_X4")}
    for record in records:
        selected = list(rankings[record["sample_id"]][:6]); gold = set(record["gold_packet_ids"])
        output["REVERSE"].append(list(reversed(selected)))
        permuted = list(selected); random.Random(f"{SEED}:{record['sample_id']}:benchmark").shuffle(permuted)
        output["RANDOM"].append(permuted)
        gold_target = next((index for index in selected if index in gold), record["gold_packet_ids"][0])
        output["GOLD_DUPLICATE_X2"].append(selected + [gold_target, gold_target])
        nongold_target = next((index for index in rankings[record["sample_id"]] if index not in gold),
                              selected[-1])
        output["NONGOLD_DUPLICATE_X2"].append(selected + [nongold_target, nongold_target])
        distractors = [index for index in rankings[record["sample_id"]]
                       if index not in gold and index not in selected]
        if not distractors:
            distractors = [nongold_target]
        output["DISTRACTOR_X4"].append(selected + [distractors[index % len(distractors)]
                                                    for index in range(4)])
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="cache/composition")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv); root = Path(args.root); output_dir = root / "benchmark"
    if output_dir.exists():
        raise RuntimeError("refusing to overwrite formal composition benchmark")
    shadow = json.loads((root / "shadow/results.json").read_text())
    if shadow["status"] != "PASS":
        raise RuntimeError("benchmark remains sealed because Shadow did not pass")
    candidate = json.loads((root / "frozen_candidate/candidate_config.json").read_text())
    if candidate != shadow["candidate_config"]:
        raise RuntimeError("frozen candidate changed after Shadow")
    if sha256(candidate["checkpoint"]) != candidate["checkpoint_sha256"]:
        raise RuntimeError("frozen candidate checkpoint hash mismatch")
    lock_path = root / "benchmark_lock.json"
    lock = assert_evaluation_lock(lock_path, "BENCHMARK_500", 1)
    # Consume the sole run before the benchmark feature cache is opened.
    lock["runs"] += 1; lock["candidate_checkpoint_sha256"] = candidate["checkpoint_sha256"]
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    cache = ControllerFeatureCache("cache/controller/features/benchmark_features")
    if len(cache) != 500 or cache.manifest["effective_split_hash"] != EXPECTED_BENCHMARK_HASH:
        raise RuntimeError("formal benchmark split mismatch")
    records = [cache[index] for index in range(len(cache))]
    device = torch.device(args.device); torch.cuda.set_device(device)
    scores = load_static_score_cache(cache, "cache/controller/static/best_short_f1/scorer.pt",
        "cache/controller/utility_predictor/features/benchmark_static_scores.pt", device)
    rankings = {record["sample_id"]: sorted(range(record["packet_count"]),
        key=lambda index: (-float(scores[record["sample_id"]][index]), index)) for record in records}
    tokenizer, generator, xrag_id, model_config = load_frozen_generator(device)
    k2 = load_frozen_k2_projector(model_config, device)
    payload = torch.load(candidate["checkpoint"], map_location="cpu", weights_only=True)
    fuser = build_fuser("C1").to(device); fuser.load_state_dict(payload["state_dict"], strict=True)
    fuser.eval(); all_rows = []; rows_by_name = {}

    def add(name, rows):
        rows_by_name[name] = rows; all_rows.extend(rows)
        print(json.dumps({name: summarize(rows)}), flush=True)

    topk3 = [record["topk_ranking"][:3] for record in records]
    static2 = [rankings[record["sample_id"]][:2] for record in records]
    static6 = [rankings[record["sample_id"]][:6] for record in records]
    static12 = [rankings[record["sample_id"]][:12] for record in records]
    all_groups = [rankings[record["sample_id"]] for record in records]
    oracle = [list(record["gold_packet_ids"]) for record in records]
    for name, groups in (("TOPK_3", topk3), ("STATIC_2", static2),
                         ("INDEPENDENT_N6", static6), ("XRAG_ORACLE", oracle),
                         ("ALL", all_groups), ("INDEPENDENT_N12", static12)):
        add(name, evaluate_independent(name, records, groups, k2, tokenizer, generator,
                                       xrag_id, device, args.batch_size))
    for name, groups in (("FUSER_N2", static2), ("FUSER_N6", static6),
                         ("FUSER_N12", static12), ("FUSER_ALL", all_groups)):
        add(name, evaluate_fuser(name, fuser, records, groups, k2, tokenizer, generator,
                                 xrag_id, device, args.batch_size))
    stresses = benchmark_stresses(records, rankings)
    for name, groups in stresses.items():
        add(f"INDEPENDENT_{name}", evaluate_independent(f"INDEPENDENT_{name}", records,
            groups, k2, tokenizer, generator, xrag_id, device, args.batch_size))
        add(f"FUSER_{name}", evaluate_fuser(f"FUSER_{name}", fuser, records, groups, k2,
            tokenizer, generator, xrag_id, device, args.batch_size))
    metrics = {name: summarize(rows) for name, rows in rows_by_name.items()}
    result = {"status": "COMPLETE_FROZEN_NO_SELECTION", "split": "BENCHMARK_500",
              "sample_count": len(records), "benchmark_hash": cache.manifest["effective_split_hash"],
              "candidate_config": candidate, "metrics": metrics,
              "checkpoint_hashes": json.loads((root / "frozen_candidate/checkpoint_hashes.json").read_text()),
              "benchmark_runs": lock["runs"], "thresholds_changed": False,
              "benchmark_used_for_tuning": False, "final_100_accessed": False}
    output_dir.mkdir(parents=True)
    (output_dir / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (output_dir / "predictions.jsonl").open("w") as stream:
        for row in all_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    ledger_path = root / "experiment_ledger.json"; ledger = json.loads(ledger_path.read_text())
    ledger["usage"]["benchmark_evaluations"] += 1
    ledger["stage8_benchmark"] = {"status": "COMPLETE_FROZEN_NO_SELECTION",
                                  "checkpoint_hash": candidate["checkpoint_sha256"]}
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
