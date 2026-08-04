#!/usr/bin/env python
"""Reproduce TOPK-3 and STATIC-2 on SEARCH_DEV only."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag.run_static_scorer_benchmark import initialize_generator
from scripts.packet_xrag.utility_predictor_training_common import load_static_score_cache
from src.packet_xrag.controller.autonomous_search import (
    EXPECTED_INTERNAL_DEV_HASH, SubsetFeatureCache, assert_only_search_dev,
    load_search_split,
)
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache
from src.packet_xrag.controller.utility_evaluation import (
    generate_rollout_answers, summarize_generation,
)


def fixed_policy(selected, scores):
    selected = list(selected)
    return {"selected_packet_ids": selected,
            "actions": [{"step": step, "packet_id": packet_id,
                         "predicted_delta": float(scores[step]),
                         "selected_packet_ids_before": selected[:step]}
                        for step, packet_id in enumerate(selected)],
            "stop": {"reason": "fixed_budget", "stop_utility": 0.0,
                     "highest_remaining_packet_id": None,
                     "highest_remaining_score": None}, "tau": None}


@torch.inference_mode()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="cache/controller/autonomous_search")
    parser.add_argument("--feature-cache", default="cache/controller/features/internal_dev_features")
    parser.add_argument("--static-checkpoint", default="cache/controller/static/best_short_f1/scorer.pt")
    parser.add_argument("--static-score-cache", default="cache/controller/utility_predictor/features/internal_dev_static_scores.pt")
    parser.add_argument("--k2-training-config", default="cache/projector/multi_token_k2/best_short_f1/training_config.json")
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args(argv); root = Path(args.root); output_dir = root / "stage0"
    output = output_dir / "search_dev_baselines.jsonl"
    if output.exists(): raise RuntimeError("refusing to overwrite SEARCH_DEV baseline generation")
    dev_split, shadow_split = load_search_split(root)
    parent = ControllerFeatureCache(args.feature_cache)
    if parent.manifest["effective_split_hash"] != EXPECTED_INTERNAL_DEV_HASH:
        raise RuntimeError("internal-dev feature hash mismatch")
    cache = SubsetFeatureCache(parent, dev_split["ordered_sample_ids"])
    assert_only_search_dev([record["sample_id"] for record in cache.records],
                           dev_split["ordered_sample_ids"], "baseline cache")
    if set(record["sample_id"] for record in cache.records) & set(shadow_split["ordered_sample_ids"]):
        raise RuntimeError("SEARCH_SHADOW entered baseline generation")
    device = torch.device(args.device); torch.cuda.set_device(device)
    static_scores = load_static_score_cache(parent, args.static_checkpoint,
                                             args.static_score_cache, device)
    policies = {"TOPK_3": {}, "STATIC_2": {}}
    for index in range(len(cache)):
        record = cache[index]; sid = record["sample_id"]
        ranking = sorted(range(record["packet_count"]),
                         key=lambda packet_id: (-float(static_scores[sid][packet_id]), packet_id))
        topk = record["topk_ranking"][:3]
        policies["TOPK_3"][sid] = fixed_policy(topk, [record["topk_scores"][i] for i in topk])
        policies["STATIC_2"][sid] = fixed_policy(
            ranking[:2], [static_scores[sid][i] for i in ranking[:2]])
    tokenizer, generator, xrag_id, hashes = initialize_generator(args.k2_training_config, device)
    rows, summary, costs = [], {}, {}
    for configuration in ("TOPK_3", "STATIC_2"):
        torch.cuda.synchronize(device); start = time.time()
        current = generate_rollout_answers(cache, policies[configuration], tokenizer,
                                           generator, xrag_id, device, configuration)
        torch.cuda.synchronize(device); elapsed = time.time() - start
        rows.extend(current); summary[configuration] = summarize_generation(current)
        costs[configuration] = {"wall_seconds": elapsed,
                                "milliseconds_per_sample": 1000 * elapsed / len(cache),
                                "generator_forward_batches": (len(cache) + 15) // 16,
                                "extra_generator_forwards_per_sample": 0}
        print(json.dumps({configuration: summary[configuration], "cost": costs[configuration]},
                         indent=2), flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        for row in rows: stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    payload = {"status": "complete", "split": "SEARCH_DEV", "sample_count": len(cache),
               "search_dev_hash": dev_split["sha256"], "search_shadow_accessed": False,
               "metrics": summary, "cost": costs, "checkpoint_hashes": hashes,
               "final_100_accessed": False, "final_100_runs": 0}
    (output_dir / "search_dev_baselines.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    ledger_path = root / "experiment_ledger.json"
    ledger = json.loads(ledger_path.read_text())
    ledger["usage"]["search_dev_generation"] = 2
    ledger["stage0_baselines"] = payload
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__": main()
