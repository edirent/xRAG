#!/usr/bin/env python
"""Evaluate a frozen set-conditioned controller at fixed packet budgets 1..6."""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag.run_static_scorer_benchmark import (
    evaluate_generator, initialize_generator, summarize_rows,
)
from scripts.packet_xrag.token_resampler_common import EXPECTED_SPLIT_HASH
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache
from src.packet_xrag.controller.sequential_controller import (
    SequentialPacketController, greedy_rollout,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="cache/controller/sequential_fixed_k/best_short_f1/controller.pt")
    parser.add_argument("--training-config", default="cache/controller/sequential_fixed_k/best_short_f1/training_config.json")
    parser.add_argument("--feature-cache", default="cache/controller/features/benchmark_features")
    parser.add_argument("--k2-training-config", default="cache/projector/multi_token_k2/best_short_f1/training_config.json")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", default="cache/controller/sequential_fixed_k/benchmark_predictions.jsonl")
    parser.add_argument("--metrics-output", default="cache/controller/sequential_fixed_k/benchmark_metrics.json")
    parser.add_argument("--allow-existing-output", action="store_true")
    return parser.parse_args(argv)


def load_controller(path, device):
    model = SequentialPacketController().to(device)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    return model


@torch.inference_mode()
def rollout_cache(cache, controller, device):
    return {cache.records[index]["sample_id"]:
            greedy_rollout(controller, cache[index], 6, device)
            for index in range(len(cache))}


@torch.inference_mode()
def main(argv=None):
    args = parse_args(argv)
    if Path(args.output).exists() and not args.allow_existing_output:
        raise RuntimeError("refusing to overwrite formal sequential benchmark output")
    device = torch.device(args.device); torch.cuda.set_device(device)
    cache = ControllerFeatureCache(args.feature_cache)
    if "benchmark_features" in args.feature_cache and cache.manifest["effective_split_hash"] != EXPECTED_SPLIT_HASH:
        raise RuntimeError("benchmark split hash mismatch")
    cache.queries = cache.queries.to(device); cache.packets = cache.packets.to(device)
    controller = load_controller(args.checkpoint, device)
    rankings = rollout_cache(cache, controller, device)
    tokenizer, generator, xrag_id, hashes = initialize_generator(args.k2_training_config, device)
    rows = evaluate_generator(cache, rankings, tokenizer, generator, xrag_id, device, args.output)
    config = json.loads(Path(args.training_config).read_text())
    payload = {
        "split_hash": cache.manifest["effective_split_hash"], "samples": len(cache),
        "metrics": {name.replace("STATIC", "SEQ"): value
                    for name, value in summarize_rows(rows).items()},
        "selected_epoch": config["selected_epoch"],
        "selected_budget": config["selected_budget"], "checkpoint_hashes": hashes,
        "benchmark_runs": 1, "final_100_accessed": False, "final_100_runs": 0,
    }
    # evaluate_generator uses generic STATIC names; normalize the formal file in place.
    for row in rows: row["configuration"] = row["configuration"].replace("STATIC", "SEQ")
    with Path(args.output).open("w") as stream:
        for row in rows: stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    Path(args.metrics_output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__": main()
