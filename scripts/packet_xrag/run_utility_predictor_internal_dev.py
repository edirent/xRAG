#!/usr/bin/env python
"""Select one checkpoint/tau per utility model using internal-dev generation only."""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag.run_static_scorer_benchmark import initialize_generator
from scripts.packet_xrag.token_resampler_common import sha256_file
from scripts.packet_xrag.utility_predictor_training_common import load_static_score_cache
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache
from src.packet_xrag.controller.utility_checkpoint import load_utility_checkpoint
from src.packet_xrag.controller.utility_evaluation import (
    cached_label_mechanism_audit, choose_rollout_configuration,
    generate_rollout_answers, run_model_rollouts, summarize_generation,
)
from src.packet_xrag.controller.utility_label_dataset import ShardedUtilityLabelDataset
from src.packet_xrag.controller.utility_rollout import THRESHOLD_GRID


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type", choices=("A", "B", "C"), required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--labels-root", default="cache/controller/utility_predictor/labels")
    parser.add_argument("--feature-cache", default="cache/controller/features/internal_dev_features")
    parser.add_argument("--static-checkpoint", default="cache/controller/static/best_short_f1/scorer.pt")
    parser.add_argument("--static-score-cache", default="cache/controller/utility_predictor/features/internal_dev_static_scores.pt")
    parser.add_argument("--k2-training-config", default="cache/projector/multi_token_k2/best_short_f1/training_config.json")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


@torch.inference_mode()
def main(argv=None):
    args = parse_args(argv); model_dir = Path(args.model_dir)
    selection_path = model_dir / "frozen_selection.json"
    if selection_path.exists():
        raise RuntimeError(f"refusing to overwrite frozen internal-dev selection: {selection_path}")
    device = torch.device(args.device); torch.cuda.set_device(device)
    cache = ControllerFeatureCache(args.feature_cache)
    if len(cache) != 500:
        raise RuntimeError("internal-dev selection requires exactly 500 samples")
    labels = ShardedUtilityLabelDataset(args.labels_root, "internal_dev")
    scores = load_static_score_cache(
        cache, args.static_checkpoint, args.static_score_cache, device
    )
    target_stats = json.loads((Path(args.labels_root).parent / "utility_target_stats.json").read_text())
    clip = target_stats["utility_clip_value"]
    candidates = json.loads((model_dir / "candidate_epochs.json").read_text())["candidate_epochs"]
    if len(candidates) != 2:
        raise RuntimeError("internal-dev rollout requires exactly two candidate epochs")
    tokenizer, generator, xrag_id, _ = initialize_generator(args.k2_training_config, device)
    grid, all_rows = [], []
    for epoch in candidates:
        checkpoint = model_dir / f"epoch_{epoch}" / "model.pt"
        model = load_utility_checkpoint(args.model_type, checkpoint, device)
        for tau in THRESHOLD_GRID:
            policies = run_model_rollouts(model, cache, scores, clip, tau, device)
            configuration = f"MODEL_{args.model_type}_E{epoch}_TAU{tau:.2f}"
            rows = generate_rollout_answers(
                cache, policies, tokenizer, generator, xrag_id, device, configuration
            )
            metrics = summarize_generation(rows)
            mechanism = cached_label_mechanism_audit(policies, labels)
            record = {"model_type": args.model_type, "epoch": epoch, "tau": tau,
                      "checkpoint": str(checkpoint.resolve()),
                      "checkpoint_sha256": sha256_file(checkpoint),
                      "configuration": configuration, "metrics": metrics,
                      "utility_mechanism": mechanism}
            grid.append(record); all_rows.extend(rows)
            print(json.dumps(record, indent=2), flush=True)
        del model
    selected = choose_rollout_configuration(grid)
    frozen = {**selected, "selection_rule": "max Short F1; within <0.25 fewer packets; larger tau; earlier epoch",
              "candidate_epochs": candidates, "threshold_grid": list(THRESHOLD_GRID),
              "selection_split": "controller_internal_dev_500",
              "selection_split_hash": cache.manifest["effective_split_hash"],
              "benchmark_used": False, "final_100_accessed": False, "final_100_runs": 0}
    selection_path.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    (model_dir / "internal_dev_grid.json").write_text(
        json.dumps({"grid": grid}, indent=2, sort_keys=True) + "\n"
    )
    with (model_dir / "internal_dev_grid_predictions.jsonl").open("w") as stream:
        for row in all_rows: stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"frozen_selection": frozen}, indent=2), flush=True)


if __name__ == "__main__": main()
