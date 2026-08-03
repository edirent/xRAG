#!/usr/bin/env python
"""Train locked Model B: static utility plus selected-set state shift."""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag.utility_predictor_training_common import (
    best_prediction_checkpoint, load_training_inputs, write_training_config,
)
from src.packet_xrag.controller.static_utility_predictor import StaticUtilityPredictor
from src.packet_xrag.controller.state_shift_utility_predictor import StateShiftUtilityPredictor
from src.packet_xrag.controller.utility_training import SEED, train_utility_model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-root", default="cache/controller/utility_predictor/labels")
    parser.add_argument("--train-cache", default="cache/controller/features/train_features")
    parser.add_argument("--dev-cache", default="cache/controller/features/internal_dev_features")
    parser.add_argument("--static-checkpoint", default="cache/controller/static/best_short_f1/scorer.pt")
    parser.add_argument("--score-root", default="cache/controller/utility_predictor/features")
    parser.add_argument("--model-a-dir", default="cache/controller/utility_predictor/model_a")
    parser.add_argument("--output-dir", default="cache/controller/utility_predictor/model_b")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv); device = torch.device(args.device); torch.cuda.set_device(device)
    torch.manual_seed(SEED)
    inputs = load_training_inputs(args.labels_root, args.train_cache, args.dev_cache,
                                  args.static_checkpoint, args.score_root, device)
    train_labels, dev_labels, train_features, dev_features, stats = inputs
    source, source_epoch = best_prediction_checkpoint(args.model_a_dir)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    base = StaticUtilityPredictor(); base.load_state_dict(
        torch.load(source, map_location="cpu", weights_only=True), strict=True
    )
    model = StateShiftUtilityPredictor(base)
    selection = train_utility_model(
        model, train_labels, dev_labels, train_features, dev_features,
        stats["utility_clip_value"], device, args.output_dir, epochs=6,
        learning_rate=1e-4,
    )
    config = {"model": "B_STATE_SHIFT", "epochs": 6, "effective_batch_size": 256,
              "optimizer": "AdamW", "learning_rate": 1e-4, "weight_decay": .01,
              "warmup_ratio": .05, "gradient_clipping": 1.0, "dtype": "BF16 autocast",
              "seed": SEED, "utility_clip_value": stats["utility_clip_value"],
              "source_model_a_checkpoint": str(source.resolve()), "source_model_a_epoch": source_epoch,
              "state_shift_final_layer_initialization": "zeros",
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "candidate_epochs": selection["candidate_epochs"],
              "benchmark_used": False, "final_100_accessed": False, "final_100_runs": 0}
    write_training_config(args.output_dir, config); print(json.dumps(config, indent=2), flush=True)


if __name__ == "__main__": main()
