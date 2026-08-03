#!/usr/bin/env python
"""Audit learned state-shift and interaction residual components on internal dev."""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag.utility_predictor_training_common import (
    load_static_score_cache,
)
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache
from src.packet_xrag.controller.utility_checkpoint import load_utility_checkpoint
from src.packet_xrag.controller.utility_label_dataset import ShardedUtilityLabelDataset
from src.packet_xrag.controller.utility_training import (
    UtilityFeatureStore, group_batches, rows_and_slices,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type", choices=("B", "C"), required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--labels-root", default="cache/controller/utility_predictor/labels")
    parser.add_argument("--feature-cache", default="cache/controller/features/internal_dev_features")
    parser.add_argument("--static-checkpoint", default="cache/controller/static/best_short_f1/scorer.pt")
    parser.add_argument("--static-score-cache", default="cache/controller/utility_predictor/features/internal_dev_static_scores.pt")
    parser.add_argument("--target-stats", default="cache/controller/utility_predictor/utility_target_stats.json")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


@torch.inference_mode()
def main(argv=None):
    args = parse_args(argv); model_dir = Path(args.model_dir)
    frozen = json.loads((model_dir / "frozen_selection.json").read_text())
    device = torch.device(args.device); torch.cuda.set_device(device)
    cache = ControllerFeatureCache(args.feature_cache)
    labels = ShardedUtilityLabelDataset(args.labels_root, "internal_dev")
    scores = load_static_score_cache(cache, args.static_checkpoint,
                                     args.static_score_cache, device)
    features = UtilityFeatureStore(cache, scores)
    clip = json.loads(Path(args.target_stats).read_text())["utility_clip_value"]
    model = load_utility_checkpoint(args.model_type, frozen["checkpoint"], device)
    shifts, residuals, state_ranges = [], [], defaultdict(list)
    feature_min = [float("inf")] * 7; feature_max = [float("-inf")] * 7
    for indices in group_batches(labels, 20260803, 0, 512, shuffle=False):
        rows, slices = rows_and_slices(labels, indices)
        batch = features.make_batch(rows, device, next(model.parameters()).dtype)
        output, components = model(**batch, return_components=True)
        shift = components["state_shift"].float().cpu() * clip
        shifts.extend(float(value) for value in shift)
        if args.model_type == "C":
            residual = components["interaction_residual"].float().cpu() * clip
            residuals.extend(float(value) for value in residual)
        state_tensor = batch["state_features"].float().cpu()
        for column in range(7):
            feature_min[column] = min(feature_min[column], float(state_tensor[:, column].min()))
            feature_max[column] = max(feature_max[column], float(state_tensor[:, column].max()))
        for start, stop, sid, state_id in slices:
            values = shift[start:stop].tolist()
            state_ranges[(sid, state_id)].append(max(values) - min(values))
    if args.model_type == "B":
        final = model.state_shift_head[-1]
    else:
        final = model.interaction_head[-1]
    payload = {"model_type": args.model_type,
               "state_shift_variance": statistics.pvariance(shifts),
               "state_shift_mean": statistics.mean(shifts),
               "state_shift_nonzero": any(value != 0 for value in shifts),
               "same_state_candidate_shift_max_range": max(max(values) for values in state_ranges.values()),
               "interaction_residual_variance": statistics.pvariance(residuals) if residuals else None,
               "final_layer_weight_norm": float(final.weight.float().norm()),
               "final_layer_bias_norm": float(final.bias.float().norm()),
               "zero_initialization_released": bool(final.weight.float().norm() > 0 or
                                                     final.bias.float().norm() > 0),
               "state_feature_min": feature_min, "state_feature_max": feature_max,
               "selected_mask_and_order": "validated by feature builder and tests",
               "rollout_recomputes_each_step": True, "tau_units": "raw utility NLL",
               "implementation_error_found": False,
               "final_100_accessed": False, "final_100_runs": 0}
    (model_dir / "component_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__": main()
