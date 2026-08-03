"""Shared setup for the three locked utility-predictor training scripts."""

import json
from pathlib import Path

import torch

from scripts.packet_xrag.run_static_scorer_benchmark import load_static_scorer
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache
from src.packet_xrag.controller.utility_label_dataset import ShardedUtilityLabelDataset
from src.packet_xrag.controller.utility_training import UtilityFeatureStore


def load_static_score_cache(feature_cache, scorer_checkpoint, output_path, device):
    output_path = Path(output_path)
    if output_path.exists():
        payload = torch.load(output_path, map_location="cpu", weights_only=True)
        if payload["split_hash"] != feature_cache.manifest["effective_split_hash"]:
            raise RuntimeError("cached STATIC scores use a different split")
        return payload["scores"]
    scorer = load_static_scorer(scorer_checkpoint, device).eval()
    for parameter in scorer.parameters(): parameter.requires_grad = False
    scores = {}
    with torch.inference_mode():
        for start in range(0, len(feature_cache), 32):
            records = [feature_cache[index] for index in
                       range(start, min(start + 32, len(feature_cache)))]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                groups = scorer.score_records(records, device)
            for record, values in zip(records, groups):
                scores[record["sample_id"]] = values.float().cpu()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"split_hash": feature_cache.manifest["effective_split_hash"],
                "scores": scores}, output_path)
    return scores


def load_training_inputs(labels_root, train_cache_path, dev_cache_path,
                         scorer_checkpoint, score_root, device):
    train_labels = ShardedUtilityLabelDataset(labels_root, "train")
    dev_labels = ShardedUtilityLabelDataset(labels_root, "internal_dev")
    train_cache = ControllerFeatureCache(train_cache_path)
    dev_cache = ControllerFeatureCache(dev_cache_path)
    if train_labels.manifest["split_hash"] != train_cache.manifest["effective_split_hash"]:
        raise RuntimeError("train labels/features mismatch")
    if dev_labels.manifest["split_hash"] != dev_cache.manifest["effective_split_hash"]:
        raise RuntimeError("dev labels/features mismatch")
    score_root = Path(score_root)
    train_scores = load_static_score_cache(
        train_cache, scorer_checkpoint, score_root / "train_static_scores.pt", device
    )
    dev_scores = load_static_score_cache(
        dev_cache, scorer_checkpoint, score_root / "internal_dev_static_scores.pt", device
    )
    stats = json.loads((Path(labels_root).parent / "utility_target_stats.json").read_text())
    return (train_labels, dev_labels, UtilityFeatureStore(train_cache, train_scores),
            UtilityFeatureStore(dev_cache, dev_scores), stats)


def best_prediction_checkpoint(model_dir):
    selection = json.loads((Path(model_dir) / "candidate_epochs.json").read_text())
    epoch = selection["candidate_epochs"][0]
    return Path(model_dir) / f"epoch_{epoch}" / "model.pt", epoch


def write_training_config(output_dir, payload):
    path = Path(output_dir); path.mkdir(parents=True, exist_ok=True)
    (path / "training_config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )

