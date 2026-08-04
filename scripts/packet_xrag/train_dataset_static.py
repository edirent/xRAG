#!/usr/bin/env python
"""Train one dataset-specific STATIC scorer with the frozen Hotpot recipe."""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag.train_static_scorer import (
    preload_embeddings, scorer_batch_loss, validation_loss,
)
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache, sha256_file
from src.packet_xrag.controller.static_scorer import StaticPacketScorer
from src.packet_xrag.generalization.dataset_static_training import (
    EFFECTIVE_BATCH_SIZE, EPOCHS, GRADIENT_CLIPPING, SEED, build_training,
    fixed_recipe, positive_indices,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("2wiki", "musique", "triviaqa"), required=True)
    parser.add_argument("--root", default="cache/generalization")
    parser.add_argument("--device", default="cuda:3")
    args = parser.parse_args(argv); root = Path(args.root); dataset_root = root / args.dataset
    output_dir = dataset_root / "static"
    if output_dir.exists(): raise RuntimeError("refusing to overwrite dataset STATIC run")
    ledger_path = root / "experiment_ledger.json"; ledger = json.loads(ledger_path.read_text())
    if ledger["datasets"][args.dataset]["static_full_runs"] != 0:
        raise RuntimeError("dataset STATIC full-run budget exhausted")
    train = ControllerFeatureCache(dataset_root / "features/train")
    dev = ControllerFeatureCache(dataset_root / "features/dev")
    train_indices, dev_indices = positive_indices(train), positive_indices(dev)
    if not train_indices or not dev_indices: raise RuntimeError("STATIC supervision has no positives")
    device = torch.device(args.device); torch.cuda.set_device(device)
    preload_embeddings(train, device); preload_embeddings(dev, device)
    random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model = StaticPacketScorer().to(device); optimizer, scheduler = build_training(
        model, len(train_indices), device)
    output_dir.mkdir(parents=True); history = []; best = None
    for epoch in range(1, EPOCHS + 1):
        model.train(); order = list(train_indices); random.Random(SEED + epoch).shuffle(order)
        losses = []
        for batch_index, start in enumerate(range(0, len(order), EFFECTIVE_BATCH_SIZE), 1):
            optimizer.zero_grad(set_to_none=True)
            records = [train[index] for index in order[start:start + EFFECTIVE_BATCH_SIZE]]
            loss, _ = scorer_batch_loss(model, records, device); loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING)
            optimizer.step(); scheduler.step(); losses.append(float(loss.detach()))
            if batch_index % 20 == 0:
                print(json.dumps({"dataset": args.dataset, "epoch": epoch, "batch": batch_index,
                                  "loss": losses[-1], "gradient_norm": float(gradient)}), flush=True)
        # DEV-only listwise loss selects the epoch without consuming a generation evaluation.
        model.eval(); dev_records = [dev[index] for index in dev_indices]
        losses_dev = []
        for start in range(0, len(dev_records), 32):
            _, values = scorer_batch_loss(model, dev_records[start:start + 32], device)
            losses_dev.extend(float(value) for value in values)
        dev_loss = sum(losses_dev) / len(losses_dev)
        checkpoint = output_dir / f"epoch_{epoch}.pt"
        torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, checkpoint)
        record = {"epoch": epoch, "mean_train_loss": sum(losses) / len(losses),
                  "dev_listwise_loss": dev_loss, "checkpoint": str(checkpoint),
                  "checkpoint_sha256": sha256_file(checkpoint)}
        history.append(record)
        if best is None or (dev_loss, epoch) < (best["dev_listwise_loss"], best["epoch"]): best = record
        (output_dir / "history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")
        print(json.dumps(record, indent=2), flush=True)
    best_path = output_dir / "best/scorer.pt"; best_path.parent.mkdir()
    shutil.copyfile(best["checkpoint"], best_path)
    report = {"status": "complete", "dataset": args.dataset, "recipe": fixed_recipe(),
        "train_samples": len(train), "train_samples_with_positive": len(train_indices),
        "dev_samples": len(dev), "dev_samples_with_positive": len(dev_indices),
        "supervision_skip_rate": 1 - len(train_indices) / len(train),
        "best_epoch": best["epoch"], "best_dev_listwise_loss": best["dev_listwise_loss"],
        "checkpoint": str(best_path), "checkpoint_sha256": sha256_file(best_path),
        "selection_split": "DEV", "selection_metric": "listwise loss",
        "generator_trainable_parameters": 0, "sfr_trainable_parameters": 0,
        "k2_trainable_parameters": 0, "benchmark_accessed": False, "final100_accessed": False}
    (output_dir / "selection.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    ledger["datasets"][args.dataset]["static_full_runs"] = 1
    ledger["datasets"][args.dataset]["static_checkpoint_sha256"] = report["checkpoint_sha256"]
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__": main()
