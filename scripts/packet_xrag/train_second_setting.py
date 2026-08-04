#!/usr/bin/env python
"""Run one of the two preregistered K4 full-training jobs."""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from statistics import mean

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag.composition_training_common import (
    build_fuser, checkpoint_payload, load_frozen_generator,
)
from scripts.packet_xrag.utility_predictor_training_common import load_static_score_cache
from src.packet_xrag.composition.fused_xrag_injection import (
    build_fused_answer_inputs, fused_answer_loss, pad_fused_answer_batch,
)
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache, sha256_file
from src.packet_xrag.generalization.protocol import SEED
from src.packet_xrag.generalization.second_setting_adapter import (
    K4_SHA256, load_frozen_k4_projector, make_k4_fused,
)

EPOCHS = 6
HOTPOT_MAIN = Path("cache/composition/full/C1_O1/epoch_6.pt")
HOTPOT_MAIN_SHA256 = "8f0f1161defb506b48dfac4249e2a395cb49cabad4ed75a3adc045ac02a9e6e3"


def task_data(task, root):
    if task == "hotpot":
        cache = ControllerFeatureCache("cache/controller/features/train_features")
        by_id = {record["sample_id"]: index for index, record in enumerate(cache.records)}
        ids = json.loads(Path("cache/composition/splits/composition_train_ids.json").read_text())[
            "ordered_sample_ids"]
        records = [cache[by_id[sample_id]] for sample_id in ids]
        checkpoint = "cache/controller/static/best_short_f1/scorer.pt"
        score_path = "cache/controller/utility_predictor/features/train_static_scores.pt"
    else:
        dataset_root = root / "musique"
        cache = ControllerFeatureCache(dataset_root / "features/train")
        records = [cache[index] for index in range(len(cache))]
        static = json.loads((dataset_root / "static/selection.json").read_text())
        checkpoint = static["checkpoint"]
        score_path = dataset_root / "static/scores/train.pt"
    return cache, records, checkpoint, score_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("hotpot", "musique"), required=True)
    parser.add_argument("--root", default="cache/generalization")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    args = parser.parse_args(argv)
    root = Path(args.root); second = root / "second_setting"
    output = second / args.task / "training"
    if output.exists():
        raise RuntimeError("refusing to overwrite K4 full-training run")
    gate = json.loads((root / "generalization_gate.json").read_text())
    if not gate["second_setting_authorized"]:
        raise RuntimeError("generalization gate did not authorize the second setting")
    ledger_path = root / "experiment_ledger.json"
    ledger = json.loads(ledger_path.read_text())
    if ledger["usage"]["second_setting_full_runs"] >= 2:
        raise RuntimeError("second-setting full-training budget exhausted")
    if args.task == "musique" and not (second / "hotpot/training/selection.json").exists():
        raise RuntimeError("Hotpot K4 run must freeze before MuSiQue K4 transfer")
    cache, records, static_checkpoint, score_path = task_data(args.task, root)
    device = torch.device(args.device); torch.cuda.set_device(device)
    scores = load_static_score_cache(cache, static_checkpoint, score_path, device)
    rankings = {record["sample_id"]: sorted(range(record["packet_count"]),
        key=lambda index: (-float(scores[record["sample_id"]][index]), index))
        for record in records}
    random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    tokenizer, generator, xrag_id, config = load_frozen_generator(device)
    k4 = load_frozen_k4_projector(config, device)
    fuser = build_fuser("C1").to(device)
    if args.task == "hotpot":
        initialization = HOTPOT_MAIN
        expected_hash = HOTPOT_MAIN_SHA256
    else:
        hotpot_selection = json.loads((second / "hotpot/training/selection.json").read_text())
        initialization = Path(hotpot_selection["checkpoint"])
        expected_hash = hotpot_selection["checkpoint_sha256"]
    if sha256_file(initialization) != expected_hash:
        raise RuntimeError("K4 fuser initialization hash mismatch")
    payload = torch.load(initialization, map_location="cpu", weights_only=True)
    fuser.load_state_dict(payload["state_dict"], strict=True)
    optimizer = torch.optim.AdamW(fuser.parameters(), lr=5e-5, weight_decay=.01)
    steps_per_epoch = math.ceil(len(records) / args.batch_size /
                                args.gradient_accumulation)
    total_steps = steps_per_epoch * EPOCHS; warmup = int(.05 * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:
        min(1.0, (step + 1) / max(1, warmup)) * max(0.0, (total_steps - step) /
        max(1, total_steps - warmup)))
    output.mkdir(parents=True)
    # Consume the full-run budget before the first optimizer update. A failed run
    # cannot be silently retried as a fresh experiment.
    ledger["usage"]["second_setting_full_runs"] += 1
    ledger.setdefault("second_setting", {})[args.task] = {"status": "training_in_progress"}
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    optimizer_steps = 0; history = []; began = time.time()
    for epoch in range(1, EPOCHS + 1):
        fuser.train(); order = list(records); random.Random(SEED + epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True); losses = []; accumulated = 0
        breadth_rng = random.Random(f"{SEED}:K4:{args.task}:{epoch}:breadth")
        for batch_index, start in enumerate(range(0, len(order), args.batch_size), 1):
            batch = order[start:start + args.batch_size]
            breadth = breadth_rng.choices((2, 4, 6), weights=(.25, .35, .40), k=1)[0]
            groups = [rankings[record["sample_id"]][:breadth] for record in batch]
            fused = make_k4_fused(fuser, batch, groups, k4, device)
            items = [build_fused_answer_inputs(tokenizer, xrag_id, record["question"],
                                               record["answer"], 4) for record in batch]
            input_ids, labels, attention = pad_fused_answer_batch(tokenizer, items, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = fused_answer_loss(generator, input_ids, attention, labels, xrag_id, fused)
                scaled = loss / args.gradient_accumulation
            if not torch.isfinite(loss):
                raise RuntimeError("K4 fuser loss became NaN/Inf")
            losses.append(float(loss.detach()))
            if loss.requires_grad:
                scaled.backward(); accumulated += 1
            if accumulated == args.gradient_accumulation:
                torch.nn.utils.clip_grad_norm_(fuser.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1; accumulated = 0
            if batch_index % 250 == 0:
                print(json.dumps({"task": args.task, "epoch": epoch,
                                  "batch": batch_index, "loss": losses[-1]}), flush=True)
        if accumulated:
            torch.nn.utils.clip_grad_norm_(fuser.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        checkpoint = output / f"epoch_{epoch}.pt"
        torch.save(checkpoint_payload("C1", fuser, {"setting": "K4", "task": args.task,
            "epoch": epoch, "initialization_sha256": expected_hash,
            "checkpoint_selection": "fixed epoch 6; no held-out generation during training"}),
            checkpoint)
        record = {"epoch": epoch, "mean_train_loss": mean(losses),
                  "checkpoint": str(checkpoint),
                  "checkpoint_sha256": sha256_file(checkpoint)}
        history.append(record)
        (output / "history.json").write_text(json.dumps(history, indent=2,
                                                          sort_keys=True) + "\n")
        print(json.dumps(record, indent=2), flush=True)
    selected = history[-1]
    report = {"status": "complete", "setting": "K4", "task": args.task,
        "checkpoint": selected["checkpoint"],
        "checkpoint_sha256": selected["checkpoint_sha256"], "selected_epoch": EPOCHS,
        "selection_rule": "fixed epoch 6, matching frozen primary-recipe endpoint",
        "heldout_generation_during_training": 0, "optimizer_steps": optimizer_steps,
        "initialization": str(initialization), "initialization_sha256": expected_hash,
        "k4_checkpoint_sha256": K4_SHA256, "output_M": 4,
        "mean_train_loss_first": history[0]["mean_train_loss"],
        "mean_train_loss_last": history[-1]["mean_train_loss"],
        "wall_seconds": time.time() - began,
        "trainable_parameters": sum(p.numel() for p in fuser.parameters() if p.requires_grad),
        "generator_trainable_parameters": 0, "k4_trainable_parameters": 0,
        "sfr_trainable_parameters": 0, "benchmark_accessed": False,
        "final100_accessed": False}
    (output / "selection.json").write_text(json.dumps(report, indent=2,
                                                        sort_keys=True) + "\n")
    ledger["second_setting"][args.task] = {"status": "training_complete",
        "checkpoint_sha256": selected["checkpoint_sha256"]}
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
