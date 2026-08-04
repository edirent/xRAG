#!/usr/bin/env python
"""Train the sole preregistered O2 variant: permute extras, keep STATIC base fixed."""

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
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag.composition_training_common import (
    SEED, build_fuser, checkpoint_payload, load_frozen_generator,
    load_frozen_k2_projector, make_fused_tokens,
)
from scripts.packet_xrag.train_full_composition import evaluate, sha256, static_ranking
from scripts.packet_xrag import train_packet_projector as v1
from scripts.packet_xrag.utility_predictor_training_common import load_static_score_cache
from src.packet_xrag.composition.fused_xrag_injection import (
    build_fused_answer_inputs, fused_answer_loss, pad_fused_answer_batch,
)
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache
from src.packet_xrag.generalization.order_robustness import permute_extras


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="cache/generalization")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    args = parser.parse_args(argv); root = Path(args.root); output_dir = root / "order/O2_training"
    if output_dir.exists(): raise RuntimeError("refusing to overwrite sole O2 full run")
    diagnostic = json.loads((root / "order/original_results.json").read_text())
    if diagnostic["variant_selected"] != "O2": raise RuntimeError("architecture audit did not select O2")
    ledger_path = root / "experiment_ledger.json"; ledger = json.loads(ledger_path.read_text())
    if ledger["usage"]["order_ablation_full_runs"] != 0:
        raise RuntimeError("order-ablation full-run budget exhausted")
    cache = ControllerFeatureCache("cache/controller/features/train_features")
    by_id = {record["sample_id"]: index for index, record in enumerate(cache.records)}
    train_ids = json.loads(Path("cache/composition/splits/composition_train_ids.json").read_text())["ordered_sample_ids"]
    dev_ids = json.loads(Path("cache/composition/splits/composition_dev_ids.json").read_text())["ordered_sample_ids"]
    train_records = [cache[by_id[sid]] for sid in train_ids]
    dev_records = [cache[by_id[sid]] for sid in dev_ids]
    device = torch.device(args.device); torch.cuda.set_device(device)
    scores = load_static_score_cache(cache, "cache/controller/static/best_short_f1/scorer.pt",
        "cache/controller/utility_predictor/features/train_static_scores.pt", device)
    rankings = {record["sample_id"]: static_ranking(record, scores[record["sample_id"]])
                for record in train_records + dev_records}
    random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    tokenizer, generator, xrag_id, config = load_frozen_generator(device)
    k2 = load_frozen_k2_projector(config, device); fuser = build_fuser("C1").to(device)
    probe = torch.load("cache/composition/probes/C1/probe.pt", map_location="cpu", weights_only=True)
    fuser.load_state_dict(probe["state_dict"], strict=True)
    optimizer = torch.optim.AdamW(fuser.parameters(), lr=5e-5, weight_decay=.01)
    steps_per_epoch = math.ceil(len(train_records) / args.batch_size / args.gradient_accumulation)
    total_steps = steps_per_epoch * 6; warmup = int(.05 * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:
        min(1.0, (step + 1) / max(1, warmup)) * max(0.0, (total_steps - step) /
        max(1, total_steps - warmup)))
    output_dir.mkdir(parents=True); history = []; best = None; optimizer_steps = 0
    began = time.time()
    for epoch in range(1, 7):
        order = list(train_records); random.Random(SEED + epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True); losses = []; accumulated = 0
        breadth_rng = random.Random(f"{SEED}:O2:{epoch}:breadth")
        for batch_index, start in enumerate(range(0, len(order), args.batch_size), 1):
            batch = order[start:start + args.batch_size]
            breadth = breadth_rng.choices((2, 4, 6), weights=(.25, .35, .40), k=1)[0]
            groups = [permute_extras(record, rankings[record["sample_id"]][:breadth],
                                     f"random_train_epoch_{epoch}") for record in batch]
            fused = make_fused_tokens(fuser, "C1", batch, groups, k2, device)
            items = [build_fused_answer_inputs(tokenizer, xrag_id, record["question"],
                                               record["answer"], 4) for record in batch]
            input_ids, labels, attention = pad_fused_answer_batch(tokenizer, items, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = fused_answer_loss(generator, input_ids, attention, labels, xrag_id, fused)
                scaled = loss / args.gradient_accumulation
            losses.append(float(loss.detach()))
            if loss.requires_grad: scaled.backward(); accumulated += 1
            if accumulated == args.gradient_accumulation:
                torch.nn.utils.clip_grad_norm_(fuser.parameters(), 1.0); optimizer.step(); scheduler.step()
                optimizer.zero_grad(set_to_none=True); optimizer_steps += 1; accumulated = 0
            if batch_index % 250 == 0:
                print(json.dumps({"variant": "O2", "epoch": epoch, "batch": batch_index,
                                  "loss": losses[-1]}), flush=True)
        if accumulated:
            torch.nn.utils.clip_grad_norm_(fuser.parameters(), 1.0); optimizer.step(); scheduler.step()
            optimizer.zero_grad(set_to_none=True); optimizer_steps += 1
        fuser.eval(); dev = evaluate(fuser, dev_records, rankings, k2, tokenizer,
                                     generator, xrag_id, device); fuser.train()
        checkpoint = output_dir / f"epoch_{epoch}.pt"
        torch.save(checkpoint_payload("C1", fuser, {"variant": "O2", "epoch": epoch,
            "dev": dev, "only_change": "random physical permutation of rank3..N extras"}), checkpoint)
        record = {"epoch": epoch, "mean_train_loss": mean(losses), "dev": dev,
                  "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint)}
        history.append(record)
        if best is None or (dev["selection_score"], dev["6"]["short_f1"], -epoch) > (
                best["dev"]["selection_score"], best["dev"]["6"]["short_f1"], -best["epoch"]):
            best = record
        (output_dir / "history.json").write_text(json.dumps(history, indent=2,
                                                              sort_keys=True) + "\n")
        print(json.dumps(record, indent=2), flush=True)
    selection = {"status": "complete", "variant": "O2", "epochs": 6,
        "optimizer_steps": optimizer_steps, "best_epoch": best["epoch"],
        "best_checkpoint": best["checkpoint"], "best_checkpoint_sha256": best["checkpoint_sha256"],
        "best_dev": best["dev"], "wall_seconds": time.time() - began,
        "trainable_parameters": sum(p.numel() for p in fuser.parameters() if p.requires_grad),
        "generator_trainable_parameters": 0, "k2_trainable_parameters": 0,
        "sfr_trainable_parameters": 0, "hotpot_primary_checkpoint_changed": False,
        "benchmark_accessed": False, "final100_accessed": False}
    (output_dir / "selection.json").write_text(json.dumps(selection, indent=2,
                                                            sort_keys=True) + "\n")
    ledger["usage"]["order_ablation_full_runs"] = 1
    ledger["order_variant"] = {"status": "trained_pending_gate", "variant": "O2",
                               "checkpoint_sha256": best["checkpoint_sha256"]}
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps(selection, indent=2), flush=True)


if __name__ == "__main__": main()
