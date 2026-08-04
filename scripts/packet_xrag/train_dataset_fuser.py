#!/usr/bin/env python
"""Fine-tune one dataset fuser from the frozen Hotpot composition checkpoint."""

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
    build_fuser, checkpoint_payload, load_frozen_generator, load_frozen_k2_projector,
    make_fused_tokens,
)
from scripts.packet_xrag.evaluate_dataset_dev_baselines import (
    HOTPOT_FUSER, HOTPOT_FUSER_SHA256,
)
from scripts.packet_xrag.utility_predictor_training_common import load_static_score_cache
from src.packet_xrag.composition.fused_xrag_injection import (
    build_fused_answer_inputs, fused_answer_loss, pad_fused_answer_batch,
)
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache, sha256_file
from src.packet_xrag.generalization.dataset_evaluation import (
    evaluate_fuser, make_c1_fused, summarize,
)
from src.packet_xrag.generalization.protocol import SEED


EPOCHS = 6
EVALUATION_EPOCHS = (2, 4, 6)


def rankings_for(cache, checkpoint, score_path, device):
    scores = load_static_score_cache(cache, checkpoint, score_path, device)
    return {record["sample_id"]: sorted(range(record["packet_count"]),
        key=lambda index: (-float(scores[record["sample_id"]][index]), index))
        for record in cache.records}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("2wiki", "musique", "triviaqa"), required=True)
    parser.add_argument("--root", default="cache/generalization")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    args = parser.parse_args(argv); root = Path(args.root); dataset_root = root / args.dataset
    output = dataset_root / "fuser/run_1_hotpot_init"
    if output.exists(): raise RuntimeError("refusing to overwrite dataset fuser run")
    ledger_path = root / "experiment_ledger.json"; ledger = json.loads(ledger_path.read_text())
    usage = ledger["datasets"][args.dataset]
    if usage["fuser_full_runs"] != 0: raise RuntimeError("first fuser run already consumed")
    if not (dataset_root / "dev_baselines/results.json").exists() or usage["dev_generation"] > 3:
        raise RuntimeError("frozen DEV baselines and zero-shot evaluation must precede training")
    if usage["dev_generation"] + len(EVALUATION_EPOCHS) > 5:
        raise RuntimeError("three checkpoint suites would leave no selected-DEV budget")
    if sha256_file(HOTPOT_FUSER) != HOTPOT_FUSER_SHA256:
        raise RuntimeError("frozen Hotpot initialization hash mismatch")
    static = json.loads((dataset_root / "static/selection.json").read_text())
    train = ControllerFeatureCache(dataset_root / "features/train")
    dev = ControllerFeatureCache(dataset_root / "features/dev")
    device = torch.device(args.device); torch.cuda.set_device(device)
    train_rankings = rankings_for(train, static["checkpoint"],
        dataset_root / "static/scores/train.pt", device)
    dev_rankings = rankings_for(dev, static["checkpoint"],
        dataset_root / "static/scores/dev.pt", device)
    train_records = [train[index] for index in range(len(train))]
    dev_records = [dev[index] for index in range(len(dev))]
    random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    tokenizer, generator, xrag_id, config = load_frozen_generator(device)
    k2 = load_frozen_k2_projector(config, device); fuser = build_fuser("C1").to(device)
    initial = torch.load(HOTPOT_FUSER, map_location="cpu", weights_only=True)
    fuser.load_state_dict(initial["state_dict"], strict=True)
    optimizer = torch.optim.AdamW(fuser.parameters(), lr=5e-5, weight_decay=.01)
    steps_per_epoch = math.ceil(len(train_records) / args.batch_size /
                                args.gradient_accumulation)
    total_steps = steps_per_epoch * EPOCHS; warmup = int(.05 * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:
        min(1.0, (step + 1) / max(1, warmup)) * max(0.0, (total_steps - step) /
        max(1, total_steps - warmup)))
    output.mkdir(parents=True); history = []; candidates = []; optimizer_steps = 0
    began = time.time()
    for epoch in range(1, EPOCHS + 1):
        fuser.train(); order = list(train_records); random.Random(SEED + epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True); losses = []; accumulated = 0
        breadth_rng = random.Random(f"{SEED}:{args.dataset}:{epoch}:breadth")
        for batch_index, start in enumerate(range(0, len(order), args.batch_size), 1):
            batch = order[start:start + args.batch_size]
            breadth = breadth_rng.choices((2, 4, 6), weights=(.25, .35, .40), k=1)[0]
            groups = [train_rankings[record["sample_id"]][:breadth] for record in batch]
            fused = make_fused_tokens(fuser, "C1", batch, groups, k2, device)
            items = [build_fused_answer_inputs(tokenizer, xrag_id, record["question"],
                                               record["answer"], 4) for record in batch]
            input_ids, labels, attention = pad_fused_answer_batch(tokenizer, items, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = fused_answer_loss(generator, input_ids, attention, labels, xrag_id, fused)
                scaled = loss / args.gradient_accumulation
            losses.append(float(loss.detach()))
            if not torch.isfinite(loss): raise RuntimeError("dataset fuser loss became NaN/Inf")
            if loss.requires_grad: scaled.backward(); accumulated += 1
            if accumulated == args.gradient_accumulation:
                torch.nn.utils.clip_grad_norm_(fuser.parameters(), 1.0); optimizer.step()
                scheduler.step(); optimizer.zero_grad(set_to_none=True)
                accumulated = 0; optimizer_steps += 1
            if batch_index % 250 == 0:
                print(json.dumps({"dataset": args.dataset, "epoch": epoch,
                                  "batch": batch_index, "loss": losses[-1]}), flush=True)
        if accumulated:
            torch.nn.utils.clip_grad_norm_(fuser.parameters(), 1.0); optimizer.step()
            scheduler.step(); optimizer.zero_grad(set_to_none=True); optimizer_steps += 1
        checkpoint = output / f"epoch_{epoch}.pt"
        dev_summary = None
        if epoch in EVALUATION_EPOCHS:
            fuser.eval(); by_breadth = {}
            for breadth in (2, 6):
                groups = [dev_rankings[record["sample_id"]][:breadth] for record in dev_records]
                rows = evaluate_fuser(f"DATASET_FUSER_{breadth}", fuser, dev_records, groups,
                    make_c1_fused, k2, tokenizer, generator, xrag_id, device, 8)
                by_breadth[str(breadth)] = summarize(rows)
                predictions = output / f"dev_epoch_{epoch}_n{breadth}.jsonl"
                with predictions.open("w") as stream:
                    for row in rows: stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            score = by_breadth["6"]["short_f1"] - .5 * max(
                0.0, by_breadth["2"]["short_f1"] - by_breadth["6"]["short_f1"])
            dev_summary = {**by_breadth, "selection_score": score}
            usage["dev_generation"] += 1
        torch.save(checkpoint_payload("C1", fuser, {"dataset": args.dataset,
            "run": 1, "initialization": HOTPOT_FUSER_SHA256, "epoch": epoch,
            "dev": dev_summary}), checkpoint)
        record = {"epoch": epoch, "mean_train_loss": mean(losses), "dev": dev_summary,
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint)}
        history.append(record)
        if dev_summary is not None: candidates.append(record)
        (output / "history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")
        ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
        print(json.dumps(record, indent=2), flush=True)
    best = max(candidates, key=lambda item: (item["dev"]["selection_score"],
        item["dev"]["6"]["short_f1"],
        -max(0.0, item["dev"]["2"]["short_f1"] - item["dev"]["6"]["short_f1"]),
        -item["epoch"], -item["dev"]["6"]["mean_total_latency_ms"]))
    gate_weight = fuser.gate.weight.detach().abs().max().item()
    gate_bias = fuser.gate.bias.detach().abs().max().item()
    baseline_metrics = json.loads((dataset_root / "dev_baselines/results.json").read_text())["metrics"]
    failures = []
    if history[-1]["mean_train_loss"] >= history[0]["mean_train_loss"]:
        failures.append("mean training loss did not decrease")
    if gate_weight == 0 and gate_bias == 0:
        failures.append("residual gate remained strictly zero")
    if best["dev"]["6"]["short_f1"] < baseline_metrics["INDEPENDENT_STATIC_6"]["short_f1"] - 5:
        failures.append("DEV Fuser N6 is more than 5 F1 below Independent N6")
    report = {"status": "complete", "dataset": args.dataset, "run": 1,
        "initialization": "Hotpot frozen fuser", "initialization_sha256": HOTPOT_FUSER_SHA256,
        "epochs": EPOCHS, "dev_evaluation_epochs": list(EVALUATION_EPOCHS),
        "optimizer_steps": optimizer_steps, "best_epoch": best["epoch"],
        "best_checkpoint": best["checkpoint"],
        "best_checkpoint_sha256": best["checkpoint_sha256"], "best_dev": best["dev"],
        "wall_seconds": time.time() - began, "residual_strictly_zero": gate_weight == 0 and gate_bias == 0,
        "max_abs_gate_weight": gate_weight, "max_abs_gate_bias": gate_bias,
        "optimization_failure": bool(failures), "optimization_failure_reasons": failures,
        "second_run_authorized": bool(failures),
        "trainable_parameters": sum(p.numel() for p in fuser.parameters() if p.requires_grad),
        "generator_trainable_parameters": 0, "k2_trainable_parameters": 0,
        "sfr_trainable_parameters": 0, "shadow_accessed": False,
        "benchmark_accessed": False, "final100_accessed": False}
    (output / "selection.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    usage["fuser_full_runs"] = 1
    usage["fuser_checkpoint_sha256"] = best["checkpoint_sha256"]
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__": main()
