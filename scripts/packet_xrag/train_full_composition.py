#!/usr/bin/env python
"""Run one preregistered six-epoch C1 residual composition training objective."""

import argparse
import hashlib
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

from scripts.packet_xrag import run_selector_calibration as selector
from scripts.packet_xrag import train_packet_projector as v1
from scripts.packet_xrag.composition_training_common import (
    SEED, build_fuser, checkpoint_payload, load_frozen_generator,
    load_frozen_k2_projector, make_fused_tokens, selected_ids,
)
from scripts.packet_xrag.utility_predictor_training_common import load_static_score_cache
from src.packet_xrag.composition.composition_losses import slot_cosine_consistency
from src.packet_xrag.composition.fused_xrag_injection import (
    build_fused_answer_inputs, fused_answer_loss, greedy_generate_fused,
    pad_fused_answer_batch,
)
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache


def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def static_ranking(record, scores):
    return sorted(range(record["packet_count"]), key=lambda index: (-float(scores[index]), index))


@torch.inference_mode()
def evaluate(fuser, records, rankings, k2, tokenizer, generator, xrag_id, device, batch_size=8):
    output = {}
    for breadth in (2, 4, 6):
        rows = []
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            groups = [selected_ids(record, rankings[record["sample_id"]], "C1", breadth)
                      for record in batch]
            fused = make_fused_tokens(fuser, "C1", batch, groups, k2, device)
            prompts = tokenizer([v1.build_prompt(record["question"], 4) for record in batch],
                                return_tensors="pt", add_special_tokens=False,
                                padding=True).to(device)
            generated = greedy_generate_fused(generator, tokenizer, prompts.input_ids,
                prompts.attention_mask, xrag_id, fused, max_new_tokens=32)
            for row_index, record in enumerate(batch):
                tokens = generated[row_index]; eos = tokens.eq(tokenizer.eos_token_id).nonzero(as_tuple=False)
                length = int(eos[0]) + 1 if len(eos) else len(tokens)
                raw = tokenizer.decode(tokens[:length], skip_special_tokens=False)
                short = selector.extract_short_answer(raw) or "[EMPTY]"
                em, f1 = selector.score_prediction(short, record["answer"])
                rows.append({"sample_id": record["sample_id"], "short_prediction": short,
                             "short_em": em, "short_f1": f1})
        output[str(breadth)] = {"short_f1": 100 * mean(row["short_f1"] for row in rows),
                                "short_em": 100 * mean(row["short_em"] for row in rows),
                                "empty": sum(row["short_prediction"] == "[EMPTY]" for row in rows)}
    output["selection_score"] = output["6"]["short_f1"] - .5 * max(
        0.0, output["2"]["short_f1"] - output["6"]["short_f1"])
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=("C1_O1", "C1_O1_O3_O4"), required=True)
    parser.add_argument("--root", default="cache/composition")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    args = parser.parse_args(argv); root = Path(args.root); output_dir = root / f"full/{args.run}"
    if output_dir.exists(): raise RuntimeError("refusing to overwrite full composition run")
    ledger_path = root / "experiment_ledger.json"; ledger = json.loads(ledger_path.read_text())
    entry = next(item for item in ledger["full_runs"] if item["experiment_id"] == args.run)
    if entry["status"] != "preregistered": raise RuntimeError("full run was not preregistered")
    cache = ControllerFeatureCache("cache/controller/features/train_features")
    by_id = {record["sample_id"]: index for index, record in enumerate(cache.records)}
    train_ids = json.loads((root / "splits/composition_train_ids.json").read_text())["ordered_sample_ids"]
    dev_ids = json.loads((root / "splits/composition_dev_ids.json").read_text())["ordered_sample_ids"]
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
    probe = torch.load(root / "probes/C1/probe.pt", map_location="cpu", weights_only=True)
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
        breadth_rng = random.Random(f"{SEED}:{args.run}:{epoch}:breadth")
        for batch_index, start in enumerate(range(0, len(order), args.batch_size), 1):
            batch = order[start:start + args.batch_size]
            breadth = breadth_rng.choices((2, 4, 6), weights=(.25, .35, .40), k=1)[0]
            groups = [rankings[record["sample_id"]][:breadth] for record in batch]
            fused = make_fused_tokens(fuser, "C1", batch, groups, k2, device)
            items = [build_fused_answer_inputs(tokenizer, xrag_id, record["question"],
                                               record["answer"], 4) for record in batch]
            input_ids, labels, attention = pad_fused_answer_batch(tokenizer, items, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                answer_loss = fused_answer_loss(generator, input_ids, attention, labels, xrag_id, fused)
                loss = answer_loss
                duplicate_loss = distractor_loss = torch.zeros((), device=device)
                if args.run == "C1_O1_O3_O4":
                    duplicate_groups = [selected + [selected[-1]] for selected in groups]
                    distractor_groups = []
                    for record, selected in zip(batch, groups):
                        gold = set(record["gold_packet_ids"])
                        distractor = next((index for index in rankings[record["sample_id"]]
                                           if index not in gold and index not in set(selected)), None)
                        distractor_groups.append(selected if distractor is None else
                                                 selected + [distractor])
                    duplicate_fused = make_fused_tokens(fuser, "C1", batch, duplicate_groups,
                                                        k2, device)
                    distractor_fused = make_fused_tokens(fuser, "C1", batch, distractor_groups,
                                                         k2, device)
                    duplicate_loss = slot_cosine_consistency(fused, duplicate_fused)
                    distractor_loss = slot_cosine_consistency(fused, distractor_fused)
                    loss = answer_loss + .1 * duplicate_loss + .1 * distractor_loss
                scaled = loss / args.gradient_accumulation
            losses.append(float(loss.detach()))
            if loss.requires_grad:
                scaled.backward(); accumulated += 1
            if accumulated == args.gradient_accumulation:
                torch.nn.utils.clip_grad_norm_(fuser.parameters(), 1.0); optimizer.step(); scheduler.step()
                optimizer.zero_grad(set_to_none=True); optimizer_steps += 1; accumulated = 0
            if batch_index % 250 == 0:
                print(json.dumps({"run": args.run, "epoch": epoch, "batch": batch_index,
                                  "loss": losses[-1], "answer": float(answer_loss.detach()),
                                  "duplicate": float(duplicate_loss.detach()),
                                  "distractor": float(distractor_loss.detach())}), flush=True)
        if accumulated:
            torch.nn.utils.clip_grad_norm_(fuser.parameters(), 1.0); optimizer.step(); scheduler.step()
            optimizer.zero_grad(set_to_none=True); optimizer_steps += 1
        fuser.eval(); dev = evaluate(fuser, dev_records, rankings, k2, tokenizer,
                                     generator, xrag_id, device); fuser.train()
        checkpoint = output_dir / f"epoch_{epoch}.pt"
        torch.save(checkpoint_payload("C1", fuser, {"run": args.run, "epoch": epoch,
                                                    "dev": dev}), checkpoint)
        record = {"epoch": epoch, "mean_train_loss": mean(losses), "dev": dev,
                  "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint)}
        history.append(record)
        if best is None or (dev["selection_score"], dev["6"]["short_f1"],
                            -max(0, dev["2"]["short_f1"] - dev["6"]["short_f1"]), -epoch) > (
                            best["dev"]["selection_score"], best["dev"]["6"]["short_f1"],
                            -max(0, best["dev"]["2"]["short_f1"] - best["dev"]["6"]["short_f1"]),
                            -best["epoch"]): best = record
        (output_dir / "history.json").write_text(json.dumps(history, indent=2,
                                                              sort_keys=True) + "\n")
        print(json.dumps({"run": args.run, **record}, indent=2), flush=True)
    selection = {"status": "complete", "run": args.run, "epochs": 6,
                 "optimizer_steps": optimizer_steps, "best_epoch": best["epoch"],
                 "best_checkpoint": best["checkpoint"], "best_checkpoint_sha256": best["checkpoint_sha256"],
                 "best_dev": best["dev"], "wall_seconds": time.time() - began,
                 "trainable_parameters": sum(p.numel() for p in fuser.parameters() if p.requires_grad),
                 "generator_trainable_parameters": 0, "k2_trainable_parameters": 0,
                 "sfr_trainable_parameters": 0, "shadow_accessed": False,
                 "benchmark_accessed": False, "final_100_accessed": False}
    (output_dir / "selection.json").write_text(json.dumps(selection, indent=2,
                                                            sort_keys=True) + "\n")
    ledger["usage"]["full_training_runs"] += 1
    ledger["usage"]["composition_dev_generation"] += 1
    entry.update({"status": "completed", "actual_metrics": best["dev"],
                  "checkpoint_hash": best["checkpoint_sha256"], "best_epoch": best["epoch"]})
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps(selection, indent=2), flush=True)


if __name__ == "__main__": main()
