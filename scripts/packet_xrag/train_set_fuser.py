#!/usr/bin/env python
"""Train one O1-only fixed-slot composition architecture probe."""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag.composition_training_common import (
    SEED, build_fuser, checkpoint_payload, load_frozen_generator,
    load_frozen_k2_projector, make_fused_tokens, selected_ids,
)
from scripts.packet_xrag.utility_predictor_training_common import load_static_score_cache
from src.packet_xrag.composition.fused_xrag_injection import (
    build_fused_answer_inputs, fused_answer_loss, pad_fused_answer_batch,
)
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache


def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=("A1", "B1", "C1", "D1"), required=True)
    parser.add_argument("--root", default="cache/composition")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    args = parser.parse_args(argv); root = Path(args.root)
    output_dir = root / f"probes/{args.branch}"; checkpoint = output_dir / "probe.pt"
    if checkpoint.exists(): raise RuntimeError("refusing to overwrite architecture probe")
    ledger = json.loads((root / "experiment_ledger.json").read_text())
    entry = next(item for item in ledger["branches"] if item["experiment_id"] == args.branch)
    if entry["status"] != "preregistered": raise RuntimeError("branch was not preregistered")
    ids = json.loads((root / "splits/composition_train_ids.json").read_text())["ordered_sample_ids"][:1000]
    cache = ControllerFeatureCache("cache/controller/features/train_features")
    by_id = {record["sample_id"]: index for index, record in enumerate(cache.records)}
    records = [cache[by_id[sid]] for sid in ids]
    device = torch.device(args.device); torch.cuda.set_device(device)
    static_scores = load_static_score_cache(cache,
        "cache/controller/static/best_short_f1/scorer.pt",
        "cache/controller/utility_predictor/features/train_static_scores.pt", device)
    random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    tokenizer, generator, xrag_id, config = load_frozen_generator(device)
    k2 = load_frozen_k2_projector(config, device); fuser = build_fuser(args.branch).to(device)
    optimizer = torch.optim.AdamW(fuser.parameters(), lr=1e-4, weight_decay=.01)
    batches = [records[start:start + args.batch_size] for start in range(0, len(records), args.batch_size)]
    optimizer.zero_grad(set_to_none=True); losses = []; began = time.time(); optimizer_steps = 0
    for batch_index, batch in enumerate(batches):
        breadth = ((4, 6)[batch_index % 2] if args.branch == "C1"
                   else (2, 4, 6)[batch_index % 3])
        groups = [selected_ids(record, sorted(range(record["packet_count"]),
                  key=lambda index: (-float(static_scores[record["sample_id"]][index]), index)),
                  args.branch, breadth) for record in batch]
        fused = make_fused_tokens(fuser, args.branch, batch, groups, k2, device)
        answer_items = [build_fused_answer_inputs(tokenizer, xrag_id, record["question"],
                                                  record["answer"], 4) for record in batch]
        input_ids, labels, attention = pad_fused_answer_batch(tokenizer, answer_items, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = fused_answer_loss(generator, input_ids, attention, labels, xrag_id, fused)
            scaled = loss / args.gradient_accumulation
        scaled.backward(); losses.append(float(loss.detach()))
        if (batch_index + 1) % args.gradient_accumulation == 0 or batch_index + 1 == len(batches):
            torch.nn.utils.clip_grad_norm_(fuser.parameters(), 1.0); optimizer.step()
            optimizer.zero_grad(set_to_none=True); optimizer_steps += 1
        if (batch_index + 1) % 50 == 0:
            print(json.dumps({"branch": args.branch, "batch": batch_index + 1,
                              "batches": len(batches), "loss": losses[-1]}), flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(args.branch, fuser, {"optimizer_steps": optimizer_steps}), checkpoint)
    report = {"status": "complete", "branch": args.branch, "samples": 1000, "epochs": 1,
              "optimizer_steps": optimizer_steps, "mean_loss": sum(losses) / len(losses),
              "last_loss": losses[-1], "wall_seconds": time.time() - began,
              "checkpoint_sha256": sha256(checkpoint),
              "generator_trainable_parameters": 0, "k2_trainable_parameters": 0,
              "sfr_trainable_parameters": 0, "final_100_accessed": False}
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2,
                                                                 sort_keys=True) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__": main()
