#!/usr/bin/env python
"""Train the locked Stage-1 static multi-positive listwise packet scorer."""

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import get_linear_schedule_with_warmup

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag.run_static_scorer_benchmark import (
    evaluate_generator,
    initialize_generator,
    rank_cache,
    summarize_rows,
)
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache, sha256_file
from src.packet_xrag.controller.static_scorer import (
    StaticPacketScorer,
    multi_positive_listwise_loss,
    negative_analysis_labels,
)


SEED = 20260803


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", default="cache/controller/features/train_features")
    parser.add_argument("--dev-cache", default="cache/controller/features/internal_dev_features")
    parser.add_argument("--quarantine-file", default="cache/controller/splits/controller_quarantine.json")
    parser.add_argument("--k2-training-config", default="cache/projector/multi_token_k2/best_short_f1/training_config.json")
    parser.add_argument("--output-dir", default="cache/controller/static")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--effective-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--gradient-clipping", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args(argv)


def assert_locked_config(args):
    expected = {
        "epochs": 8, "effective_batch_size": 32, "learning_rate": 2e-4,
        "weight_decay": 0.01, "warmup_ratio": 0.05,
        "gradient_clipping": 1.0, "seed": SEED,
    }
    actual = {key: getattr(args, key) for key in expected}
    if actual != expected:
        raise RuntimeError(f"static training protocol differs from lock: {actual}")


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gold_mask(record, device):
    mask = torch.zeros(record["packet_count"], dtype=torch.bool, device=device)
    mask[record["gold_packet_ids"]] = True
    return mask


def scorer_loss(model, record, device):
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=device.type == "cuda"):
        scores = model.score_record(record, device)
    return multi_positive_listwise_loss(scores.float(), gold_mask(record, device))


def scorer_batch_loss(model, records, device):
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=device.type == "cuda"):
        score_groups = model.score_records(records, device)
    losses = [multi_positive_listwise_loss(scores.float(), gold_mask(record, device))
              for record, scores in zip(records, score_groups)]
    return torch.stack(losses).mean(), losses


@torch.inference_mode()
def validation_loss(model, cache, device):
    model.eval()
    losses = []
    for start in range(0, len(cache), 32):
        records = [cache[index] for index in range(start, min(start + 32, len(cache)))]
        _, batch_losses = scorer_batch_loss(model, records, device)
        losses.extend(float(loss) for loss in batch_losses)
    return sum(losses) / len(losses)


def choose_budget(metrics):
    best_f1 = max(item["short_f1"] for item in metrics.values())
    close = [(int(name.split("_")[1]), item) for name, item in metrics.items()
             if best_f1 - item["short_f1"] < 0.25]
    return min(close, key=lambda pair: pair[0])[0]


def choose_checkpoint(history):
    maximum = max(item["selected_short_f1"] for item in history)
    close = [item for item in history if maximum - item["selected_short_f1"] < 0.25]
    return min(close, key=lambda item: (
        item["selected_budget"], item["validation_listwise_loss"], item["epoch"]
    ))


def save_model(model, directory, payload):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save(state, directory / "scorer.pt")
    (directory / "checkpoint_metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def negative_label_counts(cache):
    counts = Counter()
    for record in cache.records:
        counts.update(negative_analysis_labels(
            record["packets"], record["gold_packet_ids"], record["topk_ranking"]
        ).values())
    return dict(sorted(counts.items()))


def preload_embeddings(cache, device):
    """Keep the complete frozen mmap tensors on GPU for all eight epochs."""
    cache.queries = cache.queries.to(device=device, dtype=torch.bfloat16)
    cache.packets = cache.packets.to(device=device, dtype=torch.bfloat16)


def main(argv=None):
    args = parse_args(argv)
    assert_locked_config(args)
    seed_everything(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    train_cache = ControllerFeatureCache(args.train_cache)
    dev_cache = ControllerFeatureCache(args.dev_cache)
    if len(train_cache) != 4499 or len(dev_cache) != 500:
        raise RuntimeError("static training requires effective train=4499 and internal dev=500")
    if train_cache.manifest["quarantine_hash"] != sha256_file(args.quarantine_file):
        raise RuntimeError("training cache quarantine hash mismatch")
    quarantined_ids = {
        entry["sample_id"] for entry in json.loads(Path(args.quarantine_file).read_text())["entries"]
    }
    if quarantined_ids & {record["sample_id"] for record in train_cache.records}:
        raise RuntimeError("quarantined sample entered static scorer training")
    print("Preloading frozen train/internal-dev embeddings to GPU", flush=True)
    preload_embeddings(train_cache, device)
    preload_embeddings(dev_cache, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = StaticPacketScorer().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    updates_per_epoch = math.ceil(len(train_cache) / args.effective_batch_size)
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_updates * args.warmup_ratio), total_updates
    )
    tokenizer, generator, xrag_id, checkpoint_hashes = initialize_generator(
        args.k2_training_config, device
    )
    history = []
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train_cache)))
        random.Random(args.seed + epoch).shuffle(order)
        epoch_losses = []
        for batch_index, start in enumerate(range(0, len(order), args.effective_batch_size), 1):
            optimizer.zero_grad(set_to_none=True)
            indices = order[start:start + args.effective_batch_size]
            records = [train_cache[index] for index in indices]
            loss, _ = scorer_batch_loss(model, records, device)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.gradient_clipping
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            epoch_losses.append(float(loss.detach()))
            if batch_index % args.log_every == 0 or batch_index == updates_per_epoch:
                print(json.dumps({
                    "epoch": epoch, "batch": batch_index, "batches": updates_per_epoch,
                    "loss": epoch_losses[-1], "lr": scheduler.get_last_lr()[0],
                    "gradient_norm": float(gradient_norm),
                }), flush=True)
        dev_loss = validation_loss(model, dev_cache, device)
        rankings = rank_cache(dev_cache, model, device)
        epoch_dir = output_dir / f"epoch_{epoch}"
        predictions = epoch_dir / "validation_predictions.jsonl"
        rows = evaluate_generator(
            dev_cache, rankings, tokenizer, generator, xrag_id, device, predictions,
            max_new_tokens=32,
        )
        metrics = summarize_rows(rows)
        selected_budget = choose_budget(metrics)
        epoch_record = {
            "epoch": epoch, "training_listwise_loss": sum(epoch_losses) / len(epoch_losses),
            "validation_listwise_loss": dev_loss, "selected_budget": selected_budget,
            "selected_short_f1": metrics[f"STATIC_{selected_budget}"]["short_f1"],
            "metrics": metrics,
        }
        history.append(epoch_record)
        epoch_dir.mkdir(parents=True, exist_ok=True)
        (epoch_dir / "validation_metrics.json").write_text(
            json.dumps(epoch_record, indent=2, sort_keys=True) + "\n"
        )
        save_model(model, epoch_dir, epoch_record)
        print(json.dumps(epoch_record, indent=2), flush=True)
    selected = choose_checkpoint(history)
    selected_epoch_dir = output_dir / f"epoch_{selected['epoch']}"
    best_dir = output_dir / "best_short_f1"
    last_dir = output_dir / "last"
    for destination, source in ((best_dir, selected_epoch_dir),
                                (last_dir, output_dir / f"epoch_{args.epochs}")):
        destination.mkdir(parents=True, exist_ok=True)
        for filename in ("scorer.pt", "validation_predictions.jsonl",
                         "validation_metrics.json", "checkpoint_metadata.json"):
            shutil.copyfile(source / filename, destination / filename)
    config = {
        "stage": "static_scorer", "epochs": args.epochs,
        "effective_batch_size": args.effective_batch_size,
        "optimizer": "AdamW", "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "warmup_ratio": args.warmup_ratio,
        "gradient_clipping": args.gradient_clipping, "dtype": "BF16 autocast",
        "seed": args.seed, "loss": "multi-positive listwise",
        "train_samples": len(train_cache), "internal_dev_samples": len(dev_cache),
        "train_split_hash": train_cache.manifest["effective_split_hash"],
        "internal_dev_split_hash": dev_cache.manifest["effective_split_hash"],
        "quarantine_hash": train_cache.manifest["quarantine_hash"],
        "selected_epoch": selected["epoch"],
        "selected_budget": selected["selected_budget"],
        "selected_internal_dev_short_f1": selected["selected_short_f1"],
        "selection_rule": "max Short F1; within <0.25 choose smaller budget; then lower validation loss",
        "negative_analysis_counts": negative_label_counts(train_cache),
        "generator_checkpoint_hashes": checkpoint_hashes,
        "model_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "generator_trainable_parameters": sum(p.numel() for p in generator.parameters() if p.requires_grad),
        "history": history, "benchmark_accessed_during_training": False,
        "final_100_accessed": False, "final_100_runs": 0,
    }
    for directory in (best_dir, last_dir):
        (directory / "training_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n"
        )
    metadata = json.loads((best_dir / "checkpoint_metadata.json").read_text())
    metadata.update({
        "scorer_sha256": sha256_file(best_dir / "scorer.pt"),
        "selection": selected,
    })
    (best_dir / "checkpoint_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"selected": selected, "best_dir": str(best_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
