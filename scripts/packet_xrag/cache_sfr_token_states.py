#!/usr/bin/env python
"""Create deterministic memory-mapped BF16 SFR token-state shards."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag import train_packet_projector as v1
from scripts.packet_xrag.token_resampler_common import (
    EXPECTED_SPLIT_HASH, POOLED_K2_F1, audit_checkpoints, collect_used_packets,
    locked_records,
)
from src.model import SFR
from src.model.SFR.modeling_sfr import last_token_pool
from src.packet_xrag.encoding.token_state_cache import TokenStateShardWriter


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--target-shard-gib", type=float, default=1.0)
    parser.add_argument("--cache-dir", default="cache/token_states")
    parser.add_argument("--audit-json", default="cache/results/token_state_cache_audit.json")
    parser.add_argument("--split-file", default="cache/projector/packet_projector_calibration/data_split.json")
    parser.add_argument("--v1-training-config", default="cache/projector/packet_projector_calibration/last/training_config.json")
    parser.add_argument("--v1-checkpoint", default="cache/projector/packet_projector_calibration/last/projector.pt")
    parser.add_argument("--k2-checkpoint", default="cache/projector/multi_token_k2/best_short_f1/multi_token_projector.pt")
    return parser.parse_args()


def span_metadata(text, offsets, ids, special_ids):
    marker = text.find("] ")
    title_end = marker + 1 if marker >= 0 else 0
    sentence_start = marker + 2 if marker >= 0 else 0
    title_positions, sentence_positions, special_positions = [], [], []
    for index, ((start, stop), token_id) in enumerate(zip(offsets, ids)):
        if (start, stop) == (0, 0) or int(token_id) in special_ids:
            special_positions.append(index)
        elif start < title_end and stop > 0:
            title_positions.append(index)
        elif stop > sentence_start:
            sentence_positions.append(index)
    return {
        "text": text,
        "title_span": [min(title_positions), max(title_positions) + 1] if title_positions else [0, 0],
        "sentence_span": [min(sentence_positions), max(sentence_positions) + 1] if sentence_positions else [0, 0],
        "special_positions": special_positions,
    }


@torch.inference_mode()
def main():
    args = parse_args()
    audit = json.loads(Path(args.audit_json).read_text())
    if abs(audit["pooled_k2_reproduction"]["validation_short_f1"] - POOLED_K2_F1) > 0.1:
        raise RuntimeError("K2 was not reproduced within tolerance before cache extraction")
    hashes, config = audit_checkpoints(args.v1_checkpoint, args.k2_checkpoint)
    if hashes != audit["checkpoint_sha256"] or audit["validation_split_hash"] != EXPECTED_SPLIT_HASH:
        raise RuntimeError("audit/checkpoint protocol mismatch")
    train, validation = locked_records(args.split_file, args.v1_training_config)
    inventory = collect_used_packets(train, validation)
    ordered = [(key, inventory["union"][key]) for key in sorted(inventory["union"])]

    device = torch.device(args.device); torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(v1.SFR_MODEL_NAME, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("a fast SFR tokenizer is required for exact span offsets")
    model = SFR.from_pretrained(v1.SFR_MODEL_NAME, torch_dtype=torch.bfloat16).eval().to(device)
    for parameter in model.parameters(): parameter.requires_grad = False
    writer = TokenStateShardWriter(
        args.cache_dir, config.retriever_hidden_size,
        target_bytes=int(args.target_shard_gib * 1024**3),
    )
    started = time.perf_counter(); encoded = 0
    special_ids = set(tokenizer.all_special_ids)
    for batch_start in range(0, len(ordered), args.batch_size):
        batch = ordered[batch_start:batch_start + args.batch_size]
        texts = [packet["encoder_text"] for _, packet in batch]
        encoded_batch = tokenizer(
            texts, max_length=180, padding=True, truncation=True,
            return_tensors="pt", return_offsets_mapping=True,
        )
        offsets = encoded_batch.pop("offset_mapping")
        inputs = {name: value.to(device) for name, value in encoded_batch.items()}
        outputs = model(**inputs)
        pooled = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
        for row, (key, packet) in enumerate(batch):
            keep = inputs["attention_mask"][row].bool()
            hidden = outputs.last_hidden_state[row][keep]
            ids = inputs["input_ids"][row][keep]
            mask = torch.ones(hidden.shape[0], dtype=torch.bool)
            row_offsets = offsets[row][encoded_batch["attention_mask"][row].bool()].tolist()
            metadata = span_metadata(texts[row], row_offsets, ids.detach().cpu().tolist(), special_ids)
            metadata.update({"title": packet["title"], "sentence": packet["text"], "truncated": hidden.shape[0] == 180})
            writer.add(key, hidden, pooled[row], ids, mask, metadata)
            encoded += 1
        if encoded % 256 < args.batch_size:
            print(f"encoded={encoded}/{len(ordered)}", flush=True)
    torch.cuda.synchronize(); elapsed = time.perf_counter() - started
    manifest = writer.close({
        "validation_split_hash": EXPECTED_SPLIT_HASH,
        "checkpoint_sha256": hashes,
        "sfr_model": v1.SFR_MODEL_NAME,
        "max_length": 180,
        "dtype": "bfloat16",
        "num_records": len(ordered),
        "unique_train_packets": len(inventory["train"]),
        "unique_validation_packets": len(inventory["validation"]),
        "encoding_seconds": elapsed,
        "encoding_packets_per_second": encoded / elapsed,
    })
    actual = sum(path.stat().st_size for path in Path(args.cache_dir).glob("*"))
    metrics = {"actual_cache_bytes": actual, "actual_cache_gib": actual / 1024**3,
               "encoding_packets_per_second": encoded / elapsed, "num_shards": len(manifest["shards"])}
    (Path(args.cache_dir) / "cache_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
