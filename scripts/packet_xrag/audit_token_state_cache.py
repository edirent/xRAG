#!/usr/bin/env python
"""Audit locked protocol, checkpoints, K2 reproduction, and token-cache feasibility."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag import train_multi_token_projector as multi
from scripts.packet_xrag import train_packet_projector as v1
from scripts.packet_xrag.token_resampler_common import (
    EXPECTED_SPLIT_HASH, POOLED_K2_F1, audit_checkpoints, collect_used_packets,
    load_frozen_k2_model, locked_records, write_jsonl,
)
from src.language_modeling.utils import XRAG_TOKEN
from src.model import SFR
from src.model.SFR.modeling_sfr import last_token_pool


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split-file", default="cache/projector/packet_projector_calibration/data_split.json")
    parser.add_argument("--v1-training-config", default="cache/projector/packet_projector_calibration/last/training_config.json")
    parser.add_argument("--v1-checkpoint", default="cache/projector/packet_projector_calibration/last/projector.pt")
    parser.add_argument("--k2-checkpoint", default="cache/projector/multi_token_k2/best_short_f1/multi_token_projector.pt")
    parser.add_argument("--output-json", default="cache/results/token_state_cache_audit.json")
    parser.add_argument("--output-md", default="cache/results/token_state_cache_audit.md")
    parser.add_argument("--baseline-predictions", default="cache/results/token_resampler_pooled_k2_reproduction.jsonl")
    return parser.parse_args()


def quantile(values, probability):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(probability * (len(ordered) - 1)))]


@torch.inference_mode()
def main():
    args = parse_args()
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    hashes, config = audit_checkpoints(args.v1_checkpoint, args.k2_checkpoint)
    train, validation = locked_records(args.split_file, args.v1_training_config)
    inventory = collect_used_packets(train, validation)

    device = torch.device(args.device); torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        v1.XRAG_MODEL_NAME, padding_side="left", add_eos_token=False, use_fast=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id or tokenizer.eos_token_id
    xrag_id = tokenizer.convert_tokens_to_ids(XRAG_TOKEN)
    sfr_tokenizer = AutoTokenizer.from_pretrained(v1.SFR_MODEL_NAME)
    sfr = SFR.from_pretrained(v1.SFR_MODEL_NAME, torch_dtype=torch.bfloat16).eval().to(device)
    for parameter in sfr.parameters(): parameter.requires_grad = False

    # This fresh full-500 generation is the mandatory reproduction before extraction.
    model, _ = load_frozen_k2_model(
        device, tokenizer, xrag_id, args.v1_checkpoint, args.k2_checkpoint
    )
    baseline, predictions = multi.validate_generation(
        model, tokenizer, sfr_tokenizer, sfr, validation, device, 2, 32
    )
    if abs(baseline["validation_short_f1"] - POOLED_K2_F1) > 0.1:
        raise RuntimeError(f"K2 reproduction failed: {baseline}")
    write_jsonl(args.baseline_predictions, predictions)
    del model; torch.cuda.empty_cache()

    packets = inventory["union"]
    texts = [packets[key]["encoder_text"] for key in sorted(packets)]
    lengths, truncated = [], 0
    for start in range(0, len(texts), 512):
        tokenized = sfr_tokenizer(
            texts[start:start + 512], max_length=180, truncation=True,
            add_special_tokens=True, return_length=True
        )
        lengths.extend(tokenized["length"])
        raw = sfr_tokenizer(texts[start:start + 512], add_special_tokens=True, return_length=True)["length"]
        truncated += sum(length > 180 for length in raw)
    estimated_bytes = sum(length * config.retriever_hidden_size * 2 + config.retriever_hidden_size * 2 + length * 5 for length in lengths)
    if estimated_bytes >= 40 * 1024**3:
        raise RuntimeError(f"estimated cache exceeds 40 GiB: {estimated_bytes / 1024**3:.3f}")

    fixed = texts[:100]
    tokenized = sfr_tokenizer(fixed, max_length=180, padding=True, truncation=True, return_tensors="pt").to(device)
    torch.cuda.synchronize(); started = time.perf_counter()
    outputs = sfr(input_ids=tokenized.input_ids, attention_mask=tokenized.attention_mask)
    torch.cuda.synchronize(); elapsed = time.perf_counter() - started
    pooled_direct = last_token_pool(outputs.last_hidden_state, tokenized.attention_mask)
    pooled_api = sfr.get_doc_embedding(tokenized.input_ids, tokenized.attention_mask)
    pooling_max_abs = float((pooled_direct - pooled_api).abs().max())
    pooling_consistent = bool(torch.allclose(pooled_direct, pooled_api, atol=1e-4, rtol=1e-4))
    if not pooling_consistent:
        raise RuntimeError(f"SFR pooling consistency failed: max abs {pooling_max_abs}")

    result = {
        "validation_split_hash": EXPECTED_SPLIT_HASH,
        "checkpoint_sha256": hashes,
        "base_model": v1.XRAG_MODEL_NAME,
        "sfr_model": v1.SFR_MODEL_NAME,
        "base_model_compatible": True,
        "pooled_k2_reproduction": baseline,
        "unique_train_packets": len(inventory["train"]),
        "unique_validation_packets": len(inventory["validation"]),
        "unique_union_packets": len(inventory["union"]),
        "train_packet_uses_five_epochs": inventory["train_packet_uses"],
        "validation_packet_uses": inventory["validation_packet_uses"],
        "mean_tokens_per_packet": statistics.mean(lengths),
        "token_length_p50": quantile(lengths, 0.50),
        "token_length_p90": quantile(lengths, 0.90),
        "token_length_p95": quantile(lengths, 0.95),
        "token_length_p99": quantile(lengths, 0.99),
        "max_tokens": max(lengths),
        "truncated_packets": truncated,
        "truncation_rate": truncated / len(lengths),
        "estimated_cache_bytes": estimated_bytes,
        "estimated_cache_gib": estimated_bytes / 1024**3,
        "pooling_consistency_samples": 100,
        "pooling_consistent": pooling_consistent,
        "pooling_max_abs_error": pooling_max_abs,
        "sfr_encoding_packets_per_second": len(fixed) / elapsed,
        "sfr_encoding_elapsed_seconds": elapsed,
        "online_soft_tokens_per_packet": 2,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = ["# Token-state Cache Feasibility Audit", ""] + [
        f"- {key}: {value}" for key, value in result.items()
    ]
    Path(args.output_md).write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

