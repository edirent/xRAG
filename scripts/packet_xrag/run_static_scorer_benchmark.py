#!/usr/bin/env python
"""Run frozen-K2 generation for STATIC_1..6 on a complete feature cache."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag import run_selector_calibration as selector
from scripts.packet_xrag import train_packet_projector as v1
from scripts.packet_xrag.run_k2_selector_benchmark import generate_xrag
from scripts.packet_xrag.token_resampler_common import (
    EXPECTED_K2_SHA256,
    EXPECTED_SPLIT_HASH,
    EXPECTED_V1_SHA256,
    audit_checkpoints,
    load_frozen_k2_model,
)
from src.language_modeling.utils import XRAG_TOKEN
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache
from src.packet_xrag.controller.static_scorer import (
    StaticPacketScorer,
    negative_analysis_labels,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="cache/controller/static/best_short_f1/scorer.pt")
    parser.add_argument("--training-config", default="cache/controller/static/best_short_f1/training_config.json")
    parser.add_argument("--feature-cache", default="cache/controller/features/benchmark_features")
    parser.add_argument("--k2-training-config", default="cache/projector/multi_token_k2/best_short_f1/training_config.json")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output", default="cache/controller/static/benchmark_predictions.jsonl")
    parser.add_argument("--metrics-output", default="cache/controller/static/benchmark_metrics.json")
    parser.add_argument("--allow-existing-output", action="store_true")
    return parser.parse_args(argv)


def load_static_scorer(checkpoint, device):
    model = StaticPacketScorer().to(device=device, dtype=torch.bfloat16)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    return model


@torch.inference_mode()
def rank_cache(cache, scorer, device, question_batch_size=32):
    rankings = {}
    for start in range(0, len(cache), question_batch_size):
        records = [cache[index] for index in
                   range(start, min(start + question_batch_size, len(cache)))]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            score_groups = scorer.score_records(records, device)
        for record, scores in zip(records, score_groups):
            scores = scores.float().cpu()
            rankings[record["sample_id"]] = sorted(
                range(len(scores)),
                key=lambda packet_id: (-float(scores[packet_id]), packet_id),
            )
    return rankings


def initialize_generator(k2_training_config, device):
    k2_config = json.loads(Path(k2_training_config).read_text())
    v1_path = k2_config["base_projector_checkpoint"]
    k2_path = str(Path(k2_config["output_dir"]) / "best_short_f1" / "multi_token_projector.pt")
    hashes, _ = audit_checkpoints(v1_path, k2_path)
    if hashes != {"v1": EXPECTED_V1_SHA256, "k2": EXPECTED_K2_SHA256}:
        raise RuntimeError("formal frozen K2 checkpoint mismatch")
    tokenizer = AutoTokenizer.from_pretrained(
        v1.XRAG_MODEL_NAME, padding_side="left", add_eos_token=False, use_fast=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id or tokenizer.eos_token_id
    xrag_id = tokenizer.convert_tokens_to_ids(XRAG_TOKEN)
    model, _ = load_frozen_k2_model(device, tokenizer, xrag_id, v1_path, k2_path)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("generator must remain frozen")
    return tokenizer, model, xrag_id, hashes


@torch.inference_mode()
def generate_xrag_batch(tokenizer, generator, xrag_id, questions, embeddings, device,
                        max_new_tokens):
    if len(questions) != len(embeddings) or not questions:
        raise ValueError("questions and retrieval batches must be non-empty and aligned")
    packet_counts = [len(item) for item in embeddings]
    prompts = [v1.build_prompt(question, count * 2)
               for question, count in zip(questions, packet_counts)]
    tokenized = tokenizer(
        prompts, return_tensors="pt", add_special_tokens=False, padding=True
    ).to(device)
    retrieval = torch.cat([item.to(device) for item in embeddings], dim=0)
    if int((tokenized.input_ids == xrag_id).sum()) != sum(packet_counts) * 2:
        raise RuntimeError("batched XRAG prompt/retrieval count mismatch")
    generated = generator.generate(
        input_ids=tokenized.input_ids, attention_mask=tokenized.attention_mask,
        retrieval_embeds=retrieval, do_sample=False, max_new_tokens=max_new_tokens,
        use_cache=True, pad_token_id=tokenizer.pad_token_id,
    )
    new = (generated[:, tokenized.input_ids.shape[1]:]
           if generated.shape[1] > tokenized.input_ids.shape[1] else generated)
    raws, generated_lengths = [], []
    for row in new:
        eos = (row == tokenizer.eos_token_id).nonzero(as_tuple=False)
        length = int(eos[0]) + 1 if len(eos) else len(row)
        raws.append(tokenizer.decode(row[:length], skip_special_tokens=False))
        generated_lengths.append(length)
    prompt_lengths = [int(value) for value in tokenized.attention_mask.sum(dim=1)]
    return raws, prompt_lengths, generated_lengths


@torch.inference_mode()
def evaluate_generator(cache, rankings, tokenizer, generator, xrag_id, device,
                       output_path, max_new_tokens=32, log_every=50,
                       generation_batch_size=16):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with output_path.open("w") as stream:
        for budget in range(1, 7):
            for start in range(0, len(cache), generation_batch_size):
                records = [cache[index] for index in
                           range(start, min(start + generation_batch_size, len(cache)))]
                selected_groups = [rankings[record["sample_id"]][:budget]
                                   for record in records]
                embedding_groups = [record["packet_embeddings"][selected]
                                    for record, selected in zip(records, selected_groups)]
                raws, prompt_lengths, generated_lengths = generate_xrag_batch(
                    tokenizer, generator, xrag_id,
                    [record["question"] for record in records], embedding_groups,
                    device, max_new_tokens,
                )
                for record, selected, raw, prompt_tokens, generated_tokens in zip(
                        records, selected_groups, raws, prompt_lengths, generated_lengths):
                    gold = set(record["gold_packet_ids"])
                    negative_labels = negative_analysis_labels(
                        record["packets"], record["gold_packet_ids"],
                        record["topk_ranking"],
                    )
                    clean = selector.clean_prediction(raw)
                    short = selector.extract_short_answer(raw) or "[EMPTY]"
                    em, f1 = selector.score_prediction(short, record["answer"])
                    selected_set = set(selected)
                    row = {
                        "sample_id": record["sample_id"], "question": record["question"],
                        "gold_answer": record["answer"],
                        "configuration": f"STATIC_{budget}",
                        "budget": budget, "selected_packet_ids": selected,
                        "selected_negative_labels": {
                            str(packet_id): negative_labels[packet_id]
                            for packet_id in selected if packet_id in negative_labels
                        },
                        "gold_packet_ids": record["gold_packet_ids"],
                        "short_prediction": short, "clean_prediction": clean,
                        "raw_generation": raw, "short_em": em, "short_f1": f1,
                        "support_recall": len(gold & selected_set) / len(gold),
                        "full_support_coverage": float(gold.issubset(selected_set)),
                        "num_packets": len(selected),
                        "total_soft_tokens": len(selected) * 2,
                        "prompt_tokens": prompt_tokens,
                        "generated_tokens": generated_tokens,
                        "is_empty": int(short == "[EMPTY]"),
                    }
                    rows.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                completed = min(start + generation_batch_size, len(cache))
                if completed % log_every == 0 or completed == len(cache):
                    print(
                        f"STATIC_{budget} generation: {completed}/{len(cache)}",
                        flush=True,
                    )
    return rows


def summarize_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["configuration"]].append(row)
    return {
        name: {
            "samples": len(items),
            "short_em": 100 * mean(item["short_em"] for item in items),
            "short_f1": 100 * mean(item["short_f1"] for item in items),
            "support_recall": mean(item["support_recall"] for item in items),
            "full_support_coverage": mean(item["full_support_coverage"] for item in items),
            "empty": sum(item["is_empty"] for item in items),
        }
        for name, items in sorted(grouped.items())
    }


@torch.inference_mode()
def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output)
    if output.exists() and not args.allow_existing_output:
        raise RuntimeError(f"refusing to rerun or overwrite formal benchmark output: {output}")
    if args.max_new_tokens != 32:
        raise RuntimeError("generation protocol requires max_new_tokens=32")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    cache = ControllerFeatureCache(args.feature_cache)
    if ("benchmark_features" in str(args.feature_cache) and
            cache.manifest["effective_split_hash"] != EXPECTED_SPLIT_HASH):
        raise RuntimeError("frozen benchmark feature hash mismatch")
    scorer = load_static_scorer(args.checkpoint, device)
    rankings = rank_cache(cache, scorer, device)
    tokenizer, generator, xrag_id, hashes = initialize_generator(
        args.k2_training_config, device
    )
    rows = evaluate_generator(
        cache, rankings, tokenizer, generator, xrag_id, device, output,
        args.max_new_tokens,
    )
    training_config = json.loads(Path(args.training_config).read_text())
    payload = {
        "split_hash": cache.manifest["effective_split_hash"],
        "samples": len(cache), "metrics": summarize_rows(rows),
        "frozen_internal_dev_selected_budget": training_config["selected_budget"],
        "frozen_internal_dev_selected_epoch": training_config["selected_epoch"],
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_hashes": hashes, "benchmark_runs": 1,
        "final_100_accessed": False, "final_100_runs": 0,
    }
    Path(args.metrics_output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
