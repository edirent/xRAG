#!/usr/bin/env python
"""Train the locked depth-one residual token-state resampler."""

import argparse
import copy
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag import run_selector_calibration as selector
from scripts.packet_xrag import train_packet_projector as v1
from scripts.packet_xrag import train_residual_packet_projector as locked
from scripts.packet_xrag.token_resampler_common import (
    CachedTokenStateDataset, EXPECTED_SPLIT_HASH, POOLED_K2_F1,
    audit_checkpoints, load_frozen_k2_model, locked_records, make_cached_collator,
    retrieval_to_device, sha256_file, write_jsonl,
)
from src.language_modeling.utils import XRAG_TOKEN
from src.packet_xrag.encoding.token_state_cache import TokenStateCache, packet_key
from src.packet_xrag.encoding.token_state_resampler import (
    PooledResidualResamplerControl, ResidualTokenStateResampler,
)
from src.packet_xrag.modeling.token_state_xrag import install_token_state_injection


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--validation-samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--gradient-clipping", type=float, default=1.0)
    parser.add_argument("--latent-size", type=int, default=512)
    parser.add_argument("--num-latents", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--cache-dir", default="cache/token_states")
    parser.add_argument("--audit-json", default="cache/results/token_state_cache_audit.json")
    parser.add_argument("--output-dir", default="cache/resampler/token_state_depth1")
    parser.add_argument("--split-file", default="cache/projector/packet_projector_calibration/data_split.json")
    parser.add_argument("--v1-training-config", default="cache/projector/packet_projector_calibration/last/training_config.json")
    parser.add_argument("--v1-checkpoint", default="cache/projector/packet_projector_calibration/last/projector.pt")
    parser.add_argument("--k2-checkpoint", default="cache/projector/multi_token_k2/best_short_f1/multi_token_projector.pt")
    return parser.parse_args()


def pack_packet_keys(cache, keys, device):
    records = [cache.get(key) for key in keys]
    max_length = max(record["hidden"].shape[0] for record in records)
    hidden_size = records[0]["hidden"].shape[1]
    hidden = torch.zeros(len(records), max_length, hidden_size, dtype=torch.bfloat16)
    mask = torch.zeros(len(records), max_length, dtype=torch.bool)
    for index, record in enumerate(records):
        length = record["hidden"].shape[0]
        hidden[index, :length].copy_(record["hidden"])
        mask[index, :length].copy_(record["mask"])
    pooled = torch.stack([record["pooled"] for record in records])
    return retrieval_to_device({
        "token_states": hidden, "token_mask": mask, "pooled_embeddings": pooled,
    }, device)


@torch.inference_mode()
def validate_loss(model, loader, device):
    model.eval(); losses = []
    for batch in loader:
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
            retrieval_embeds=retrieval_to_device(batch["retrieval"], device),
        )
        losses.append(float(outputs.loss))
    return mean(losses)


@torch.inference_mode()
def validate_generation(model, tokenizer, cache, records, device, max_new_tokens):
    model.eval(); predictions = []
    for sample, gold, _ in records:
        prompt = v1.build_prompt(sample["question"], len(gold) * 2)
        tokenized = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        keys = [packet_key(packet["encoder_text"]) for packet in gold]
        retrieval = pack_packet_keys(cache, keys, device)
        generated = model.generate(
            input_ids=tokenized.input_ids,
            attention_mask=tokenized.attention_mask,
            retrieval_embeds=retrieval,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        new = generated[:, tokenized.input_ids.shape[1]:] if generated.shape[1] > tokenized.input_ids.shape[1] else generated
        raw = tokenizer.batch_decode(new, skip_special_tokens=False)[0]
        clean = selector.clean_prediction(raw)
        short = selector.extract_short_answer(raw) or "[EMPTY]"
        em, f1 = selector.score_prediction(short, sample["answer"])
        _, clean_f1 = selector.score_prediction(clean, sample["answer"])
        predictions.append({
            "sample_id": locked.sample_id(sample), "gold_answer": sample["answer"],
            "raw_prediction": raw, "clean_prediction": clean, "short_prediction": short,
            "short_em": em, "short_f1": f1, "clean_f1": clean_f1,
            "num_packets": len(gold), "tokens_per_packet": 2,
            "total_soft_tokens": 2 * len(gold), "is_empty": short == "[EMPTY]",
        })
    return {
        "validation_short_em": 100 * mean(row["short_em"] for row in predictions),
        "validation_short_f1": 100 * mean(row["short_f1"] for row in predictions),
        "validation_clean_f1": 100 * mean(row["clean_f1"] for row in predictions),
        "validation_empty_count": sum(row["is_empty"] for row in predictions),
        "average_selected_packets": mean(row["num_packets"] for row in predictions),
        "average_online_soft_tokens": mean(row["total_soft_tokens"] for row in predictions),
    }, predictions


def save_checkpoint(directory, model, config, history, metadata, predictions):
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {name: value.detach().cpu() for name, value in model.projector.state_dict().items()},
        directory / "resampler.pt",
    )
    (directory / "training_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (directory / "validation_metrics.json").write_text(json.dumps({"history": history}, indent=2, sort_keys=True) + "\n")
    (directory / "checkpoint_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    write_jsonl(directory / "validation_predictions.jsonl", predictions)


def main(memory_source="token_states"):
    args = parse_args()
    assert (args.train_samples, args.validation_samples, args.epochs, args.batch_size) == (5000, 500, 5, 8)
    assert (args.learning_rate, args.weight_decay, args.warmup_ratio, args.gradient_clipping) == (2e-4, 0.01, 0.05, 1.0)
    assert (args.latent_size, args.num_latents, args.num_heads, args.ffn_size) == (512, 2, 8, 2048)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device); torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats()
    hashes, _ = audit_checkpoints(args.v1_checkpoint, args.k2_checkpoint)
    audit = json.loads(Path(args.audit_json).read_text())
    online_reproduction = audit["pooled_k2_reproduction"]
    if abs(online_reproduction["validation_short_f1"] - POOLED_K2_F1) > 0.1:
        raise RuntimeError(f"online K2 reproduction gate failed: {online_reproduction}")
    cache = TokenStateCache(args.cache_dir)
    if cache.manifest["validation_split_hash"] != EXPECTED_SPLIT_HASH or cache.manifest["checkpoint_sha256"] != hashes:
        raise RuntimeError("token-state cache protocol mismatch")
    train_records, validation_records = locked_records(args.split_file, args.v1_training_config)
    tokenizer = AutoTokenizer.from_pretrained(
        v1.XRAG_MODEL_NAME, padding_side="left", add_eos_token=False, use_fast=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id or tokenizer.eos_token_id
    xrag_id = tokenizer.convert_tokens_to_ids(XRAG_TOKEN)
    model, config = load_frozen_k2_model(
        device, tokenizer, xrag_id, args.v1_checkpoint, args.k2_checkpoint
    )
    pooled_k2 = model.projector
    resampler_class = (ResidualTokenStateResampler if memory_source == "token_states"
                       else PooledResidualResamplerControl)
    model.projector = resampler_class(
        pooled_k2, config.retriever_hidden_size, config.hidden_size,
        args.latent_size, args.num_latents, args.num_heads, args.ffn_size,
    ).to(device=device, dtype=torch.bfloat16)
    install_token_state_injection(model)
    trainable = [parameter for parameter in model.projector.parameters() if parameter.requires_grad]
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable_names or not all(name.startswith("projector.") and not name.startswith("projector.pooled_k2_projector.") for name in trainable_names):
        raise RuntimeError(f"invalid trainable parameter audit: {trainable_names}")
    frozen_probe_name, frozen_probe_parameter = next((name, p) for name, p in model.named_parameters() if not p.requires_grad)
    frozen_probe = frozen_probe_parameter.detach().flatten()[:1024].cpu().clone()

    train_dataset = CachedTokenStateDataset(train_records, tokenizer, xrag_id, args.seed, training=True, tokens_per_packet=2)
    validation_dataset = CachedTokenStateDataset(validation_records, tokenizer, xrag_id, args.seed, training=False, tokens_per_packet=2)
    collator = make_cached_collator(tokenizer, cache)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed), collate_fn=collator)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    # The mandatory online K2 reproduction in the audit is checked above.  This
    # second pass audits the offline BF16 cache before optimizer creation.
    initial_generation, initial_predictions = validate_generation(
        model, tokenizer, cache, validation_records, device, args.max_new_tokens
    )
    probe = pack_packet_keys(cache, [packet_key(validation_records[0][1][0]["encoder_text"])], device)
    with torch.inference_mode():
        resampled = model.projector(probe["token_states"], probe["token_mask"], probe["pooled_embeddings"])
        pooled_only = model.projector.pooled_k2_projector(probe["pooled_embeddings"])
    zero_residual_max_abs = float((resampled - pooled_only).abs().max())
    if zero_residual_max_abs != 0.0:
        raise RuntimeError(f"zero-residual output is not exactly K2: {zero_residual_max_abs}")
    print(json.dumps({"online_pre_optimizer_k2_reproduction": online_reproduction,
                      "cached_zero_residual_generation": initial_generation,
                      "zero_residual_max_abs": zero_residual_max_abs}, sort_keys=True), flush=True)

    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_updates * args.warmup_ratio), total_updates
    )
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    run_config = vars(args) | {
        "prompt": "P2_SHORT", "answer_extraction": "run_selector_calibration.extract_short_answer",
        "variant_weights": v1.VARIANT_WEIGHTS, "validation_split_hash": EXPECTED_SPLIT_HASH,
        "checkpoint_sha256": hashes, "dtype": "bfloat16", "depth": 1,
        "memory_source": memory_source,
        "trainable_names": trainable_names,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "online_pre_optimizer_k2_reproduction": online_reproduction,
        "cached_zero_residual_generation": initial_generation,
        "cached_zero_residual_delta_from_online_f1": initial_generation["validation_short_f1"] - online_reproduction["validation_short_f1"],
        "zero_residual_max_abs": zero_residual_max_abs,
    }
    history = []; best_f1 = float("-inf"); best_nll = float("inf"); best_epoch = None
    global_update = 0; optimizer.zero_grad(set_to_none=True); started = time.perf_counter()
    for epoch in range(args.epochs):
        train_dataset.set_epoch(epoch); model.train(); running_loss = 0.0; variants = Counter()
        epoch_started = time.perf_counter()
        for micro_step, batch in enumerate(train_loader):
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
                retrieval_embeds=retrieval_to_device(batch["retrieval"], device),
            )
            loss = outputs.loss / args.gradient_accumulation
            loss.backward(); running_loss += float(loss.detach()) * args.gradient_accumulation
            variants.update(batch["variants"])
            if (micro_step + 1) % args.gradient_accumulation == 0 or micro_step + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clipping)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); global_update += 1
                if global_update % args.log_every == 0:
                    print(f"epoch={epoch+1} update={global_update}/{total_updates} loss={running_loss/(micro_step+1):.4f} lr={scheduler.get_last_lr()[0]:.2e}", flush=True)
        nll = validate_loss(model, validation_loader, device)
        generation, predictions = validate_generation(
            model, tokenizer, cache, validation_records, device, args.max_new_tokens
        )
        metrics = {
            "epoch": epoch + 1, "global_update": global_update,
            "train_loss": running_loss / len(train_loader), "validation_nll": nll,
            **generation, "variant_counts": dict(variants),
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(metrics)
        metadata = {"epoch": epoch + 1, "selection_metric": "validation_short_f1",
                    "tie_break_metric": "validation_nll", "validation_examples": 500}
        save_checkpoint(output_dir / "last", model, run_config, history, metadata, predictions)
        f1 = generation["validation_short_f1"]
        if f1 > best_f1 + 1e-8 or (abs(f1 - best_f1) <= 1e-8 and nll < best_nll - 1e-8):
            best_f1, best_nll, best_epoch = f1, nll, epoch + 1
            save_checkpoint(output_dir / "best_short_f1", model, run_config, history,
                            metadata | {"selected": True}, predictions)
        print(json.dumps(metrics, sort_keys=True), flush=True)

    if not torch.equal(dict(model.named_parameters())[frozen_probe_name].detach().flatten()[:1024].cpu(), frozen_probe):
        raise RuntimeError(f"frozen parameter changed: {frozen_probe_name}")
    if any(parameter.grad is not None for parameter in model.projector.pooled_k2_projector.parameters()):
        raise RuntimeError("frozen pooled K2 received gradients")
    summary = {
        "best_epoch": best_epoch, "best_validation_short_f1": best_f1,
        "best_validation_nll": best_nll,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "runtime_seconds": time.perf_counter() - started,
        "best_checkpoint_sha256": sha256_file(output_dir / "best_short_f1" / "resampler.pt"),
        "final_100_accessed": False,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
