#!/usr/bin/env python
"""Train only a residual adapter on top of the calibrated packet projector."""

import argparse
import copy
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag import run_selector_calibration as selector
from scripts.packet_xrag import train_packet_projector as v1
from src.language_modeling.utils import XRAG_TOKEN
from src.model import SFR, XMistralForCausalLM
from src.packet_xrag.encoding.residual_projector import ResidualPacketProjector


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
    parser.add_argument("--max-packets", type=int, default=4)
    parser.add_argument("--bottleneck-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--base-projector-checkpoint",
        default="cache/projector/packet_projector_calibration/last/projector.pt",
    )
    parser.add_argument(
        "--v1-training-config",
        default="cache/projector/packet_projector_calibration/last/training_config.json",
    )
    parser.add_argument(
        "--data-split",
        default="cache/projector/packet_projector_calibration/data_split.json",
    )
    parser.add_argument(
        "--output-dir", default="cache/projector/residual_packet_projector"
    )
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def sample_id(sample):
    return str(sample.get("id", sample.get("_id")))


def load_locked_split(args):
    split_path = Path(args.data_split)
    v1_config = json.loads(Path(args.v1_training_config).read_text())
    recovered = {
        "train_sample_ids": v1_config["train_sample_ids"],
        "validation_sample_ids": v1_config["validation_sample_ids"],
        "seed": v1_config["seed"],
        "split_policy": v1_config["split_policy"],
    }
    if split_path.exists():
        locked = json.loads(split_path.read_text())
        if locked != recovered:
            raise ValueError(f"locked split differs from V1 training record: {split_path}")
    else:
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(json.dumps(recovered, indent=2) + "\n")

    train_ids = recovered["train_sample_ids"]
    validation_ids = recovered["validation_sample_ids"]
    if len(train_ids) != args.train_samples or len(validation_ids) != args.validation_samples:
        raise ValueError("requested sample counts do not match the locked V1 split")
    if recovered["seed"] != args.seed:
        raise ValueError("seed does not match V1")
    all_ids = train_ids + validation_ids
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("duplicate sample IDs in locked split")

    wanted = set(all_ids)
    found = {}
    dataset = load_dataset(
        "hotpotqa/hotpot_qa", "distractor", split="train", trust_remote_code=True
    )
    for sample in dataset:
        sid = sample_id(sample)
        if sid not in wanted:
            continue
        gold, distractors = v1.supporting_and_distractor_packets(sample)
        if not 1 <= len(gold) <= args.max_packets:
            raise ValueError(f"locked sample is no longer eligible: {sid}")
        found[sid] = (sample, gold, distractors)
        if len(found) == len(wanted):
            break
    missing = wanted - set(found)
    if missing:
        raise ValueError(f"could not recover {len(missing)} locked samples")
    return [found[sid] for sid in train_ids], [found[sid] for sid in validation_ids]


@torch.no_grad()
def validate_generation(model, tokenizer, sfr_tokenizer, sfr_model, records, device, max_new_tokens):
    model.eval()
    clean_f1s, short_ems, short_f1s = [], [], []
    for sample, gold, _ in records:
        prompt = v1.build_prompt(sample["question"], len(gold))
        tokenized = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        retrieval = v1.encode_packets(
            sfr_tokenizer, sfr_model, [packet["encoder_text"] for packet in gold], device
        )
        generated = model.generate(
            input_ids=tokenized.input_ids,
            attention_mask=tokenized.attention_mask,
            retrieval_embeds=retrieval,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        raw = tokenizer.batch_decode(generated, skip_special_tokens=False)[0]
        clean = selector.clean_prediction(raw)
        short = selector.extract_short_answer(raw)
        _, clean_f1 = selector.score_prediction(clean, sample["answer"])
        short_em, short_f1 = selector.score_prediction(short, sample["answer"])
        clean_f1s.append(clean_f1)
        short_ems.append(short_em)
        short_f1s.append(short_f1)
    count = len(records)
    return {
        "validation_short_em": 100 * sum(short_ems) / count,
        "validation_short_f1": 100 * sum(short_f1s) / count,
        "validation_clean_f1": 100 * sum(clean_f1s) / count,
    }


def save_checkpoint(directory, model, config, history, metadata):
    directory.mkdir(parents=True, exist_ok=True)
    state = {
        name: tensor.detach().cpu() for name, tensor in model.projector.state_dict().items()
    }
    torch.save(state, directory / "projector.pt")
    (directory / "training_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    (directory / "validation_metrics.json").write_text(
        json.dumps({"history": history}, indent=2, sort_keys=True) + "\n"
    )
    (directory / "checkpoint_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def main():
    args = parse_args()
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert args.max_packets == 4
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained(
        v1.XRAG_MODEL_NAME, padding_side="left", add_eos_token=False, use_fast=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = (
            tokenizer.unk_token_id
            if tokenizer.unk_token_id is not None
            else tokenizer.eos_token_id
        )
    xrag_token_id = tokenizer.convert_tokens_to_ids(XRAG_TOKEN)
    sfr_tokenizer = AutoTokenizer.from_pretrained(v1.SFR_MODEL_NAME)
    sfr_model = SFR.from_pretrained(
        v1.SFR_MODEL_NAME, torch_dtype=torch.bfloat16
    ).eval().to(device)
    for parameter in sfr_model.parameters():
        parameter.requires_grad = False

    config = AutoConfig.from_pretrained(v1.XRAG_MODEL_NAME)
    model = XMistralForCausalLM.from_pretrained(
        v1.XRAG_MODEL_NAME,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.set_xrag_token_id(xrag_token_id)
    calibrated_state = torch.load(
        args.base_projector_checkpoint, map_location="cpu", weights_only=True
    )
    model.projector.load_state_dict(calibrated_state, strict=True)
    calibrated_snapshot = copy.deepcopy(calibrated_state)
    for parameter in model.parameters():
        parameter.requires_grad = False
    residual_projector = ResidualPacketProjector(
        model.projector,
        hidden_size=config.hidden_size,
        bottleneck_size=args.bottleneck_size,
    ).to(device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        probe = torch.randn(2, config.retriever_hidden_size, device=device, dtype=torch.bfloat16)
        base_output = residual_projector.base_projector(probe)
        v2_output = residual_projector(probe)
    assert torch.allclose(base_output, v2_output, atol=1e-5, rtol=1e-5)
    model.projector = residual_projector

    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable_names
    assert all(
        name.startswith("projector.input_norm.") or name.startswith("projector.adapter.")
        for name in trainable_names
    )
    assert all(not p.requires_grad for p in model.projector.base_projector.parameters())
    assert all(not p.requires_grad for p in sfr_model.parameters())

    print("Loading exact V1 HotpotQA train/projector-validation split", flush=True)
    train_records, validation_records = load_locked_split(args)
    train_dataset = v1.ProjectorDataset(
        train_records, tokenizer, xrag_token_id, args.seed, training=True
    )
    validation_dataset = v1.ProjectorDataset(
        validation_records, tokenizer, xrag_token_id, args.seed, training=False
    )
    collator = v1.make_collator(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * args.warmup_ratio),
        num_training_steps=total_updates,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = vars(args) | {
        "sfr_model": v1.SFR_MODEL_NAME,
        "xrag_model": v1.XRAG_MODEL_NAME,
        "prompt": "P2_SHORT",
        "variant_weights": v1.VARIANT_WEIGHTS,
        "dtype": "bfloat16",
        "trainable_parameters": sum(p.numel() for p in trainable),
        "trainable_names": trainable_names,
        "data_split": args.data_split,
    }

    total_parameters = sum(p.numel() for p in model.parameters()) + sum(
        p.numel() for p in sfr_model.parameters()
    )
    frozen_base_parameters = sum(p.numel() for p in model.projector.base_projector.parameters())
    frozen_sfr_parameters = sum(p.numel() for p in sfr_model.parameters())
    frozen_llm_parameters = sum(
        p.numel() for name, p in model.named_parameters() if not name.startswith("projector.")
    )
    print(f"Total model parameters: {total_parameters}", flush=True)
    print(f"Trainable parameters: {sum(p.numel() for p in trainable)}", flush=True)
    print(f"Frozen base projector parameters: {frozen_base_parameters}", flush=True)
    print(f"Frozen SFR parameters: {frozen_sfr_parameters}", flush=True)
    print(f"Frozen LLM parameters: {frozen_llm_parameters}", flush=True)

    history = []
    best_short_f1 = float("-inf")
    best_validation_nll = float("inf")
    best_epoch = None
    optimizer.zero_grad(set_to_none=True)
    global_update = 0
    start = time.perf_counter()
    for epoch in range(args.epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        variant_counts = Counter()
        running_loss = 0.0
        for micro_step, batch in enumerate(train_loader):
            retrieval = v1.encode_packets(
                sfr_tokenizer, sfr_model, batch["packet_texts"], device
            )
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
                retrieval_embeds=retrieval,
            )
            loss = outputs.loss / args.gradient_accumulation
            loss.backward()
            running_loss += float(loss.detach()) * args.gradient_accumulation
            variant_counts.update(batch["variants"])
            should_step = (
                (micro_step + 1) % args.gradient_accumulation == 0
                or micro_step + 1 == len(train_loader)
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clipping)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1
                if global_update % args.log_every == 0:
                    print(
                        f"epoch={epoch + 1} update={global_update}/{total_updates} "
                        f"loss={running_loss / (micro_step + 1):.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e}",
                        flush=True,
                    )

        validation_nll = v1.validate_loss(
            model, sfr_tokenizer, sfr_model, validation_loader, device
        )
        generation_metrics = validate_generation(
            model,
            tokenizer,
            sfr_tokenizer,
            sfr_model,
            validation_records,
            device,
            args.max_new_tokens,
        )
        metrics = {
            "epoch": epoch + 1,
            "train_loss": running_loss / len(train_loader),
            "validation_nll": validation_nll,
            **generation_metrics,
            "variant_counts": dict(variant_counts),
            "global_update": global_update,
        }
        history.append(metrics)
        metadata = {
            "epoch": epoch + 1,
            "selection_metric": "validation_short_f1",
            "tie_break_metric": "validation_nll",
            "base_projector_checkpoint": args.base_projector_checkpoint,
        }
        save_checkpoint(output_dir / "last", model, run_config, history, metadata)
        f1 = metrics["validation_short_f1"]
        is_best = f1 > best_short_f1 + 1e-8 or (
            abs(f1 - best_short_f1) <= 1e-8
            and validation_nll < best_validation_nll - 1e-8
        )
        if is_best:
            best_short_f1 = f1
            best_validation_nll = validation_nll
            best_epoch = epoch + 1
            save_checkpoint(
                output_dir / "best_short_f1",
                model,
                run_config,
                history,
                metadata | {"selected": True},
            )
        print(json.dumps(metrics, sort_keys=True), flush=True)

    current_base = model.projector.base_projector.state_dict()
    assert all(
        torch.equal(calibrated_snapshot[name], current_base[name].detach().cpu())
        for name in calibrated_snapshot
    )
    assert all(parameter.grad is None for parameter in model.projector.base_projector.parameters())
    assert all(parameter.grad is None for parameter in sfr_model.parameters())
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not name.startswith("projector.")
    )
    print(f"Peak allocated VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.3f} GB", flush=True)
    print(f"Best epoch: {best_epoch}", flush=True)
    print(f"Best validation Short F1: {best_short_f1:.6f}", flush=True)
    print(f"Best validation NLL: {best_validation_nll:.6f}", flush=True)
    print(f"Last validation Short F1: {history[-1]['validation_short_f1']:.6f}", flush=True)
    print(f"Runtime seconds: {time.perf_counter() - start:.3f}", flush=True)


if __name__ == "__main__":
    main()
