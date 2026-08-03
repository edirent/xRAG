#!/usr/bin/env python
"""Train LoRA adapters only in the early decoder layers and enforce the Stage-A gate."""

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag import train_packet_projector as v1
from scripts.packet_xrag import train_residual_packet_projector as residual
from src.language_modeling.utils import XRAG_TOKEN
from src.model import SFR, XMistralForCausalLM
from src.packet_xrag.encoding.residual_projector import ResidualPacketProjector


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        self.base = base
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.lora_a.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_b.to(device=base.weight.device, dtype=base.weight.dtype)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, inputs):
        return self.base(inputs) + self.lora_b(self.lora_a(self.dropout(inputs))) * self.scaling


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--validation-samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--gradient-clipping", type=float, default=1.0)
    parser.add_argument("--early-layers", type=int, default=8)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--max-packets", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--gate-short-f1", type=float, default=66.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--output-dir", default="cache/lora/early_layer_lora")
    parser.add_argument("--data-split", default="cache/projector/packet_projector_calibration/data_split.json")
    parser.add_argument("--v1-training-config", default="cache/projector/packet_projector_calibration/last/training_config.json")
    parser.add_argument("--projector-checkpoint", default="cache/projector/residual_packet_projector/best_short_f1/projector.pt")
    return parser.parse_args()


def install_lora(model, layer_count, targets, rank, alpha, dropout):
    if not 1 <= layer_count <= len(model.model.layers):
        raise ValueError("early-layers must select a non-empty decoder prefix")
    replaced = []
    for layer_index in range(layer_count):
        attention = model.model.layers[layer_index].self_attn
        for target in targets:
            module = getattr(attention, target, None)
            if not isinstance(module, nn.Linear):
                raise ValueError(f"unsupported LoRA target: layer {layer_index} {target}")
            setattr(attention, target, LoRALinear(module, rank, alpha, dropout))
            replaced.append(f"model.layers.{layer_index}.self_attn.{target}")
    return replaced


def adapter_state(model):
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if ".lora_a." in name or ".lora_b." in name
    }


def save_checkpoint(directory, model, config, history, metadata):
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(adapter_state(model), directory / "adapter.pt")
    (directory / "training_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (directory / "validation_metrics.json").write_text(json.dumps({"history": history}, indent=2, sort_keys=True) + "\n")
    (directory / "checkpoint_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def write_gate(output_dir, best_epoch, best_f1, best_nll, threshold):
    passed = best_f1 >= threshold
    text = "\n".join([
        "# Early-layer LoRA Internal Gate", "",
        f"- Best epoch: {best_epoch}",
        f"- Best validation Short F1: {best_f1:.6f}",
        f"- Best validation NLL: {best_nll:.6f}",
        f"- Required validation Short F1: {threshold}",
        f"- Passed: {'Yes' if passed else 'No'}", "",
        ("Internal gate passed. The final 100 samples remain untouched in this Stage-A run."
         if passed else "Gate not reached. Stop here and do not evaluate the final 100 samples."), ""
    ])
    (output_dir / "internal_gate_decision.md").write_text(text)
    return passed


def main():
    args = parse_args()
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert args.train_samples == 5000 and args.validation_samples == 500
    assert args.max_packets == 4
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained(v1.XRAG_MODEL_NAME, padding_side="left", add_eos_token=False, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.eos_token_id
    xrag_token_id = tokenizer.convert_tokens_to_ids(XRAG_TOKEN)
    sfr_tokenizer = AutoTokenizer.from_pretrained(v1.SFR_MODEL_NAME)
    sfr_model = SFR.from_pretrained(v1.SFR_MODEL_NAME, torch_dtype=torch.bfloat16).eval().to(device)
    for parameter in sfr_model.parameters():
        parameter.requires_grad = False

    config = AutoConfig.from_pretrained(v1.XRAG_MODEL_NAME)
    model = XMistralForCausalLM.from_pretrained(v1.XRAG_MODEL_NAME, config=config, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device)
    model.set_xrag_token_id(xrag_token_id)
    projector = ResidualPacketProjector(model.projector, hidden_size=config.hidden_size, bottleneck_size=1024)
    projector.load_state_dict(torch.load(args.projector_checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.projector = projector.to(device=device, dtype=torch.bfloat16)
    for parameter in model.parameters():
        parameter.requires_grad = False
    replaced = install_lora(model, args.early_layers, args.target_modules, args.rank, args.lora_alpha, args.lora_dropout)
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable_names and all(".lora_" in name for name in trainable_names)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    locked_args = argparse.Namespace(**vars(args))
    train_records, validation_records = residual.load_locked_split(locked_args)
    train_dataset = v1.ProjectorDataset(train_records, tokenizer, xrag_token_id, args.seed, training=True)
    validation_dataset = v1.ProjectorDataset(validation_records, tokenizer, xrag_token_id, args.seed, training=False)
    collator = v1.make_collator(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed), collate_fn=collator)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_updates * args.warmup_ratio), total_updates)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = vars(args) | {
        "prompt": "P2_SHORT", "variant_weights": v1.VARIANT_WEIGHTS,
        "answer_extraction": "run_selector_calibration.extract_short_answer",
        "selection_metric": "validation_short_f1_on_all_500",
        "selected_layers": list(range(args.early_layers)), "replaced_modules": replaced,
        "trainable_names": trainable_names, "trainable_parameters": sum(p.numel() for p in trainable),
        "final_100_accessed": False, "dtype": "bfloat16",
    }
    (output_dir / "stage_a_protocol.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n")

    history, best_f1, best_nll, best_epoch = [], float("-inf"), float("inf"), None
    global_update = 0
    optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter()
    for epoch in range(args.epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        running_loss, variant_counts = 0.0, Counter()
        for micro_step, batch in enumerate(train_loader):
            retrieval = v1.encode_packets(sfr_tokenizer, sfr_model, batch["packet_texts"], device)
            outputs = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device), labels=batch["labels"].to(device), retrieval_embeds=retrieval, use_cache=False)
            loss = outputs.loss / args.gradient_accumulation
            loss.backward()
            running_loss += float(loss.detach()) * args.gradient_accumulation
            variant_counts.update(batch["variants"])
            if (micro_step + 1) % args.gradient_accumulation == 0 or micro_step + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clipping)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                global_update += 1
                if global_update % args.log_every == 0:
                    print(f"epoch={epoch + 1} update={global_update}/{total_updates} loss={running_loss/(micro_step+1):.4f} lr={scheduler.get_last_lr()[0]:.2e}", flush=True)
        validation_nll = v1.validate_loss(model, sfr_tokenizer, sfr_model, validation_loader, device)
        generation = residual.validate_generation(model, tokenizer, sfr_tokenizer, sfr_model, validation_records, device, args.max_new_tokens)
        metrics = {"epoch": epoch + 1, "global_update": global_update, "train_loss": running_loss / len(train_loader), "validation_nll": validation_nll, **generation, "variant_counts": dict(variant_counts)}
        history.append(metrics)
        metadata = {"epoch": epoch + 1, "selection_metric": "validation_short_f1", "tie_break_metric": "validation_nll", "validation_examples": 500}
        save_checkpoint(output_dir / "last", model, run_config, history, metadata)
        f1 = metrics["validation_short_f1"]
        if f1 > best_f1 + 1e-8 or (abs(f1 - best_f1) <= 1e-8 and validation_nll < best_nll - 1e-8):
            best_f1, best_nll, best_epoch = f1, validation_nll, epoch + 1
            save_checkpoint(output_dir / "best_short_f1", model, run_config, history, metadata | {"selected": True})
        print(json.dumps(metrics, sort_keys=True), flush=True)

    passed = write_gate(output_dir, best_epoch, best_f1, best_nll, args.gate_short_f1)
    print(f"Best epoch: {best_epoch}\nBest validation Short F1: {best_f1:.6f}\nInternal gate passed: {passed}\nFinal 100 accessed: False\nRuntime seconds: {time.perf_counter()-start:.3f}", flush=True)


if __name__ == "__main__":
    main()
