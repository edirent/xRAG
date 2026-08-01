#!/usr/bin/env python
"""Train only zero-initialized extra soft-token heads for each sentence packet."""

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag import run_selector_calibration as selector
from scripts.packet_xrag import train_packet_projector as v1
from scripts.packet_xrag import train_residual_packet_projector as locked
from src.language_modeling.utils import XRAG_TOKEN
from src.model import SFR, XMistralForCausalLM
from src.packet_xrag.encoding.multi_token_projector import MultiTokenPacketProjector
from src.packet_xrag.modeling.multi_token_xrag import install_multi_token_injection

EXPECTED_SPLIT_HASH = "8f925ff8ababf1efc6bb8a913e6d5431437610b0bb30fa8357a57dfbb5f24052"
V1_FULL_500_F1 = 58.589785360838036


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens-per-packet", type=int, required=True, choices=[2, 4])
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--validation-samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--gradient-clipping", type=float, default=1.0)
    parser.add_argument("--residual-hidden-size", type=int, default=1024)
    parser.add_argument("--max-packets", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--base-projector-checkpoint", default="cache/projector/packet_projector_calibration/last/projector.pt")
    parser.add_argument("--split-file", default="cache/projector/packet_projector_calibration/data_split.json")
    parser.add_argument("--v1-training-config", default="cache/projector/packet_projector_calibration/last/training_config.json")
    parser.add_argument("--baseline-results", default="cache/results/packet_representation_ablation.jsonl")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


class MultiTokenDataset(v1.ProjectorDataset):
    def __init__(self, *args, tokens_per_packet, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokens_per_packet = tokens_per_packet

    def __getitem__(self, index):
        item = super().__getitem__(index)
        sample, gold, distractors = self.records[index]
        if self.training:
            variant, selected = v1.choose_variant(index, self.epoch, gold, distractors, self.seed)
        else:
            variant, selected = "all_gold", list(gold)
        prompt = v1.build_prompt(sample["question"], len(selected) * self.tokens_per_packet)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(prompt + " " + sample["answer"], add_special_tokens=False)["input_ids"]
        full_ids.append(self.tokenizer.eos_token_id)
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        assert full_ids[:len(prompt_ids)] == prompt_ids
        assert sum(token == self.xrag_token_id for token in full_ids) == len(selected) * self.tokens_per_packet
        item.update({"variant": variant, "input_ids": torch.tensor(full_ids), "labels": torch.tensor(labels),
                     "packet_texts": [packet["encoder_text"] for packet in selected]})
        return item


def split_hash(ids):
    return hashlib.sha256("".join(ids).encode()).hexdigest()


def audit_baseline(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row["variant"] == "V1_TITLE_SENTENCE"]
    assert len(rows) == 500
    f1 = 100 * sum(row["short_f1"] for row in rows) / 500
    assert abs(f1 - V1_FULL_500_F1) <= 0.1
    return [row["sample_id"] for row in rows], f1


@torch.inference_mode()
def validate_generation(model, tokenizer, sfr_tokenizer, sfr_model, records, device, k, max_new_tokens):
    model.eval(); predictions = []
    for sample, gold, _ in records:
        prompt = v1.build_prompt(sample["question"], len(gold) * k)
        tokenized = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        retrieval = v1.encode_packets(sfr_tokenizer, sfr_model, [p["encoder_text"] for p in gold], device)
        assert int((tokenized.input_ids == model.xrag_token_id).sum()) == len(gold) * k
        generated = model.generate(input_ids=tokenized.input_ids, attention_mask=tokenized.attention_mask,
            retrieval_embeds=retrieval, do_sample=False, max_new_tokens=max_new_tokens, use_cache=True,
            pad_token_id=tokenizer.pad_token_id)
        new_tokens = generated[:, tokenized.input_ids.shape[1]:] if generated.shape[1] > tokenized.input_ids.shape[1] else generated
        raw = tokenizer.batch_decode(new_tokens, skip_special_tokens=False)[0]
        clean, short = selector.clean_prediction(raw), selector.extract_short_answer(raw)
        if not short: short = "[EMPTY]"
        short_em, short_f1 = selector.score_prediction(short, sample["answer"])
        _, clean_f1 = selector.score_prediction(clean, sample["answer"])
        predictions.append({"sample_id": locked.sample_id(sample), "gold_answer": sample["answer"], "raw_prediction": raw,
            "clean_prediction": clean, "short_prediction": short, "short_em": short_em, "short_f1": short_f1,
            "clean_f1": clean_f1, "num_packets": len(gold), "tokens_per_packet": k, "total_soft_tokens": len(gold) * k})
    assert len(predictions) == 500
    return {"validation_short_em": 100 * sum(r["short_em"] for r in predictions) / 500,
        "validation_short_f1": 100 * sum(r["short_f1"] for r in predictions) / 500,
        "validation_clean_f1": 100 * sum(r["clean_f1"] for r in predictions) / 500,
        "validation_empty_count": sum(r["short_prediction"] == "[EMPTY]" for r in predictions)}, predictions


def save_checkpoint(directory, model, config, history, metadata, predictions):
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({name: value.detach().cpu() for name, value in model.projector.state_dict().items()}, directory / "multi_token_projector.pt")
    (directory / "training_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (directory / "validation_metrics.json").write_text(json.dumps({"history": history}, indent=2, sort_keys=True) + "\n")
    (directory / "checkpoint_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    with (directory / "validation_predictions.jsonl").open("w") as stream:
        for row in predictions: stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert args.train_samples == 5000 and args.validation_samples == 500 and args.epochs == 5
    baseline_ids, baseline_f1 = audit_baseline(args.baseline_results)
    split_record = json.loads(Path(args.split_file).read_text())
    assert split_hash(split_record["validation_sample_ids"]) == EXPECTED_SPLIT_HASH
    assert baseline_ids == split_record["validation_sample_ids"]
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device); torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(v1.XRAG_MODEL_NAME, padding_side="left", add_eos_token=False, use_fast=False)
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.eos_token_id
    xrag_token_id = tokenizer.convert_tokens_to_ids(XRAG_TOKEN)
    sfr_tokenizer = AutoTokenizer.from_pretrained(v1.SFR_MODEL_NAME)
    sfr_model = SFR.from_pretrained(v1.SFR_MODEL_NAME, torch_dtype=torch.bfloat16).eval().to(device)
    for parameter in sfr_model.parameters(): parameter.requires_grad = False
    config = AutoConfig.from_pretrained(v1.XRAG_MODEL_NAME)
    model = XMistralForCausalLM.from_pretrained(v1.XRAG_MODEL_NAME, config=config, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device)
    model.set_xrag_token_id(xrag_token_id)
    model.projector.load_state_dict(torch.load(args.base_projector_checkpoint, map_location="cpu", weights_only=True), strict=True)
    base_snapshot = copy.deepcopy({name: value.detach().cpu() for name, value in model.projector.state_dict().items()})
    llm_probe_name, llm_probe_parameter = next((name, value) for name, value in model.named_parameters() if not name.startswith("projector."))
    sfr_probe_name, sfr_probe_parameter = next(iter(sfr_model.named_parameters()))
    llm_probe = llm_probe_parameter.detach().flatten()[:1024].cpu().clone()
    sfr_probe = sfr_probe_parameter.detach().flatten()[:1024].cpu().clone()
    for parameter in model.parameters(): parameter.requires_grad = False
    model.projector = MultiTokenPacketProjector(model.projector, config.retriever_hidden_size, config.hidden_size,
        args.tokens_per_packet, args.residual_hidden_size).to(device=device, dtype=torch.bfloat16)
    install_multi_token_injection(model)
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable_names and all(name.startswith("projector.extra_heads.") for name in trainable_names)
    args.data_split = args.split_file
    train_records, validation_records = locked.load_locked_split(args)
    assert [locked.sample_id(sample) for sample, _, _ in validation_records] == baseline_ids
    train_dataset = MultiTokenDataset(train_records, tokenizer, xrag_token_id, args.seed, training=True, tokens_per_packet=args.tokens_per_packet)
    validation_dataset = MultiTokenDataset(validation_records, tokenizer, xrag_token_id, args.seed, training=False, tokens_per_packet=args.tokens_per_packet)
    collator = v1.make_collator(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed), collate_fn=collator)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation); total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_updates * args.warmup_ratio), total_updates)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    run_config = vars(args) | {"prompt": "P2_SHORT", "answer_extraction": "run_selector_calibration.extract_short_answer",
        "variant_weights": v1.VARIANT_WEIGHTS, "validation_split_hash": EXPECTED_SPLIT_HASH, "validation_generated_samples": 500,
        "v1_full_500_short_f1": baseline_f1, "trainable_names": trainable_names,
        "trainable_parameters": sum(p.numel() for p in trainable), "dtype": "bfloat16"}
    total_parameters = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in sfr_model.parameters())
    print(json.dumps({"K": args.tokens_per_packet, "total_parameters": total_parameters,
        "trainable_parameters": sum(p.numel() for p in trainable), "frozen_sfr_parameters": sum(p.numel() for p in sfr_model.parameters()),
        "frozen_llm_parameters": sum(p.numel() for n,p in model.named_parameters() if not n.startswith("projector.")),
        "frozen_base_projector_parameters": sum(p.numel() for p in model.projector.base_projector.parameters())}), flush=True)
    history=[]; best_f1=float("-inf"); best_nll=float("inf"); best_epoch=None; global_update=0; optimizer.zero_grad(set_to_none=True); start=time.perf_counter()
    for epoch in range(args.epochs):
        train_dataset.set_epoch(epoch); model.train(); running_loss=0.; counts=Counter()
        for micro_step,batch in enumerate(train_loader):
            retrieval=v1.encode_packets(sfr_tokenizer,sfr_model,batch["packet_texts"],device)
            outputs=model(input_ids=batch["input_ids"].to(device),attention_mask=batch["attention_mask"].to(device),labels=batch["labels"].to(device),retrieval_embeds=retrieval)
            loss=outputs.loss/args.gradient_accumulation; loss.backward(); running_loss+=float(loss.detach())*args.gradient_accumulation; counts.update(batch["variants"])
            if (micro_step+1)%args.gradient_accumulation==0 or micro_step+1==len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable,args.gradient_clipping); optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); global_update+=1
                if global_update%args.log_every==0: print(f"K={args.tokens_per_packet} epoch={epoch+1} update={global_update}/{total_updates} loss={running_loss/(micro_step+1):.4f} lr={scheduler.get_last_lr()[0]:.2e}",flush=True)
        validation_nll=v1.validate_loss(model,sfr_tokenizer,sfr_model,validation_loader,device)
        generation,predictions=validate_generation(model,tokenizer,sfr_tokenizer,sfr_model,validation_records,device,args.tokens_per_packet,args.max_new_tokens)
        metrics={"epoch":epoch+1,"global_update":global_update,"train_loss":running_loss/len(train_loader),"validation_nll":validation_nll,**generation,"variant_counts":dict(counts)}
        history.append(metrics); metadata={"epoch":epoch+1,"selection_metric":"validation_short_f1","tie_break_metric":"validation_nll","validation_examples":500,"tokens_per_packet":args.tokens_per_packet}
        save_checkpoint(output_dir/"last",model,run_config,history,metadata,predictions)
        f1=metrics["validation_short_f1"]
        if f1>best_f1+1e-8 or (abs(f1-best_f1)<=1e-8 and validation_nll<best_nll-1e-8):
            best_f1,best_nll,best_epoch=f1,validation_nll,epoch+1
            save_checkpoint(output_dir/"best_short_f1",model,run_config,history,metadata|{"selected":True},predictions)
        print(json.dumps(metrics,sort_keys=True),flush=True)
    assert all(torch.equal(base_snapshot[n],v.detach().cpu()) for n,v in model.projector.base_projector.state_dict().items())
    assert all(p.grad is None for p in model.projector.base_projector.parameters()) and all(p.grad is None for p in sfr_model.parameters())
    assert torch.equal(dict(model.named_parameters())[llm_probe_name].detach().flatten()[:1024].cpu(), llm_probe)
    assert torch.equal(dict(sfr_model.named_parameters())[sfr_probe_name].detach().flatten()[:1024].cpu(), sfr_probe)
    assert all(p.grad is None for name,p in model.named_parameters() if not name.startswith("projector."))
    print(json.dumps({"K":args.tokens_per_packet,"peak_vram_gb":torch.cuda.max_memory_allocated()/1024**3,"best_epoch":best_epoch,
        "best_validation_short_f1":best_f1,"best_validation_nll":best_nll,"last_validation_short_f1":history[-1]["validation_short_f1"],
        "best_empty_count":json.loads((output_dir/"best_short_f1"/"validation_metrics.json").read_text())["history"][-1]["validation_empty_count"],
        "runtime_seconds":time.perf_counter()-start}),flush=True)


if __name__ == "__main__": main()
