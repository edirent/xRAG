#!/usr/bin/env python
import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.language_modeling.utils import XRAG_TOKEN
from src.model import SFR, XMistralForCausalLM


SFR_MODEL_NAME = "Salesforce/SFR-Embedding-Mistral"
XRAG_MODEL_NAME = "Hannibal046/xrag-7b"
VARIANT_WEIGHTS = {
    "all_gold": 0.20,
    "all_gold_shuffled": 0.20,
    "first_two_gold": 0.20,
    "single_gold": 0.10,
    "gold_plus_1": 0.20,
    "gold_plus_2": 0.10,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate only the xRAG packet projector.")
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--validation-samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--gradient-clipping", type=float, default=1.0)
    parser.add_argument("--max-packets", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="cache/projector/packet_projector_calibration")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--validation-generation-samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def stable_rng(*parts):
    value = ":".join(str(part) for part in parts)
    seed = int(hashlib.sha256(value.encode()).hexdigest()[:16], 16)
    return random.Random(seed)


def supporting_and_distractor_packets(sample):
    supporting_pairs = set(
        zip(sample["supporting_facts"]["title"], sample["supporting_facts"]["sent_id"])
    )
    gold, distractors = [], []
    for doc_id, (title, sentences) in enumerate(
        zip(sample["context"]["title"], sample["context"]["sentences"])
    ):
        for sentence_id, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            packet = {
                "doc_id": doc_id,
                "sentence_id": sentence_id,
                "title": title,
                "text": sentence,
                "encoder_text": f"[{title}] {sentence}",
            }
            (gold if (title, sentence_id) in supporting_pairs else distractors).append(packet)
    gold.sort(key=lambda packet: (packet["doc_id"], packet["sentence_id"]))
    assert gold
    return gold, distractors


def load_splits(train_samples, validation_samples, max_packets):
    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="train",
        trust_remote_code=True,
    )
    accepted = []
    for sample in dataset:
        gold, distractors = supporting_and_distractor_packets(sample)
        if len(gold) > max_packets:
            continue
        accepted.append((sample, gold, distractors))
        if len(accepted) == train_samples + validation_samples:
            break
    assert len(accepted) == train_samples + validation_samples
    return accepted[:train_samples], accepted[train_samples:]


def choose_variant(index, epoch, gold, distractors, seed):
    rng = stable_rng(seed, epoch, index, "variant")
    value = rng.random()
    cumulative = 0.0
    variant = None
    for name, weight in VARIANT_WEIGHTS.items():
        cumulative += weight
        if value < cumulative:
            variant = name
            break
    assert variant is not None

    if variant == "single_gold":
        selected = [gold[rng.randrange(len(gold))]]
    elif variant == "first_two_gold":
        selected = gold[:2]
    elif variant == "all_gold_shuffled":
        selected = list(gold)
        rng.shuffle(selected)
    elif variant in {"gold_plus_1", "gold_plus_2"}:
        requested = 1 if variant.endswith("_1") else 2
        room = 4 - len(gold)
        count = min(requested, room, len(distractors))
        if count == 0:
            variant = "all_gold"
            selected = list(gold)
        else:
            selected = list(gold) + rng.sample(distractors, count)
            rng.shuffle(selected)
    else:
        selected = list(gold)
    assert 1 <= len(selected) <= 4
    return variant, selected


def build_prompt(question, num_packets):
    background = " ".join([XRAG_TOKEN] * num_packets)
    content = (
        "Refer to the background document and answer the question. "
        "Respond only with the shortest possible answer. "
        "Do not provide an explanation."
        "\n\n"
        f"Background: {background}"
        "\n\n"
        f"Question: {question}"
    )
    return f"[INST] {content} [/INST] The answer is:"


class ProjectorDataset(Dataset):
    def __init__(self, records, tokenizer, xrag_token_id, seed, training):
        self.records = records
        self.tokenizer = tokenizer
        self.xrag_token_id = xrag_token_id
        self.seed = seed
        self.training = training
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        sample, gold, distractors = self.records[index]
        if self.training:
            variant, selected = choose_variant(
                index, self.epoch, gold, distractors, self.seed
            )
        else:
            variant, selected = "all_gold", list(gold)
        prompt = build_prompt(sample["question"], len(selected))
        prompt_ids = self.tokenizer(
            prompt, add_special_tokens=False
        )["input_ids"]
        full_ids = self.tokenizer(
            prompt + " " + sample["answer"], add_special_tokens=False
        )["input_ids"]
        full_ids.append(self.tokenizer.eos_token_id)
        assert full_ids[: len(prompt_ids)] == prompt_ids
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        assert sum(token != -100 for token in labels) >= 2
        assert sum(token == self.xrag_token_id for token in full_ids) == len(selected)
        return {
            "sample_id": str(sample.get("id", sample.get("_id", index))),
            "question": sample["question"],
            "gold_answer": sample["answer"],
            "variant": variant,
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "packet_texts": [packet["encoder_text"] for packet in selected],
        }


def make_collator(tokenizer):
    def collate(items):
        input_ids = pad_sequence(
            [item["input_ids"] for item in items],
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
            padding_side="left",
        )
        labels = pad_sequence(
            [item["labels"] for item in items],
            batch_first=True,
            padding_value=-100,
            padding_side="left",
        )
        attention_mask = input_ids.ne(tokenizer.pad_token_id)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "packet_texts": [
                text for item in items for text in item["packet_texts"]
            ],
            "variants": [item["variant"] for item in items],
            "sample_ids": [item["sample_id"] for item in items],
            "questions": [item["question"] for item in items],
            "gold_answers": [item["gold_answer"] for item in items],
        }
    return collate


@torch.no_grad()
def encode_packets(tokenizer, model, texts, device):
    tokenized = tokenizer(
        texts,
        max_length=180,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    embeddings = model.get_doc_embedding(
        input_ids=tokenized.input_ids,
        attention_mask=tokenized.attention_mask,
    )
    return embeddings.view(-1, embeddings.shape[-1]).detach()


def normalize_answer(text):
    import re
    import string
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def short_answer(text):
    text = text.replace("</s>", " ").replace("<s>", " ").strip()
    text = " ".join(text.split()).strip() or "[EMPTY]"
    first = text.splitlines()[0].strip()
    for prefix in ["the answer is ", "answer: ", "it is "]:
        if first.lower().startswith(prefix):
            first = first[len(prefix) :].strip()
            break
    if "." in first:
        first = first.split(".", 1)[0].strip() or first
    return first.strip(" \t\n\"'")


def answer_metrics(prediction, gold):
    pred = normalize_answer(prediction).split()
    target = normalize_answer(gold).split()
    em = float(pred == target)
    common = Counter(pred) & Counter(target)
    same = sum(common.values())
    if not pred or not target:
        f1 = float(pred == target)
    elif not same:
        f1 = 0.0
    else:
        precision, recall = same / len(pred), same / len(target)
        f1 = 2 * precision * recall / (precision + recall)
    return em, f1


@torch.no_grad()
def validate_loss(model, sfr_tokenizer, sfr_model, loader, device):
    model.eval()
    losses = []
    for batch in loader:
        retrieval = encode_packets(
            sfr_tokenizer, sfr_model, batch["packet_texts"], device
        )
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
            retrieval_embeds=retrieval,
        )
        losses.append(float(outputs.loss))
    return sum(losses) / len(losses)


@torch.no_grad()
def validate_generation(
    model, tokenizer, sfr_tokenizer, sfr_model, records, device, max_samples, max_new_tokens
):
    model.eval()
    ems, f1s = [], []
    for sample, gold, _ in records[:max_samples]:
        prompt = build_prompt(sample["question"], len(gold))
        tokenized = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).to(device)
        retrieval = encode_packets(
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
        prediction = short_answer(raw)
        em, f1 = answer_metrics(prediction, sample["answer"])
        ems.append(em)
        f1s.append(f1)
    return 100 * sum(ems) / len(ems), 100 * sum(f1s) / len(f1s)


def save_checkpoint(output_dir, model, optimizer, scheduler, config, metrics):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {name: tensor.detach().cpu() for name, tensor in model.projector.state_dict().items()},
        output_dir / "projector.pt",
    )
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        output_dir / "optimizer.pt",
    )
    (output_dir / "training_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True)
    )
    (output_dir / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True)
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
        XRAG_MODEL_NAME,
        padding_side="left",
        add_eos_token=False,
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = (
            tokenizer.unk_token_id
            if tokenizer.unk_token_id is not None
            else tokenizer.eos_token_id
        )
    xrag_token_id = tokenizer.convert_tokens_to_ids(XRAG_TOKEN)

    sfr_tokenizer = AutoTokenizer.from_pretrained(SFR_MODEL_NAME)
    sfr_model = SFR.from_pretrained(
        SFR_MODEL_NAME, torch_dtype=torch.bfloat16
    ).eval().to(device)
    for parameter in sfr_model.parameters():
        parameter.requires_grad = False

    config = AutoConfig.from_pretrained(XRAG_MODEL_NAME)
    model = XMistralForCausalLM.from_pretrained(
        XRAG_MODEL_NAME,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.set_xrag_token_id(xrag_token_id)
    initial_projector = {
        name: value.detach().cpu().clone()
        for name, value in model.projector.state_dict().items()
    }
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("projector.")
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    assert trainable_names and all(name.startswith("projector.") for name in trainable_names)
    assert all(
        torch.equal(initial_projector[name], model.projector.state_dict()[name].detach().cpu())
        for name in initial_projector
    )

    print("Loading fixed HotpotQA train/projector-validation split", flush=True)
    train_records, validation_records = load_splits(
        args.train_samples, args.validation_samples, args.max_packets
    )
    train_dataset = ProjectorDataset(
        train_records, tokenizer, xrag_token_id, args.seed, training=True
    )
    validation_dataset = ProjectorDataset(
        validation_records, tokenizer, xrag_token_id, args.seed, training=False
    )
    collator = make_collator(tokenizer)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
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
    updates_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation
    )
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * args.warmup_ratio),
        num_training_steps=total_updates,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = vars(args) | {
        "sfr_model": SFR_MODEL_NAME,
        "xrag_model": XRAG_MODEL_NAME,
        "prompt": "P2_SHORT",
        "variant_weights": VARIANT_WEIGHTS,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "trainable_names": trainable_names,
        "split_policy": "first eligible HotpotQA train rows with 1..4 gold packets",
        "train_sample_ids": [
            str(sample.get("id", sample.get("_id", index)))
            for index, (sample, _, _) in enumerate(train_records)
        ],
        "validation_sample_ids": [
            str(sample.get("id", sample.get("_id", index)))
            for index, (sample, _, _) in enumerate(validation_records)
        ],
    }
    history = []
    optimizer.zero_grad(set_to_none=True)
    global_update = 0
    start = time.perf_counter()
    for epoch in range(args.epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        variant_counts = Counter()
        running_loss = 0.0
        for micro_step, batch in enumerate(train_loader):
            retrieval = encode_packets(
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

        validation_nll = validate_loss(
            model, sfr_tokenizer, sfr_model, validation_loader, device
        )
        short_em, short_f1 = validate_generation(
            model,
            tokenizer,
            sfr_tokenizer,
            sfr_model,
            validation_records,
            device,
            args.validation_generation_samples,
            args.max_new_tokens,
        )
        metrics = {
            "epoch": epoch + 1,
            "train_loss": running_loss / len(train_loader),
            "validation_nll": validation_nll,
            "validation_short_em": short_em,
            "validation_short_f1": short_f1,
            "variant_counts": dict(variant_counts),
            "global_update": global_update,
        }
        history.append(metrics)
        save_checkpoint(
            output_dir / f"epoch_{epoch + 1}",
            model,
            optimizer,
            scheduler,
            run_config,
            {"history": history},
        )
        print(json.dumps(metrics, sort_keys=True), flush=True)

    save_checkpoint(
        output_dir / "last",
        model,
        optimizer,
        scheduler,
        run_config,
        {"history": history, "runtime_seconds": time.perf_counter() - start},
    )
    changed = any(
        not torch.equal(initial_projector[name], model.projector.state_dict()[name].detach().cpu())
        for name in initial_projector
    )
    assert changed
    print(f"Projector checkpoint: {output_dir / 'last' / 'projector.pt'}", flush=True)
    print(f"Peak allocated GB: {torch.cuda.max_memory_allocated() / 1024**3:.3f}", flush=True)


if __name__ == "__main__":
    main()
