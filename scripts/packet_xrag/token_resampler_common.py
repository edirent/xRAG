"""Locked protocol helpers shared by token-state resampler experiments."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig

from scripts.packet_xrag import train_multi_token_projector as multi
from scripts.packet_xrag import train_packet_projector as v1
from scripts.packet_xrag import train_residual_packet_projector as locked
from src.model import XMistralForCausalLM
from src.model.xMistral.modeling_xmistral import Projector
from src.packet_xrag.encoding.multi_token_projector import MultiTokenPacketProjector
from src.packet_xrag.encoding.token_state_cache import packet_key
from src.packet_xrag.modeling.multi_token_xrag import install_multi_token_injection


EXPECTED_SPLIT_HASH = "8f925ff8ababf1efc6bb8a913e6d5431437610b0bb30fa8357a57dfbb5f24052"
EXPECTED_V1_SHA256 = "fa1a9ba443960acc176dc387989fe7c2e3fa1ef1cf24a38db265da4c1c60f760"
EXPECTED_K2_SHA256 = "c40aa3dc297f57b5be73649f1754dc292ef98fefbc5c8a338b90e108427fd8a4"
POOLED_K2_F1 = 62.56321637426902
POOLED_K2_EM = 50.6


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_hash(ids):
    return hashlib.sha256("".join(ids).encode()).hexdigest()


def locked_records(split_file, v1_training_config):
    args = SimpleNamespace(
        data_split=split_file,
        v1_training_config=v1_training_config,
        train_samples=5000,
        validation_samples=500,
        seed=42,
        max_packets=4,
    )
    train, validation = locked.load_locked_split(args)
    ids = [locked.sample_id(sample) for sample, _, _ in validation]
    if split_hash(ids) != EXPECTED_SPLIT_HASH:
        raise RuntimeError("locked validation split hash mismatch")
    return train, validation


def collect_used_packets(train_records, validation_records, epochs=5):
    train_packets, validation_packets = {}, {}
    train_uses = 0
    for epoch in range(epochs):
        for index, (_, gold, distractors) in enumerate(train_records):
            _, selected = v1.choose_variant(index, epoch, gold, distractors, 42)
            train_uses += len(selected)
            for packet in selected:
                key = packet_key(packet["encoder_text"])
                train_packets[key] = packet
    validation_uses = 0
    for _, gold, _ in validation_records:
        validation_uses += len(gold)
        for packet in gold:
            validation_packets[packet_key(packet["encoder_text"])] = packet
    union = dict(train_packets)
    union.update(validation_packets)
    return {
        "train": train_packets,
        "validation": validation_packets,
        "union": union,
        "train_packet_uses": train_uses,
        "validation_packet_uses": validation_uses,
    }


def audit_checkpoints(v1_path, k2_path):
    values = {"v1": sha256_file(v1_path), "k2": sha256_file(k2_path)}
    if values["v1"] != EXPECTED_V1_SHA256 or values["k2"] != EXPECTED_K2_SHA256:
        raise RuntimeError(f"checkpoint SHA256 mismatch: {values}")
    v1_state = torch.load(v1_path, map_location="cpu", weights_only=True)
    k2_state = torch.load(k2_path, map_location="cpu", weights_only=True)
    config = AutoConfig.from_pretrained(v1.XRAG_MODEL_NAME)
    base = Projector(config)
    base.load_state_dict(v1_state, strict=True)
    k2 = MultiTokenPacketProjector(
        base, config.retriever_hidden_size, config.hidden_size, 2, 1024
    )
    k2.load_state_dict(k2_state, strict=True)
    return values, config


def load_frozen_k2_model(device, tokenizer, xrag_id, v1_path, k2_path):
    config = AutoConfig.from_pretrained(v1.XRAG_MODEL_NAME)
    model = XMistralForCausalLM.from_pretrained(
        v1.XRAG_MODEL_NAME,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.set_xrag_token_id(xrag_id)
    model.projector.load_state_dict(
        torch.load(v1_path, map_location="cpu", weights_only=True), strict=True
    )
    model.projector = MultiTokenPacketProjector(
        model.projector, config.retriever_hidden_size, config.hidden_size, 2, 1024
    ).to(device=device, dtype=torch.bfloat16)
    model.projector.load_state_dict(
        torch.load(k2_path, map_location="cpu", weights_only=True), strict=True
    )
    install_multi_token_injection(model)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, config


class CachedTokenStateDataset(multi.MultiTokenDataset):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        item["packet_keys"] = [packet_key(text) for text in item["packet_texts"]]
        return item


def make_cached_collator(tokenizer, cache):
    base_collator = v1.make_collator(tokenizer)

    def collate(items):
        batch = base_collator(items)
        records = [cache.get(key) for item in items for key in item["packet_keys"]]
        hidden = pad_sequence(
            [record["hidden"] for record in records], batch_first=True, padding_value=0.0
        )
        mask = pad_sequence(
            [record["mask"] for record in records], batch_first=True, padding_value=False
        )
        pooled = torch.stack([record["pooled"] for record in records])
        batch["retrieval"] = {
            "token_states": hidden,
            "token_mask": mask,
            "pooled_embeddings": pooled,
        }
        batch["packet_keys"] = [key for item in items for key in item["packet_keys"]]
        return batch

    return collate


def retrieval_to_device(retrieval, device):
    return {name: tensor.to(device, non_blocking=True) for name, tensor in retrieval.items()}


def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
