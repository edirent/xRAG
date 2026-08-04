"""Shared frozen-model setup and batching for bounded composition training."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

from scripts.packet_xrag import train_packet_projector as v1
from src.language_modeling.utils import XRAG_TOKEN
from src.model import XMistralForCausalLM
from src.model.xMistral.modeling_xmistral import Projector
from src.packet_xrag.composition.direct_embedding_fuser import DirectEmbeddingFuser
from src.packet_xrag.composition.residual_set_fuser import ResidualSetFuser
from src.packet_xrag.composition.set_fuser import QueryConditionedSetFuser
from src.packet_xrag.encoding.multi_token_projector import MultiTokenPacketProjector


SEED = 20260804
BRANCHES = {
    "A1": {"selector": "TOPK", "architecture": "K2_SET_FUSER"},
    "B1": {"selector": "STATIC", "architecture": "K2_SET_FUSER"},
    "C1": {"selector": "STATIC", "architecture": "RESIDUAL_K2_SET_FUSER"},
    "D1": {"selector": "STATIC", "architecture": "DIRECT_SFR_SET_FUSER"},
}


def load_frozen_generator(device):
    tokenizer = AutoTokenizer.from_pretrained(v1.XRAG_MODEL_NAME, padding_side="left",
                                               add_eos_token=False, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id or tokenizer.eos_token_id
    xrag_id = tokenizer.convert_tokens_to_ids(XRAG_TOKEN)
    config = AutoConfig.from_pretrained(v1.XRAG_MODEL_NAME)
    model = XMistralForCausalLM.from_pretrained(v1.XRAG_MODEL_NAME, config=config,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device)
    model.set_xrag_token_id(xrag_id); model.eval()
    for parameter in model.parameters(): parameter.requires_grad = False
    return tokenizer, model, xrag_id, config


def load_frozen_k2_projector(config, device,
                             v1_path="cache/projector/packet_projector_calibration/last/projector.pt",
                             k2_path="cache/projector/multi_token_k2/best_short_f1/multi_token_projector.pt"):
    base = Projector(config)
    base.load_state_dict(torch.load(v1_path, map_location="cpu", weights_only=True), strict=True)
    projector = MultiTokenPacketProjector(base, config.retriever_hidden_size,
                                           config.hidden_size, 2, 1024)
    projector.load_state_dict(torch.load(k2_path, map_location="cpu", weights_only=True), strict=True)
    projector.to(device=device, dtype=torch.bfloat16).eval()
    for parameter in projector.parameters(): parameter.requires_grad = False
    return projector


def build_fuser(branch_id):
    if branch_id in {"A1", "B1"}:
        return QueryConditionedSetFuser(output_slots=4, depth=1)
    if branch_id == "C1": return ResidualSetFuser(output_slots=4)
    if branch_id == "D1": return DirectEmbeddingFuser(output_slots=4, depth=1)
    raise ValueError(f"unknown composition branch: {branch_id}")


def selected_ids(record, static_ranking, branch_id, breadth):
    ranking = record["topk_ranking"] if BRANCHES[branch_id]["selector"] == "TOPK" else static_ranking
    return list(ranking[:min(breadth, len(ranking))])


@torch.no_grad()
def make_fused_tokens(fuser, branch_id, records, selected_groups, k2_projector, device):
    query = torch.stack([record["query_embedding"] for record in records]).to(
        device=device, dtype=torch.float32)
    breadth = max(len(selected) for selected in selected_groups)
    mask = torch.zeros(len(records), breadth, device=device, dtype=torch.bool)
    if branch_id == "D1":
        packets = torch.zeros(len(records), breadth, 4096, device=device, dtype=torch.float32)
        for row, (record, selected) in enumerate(zip(records, selected_groups)):
            packets[row, :len(selected)] = record["packet_embeddings"][selected].to(
                device=device, dtype=torch.float32); mask[row, :len(selected)] = True
        with torch.enable_grad(): return fuser(query, packets, mask)
    projected_groups = []
    for record, selected in zip(records, selected_groups):
        projected = k2_projector(record["packet_embeddings"][selected].to(
            device=device, dtype=torch.bfloat16)).float()
        projected_groups.append(projected)
    if branch_id == "C1":
        base = torch.stack([tokens[:2].reshape(4, 4096) for tokens in projected_groups])
        extra_count = max(max(0, len(selected) - 2) for selected in selected_groups)
        if extra_count == 0:
            return fuser(query, base, torch.empty(len(records), 0, 2, 4096, device=device),
                         torch.empty(len(records), 0, dtype=torch.bool, device=device))[0]
        extras = torch.zeros(len(records), extra_count, 2, 4096, device=device)
        extra_mask = torch.zeros(len(records), extra_count, dtype=torch.bool, device=device)
        for row, tokens in enumerate(projected_groups):
            count = max(0, tokens.shape[0] - 2)
            if count: extras[row, :count] = tokens[2:]; extra_mask[row, :count] = True
        with torch.enable_grad(): return fuser(query, base, extras, extra_mask)[0]
    packets = torch.zeros(len(records), breadth, 2, 4096, device=device)
    for row, tokens in enumerate(projected_groups):
        packets[row, :tokens.shape[0]] = tokens; mask[row, :tokens.shape[0]] = True
    with torch.enable_grad(): return fuser(query, packets, mask)


def checkpoint_payload(branch_id, fuser, extra=None):
    return {"branch_id": branch_id, "branch": BRANCHES[branch_id], "output_slots": 4,
            "latent_dim": 512, "depth": 1, "heads": 8, "seed": SEED,
            "state_dict": {key: value.detach().cpu() for key, value in fuser.state_dict().items()},
            **(extra or {})}

