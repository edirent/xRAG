"""Frozen K4 representation adapter with preregistered STATIC-2 residual base."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from src.model.xMistral.modeling_xmistral import Projector
from src.packet_xrag.encoding.multi_token_projector import MultiTokenPacketProjector


OUTPUT_M = 4
TOKENS_PER_PACKET = 4
BASE_PACKETS = 2
BASE_REDUCTION = "slotwise mean of STATIC rank-1 and rank-2 K4 tokens"
K4_CHECKPOINT = Path("cache/projector/multi_token_k4/best_short_f1/multi_token_projector.pt")
K4_SHA256 = "971df4f4c516dc8945b2a5e2ba9f80f69279ac3cc0691635beeff9e21498c7d4"
V1_CHECKPOINT = Path("cache/projector/packet_projector_calibration/last/projector.pt")


def load_frozen_k4_projector(config, device, v1_path=V1_CHECKPOINT,
                             k4_path=K4_CHECKPOINT):
    """Load the preregistered compatible K4 asset and keep it strictly frozen."""
    v1_path, k4_path = Path(v1_path), Path(k4_path)
    if hashlib.sha256(k4_path.read_bytes()).hexdigest() != K4_SHA256:
        raise RuntimeError("frozen K4 checkpoint hash mismatch")
    base = Projector(config)
    base.load_state_dict(torch.load(v1_path, map_location="cpu", weights_only=True), strict=True)
    projector = MultiTokenPacketProjector(base, config.retriever_hidden_size,
                                           config.hidden_size, TOKENS_PER_PACKET, 1024)
    projector.load_state_dict(torch.load(k4_path, map_location="cpu", weights_only=True),
                              strict=True)
    projector.to(device=device, dtype=torch.bfloat16).eval()
    for parameter in projector.parameters():
        parameter.requires_grad = False
    return projector


def k4_residual_inputs(projected_groups, device):
    """Map variable K4 packet groups to a fixed four-token STATIC-2 base."""
    if not projected_groups or any(tokens.ndim != 3 or tokens.shape[1] != TOKENS_PER_PACKET
                                   for tokens in projected_groups):
        raise ValueError("K4 groups must be non-empty [P,4,D] tensors")
    if any(tokens.shape[0] < BASE_PACKETS for tokens in projected_groups):
        raise ValueError("K4 residual composition requires STATIC rank 1-2")
    dimension = projected_groups[0].shape[-1]
    base = torch.stack([tokens[:BASE_PACKETS].mean(dim=0) for tokens in projected_groups])
    if base.shape[1:] != (OUTPUT_M, dimension):
        raise AssertionError("K4 base did not preserve fixed M=4")
    extra_count = max(tokens.shape[0] - BASE_PACKETS for tokens in projected_groups)
    extras = torch.zeros(len(projected_groups), extra_count, TOKENS_PER_PACKET, dimension,
                         device=device, dtype=projected_groups[0].dtype)
    mask = torch.zeros(len(projected_groups), extra_count, device=device, dtype=torch.bool)
    for row, tokens in enumerate(projected_groups):
        count = tokens.shape[0] - BASE_PACKETS
        if count: extras[row, :count] = tokens[BASE_PACKETS:]; mask[row, :count] = True
    return base, extras, mask


def make_k4_fused(fuser, records, selected_groups, k4_projector, device):
    query = torch.stack([record["query_embedding"] for record in records]).to(
        device=device, dtype=torch.float32)
    projected = [k4_projector(record["packet_embeddings"][selected].to(
        device=device, dtype=torch.bfloat16)).float()
        for record, selected in zip(records, selected_groups)]
    base, extras, mask = k4_residual_inputs(projected, device)
    if extras.shape[1] == 0:
        return fuser(query, base, extras, mask)[0]
    with torch.enable_grad():
        output = fuser(query, base, extras, mask)[0]
    if output.shape[1] != OUTPUT_M: raise AssertionError("K4 fuser output M changed")
    return output
