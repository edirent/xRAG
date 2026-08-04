"""Fixed Hotpot-recipe STATIC scorer training for adapted QA datasets."""

from __future__ import annotations

import math
import random

import torch
from transformers import get_linear_schedule_with_warmup

from scripts.packet_xrag.train_static_scorer import (
    gold_mask, preload_embeddings, scorer_batch_loss, validation_loss,
)
from src.packet_xrag.controller.static_scorer import StaticPacketScorer


SEED = 20260804
EPOCHS = 8
EFFECTIVE_BATCH_SIZE = 32
LEARNING_RATE = 2e-4
WEIGHT_DECAY = .01
WARMUP_RATIO = .05
GRADIENT_CLIPPING = 1.0


def positive_indices(cache):
    return [index for index, record in enumerate(cache.records) if record["gold_packet_ids"]]


def build_training(model, sample_count, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY, fused=device.type == "cuda")
    updates = math.ceil(sample_count / EFFECTIVE_BATCH_SIZE) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(updates * WARMUP_RATIO), updates)
    return optimizer, scheduler


def fixed_recipe():
    return {"architecture": "StaticPacketScorer", "loss": "multi-positive listwise",
            "epochs": EPOCHS, "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "optimizer": "AdamW", "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY, "warmup_ratio": WARMUP_RATIO,
            "gradient_clipping": GRADIENT_CLIPPING, "seed": SEED}

