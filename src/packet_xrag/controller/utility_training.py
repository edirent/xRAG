"""Shared deterministic training and label-level evaluation for utility models."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup

from src.packet_xrag.controller.static_scorer import packet_position_features
from src.packet_xrag.controller.utility_label_dataset import (
    ShardedUtilityLabelDataset, normalize_delta,
)
from src.packet_xrag.controller.utility_prediction_loss import utility_prediction_loss
from src.packet_xrag.controller.utility_rollout import (
    candidate_relation_features, state_scalar_features,
)


SEED = 20260803


def average_ranks(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values); start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + stop - 1) / 2 + 1
        for position in range(start, stop): ranks[order[position]] = rank
        start = stop
    return ranks


def spearman(left, right):
    if len(left) < 2:
        return 0.0
    x, y = average_ranks(left), average_ranks(right)
    xm, ym = mean(x), mean(y)
    numerator = sum((a - xm) * (b - ym) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - xm) ** 2 for a in x) *
                            sum((b - ym) ** 2 for b in y))
    return numerator / denominator if denominator else 0.0


class UtilityFeatureStore:
    def __init__(self, feature_cache, static_scores):
        self.cache = feature_cache
        self.by_id = {record["sample_id"]: index
                      for index, record in enumerate(feature_cache.records)}
        self.static_scores = static_scores
        if set(self.by_id) != set(static_scores):
            raise ValueError("STATIC score cache does not align with feature cache")
        self._records = {}
        self._positions = {}
        self._state_features = {}
        self._relations = {}

    def record(self, sample_id):
        if sample_id not in self._records:
            self._records[sample_id] = self.cache[self.by_id[sample_id]]
        return self._records[sample_id]

    def make_batch(self, rows, device, dtype=torch.float32):
        query, packet, positions, static = [], [], [], []
        selected_values, masks, state_values, relations = [], [], [], []
        maximum_selected = max(1, max(len(row["selected_packet_ids"]) for row in rows))
        hidden = self.cache.hidden_size
        for row in rows:
            record = self.record(row["sample_id"]); candidate = row["candidate_packet_id"]
            selected = row["selected_packet_ids"]; scores = self.static_scores[row["sample_id"]]
            query.append(record["query_embedding"]); packet.append(record["packet_embeddings"][candidate])
            if row["sample_id"] not in self._positions:
                self._positions[row["sample_id"]] = packet_position_features(record["packets"])
            positions.append(self._positions[row["sample_id"]][candidate])
            static.append(float(scores[candidate]))
            selected_tensor = torch.zeros(maximum_selected, hidden, dtype=torch.bfloat16)
            mask = torch.zeros(maximum_selected, dtype=torch.bool)
            if selected:
                selected_tensor[:len(selected)] = record["packet_embeddings"][selected]
                mask[:len(selected)] = True
            selected_values.append(selected_tensor); masks.append(mask)
            state_key = (row["sample_id"], tuple(selected))
            if state_key not in self._state_features:
                self._state_features[state_key] = state_scalar_features(record, selected, scores)
            relation_key = (*state_key, candidate)
            if relation_key not in self._relations:
                self._relations[relation_key] = candidate_relation_features(
                    record, candidate, selected
                )
            state_values.append(self._state_features[state_key])
            relations.append(self._relations[relation_key])
        return {
            "query_embeddings": torch.stack(query).to(device=device, dtype=dtype),
            "packet_embeddings": torch.stack(packet).to(device=device, dtype=dtype),
            "position_features": torch.stack(positions).to(device=device, dtype=dtype),
            "static_scores": torch.tensor(static, device=device, dtype=dtype),
            "selected_embeddings": torch.stack(selected_values).to(device=device, dtype=dtype),
            "selected_mask": torch.stack(masks).to(device),
            "state_features": torch.tensor(state_values, device=device, dtype=dtype),
            "relation_features": torch.tensor(relations, device=device, dtype=dtype),
        }


def group_batches(dataset: ShardedUtilityLabelDataset, seed: int, epoch: int,
                  max_records=256, shuffle=True):
    groups = dataset.state_groups()
    if shuffle:
        random.Random(seed + epoch).shuffle(groups)
    batches, current = [], []
    for group in groups:
        if current and len(current) + len(group) > max_records:
            batches.append(current); current = []
        current.extend(group)
    if current: batches.append(current)
    return batches


def rows_and_slices(dataset, indices):
    rows = [dataset.rows[index] for index in indices]
    slices, start = [], 0
    while start < len(rows):
        first = rows[start]; key = (first["sample_id"], tuple(first["selected_packet_ids"]))
        stop = start + 1
        while stop < len(rows):
            row = rows[stop]
            if (row["sample_id"], tuple(row["selected_packet_ids"])) != key: break
            stop += 1
        slices.append((start, stop, first["sample_id"], first["state_id"]))
        start = stop
    return rows, slices


@torch.inference_mode()
def predict_label_cache(model, dataset, features, clip_value, device, batch_records=512):
    model.eval(); predictions, targets, metadata = [], [], []
    for indices in group_batches(dataset, SEED, 0, batch_records, shuffle=False):
        rows, _ = rows_and_slices(dataset, indices)
        batch = features.make_batch(rows, device, next(model.parameters()).dtype)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            values = model(**batch).float().cpu() * clip_value
        predictions.extend(float(value) for value in values)
        targets.extend(float(row["delta_utility"]) for row in rows)
        metadata.extend(rows)
    return predictions, targets, metadata


def precision_recall(predicted, actual):
    tp = sum(p and a for p, a in zip(predicted, actual))
    fp = sum(p and not a for p, a in zip(predicted, actual))
    fn = sum(not p and a for p, a in zip(predicted, actual))
    return (tp / (tp + fp) if tp + fp else 0.0,
            tp / (tp + fn) if tp + fn else 0.0)


def label_prediction_metrics(predictions, targets, rows, clip_value):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(row["sample_id"], tuple(row["selected_packet_ids"]))].append(index)
    per_state_rho, pair_correct, pair_total = [], 0, 0
    best_correct = stop_correct = false_stop = missed_positive = harmful = 0
    regrets = []
    for indices in groups.values():
        pred = [predictions[index] for index in indices]
        true = [targets[index] for index in indices]
        per_state_rho.append(spearman(pred, true))
        for left in range(len(indices)):
            for right in range(left + 1, len(indices)):
                if abs(true[left] - true[right]) >= .05:
                    pair_correct += (pred[left] - pred[right]) * (true[left] - true[right]) > 0
                    pair_total += 1
        teacher_stop = max(true) <= 0
        predicted_stop = max(pred) <= 0
        teacher_index = None if teacher_stop else max(range(len(true)), key=lambda i: (true[i], -i))
        predicted_index = None if predicted_stop else max(range(len(pred)), key=lambda i: (pred[i], -i))
        best_correct += teacher_index == predicted_index
        stop_correct += teacher_stop == predicted_stop
        false_stop += predicted_stop and not teacher_stop
        missed_positive += predicted_stop and max(true) > 0
        chosen_utility = 0.0 if predicted_stop else true[predicted_index]
        regrets.append(max(0.0, max(true)) - chosen_utility)
        harmful += not predicted_stop and chosen_utility < -.02
    positive_actual = [value > .02 for value in targets]
    negative_actual = [value < -.02 for value in targets]
    positive_pred = [value > 0 for value in predictions]
    negative_pred = [value < 0 for value in predictions]
    pos_precision, pos_recall = precision_recall(positive_pred, positive_actual)
    neg_precision, neg_recall = precision_recall(negative_pred, negative_actual)
    nonzero = [index for index, value in enumerate(targets) if abs(value) > .02]
    normalized_pred = torch.tensor(predictions) / clip_value
    normalized_true = normalize_delta(torch.tensor(targets), clip_value)
    return {
        "mae": mean(abs(pred - true) for pred, true in zip(predictions, targets)),
        "huber_loss": float(F.huber_loss(normalized_pred, normalized_true, delta=.1)),
        "global_spearman": spearman(predictions, targets),
        "mean_per_state_spearman": mean(per_state_rho),
        "pairwise_ranking_accuracy": pair_correct / pair_total,
        "sign_accuracy": (sum((predictions[index] > 0) == (targets[index] > 0)
                              for index in nonzero) / len(nonzero)),
        "negative_precision": neg_precision, "negative_recall": neg_recall,
        "positive_precision": pos_precision, "positive_recall": pos_recall,
        "best_action_top1_accuracy": best_correct / len(groups),
        "teacher_policy_regret": mean(regrets),
        "stop_decision_accuracy": stop_correct / len(groups),
        "false_stop_rate": false_stop / len(groups),
        "missed_positive_stop_rate": missed_positive / len(groups),
        "harmful_addition_decision_rate": harmful / len(groups),
        "states": len(groups), "labels": len(rows),
    }


def save_checkpoint(model, output_dir, epoch, payload):
    directory = Path(output_dir) / f"epoch_{epoch}"
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()},
               directory / "model.pt")
    (directory / "label_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return directory


def train_utility_model(model, train_dataset, dev_dataset, train_features, dev_features,
                        clip_value, device, output_dir, epochs, learning_rate,
                        seed=SEED, batch_records=256, log_every=100):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model.to(device); model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=.01,
        fused=device.type == "cuda",
    )
    total_steps = sum(len(group_batches(train_dataset, seed, epoch, batch_records))
                      for epoch in range(1, epochs + 1))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * .05), total_steps
    )
    history = []; global_step = 0
    for epoch in range(1, epochs + 1):
        model.train(); losses = []
        batches = group_batches(train_dataset, seed, epoch, batch_records)
        for batch_index, indices in enumerate(batches, 1):
            rows, slices = rows_and_slices(train_dataset, indices)
            batch = train_features.make_batch(rows, device, next(model.parameters()).dtype)
            raw = torch.tensor([row["delta_utility"] for row in rows], device=device)
            target = normalize_delta(raw, clip_value)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                predicted = model(**batch)
                loss, components = utility_prediction_loss(
                    predicted.float(), target, raw, slices, epoch
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); global_step += 1
            losses.append(float(loss.detach()))
            if batch_index % log_every == 0 or batch_index == len(batches):
                print(json.dumps({"epoch": epoch, "batch": batch_index,
                                  "batches": len(batches), "loss": losses[-1],
                                  "regression": float(components["regression"].detach()),
                                  "ranking": float(components["ranking"].detach()),
                                  "sign": float(components["sign"].detach()),
                                  "lr": scheduler.get_last_lr()[0]}), flush=True)
        predictions, targets, rows = predict_label_cache(
            model, dev_dataset, dev_features, clip_value, device
        )
        metrics = label_prediction_metrics(predictions, targets, rows, clip_value)
        record = {"epoch": epoch, "training_loss": mean(losses),
                  "global_step": global_step, "internal_dev": metrics}
        history.append(record); save_checkpoint(model, output_dir, epoch, record)
        print(json.dumps(record, indent=2), flush=True)
    selected = sorted(history, key=lambda item: (
        item["internal_dev"]["teacher_policy_regret"], item["epoch"]
    ))[:2]
    selection = {"selection_metric": "minimum internal-dev teacher-policy regret",
                 "candidate_epochs": [item["epoch"] for item in selected],
                 "candidates": selected, "history": history}
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "candidate_epochs.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n"
    )
    return selection
