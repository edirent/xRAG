"""Deployable features and metrics for autonomous controller discovery."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from statistics import mean

import torch

from src.packet_xrag.controller.utility_rollout import (
    candidate_relation_features, state_scalar_features,
)


TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
CONTROL_TOKEN_RE = re.compile(r"<[^<>]{1,32}>")


def word_tokens(text):
    return set(TOKEN_RE.findall(str(text).casefold()))


def sanitize_provisional_answer(text):
    """Prevent generated control-token strings from becoming prompt controls."""
    value = " ".join(CONTROL_TOKEN_RE.sub(" ", str(text)).split()).strip()
    return value or "unknown"


def overlap(left, right):
    left, right = word_tokens(left), word_tokens(right)
    return len(left & right) / max(1, len(left | right))


def repetition_fraction(tokens):
    tokens = list(tokens)
    return 1.0 - len(set(tokens)) / len(tokens) if tokens else 0.0


def retriever_state_features(record, selected_ids, static_scores):
    """Small state vector available without an extra generator invocation."""
    selected = list(selected_ids); selected_set = set(selected)
    remaining = [index for index in range(record["packet_count"])
                 if index not in selected_set]
    ranked_static = sorted((float(static_scores[index]) for index in remaining), reverse=True)
    ranked_cosine = sorted((float(record["topk_scores"][index]) for index in remaining),
                           reverse=True)
    selected_static = [float(static_scores[index]) for index in selected]
    selected_cosine = [float(record["topk_scores"][index]) for index in selected]
    return state_scalar_features(record, selected, static_scores) + [
        ranked_static[0] if ranked_static else 0.0,
        ranked_static[0] - ranked_static[1] if len(ranked_static) > 1 else 0.0,
        mean(ranked_static[:3]) if ranked_static else 0.0,
        ranked_cosine[0] if ranked_cosine else 0.0,
        ranked_cosine[0] - ranked_cosine[1] if len(ranked_cosine) > 1 else 0.0,
        mean(ranked_cosine[:3]) if ranked_cosine else 0.0,
        mean(selected_static) if selected_static else 0.0,
        mean(selected_cosine) if selected_cosine else 0.0,
    ]


GENERATOR_NUMERIC_KEYS = (
    "mean_token_logprob", "minimum_token_logprob", "mean_token_entropy",
    "mean_top1_top2_margin", "mean_eos_probability", "generated_length",
    "empty_indicator", "repetition_fraction",
)


def generator_numeric_features(row):
    return [float(row[key]) for key in GENERATOR_NUMERIC_KEYS]


def candidate_retriever_features(record, candidate_id, selected_ids, static_scores):
    packet = record["packets"][candidate_id]
    return retriever_state_features(record, selected_ids, static_scores) + [
        float(static_scores[candidate_id]), float(record["topk_scores"][candidate_id]),
        int(packet["doc_id"]) / 10.0, int(packet["sentence_id"]) / 20.0,
        *candidate_relation_features(record, candidate_id, selected_ids),
    ]


def candidate_text_features(record, candidate_id, selected_ids, provisional_answer,
                            static_scores):
    """Answer-conditioned lexical relations with no annotation-derived input."""
    candidate = record["packets"][candidate_id]
    selected = [record["packets"][index] for index in selected_ids]
    candidate_text = f"{candidate['title']} {candidate['text']}"
    selected_text = " ".join(f"{item['title']} {item['text']}" for item in selected)
    answer = str(provisional_answer)
    answer_tokens = word_tokens(answer)
    candidate_tokens = word_tokens(candidate_text)
    selected_tokens = word_tokens(selected_text)
    negations = {"no", "not", "never", "none", "neither", "nor", "n't"}
    maximum_selected_overlap = max(
        (overlap(candidate_text, f"{item['title']} {item['text']}") for item in selected),
        default=0.0,
    )
    return [
        overlap(candidate_text, record["question"]),
        overlap(candidate_text, answer),
        overlap(candidate_text, selected_text),
        maximum_selected_overlap,
        len(answer_tokens & candidate_tokens) / max(1, len(answer_tokens)),
        len(candidate_tokens & selected_tokens) / max(1, len(candidate_tokens)),
        float(bool(answer.strip()) and answer.casefold() in candidate_text.casefold()),
        float(bool((candidate_tokens & negations)) != bool((answer_tokens & negations))),
        min(len(candidate_tokens), 100.0) / 100.0,
        float(static_scores[candidate_id]),
        float(record["topk_scores"][candidate_id]),
        *candidate_relation_features(record, candidate_id, selected_ids),
    ]


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
    if len(left) < 2: return 0.0
    left, right = average_ranks(left), average_ranks(right)
    lm, rm = mean(left), mean(right)
    numerator = sum((a - lm) * (b - rm) for a, b in zip(left, right))
    denominator = math.sqrt(sum((a - lm) ** 2 for a in left) *
                            sum((b - rm) ** 2 for b in right))
    return numerator / denominator if denominator else 0.0


def binary_auc(scores, labels):
    positives = [index for index, value in enumerate(labels) if value]
    negatives = [index for index, value in enumerate(labels) if not value]
    if not positives or not negatives: return 0.0
    ranks = average_ranks(scores)
    rank_sum = sum(ranks[index] for index in positives)
    return ((rank_sum - len(positives) * (len(positives) + 1) / 2) /
            (len(positives) * len(negatives)))


def average_precision(scores, labels):
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    total = sum(bool(value) for value in labels)
    if not total: return 0.0
    correct, value = 0, 0.0
    for rank, index in enumerate(order, 1):
        if labels[index]:
            correct += 1; value += correct / rank
    return value / total


def expected_calibration_error(probabilities, labels, bins=10):
    total, value = len(labels), 0.0
    for bin_id in range(bins):
        lower, upper = bin_id / bins, (bin_id + 1) / bins
        indices = [index for index, prob in enumerate(probabilities)
                   if lower <= prob < upper or (bin_id == bins - 1 and prob == 1)]
        if indices:
            confidence = mean(probabilities[index] for index in indices)
            accuracy = mean(float(labels[index]) for index in indices)
            value += len(indices) / total * abs(confidence - accuracy)
    return value


def stop_metrics(probabilities, labels):
    predicted = [value >= .5 for value in probabilities]
    tp = sum(a and p for a, p in zip(labels, predicted)); tn = sum(not a and not p for a, p in zip(labels, predicted))
    false_stops = sum(not actual and value for actual, value in zip(labels, predicted))
    unnecessary_continues = sum(actual and not value for actual, value in zip(labels, predicted))
    positives = sum(labels); negatives = len(labels) - positives
    return {
        "auroc": binary_auc(probabilities, labels),
        "auprc": average_precision(probabilities, labels),
        "balanced_accuracy": .5 * (tp / positives + tn / negatives),
        "accuracy": mean(float(a == p) for a, p in zip(labels, predicted)),
        "false_stop_rate": false_stops / negatives,
        "unnecessary_continue_rate": unnecessary_continues / positives,
        "ece_10": expected_calibration_error(probabilities, labels),
        "positive_rate": positives / len(labels), "states": len(labels),
    }


def candidate_metrics(predictions, rows):
    groups = defaultdict(list)
    for prediction, row in zip(predictions, rows):
        groups[(row["sample_id"], tuple(row["selected_packet_ids"]))].append(
            (float(prediction), float(row["delta_utility"]), row["candidate_packet_id"]))
    rhos, regrets, top1, top3, harm = [], [], 0, 0, 0
    pair_correct, pair_total = 0, 0
    for items in groups.values():
        pred = [item[0] for item in items]; true = [item[1] for item in items]
        rhos.append(spearman(pred, true))
        chosen = max(range(len(items)), key=lambda index: (pred[index], -items[index][2]))
        best = max(range(len(items)), key=lambda index: (true[index], -items[index][2]))
        top1 += chosen == best
        predicted_top3 = sorted(range(len(items)), key=lambda index: (-pred[index], items[index][2]))[:3]
        top3 += best in predicted_top3
        for left in range(len(items)):
            for right in range(left + 1, len(items)):
                if true[left] != true[right]:
                    pair_correct += (pred[left] - pred[right]) * (true[left] - true[right]) > 0
                    pair_total += 1
        regrets.append(max(0.0, true[best]) - true[chosen])
        harm += true[chosen] < -.02
    def sign(value):
        return 1 if value > .02 else -1 if value < -.02 else 0
    return {"within_state_spearman": mean(rhos),
            "pairwise_ranking_accuracy": pair_correct / pair_total if pair_total else 0.0,
            "best_action_top1_accuracy": top1 / len(groups),
            "best_action_top3_recall": top3 / len(groups),
            "utility_sign_accuracy": mean(float(sign(prediction) == sign(row["delta_utility"]))
                                          for prediction, row in zip(predictions, rows)),
            "teacher_policy_regret": mean(regrets),
            "harmful_choice_rate": harm / len(groups), "states": len(groups),
            "labels": len(rows)}


def fit_torch_probe(train_x, train_y, eval_x, hidden=False, classification=True,
                    seed=20260804, epochs=300):
    """Deterministic standardized linear/MLP probe used only for feasibility."""
    torch.manual_seed(seed)
    train_x = torch.as_tensor(train_x, dtype=torch.float32)
    eval_x = torch.as_tensor(eval_x, dtype=torch.float32)
    train_y = torch.as_tensor(train_y, dtype=torch.float32)
    center = train_x.mean(0); scale = train_x.std(0).clamp_min(1e-5)
    train_x = (train_x - center) / scale; eval_x = (eval_x - center) / scale
    layers = ([torch.nn.Linear(train_x.shape[1], 64), torch.nn.ReLU(),
               torch.nn.Dropout(.1), torch.nn.Linear(64, 1)] if hidden else
              [torch.nn.Linear(train_x.shape[1], 1)])
    model = torch.nn.Sequential(*layers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01 if not hidden else .003,
                                  weight_decay=.01)
    if classification:
        positives = float(train_y.sum()); negatives = len(train_y) - positives
        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(negatives / max(1.0, positives)))
    else:
        loss_fn = torch.nn.HuberLoss(delta=.1)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        output = model(train_x).squeeze(-1)
        loss = loss_fn(output, train_y)
        loss.backward(); optimizer.step()
    model.eval()
    with torch.inference_mode():
        output = model(eval_x).squeeze(-1)
        if classification: output = output.sigmoid()
    return output.tolist(), {"feature_center": center.tolist(), "feature_scale": scale.tolist(),
                             "state_dict": {key: value.detach().tolist()
                                            for key, value in model.state_dict().items()}}
