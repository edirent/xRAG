"""Dynamic rollout, generation metrics, and cached-label mechanism audits."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, median

import torch

from scripts.packet_xrag import run_selector_calibration as selector
from scripts.packet_xrag.run_k2_selector_benchmark import substring_score
from scripts.packet_xrag.run_static_scorer_benchmark import generate_xrag_batch
from src.packet_xrag.controller.generator_utility import utility_label
from src.packet_xrag.controller.utility_rollout import (
    length_distribution, rollout_utility_policy,
)


@torch.inference_mode()
def run_model_rollouts(model, feature_cache, static_scores, clip_value, tau, device):
    policies = {}
    for index in range(len(feature_cache)):
        record = feature_cache[index]; sid = record["sample_id"]
        policies[sid] = rollout_utility_policy(
            model, record, static_scores[sid], clip_value, tau, device
        )
    return policies


def generation_row(record, configuration, policy, raw, prompt_tokens, generated_tokens):
    selected = policy["selected_packet_ids"]
    gold = set(record["gold_packet_ids"]); selected_set = set(selected)
    clean = selector.clean_prediction(raw); short = selector.extract_short_answer(raw) or "[EMPTY]"
    em, f1 = selector.score_prediction(short, record["answer"])
    _, clean_f1 = selector.score_prediction(clean, record["answer"])
    return {
        "sample_id": record["sample_id"], "configuration": configuration,
        "selected_packet_ids": selected,
        "selection_scores": [action["predicted_delta"] for action in policy.get("actions", [])],
        "stop_score_threshold": policy.get("tau"), "num_packets": len(selected),
        "num_soft_tokens": 2 * len(selected), "raw_prediction": raw,
        "short_prediction": short, "clean_prediction": clean,
        "gold_answer": record["answer"], "short_em": em, "short_f1": f1,
        "clean_f1": clean_f1, "substring_match": substring_score(short, record["answer"]),
        "support_recall": len(gold & selected_set) / len(gold),
        "full_support": gold.issubset(selected_set), "is_empty": short == "[EMPTY]",
        "prompt_tokens": prompt_tokens, "generated_tokens": generated_tokens,
        "actions": policy.get("actions", []), "stop": policy.get("stop"),
    }


@torch.inference_mode()
def generate_rollout_answers(feature_cache, policies, tokenizer, generator, xrag_id,
                             device, configuration, batch_size=16, max_new_tokens=32):
    rows = []
    for start in range(0, len(feature_cache), batch_size):
        records = [feature_cache[index] for index in
                   range(start, min(start + batch_size, len(feature_cache)))]
        selected = [policies[record["sample_id"]]["selected_packet_ids"] for record in records]
        embeddings = [record["packet_embeddings"][packet_ids]
                      for record, packet_ids in zip(records, selected)]
        raws, prompt_lengths, generated_lengths = generate_xrag_batch(
            tokenizer, generator, xrag_id, [record["question"] for record in records],
            embeddings, device, max_new_tokens,
        )
        for record, raw, prompt, generated in zip(
                records, raws, prompt_lengths, generated_lengths):
            rows.append(generation_row(
                record, configuration, policies[record["sample_id"]], raw, prompt, generated
            ))
    return rows


def summarize_generation(rows):
    packet_counts = sorted(row["num_packets"] for row in rows)
    return {
        "samples": len(rows), "short_em": 100 * mean(row["short_em"] for row in rows),
        "short_f1": 100 * mean(row["short_f1"] for row in rows),
        "clean_f1": 100 * mean(row["clean_f1"] for row in rows),
        "avg_packets": mean(packet_counts), "median_packets": median(packet_counts),
        "p90_packets": packet_counts[min(len(packet_counts) - 1, int(.9 * len(packet_counts)))],
        "avg_soft_tokens": 2 * mean(packet_counts),
        "empty": sum(row["is_empty"] for row in rows),
        "support_recall": mean(row["support_recall"] for row in rows),
        "full_support": mean(row["full_support"] for row in rows),
        "length_distribution": length_distribution(
            [row["selected_packet_ids"] for row in rows]
        ),
    }


def cached_label_mechanism_audit(policies, label_dataset):
    # The full-cache protocol defines a state as the *ordered* selected tuple.
    # Do not reuse the earlier feasibility-cache key, which canonicalizes sets.
    def ordered_key(sample_id, selected, candidate_id):
        return str(sample_id), tuple(selected), int(candidate_id)

    lookup = {ordered_key(row["sample_id"], row["selected_packet_ids"],
                          row["candidate_packet_id"]): row
              for row in label_dataset.rows}
    state_rows = defaultdict(list)
    for row in label_dataset.rows:
        state_rows[(row["sample_id"], tuple(row["selected_packet_ids"]))].append(row)
    audited_actions, total_actions = [], 0
    decision_records = []; stop_records = []
    for sid, policy in policies.items():
        for action in policy["actions"]:
            total_actions += 1
            selected = action["selected_packet_ids_before"]
            key = ordered_key(sid, selected, action["packet_id"])
            row = lookup.get(key)
            group = state_rows.get((sid, tuple(selected)))
            if row is not None and group:
                audited_actions.append(row)
                best = max(group, key=lambda item: (item["delta_utility"],
                                                    -item["candidate_packet_id"]))
                decision_records.append({"predicted_stop": False,
                                         "teacher_stop": best["delta_utility"] <= 0,
                                         "regret": max(0.0, best["delta_utility"]) - row["delta_utility"],
                                         "best_agreement": row["candidate_packet_id"] == best["candidate_packet_id"]})
        stop = policy["stop"]
        candidate = stop.get("highest_remaining_packet_id") if stop else None
        selected = policy["selected_packet_ids"]
        group = state_rows.get((sid, tuple(selected)))
        row = lookup.get(ordered_key(sid, selected, candidate)) if candidate is not None else None
        if row is not None and group:
            best = max(group, key=lambda item: (item["delta_utility"],
                                                -item["candidate_packet_id"]))
            stop_records.append(row)
            decision_records.append({"predicted_stop": True,
                                     "teacher_stop": best["delta_utility"] <= 0,
                                     "regret": max(0.0, best["delta_utility"]),
                                     "best_agreement": best["delta_utility"] <= 0})
    labels = [utility_label(row["delta_utility"]) for row in audited_actions]
    predicted_stops = [item for item in decision_records if item["predicted_stop"]]
    teacher_stops = [item for item in decision_records if item["teacher_stop"]]
    true_stops = [item for item in decision_records if item["predicted_stop"] and item["teacher_stop"]]
    coverage = len(audited_actions) / total_actions if total_actions else 0.0
    return {
        "utility_auditable_action_coverage": coverage,
        "coverage_requirement_met": coverage >= .50,
        "selected_actions_total": total_actions, "selected_actions_audited": len(audited_actions),
        "harmful_addition_rate": labels.count("negative") / len(labels) if labels else None,
        "near_zero_addition_rate": labels.count("near-zero") / len(labels) if labels else None,
        "positive_addition_rate": labels.count("positive") / len(labels) if labels else None,
        "missed_positive_utility_at_stop": (sum(row["delta_utility"] > .02 for row in stop_records) /
                                             len(stop_records) if stop_records else None),
        "teacher_policy_regret": mean(item["regret"] for item in decision_records) if decision_records else None,
        "teacher_best_action_agreement": mean(item["best_agreement"] for item in decision_records) if decision_records else None,
        "stop_precision": len(true_stops) / len(predicted_stops) if predicted_stops else None,
        "stop_recall": len(true_stops) / len(teacher_stops) if teacher_stops else None,
        "utility_not_available_actions": total_actions - len(audited_actions),
        "limitation": None if coverage >= .50 else "Below 50% preregistered-state action coverage",
    }


def choose_rollout_configuration(items):
    best_f1 = max(item["metrics"]["short_f1"] for item in items)
    close = [item for item in items if best_f1 - item["metrics"]["short_f1"] < .25]
    return min(close, key=lambda item: (
        item["metrics"]["avg_packets"], -item["tau"], item["epoch"]
    ))
