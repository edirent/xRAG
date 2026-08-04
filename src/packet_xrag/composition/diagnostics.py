"""Deterministic evidence-set perturbations for composition diagnostics."""

from __future__ import annotations

import random


def document_order(record, packet_ids):
    return sorted(packet_ids, key=lambda index: (
        int(record["packets"][index]["doc_id"]),
        int(record["packets"][index]["sentence_id"]), index))


def fixed_permutation(packet_ids, sample_id, permutation_index, seed=20260804):
    values = list(packet_ids)
    random.Random(f"{seed}:{sample_id}:{permutation_index}").shuffle(values)
    return values


def duplicate_stress_sets(record, base_ids, seed=20260804):
    """Append 1/2/4 copies of fixed gold, non-gold, and random distractors."""
    base = list(base_ids); gold = list(record["gold_packet_ids"]); gold_set = set(gold)
    top_non_gold = next((index for index in record["topk_ranking"] if index not in gold_set), None)
    distractors = [index for index in range(record["packet_count"])
                   if index not in gold_set and index != top_non_gold]
    random.Random(f"{seed}:{record['sample_id']}:random-distractor").shuffle(distractors)
    targets = {"GOLD": gold[0], "NONGOLD": top_non_gold, "RANDOM": distractors[0]}
    output = {}
    for label, packet_id in targets.items():
        if packet_id is None: raise RuntimeError(f"no {label} duplicate target")
        for count in (1, 2, 4): output[f"DUP_{label}_X{count}"] = base + [packet_id] * count
    return output

