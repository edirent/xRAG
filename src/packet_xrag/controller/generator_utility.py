"""Frozen-generator marginal-utility helpers.

This module deliberately contains no training path.  It builds deterministic
candidate/state panels and measures mean teacher-forced answer-token NLL under
``torch.inference_mode``.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from scripts.packet_xrag import train_packet_projector as v1


UTILITY_POSITIVE_THRESHOLD = 0.02
UTILITY_NEGATIVE_THRESHOLD = -0.02
TOKENS_PER_PACKET = 2


def ordered_ids_sha256(ids: Sequence[str]) -> str:
    return hashlib.sha256("".join(ids).encode("utf-8")).hexdigest()


def utility_label(delta: float) -> str:
    if delta > UTILITY_POSITIVE_THRESHOLD:
        return "positive"
    if delta < UTILITY_NEGATIVE_THRESHOLD:
        return "negative"
    return "near-zero"


def delta_utility(base_answer_nll: float, candidate_answer_nll: float) -> float:
    """Positive values mean that appending the candidate reduced answer NLL."""
    return float(base_answer_nll) - float(candidate_answer_nll)


def utility_cache_key(sample_id: str, selected: Sequence[int], candidate_packet_id: int) -> tuple:
    """Canonical unique key shared by the builder and validity audit."""
    return str(sample_id), tuple(sorted(selected)), int(candidate_packet_id)


def candidate_addition_groups(selected: Sequence[int], candidates: Sequence[int]) -> list[list[int]]:
    """Return the base state followed by valid candidate-appended states."""
    if len(selected) != len(set(selected)):
        raise ValueError("selected state contains duplicate packet IDs")
    selected_set = set(selected)
    remaining = [packet_id for packet_id in candidates if packet_id not in selected_set]
    if len(remaining) != len(set(remaining)):
        raise ValueError("candidate pool contains duplicate packet IDs")
    return [list(selected)] + [list(selected) + [packet_id] for packet_id in remaining]


def _balanced_take(items: Sequence[dict], count: int, seed: int, label: str) -> list[dict]:
    """Take an approximately correct/wrong-balanced deterministic sample."""
    groups = {
        True: [item for item in items if item["static_correct"]],
        False: [item for item in items if not item["static_correct"]],
    }
    for correct, group in groups.items():
        random.Random(f"{seed}:{label}:{int(correct)}").shuffle(group)
    desired_true = min(count // 2, len(groups[True]))
    desired_false = min(count - desired_true, len(groups[False]))
    selected = groups[True][:desired_true] + groups[False][:desired_false]
    if len(selected) < count:
        used = {item["sample_id"] for item in selected}
        remainder = [item for item in items if item["sample_id"] not in used]
        random.Random(f"{seed}:{label}:fill").shuffle(remainder)
        selected.extend(remainder[: count - len(selected)])
    random.Random(f"{seed}:{label}:order").shuffle(selected)
    return selected


def choose_feasibility_records(
    records: Sequence[dict], static_correct: dict[str, bool], count: int = 200,
    seed: int = 20260803,
) -> list[dict]:
    """Select the single/multi-gold and STATIC-correct/wrong stratified subset."""
    if count <= 0 or count > len(records):
        raise ValueError("invalid feasibility subset size")
    annotated = []
    for record in records:
        sid = record["sample_id"]
        if sid not in static_correct:
            raise ValueError(f"missing STATIC_2 outcome for {sid}")
        annotated.append({
            "sample_id": sid,
            "record": record,
            "single_gold": len(set(record["gold_packet_ids"])) == 1,
            "static_correct": bool(static_correct[sid]),
        })
    single = [item for item in annotated if item["single_gold"]]
    multi = [item for item in annotated if not item["single_gold"]]
    single_count = min(100, len(single), count)
    multi_count = min(100, len(multi), count - single_count)
    selected = (
        _balanced_take(single, single_count, seed, "single")
        + _balanced_take(multi, multi_count, seed, "multi")
    )
    if len(selected) < count:
        used = {item["sample_id"] for item in selected}
        remainder = [item for item in annotated if item["sample_id"] not in used]
        selected.extend(_balanced_take(remainder, count - len(selected), seed, "fill"))
    if len(selected) != count or len({item["sample_id"] for item in selected}) != count:
        raise AssertionError("feasibility subset selection failed")
    random.Random(f"{seed}:final-order").shuffle(selected)
    return selected


def _source_memberships(record: dict, static_ranking: Sequence[int], seed: int) -> dict[int, set[str]]:
    gold = set(record["gold_packet_ids"])
    memberships: dict[int, set[str]] = defaultdict(set)
    for packet_id in record["gold_packet_ids"]:
        memberships[packet_id].add("gold")
    for name, ranking in (
        ("STATIC", static_ranking), ("TOPK", record["topk_ranking"]),
        ("MMR", record["mmr_ranking"]),
    ):
        for packet_id in ranking[:4]:
            memberships[packet_id].add(name)
    support_docs = {record["packets"][packet_id]["doc_id"] for packet_id in gold}
    same_document = [
        packet["packet_id"] for packet in record["packets"]
        if packet["packet_id"] not in gold and packet["doc_id"] in support_docs
    ][:2]
    for packet_id in same_document:
        memberships[packet_id].add("same-document")
    random_pool = [
        packet["packet_id"] for packet in record["packets"]
        if packet["packet_id"] not in gold
    ]
    random.Random(f"{seed}:{record['sample_id']}:random-negative").shuffle(random_pool)
    for packet_id in random_pool[:2]:
        memberships[packet_id].add("random")
    return memberships


def build_candidate_pool(
    record: dict, static_scores: Sequence[float], static_ranking: Sequence[int],
    seed: int = 20260803, max_candidates: int = 12,
) -> list[dict]:
    """Build the locked, gold-preserving, at-most-12 candidate pool."""
    if len(static_scores) != record["packet_count"]:
        raise ValueError("STATIC scores do not align with packets")
    memberships = _source_memberships(record, static_ranking, seed)
    gold = list(dict.fromkeys(record["gold_packet_ids"]))
    support_docs = {record["packets"][packet_id]["doc_id"] for packet_id in gold}
    same_document = [
        packet["packet_id"] for packet in record["packets"]
        if packet["packet_id"] not in set(gold) and packet["doc_id"] in support_docs
    ][:2]
    random_ids = sorted(
        packet_id for packet_id, tags in memberships.items() if "random" in tags
    )
    priority_groups = [
        gold, list(static_ranking[:4]), list(record["topk_ranking"][:4]),
        list(record["mmr_ranking"][:4]), same_document, random_ids,
    ]
    ordered = []
    for group in priority_groups:
        for packet_id in group:
            if packet_id not in ordered:
                ordered.append(packet_id)
    if len(gold) > max_candidates:
        raise ValueError("gold packets exceed candidate cap")
    kept = ordered[:max_candidates]
    missing_gold = set(gold) - set(kept)
    if missing_gold:
        raise AssertionError(f"candidate cap removed gold packets: {sorted(missing_gold)}")
    static_rank = {packet_id: rank + 1 for rank, packet_id in enumerate(static_ranking)}
    topk_rank = {packet_id: rank + 1 for rank, packet_id in enumerate(record["topk_ranking"])}
    mmr_rank = {packet_id: rank + 1 for rank, packet_id in enumerate(record["mmr_ranking"])}
    gold_set = set(gold)
    candidates = []
    for packet_id in kept:
        packet = record["packets"][packet_id]
        candidates.append({
            "packet_id": packet_id,
            "doc_id": int(packet["doc_id"]),
            "title": packet["title"],
            "sentence_id": int(packet["sentence_id"]),
            "text": packet["text"],
            "is_gold": packet_id in gold_set,
            "static_rank": static_rank.get(packet_id),
            "topk_rank": topk_rank.get(packet_id),
            "mmr_rank": mmr_rank.get(packet_id),
            "query_cosine": float(record["topk_scores"][packet_id]),
            "static_score": float(static_scores[packet_id]),
            "source_tags": sorted(memberships[packet_id]),
        })
    return candidates


def build_state_pool(record: dict, static_ranking: Sequence[int]) -> list[dict]:
    """Construct and deduplicate the five preregistered selected-set states."""
    gold = list(dict.fromkeys(record["gold_packet_ids"]))
    gold_set = set(gold)
    raw_states: list[tuple[str, list[int]]] = [
        ("S0_EMPTY", []),
        ("S_GOLD1", [gold[0]]),
        ("S_STATIC1", [static_ranking[0]]),
    ]
    wrong = next((packet_id for packet_id in static_ranking if packet_id not in gold_set), None)
    if wrong is None:
        wrong = next((packet_id for packet_id in record["topk_ranking"] if packet_id not in gold_set), None)
    if wrong is None:
        wrong = next((packet["packet_id"] for packet in record["packets"]
                      if packet["packet_id"] not in gold_set), None)
    if wrong is not None:
        raw_states.append(("S_WRONG1", [wrong]))
    raw_states.append(("S_SUFFICIENT", sorted(gold_set)))
    merged: dict[tuple[int, ...], dict] = {}
    for source_tag, selected in raw_states:
        if len(selected) != len(set(selected)):
            raise ValueError("state contains duplicate packet IDs")
        key = tuple(sorted(selected))
        if key not in merged:
            merged[key] = {
                "state_id": source_tag,
                "selected_packet_ids": list(selected),
                "state_source_tags": [],
            }
        merged[key]["state_source_tags"].append(source_tag)
    states = []
    for state in merged.values():
        selected_set = set(state["selected_packet_ids"])
        recall = len(gold_set & selected_set) / len(gold_set)
        state.update({
            "state_source_tags": sorted(state["state_source_tags"]),
            "num_selected": len(selected_set),
            "gold_support_recall": recall,
            "full_gold_support": gold_set.issubset(selected_set),
        })
        states.append(state)
    if not any("S0_EMPTY" in state["state_source_tags"] for state in states):
        raise AssertionError("S0 state is mandatory")
    if len(states) > 5:
        raise AssertionError("state cap exceeded")
    return states


def build_gold_answer_inputs(tokenizer, xrag_token_id: int, question: str, answer: str,
                             num_packets: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the exact projector-training prompt and gold-answer label mask."""
    prompt = v1.build_prompt(question, num_packets * TOKENS_PER_PACKET)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + " " + answer, add_special_tokens=False)["input_ids"]
    full_ids.append(tokenizer.eos_token_id)
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise RuntimeError("gold answer tokenization changed the prompt prefix")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    if sum(value != -100 for value in labels) < 2:
        raise RuntimeError("gold-answer mask must include answer and EOS")
    if sum(value == xrag_token_id for value in full_ids) != num_packets * TOKENS_PER_PACKET:
        raise RuntimeError("XRAG token count does not equal 2 * selected packets")
    return torch.tensor(full_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def mean_answer_nll_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return one mean shifted-token NLL per batch row."""
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("logits and labels must be batch/sequence aligned")
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    losses = F.cross_entropy(
        shifted_logits.transpose(1, 2), shifted_labels, ignore_index=-100,
        reduction="none",
    )
    mask = shifted_labels.ne(-100)
    counts = mask.sum(dim=1)
    if not bool((counts > 0).all()):
        raise RuntimeError("empty answer-token mask")
    return (losses * mask).sum(dim=1) / counts


@torch.inference_mode()
def gold_answer_nll_batch(
    model, tokenizer, xrag_token_id: int, question: str, answer: str,
    packet_embeddings: torch.Tensor, selected_groups: Sequence[Sequence[int]],
    device: torch.device,
) -> list[float]:
    """Measure exact mean gold-answer NLL for aligned selected packet sets."""
    if not selected_groups:
        raise ValueError("selected_groups must not be empty")
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("generator contains trainable parameters")
    pairs = [
        build_gold_answer_inputs(tokenizer, xrag_token_id, question, answer, len(selected))
        for selected in selected_groups
    ]
    input_ids = pad_sequence(
        [pair[0] for pair in pairs], batch_first=True,
        padding_value=tokenizer.pad_token_id, padding_side="left",
    ).to(device)
    labels = pad_sequence(
        [pair[1] for pair in pairs], batch_first=True,
        padding_value=-100, padding_side="left",
    ).to(device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    retrieval_groups = [packet_embeddings[list(selected)] for selected in selected_groups if selected]
    retrieval = torch.cat(retrieval_groups).to(device) if retrieval_groups else None
    expected_xrag = TOKENS_PER_PACKET * sum(len(selected) for selected in selected_groups)
    if int(input_ids.eq(xrag_token_id).sum()) != expected_xrag:
        raise RuntimeError("batched XRAG token count mismatch")
    outputs = model(
        input_ids=input_ids, attention_mask=attention_mask,
        retrieval_embeds=retrieval,
    )
    values = mean_answer_nll_from_logits(outputs.logits, labels)
    if not bool(torch.isfinite(values).all()):
        raise RuntimeError("non-finite gold-answer NLL")
    return [float(value) for value in values.cpu()]


def static_utility_rollout(
    s0_utilities: dict[int, float], fixed_k: int | None = None,
    min_packets: int = 1, max_packets: int = 6,
) -> tuple[list[int], list[dict]]:
    """Select by frozen S0 utility, optionally at a fixed packet budget."""
    ranking = sorted(s0_utilities, key=lambda packet_id: (-s0_utilities[packet_id], packet_id))
    limit = min(max_packets, len(ranking))
    if fixed_k is not None:
        limit = min(fixed_k, limit)
    selected, actions = [], []
    for packet_id in ranking[:limit]:
        delta = float(s0_utilities[packet_id])
        if fixed_k is None and len(selected) >= min_packets and delta <= 0:
            break
        selected.append(packet_id)
        actions.append({"packet_id": packet_id, "delta_utility": delta})
    if not selected and ranking and min_packets:
        packet_id = ranking[0]
        selected = [packet_id]
        actions = [{"packet_id": packet_id, "delta_utility": float(s0_utilities[packet_id])}]
    return selected, actions


def choose_state_utility_action(
    current_utilities: dict[int, float], selected_count: int,
    min_packets: int = 1,
) -> tuple[int | None, float | None]:
    """Apply the deterministic state-utility STOP rule for one rollout step."""
    if not current_utilities:
        return None, None
    packet_id = min(current_utilities, key=lambda item: (-current_utilities[item], item))
    delta = float(current_utilities[packet_id])
    if selected_count >= min_packets and delta <= 0:
        return None, delta
    return packet_id, delta


def paired_bootstrap(
    left: Sequence[dict], right: Sequence[dict], samples: int = 10_000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap for Short F1 and packet-count differences."""
    left_by_id = {row["sample_id"]: row for row in left}
    right_by_id = {row["sample_id"]: row for row in right}
    if set(left_by_id) != set(right_by_id) or len(left_by_id) != len(left):
        raise ValueError("paired bootstrap sample alignment failed")
    ids = sorted(left_by_id)
    if not ids:
        raise ValueError("paired bootstrap requires samples")
    f1 = torch.tensor([
        100.0 * (left_by_id[sid]["short_f1"] - right_by_id[sid]["short_f1"])
        for sid in ids
    ], dtype=torch.float64)
    packets = torch.tensor([
        left_by_id[sid]["num_packets"] - right_by_id[sid]["num_packets"]
        for sid in ids
    ], dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(ids), (samples, len(ids)), generator=generator)
    f1_boot = f1[indices].mean(dim=1)
    packet_boot = packets[indices].mean(dim=1)
    quantiles = torch.tensor([0.025, 0.975], dtype=torch.float64)
    f1_ci = torch.quantile(f1_boot, quantiles)
    packet_ci = torch.quantile(packet_boot, quantiles)
    return {
        "sample_count": len(ids), "bootstrap_samples": samples, "seed": seed,
        "short_f1_delta": float(f1.mean()),
        "short_f1_ci95": [float(f1_ci[0]), float(f1_ci[1])],
        "p_delta_gt_0": float((f1_boot > 0).double().mean()),
        "p_delta_ge_1": float((f1_boot >= 1).double().mean()),
        "p_delta_ge_2": float((f1_boot >= 2).double().mean()),
        "avg_packet_delta": float(packets.mean()),
        "avg_packet_delta_ci95": [float(packet_ci[0]), float(packet_ci[1])],
    }
