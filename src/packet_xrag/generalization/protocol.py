"""Deterministic split, hashing, locking, and leakage rules."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from src.packet_xrag.data.base_qa_adapter import normalized_question, stable_hash


SEED = 20260804
MAX_PACKETS = 48
FORBIDDEN_INFERENCE_FIELDS = {"answer", "answers", "gold_answer", "gold_answer_tokens",
    "support_labels", "contains_answer", "is_support", "is_supporting", "true_utility",
    "gold_answer_nll", "benchmark_metric", "gold_packet_ids"}


def ordered_hash(values):
    return hashlib.sha256("".join(str(value) for value in values).encode()).hexdigest()


def deterministic_partition(train_ids, validation_ids, train_count=3999, dev_count=250,
                            shadow_count=250, benchmark_count=500, seed=SEED):
    train_ids, validation_ids = list(train_ids), list(validation_ids)
    if len(train_ids) != len(set(train_ids)) or len(validation_ids) != len(set(validation_ids)):
        raise ValueError("dataset split contains duplicate sample IDs")
    if set(train_ids) & set(validation_ids):
        raise RuntimeError("official train/validation sample-ID overlap")
    shuffled = list(train_ids); random.Random(seed).shuffle(shuffled)
    needed = train_count + dev_count + shadow_count
    if len(shuffled) < needed or len(validation_ids) < benchmark_count:
        raise ValueError("dataset is too small for preregistered split sizes")
    benchmark = sorted(validation_ids, key=lambda value: stable_hash(f"{seed}:{value}"))[:benchmark_count]
    return {"train": shuffled[:train_count],
            "dev": shuffled[train_count:train_count + dev_count],
            "shadow": shuffled[train_count + dev_count:needed], "benchmark": benchmark}


def split_audit(named_samples):
    views = {}
    for name, samples in named_samples.items():
        ids = [str(sample["id"]) for sample in samples]
        questions = [sample["question"] for sample in samples]
        documents = {stable_hash(str(document.get("title", document["document_id"])).casefold())
                     for sample in samples for document in sample["documents"]}
        answers = {answer.casefold() for sample in samples for answer in sample["answers"]}
        views[name] = {"ids": set(ids), "exact": {stable_hash(value) for value in questions},
                       "normalized": {stable_hash(normalized_question(value)) for value in questions},
                       "documents": documents, "answers": answers}
    output = {}
    names = list(named_samples)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            output[f"{left}__{right}"] = {
                "sample_id_overlap": len(views[left]["ids"] & views[right]["ids"]),
                "exact_question_overlap": len(views[left]["exact"] & views[right]["exact"]),
                "normalized_question_overlap": len(views[left]["normalized"] & views[right]["normalized"]),
                "context_document_overlap": len(views[left]["documents"] & views[right]["documents"]),
                "answer_overlap": len(views[left]["answers"] & views[right]["answers"])}
    failures = {key: value for key, value in output.items()
                if value["sample_id_overlap"] or value["exact_question_overlap"] or
                value["normalized_question_overlap"]}
    if failures:
        raise RuntimeError(f"cross-dataset split leakage: {failures}")
    return output


def assert_inference_view(view):
    overlap = FORBIDDEN_INFERENCE_FIELDS & set(view)
    if overlap:
        raise RuntimeError(f"supervision leaked into inference view: {sorted(overlap)}")
    return True


def consume_single_run_lock(path, split, checkpoint_hash=None):
    path = Path(path); payload = json.loads(path.read_text())
    if payload != {"split": split, "runs": 0, "maximum_runs": 1}:
        raise RuntimeError(f"{split} single-run lock is not pristine")
    payload["runs"] = 1
    if checkpoint_hash: payload["checkpoint_sha256"] = checkpoint_hash
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def assert_manifest_immutable(path, expected_hash):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise RuntimeError("frozen evaluation manifest changed")
    return True
