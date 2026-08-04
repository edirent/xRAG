"""Data isolation and single-run locks for composition-repair experiments."""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path


COMPOSITION_SEED = 20260804
TRAIN_COUNT = 3999
DEV_COUNT = 250
SHADOW_COUNT = 250
DIAGNOSTIC_COUNT = 150
EXPECTED_EFFECTIVE_TRAIN_HASH = "91a47f422aeebe7f212058330e4c1357651b8c729b62bdec66379df413104840"
EXPECTED_BENCHMARK_HASH = "8f925ff8ababf1efc6bb8a913e6d5431437610b0bb30fa8357a57dfbb5f24052"
QUARANTINED_ID = "5a7b23ca554299042af8f703"


def ordered_ids_sha256(ids):
    return hashlib.sha256("".join(str(value) for value in ids).encode()).hexdigest()


def split_composition_ids(effective_ids, seed=COMPOSITION_SEED):
    ids = list(effective_ids)
    if len(ids) != 4499 or len(ids) != len(set(ids)):
        raise ValueError("composition split requires exactly 4,499 unique effective-train IDs")
    if ordered_ids_sha256(ids) != EXPECTED_EFFECTIVE_TRAIN_HASH:
        raise ValueError("effective-train hash mismatch")
    if QUARANTINED_ID in ids:
        raise RuntimeError("quarantined sample entered composition source")
    random.Random(seed).shuffle(ids)
    train = ids[:TRAIN_COUNT]
    dev = ids[TRAIN_COUNT:TRAIN_COUNT + DEV_COUNT]
    shadow = ids[TRAIN_COUNT + DEV_COUNT:]
    if len(shadow) != SHADOW_COUNT or set(train) & set(dev) or set(train) & set(shadow) or set(dev) & set(shadow):
        raise RuntimeError("composition split overlap/cardinality failure")
    return train, dev, shadow


def diagnostic_ids(dev_ids, seed=COMPOSITION_SEED):
    values = list(dev_ids)
    if len(values) != DEV_COUNT or len(values) != len(set(values)):
        raise ValueError("diagnostic selection requires locked COMPOSITION_DEV")
    random.Random(seed).shuffle(values)
    return values[:DIAGNOSTIC_COUNT]


def normalized_question(value):
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"[^\w]+", "", value, flags=re.UNICODE)


def question_hash(value, normalized=False):
    value = normalized_question(value) if normalized else str(value)
    return hashlib.sha256(value.encode()).hexdigest()


def overlap_audit(named_records):
    views = {}
    for name, records in named_records.items():
        ids = [record["sample_id"] for record in records]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"duplicate sample IDs in {name}")
        views[name] = {"ids": set(ids),
                       "exact": {question_hash(record["question"]) for record in records},
                       "normalized": {question_hash(record["question"], True) for record in records}}
    output = {}
    names = list(named_records)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            output[f"{left}__{right}"] = {
                "sample_id_overlap": sorted(views[left]["ids"] & views[right]["ids"]),
                "exact_question_overlap": sorted(views[left]["exact"] & views[right]["exact"]),
                "normalized_question_overlap": sorted(views[left]["normalized"] & views[right]["normalized"]),
            }
    return output


def assert_no_overlap(audit):
    failures = {name: value for name, value in audit.items() if any(value.values())}
    if failures: raise RuntimeError(f"MANDATORY STOP: composition split overlap: {failures}")
    return True


def assert_evaluation_lock(lock_path, split, maximum_runs):
    path = Path(lock_path)
    payload = json.loads(path.read_text())
    if payload["split"] != split or payload["maximum_runs"] != maximum_runs:
        raise RuntimeError("composition evaluation lock configuration mismatch")
    if payload["runs"] >= maximum_runs:
        raise RuntimeError(f"{split} evaluation budget exhausted")
    return payload

